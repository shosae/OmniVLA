#!/usr/bin/env python3
import sys
import os
import queue
import time
import math
import re
import glob
import faulthandler
import io
import requests
import threading

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

DEBUG_LOG_PATH = os.environ.get("OMNIVLA_DEBUG_LOG", "/tmp/omnivla_gui_debug.log")
_DEBUG_LOG_FILE = open(DEBUG_LOG_PATH, "a", buffering=1)
faulthandler.enable(_DEBUG_LOG_FILE, all_threads=True)

def debug_log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    _DEBUG_LOG_FILE.write(line + "\n")
    _DEBUG_LOG_FILE.flush()

def preload_jpeg_compat():
    import ctypes
    candidates = [
        os.environ.get("OMNIVLA_JPEG_COMPAT_LIB"),
        "/usr/NX/lib/libjpeg.so",
        "/usr/lib/aarch64-linux-gnu/libjpeg.so.8",
        "/usr/lib/aarch64-linux-gnu/libjpeg.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not os.path.exists(candidate):
            continue
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return candidate
        except OSError:
            continue
    return None

preload_jpeg_compat()

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from PyQt5 import QtWidgets, QtGui, QtCore

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.task import Future
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ROSImage
    from std_srvs.srv import SetBool
except ImportError:
    raise ImportError("ROS2 dependencies are required for this integrated client")

class OmniVlaNode(Node):
    def __init__(self):
        super().__init__("omnivla_gui_node")
        self.bridge = CvBridge()
        self.latest_frame_bgr = None
        
        self.image_source = os.environ.get("OMNIVLA_IMAGE_SOURCE", "camera").lower()
        self.image_topic = os.environ.get("OMNIVLA_IMAGE_TOPIC", "/camera/image_raw")
        self.cmd_vel_topic = os.environ.get("OMNIVLA_CMD_VEL_TOPIC", "/cmd_vel")
        self.drive_enable_service = os.environ.get(
            "OMNIVLA_DRIVE_ENABLE_SERVICE",
            "/omni_drive_controller/set_enabled",
        )

        self.image_sub = None
        if self.image_source == "topic":
            self.image_sub = self.create_subscription(
                ROSImage,
                self.image_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.drive_enable_client = self.create_client(SetBool, self.drive_enable_service)

    def image_callback(self, msg: ROSImage):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_frame_bgr = frame_bgr.copy()
        except Exception as e:
            self.get_logger().error(f"cv_bridge exception: {e}")

class CameraCapture:
    def __init__(self):
        self.cap = None
        self.read_count = 0

    def open(self):
        preloaded_jpeg_lib = preload_jpeg_compat()
        if preloaded_jpeg_lib is None:
            debug_log("Warning: Failed to preload a libjpeg compatibility library; nvarguscamerasrc may fail to load")

        width = int(os.environ.get("OMNIVLA_CAMERA_WIDTH", "1280"))
        height = int(os.environ.get("OMNIVLA_CAMERA_HEIGHT", "720"))
        fps = os.environ.get("OMNIVLA_CAMERA_FPS")
        rotate = os.environ.get("OMNIVLA_CAMERA_ROTATE", "1").lower() not in {"0", "false", "no"}
        flip_method = 2 if rotate else 0
        source_caps = f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12"
        rate_caps = ""
        rate_desc = "default"

        if fps:
            fps_value = int(fps)
            source_caps += f", framerate={fps_value}/1"
            rate_caps = f"videorate ! video/x-raw, framerate={fps_value}/1 ! "
            rate_desc = f"{fps_value}fps"

        gst_str = (
            f"nvarguscamerasrc ! "
            f"{source_caps} ! "
            f"nvvidconv flip-method={flip_method} ! video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"{rate_caps}"
            f"appsink max-buffers=1 drop=true sync=false"
        )

        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            debug_log("Internal camera not opened")
            raise RuntimeError("Internal camera not opened")

        debug_log(f"Camera opened with output {width}x{height} @ {rate_desc}")

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame_bgr = self.cap.read()
        if not ret:
            debug_log("Camera read failed")
            return None
        self.read_count += 1
        # if self.read_count <= 3 or self.read_count % 60 == 0:
        #     debug_log(f"Camera read ok count={self.read_count} shape={frame_bgr.shape}")
        return frame_bgr.copy()

    def stop(self):
        if self.cap is not None and self.cap.isOpened():
            debug_log("Releasing camera capture")
            self.cap.release()

class VisualizationWorker:
    def __init__(self, save_dir, instruction):
        self.save_dir = save_dir
        self.instruction = instruction
        self.task_queue = queue.Queue(maxsize=int(os.environ.get("OMNIVLA_VIS_QUEUE_SIZE", "2")))
        self.result_queue = queue.Queue()
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        debug_log("Visualization worker start")
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def enqueue_task(self, task):
        try:
            self.task_queue.put_nowait(task)
            debug_log(f"Visualization enqueue count_id={task['count_id']}")
        except queue.Full:
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                pass
            try:
                self.task_queue.put_nowait(task)
                debug_log(f"Visualization queue dropped oldest and enqueued count_id={task['count_id']}")
            except queue.Full:
                debug_log(f"Visualization queue still full, dropping count_id={task['count_id']}")
                pass

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                self.task_queue.task_done()
                break

            try:
                save_path = self._save_visualization(task)
                self.result_queue.put((save_path, task))
                debug_log(f"Visualization rendered count_id={task['count_id']} path={save_path}")
            except Exception as exc:
                debug_log(f"Failed to save visualization image: {exc}")
            finally:
                self.task_queue.task_done()

    def poll_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
                
    def _save_visualization(self, task):
        fig = Figure(figsize=(34, 16), dpi=80)
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        frame_rgb = cv2.cvtColor(task["frame_bgr"], cv2.COLOR_BGR2RGB)
        text = f"CMD: [{task['linear']:.3f}, {task['angular']:.3f}] | WP: {task['waypoint_select']}"
        cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        ax_ob.imshow(frame_rgb)
        ax_ob.set_title("Egocentric current image", fontsize=18)

        ax_goal.imshow(np.zeros_like(frame_rgb))
        ax_goal.set_title("Egocentric goal image (None)", fontsize=18)

        waypoints = task["waypoints"]
        goal_pose = task["goal_pose"]
        modality_id = task["modality_id"]

        if waypoints:
            waypoints_np = np.array(waypoints)
            x_seq = waypoints_np[:, 0].tolist()
            y_seq = waypoints_np[:, 1].tolist()
            x_seq = [0.0] + x_seq
            y_seq = [0.0] + y_seq
            y_seq_inv = [-vy for vy in y_seq]
        else:
            x, y, theta = 0.0, 0.0, 0.0
            x_seq, y_seq = [0.0], [0.0]
            dt = 1.2
            for _ in range(8):
                x += task["linear"] * math.cos(theta) * dt
                y += task["linear"] * math.sin(theta) * dt
                theta += task["angular"] * dt
                x_seq.append(x)
                y_seq.append(y)
            y_seq_inv = [-vy for vy in y_seq]

        ax_graph_pos.plot(y_seq_inv, x_seq, linewidth=4.0, markersize=12, marker='o', color='blue')

        mask_type = int(modality_id[0]) if len(modality_id) > 0 else 7
        if goal_pose and len(goal_pose) >= 2 and mask_type in [1, 3, 4, 5, 8]:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker='*', color='red', markersize=15)

        ax_graph_pos.set_xlim(-3.0, 3.0)
        ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)
        ax_graph_pos.tick_params(axis='x', labelsize=15)
        ax_graph_pos.tick_params(axis='y', labelsize=15)

        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            annot_text = f"{mask_texts[mask_type]} | {self.instruction}"
            ax_graph_pos.annotate(annot_text, xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')

        save_path = os.path.join(self.save_dir, f"{task['count_id']}_ex.jpg")
        fig.savefig(save_path, bbox_inches='tight')
        fig.clear()
        return save_path

    def stop(self):
        debug_log("Visualization worker stop")
        self.stop_event.set()
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)


class InferenceMonitorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OmniVLA Real-Time Inference Monitor (Integrated)")
        self.resize(1200, 700)
        
        # Paths
        inference_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_root = os.environ.get("OMNIVLA_SAVE_DIR", os.path.join(inference_dir, "runs"))
        
        # State variables
        self.is_inferencing = False
        self.latest_frame_bgr = None
        self.frame_lock = threading.Lock()
        self.count_id = 0
        self.save_dir = ""
        self.vis_worker = None
        self.display_count = 0
        self.drive_enabled = False
        self.last_drive_enable_request = None
        self.pending_drive_enable_future = None
        self.ros_active = True
        
        self.server_ip = os.environ.get("OMNIVLA_SERVER_IP", "100.69.46.104")
        self.server_port = os.environ.get("OMNIVLA_SERVER_PORT", "8000")
        self.server_url = f"http://{self.server_ip}:{self.server_port}/predict"
        self.tick_rate = float(os.environ.get("OMNIVLA_TICK_RATE", "3.0"))

        # Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()
        self.node = OmniVlaNode()
        
        # Setup camera thread if needed
        self.camera_capture = None
        if self.node.image_source == "camera":
            self.camera_capture = CameraCapture()
            self.camera_capture.open()

        # Initialize UI
        self.init_ui()
        
        # Timer for displaying live camera feed
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(33)  # ~30 Hz

        # Timer for spinning ROS 2
        self.ros_timer = QtCore.QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(10)  # ~100 Hz
        
        # Timer for inference HTTP requests
        self.inference_timer = QtCore.QTimer(self)
        self.inference_timer.timeout.connect(self.inference_step)
        self.vis_result_timer = QtCore.QTimer(self)
        self.vis_result_timer.timeout.connect(self.poll_visualization_results)
        self.vis_result_timer.start(50)
        self.drive_sync_timer = QtCore.QTimer(self)
        self.drive_sync_timer.timeout.connect(self.sync_drive_enable_state)
        self.drive_sync_timer.start(500)
        debug_log("InferenceMonitorApp initialized")

    def spin_ros(self):
        if not self.ros_active or not rclpy.ok():
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0)
        except Exception as exc:
            debug_log(f"spin_ros failed; disabling ROS integration: {exc}")
            self.disable_ros()
            return
        if self.node.image_source == "topic" and self.node.latest_frame_bgr is not None:
            with self.frame_lock:
                self.latest_frame_bgr = self.node.latest_frame_bgr.copy()

    def disable_ros(self):
        if not self.ros_active:
            return
        self.ros_active = False
        if hasattr(self, "ros_timer"):
            self.ros_timer.stop()
        if hasattr(self, "drive_sync_timer"):
            self.drive_sync_timer.stop()
        debug_log("ROS integration disabled")

    def refresh_latest_frame(self):
        if self.node.image_source != "camera" or self.camera_capture is None:
            return

        frame_bgr = self.camera_capture.read()
        if frame_bgr is None:
            return

        with self.frame_lock:
            self.latest_frame_bgr = frame_bgr
        self.display_count += 1
        # if self.display_count <= 3 or self.display_count % 60 == 0:
        #     debug_log(f"Latest frame refreshed count={self.display_count} shape={frame_bgr.shape}")

    def init_ui(self):
        # Dark Theme Palette
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget#headerCard {
                background-color: #1E1E1E;
                border-radius: 8px;
                border: 1px solid #2D2D2D;
            }
            QLabel {
                color: #E0E0E0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel#titleLabel {
                color: #00E5FF;
                font-size: 18px;
                font-weight: bold;
            }
            QLabel#statusDot {
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #2979FF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2962FF;
            }
            QPushButton#infBtn {
                background-color: #2979FF;
            }
            QPushButton#infBtn:hover {
                background-color: #2962FF;
            }
            QPushButton#infBtn[running="true"] {
                background-color: #D50000;
            }
            QPushButton#infBtn[running="true"]:hover {
                background-color: #FF1744;
            }
            QScrollArea {
                border: none;
                background-color: #121212;
            }
        """)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Header Card
        header_card = QtWidgets.QWidget()
        header_card.setObjectName("headerCard")
        header_layout = QtWidgets.QHBoxLayout(header_card)
        header_layout.setContentsMargins(15, 12, 15, 12)
        
        title_vbox = QtWidgets.QVBoxLayout()
        title_vbox.setSpacing(4)
        
        title_hbox = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("OmniVLA Real-Time Monitor (Integrated)")
        self.title_label.setObjectName("titleLabel")
        
        self.status_dot = QtWidgets.QLabel("■ IDLE")
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setStyleSheet("color: #FFA000; margin-left: 10px;")
        
        title_hbox.addWidget(self.title_label)
        title_hbox.addWidget(self.status_dot)
        title_hbox.addStretch()
        
        self.instruction_label = QtWidgets.QLabel("Instruction: Waiting for run...")
        self.instruction_label.setStyleSheet("font-size: 14px; font-weight: 500; color: #B0BEC5;")
        
        title_vbox.addLayout(title_hbox)
        title_vbox.addWidget(self.instruction_label)
        header_layout.addLayout(title_vbox)
        header_layout.addStretch()

        self.info_label = QtWidgets.QLabel("Folder: -\nStep: -")
        self.info_label.setStyleSheet("font-size: 12px; color: #90A4AE; line-height: 1.4;")
        self.info_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self.info_label)
        main_layout.addWidget(header_card)

        # Image Viewer Area
        content_layout = QtWidgets.QHBoxLayout()
        
        # Left Panel (Live Camera)
        self.live_camera_label = QtWidgets.QLabel("Waiting for camera feed...")
        self.live_camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.live_camera_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.live_camera_label.setMinimumSize(200, 200)
        self.live_camera_label.setStyleSheet("font-size: 16px; color: #78909C; font-weight: bold; border: 1px solid #2D2D2D; border-radius: 4px; background-color: #000000;")
        content_layout.addWidget(self.live_camera_label, stretch=1)
        
        # Right Panel (Trajectory / Result)
        self.result_label = QtWidgets.QLabel("Waiting for visualization...")
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.result_label.setMinimumSize(200, 200)
        self.result_label.setStyleSheet("font-size: 16px; color: #78909C; font-weight: bold; border: 1px solid #2D2D2D; border-radius: 4px; background-color: #000000;")
        content_layout.addWidget(self.result_label, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)

        # Inference Control Panel
        inf_layout = QtWidgets.QHBoxLayout()
        
        inf_label1 = QtWidgets.QLabel("Instruction:")
        self.instr_input = QtWidgets.QLineEdit("move toward black office chair")
        self.instr_input.setStyleSheet("background-color: #2D2D2D; color: white; border-radius: 4px; padding: 4px; font-size: 14px;")
        
        inf_label2 = QtWidgets.QLabel("Waypoint:")
        self.wp_input = QtWidgets.QSpinBox()
        self.wp_input.setRange(0, 9)
        self.wp_input.setValue(4)
        self.wp_input.setStyleSheet("background-color: #2D2D2D; color: white; border-radius: 4px; padding: 4px; font-size: 14px; min-width: 50px;")
        
        self.drive_btn = QtWidgets.QPushButton("Drive: OFF")
        self.drive_btn.setCheckable(True)
        self.drive_btn.setStyleSheet("background-color: #D50000; color: white; border-radius: 4px; padding: 8px 16px; font-weight: bold; font-size: 13px;")
        self.drive_btn.clicked.connect(self.toggle_drive)

        self.inf_btn = QtWidgets.QPushButton("Start Inference")
        self.inf_btn.setObjectName("infBtn")
        self.inf_btn.setProperty("running", "false")
        self.inf_btn.clicked.connect(self.toggle_inference)
        
        inf_layout.addWidget(inf_label1)
        inf_layout.addWidget(self.instr_input, stretch=1)
        inf_layout.addWidget(inf_label2)
        inf_layout.addWidget(self.wp_input)
        inf_layout.addWidget(self.drive_btn)
        inf_layout.addWidget(self.inf_btn)
        
        main_layout.addLayout(inf_layout)

    def update_display(self):
        self.refresh_latest_frame()

        with self.frame_lock:
            if self.latest_frame_bgr is None:
                return
            frame_bgr = self.latest_frame_bgr.copy()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(q_img)

        lbl_w = self.live_camera_label.width()
        lbl_h = self.live_camera_label.height()
        self.live_camera_label.setPixmap(pixmap.scaled(lbl_w, lbl_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def poll_visualization_results(self):
        if self.vis_worker is None:
            return

        result = self.vis_worker.poll_result()
        if result is None:
            return

        save_path, task = result
        self.on_visualization_ready(save_path, task)

    def toggle_drive(self, checked):
        # debug_log(f"Drive button toggled checked={checked}")
        self.drive_enabled = checked
        self.last_drive_enable_request = checked
        if checked:
            self.drive_btn.setText("Drive: ON")
            self.drive_btn.setStyleSheet("background-color: #00C853; color: white; border-radius: 4px; padding: 8px 16px; font-weight: bold; font-size: 13px;")
        else:
            self.drive_btn.setText("Drive: OFF")
            self.drive_btn.setStyleSheet("background-color: #D50000; color: white; border-radius: 4px; padding: 8px 16px; font-weight: bold; font-size: 13px;")
            cmd = Twist()
            # debug_log("Drive button OFF: publishing immediate zero cmd_vel")
            self.publish_cmd_vel(cmd)

        self.sync_drive_enable_state(force=True)

    def on_drive_enable_response(self, future: Future):
        self.pending_drive_enable_future = None
        try:
            response = future.result()
            if response is None:
                # debug_log("Drive enable service returned no response")
                return
            # debug_log(
            #     f"Drive enable service response success={response.success} message={response.message!r}"
            # )
        except Exception as exc:
            debug_log(f"Drive enable service failed: {exc}")

    def sync_drive_enable_state(self, force=False):
        if not self.ros_active or not rclpy.ok():
            return
        if self.last_drive_enable_request is None:
            return
        if self.pending_drive_enable_future is not None:
            return

        try:
            if not self.node.drive_enable_client.wait_for_service(timeout_sec=0.0):
                if force:
                    # debug_log(
                    #     f"Drive enable service `{self.node.drive_enable_service}` not ready; GUI gate only"
                    # )
                    pass
                return
        except Exception as exc:
            debug_log(f"Drive enable service readiness check failed; disabling ROS integration: {exc}")
            self.disable_ros()
            return

        req = SetBool.Request()
        req.data = self.last_drive_enable_request
        try:
            self.pending_drive_enable_future = self.node.drive_enable_client.call_async(req)
        except Exception as exc:
            debug_log(f"Drive enable service request failed; disabling ROS integration: {exc}")
            self.disable_ros()
            return
        self.pending_drive_enable_future.add_done_callback(self.on_drive_enable_response)
        # debug_log(
        #     f"Drive enable service requested checked={self.last_drive_enable_request} service={self.node.drive_enable_service}"
        # )

    def publish_cmd_vel(self, cmd: Twist):
        if not self.ros_active or not rclpy.ok():
            debug_log("Skipping cmd_vel publish because ROS integration is inactive")
            return False
        try:
            self.node.cmd_vel_pub.publish(cmd)
            return True
        except Exception as exc:
            debug_log(f"cmd_vel publish failed; disabling ROS integration: {exc}")
            self.disable_ros()
            return False

    def toggle_inference(self):
        self.is_inferencing = not self.is_inferencing
        if self.is_inferencing:
            # START
            instruction = self.instr_input.text()
            debug_log(f"Start inference requested instruction={instruction!r} waypoint={self.wp_input.value()}")
            safe_instr = instruction.replace(" ", "_").replace("/", "")
            self.save_dir = os.path.join(self.save_root, f"{safe_instr}_{int(time.time())}")
            os.makedirs(self.save_dir, exist_ok=True)
            self.count_id = 0

            self.vis_worker = VisualizationWorker(self.save_dir, instruction)
            self.vis_worker.start()
            
            self.instruction_label.setText(f"Instruction: {instruction}")
            self.status_dot.setText("● ACTIVE")
            self.status_dot.setStyleSheet("color: #00C853; margin-left: 10px;")
            self.inf_btn.setText("Stop Inference")
            self.inf_btn.setProperty("running", "true")
            self.instr_input.setEnabled(False)
            self.wp_input.setEnabled(False)
            
            self.inference_timer.start(int(1000.0 / self.tick_rate))
            debug_log(f"Inference timer started tick_rate={self.tick_rate}")
        else:
            # STOP
            debug_log("Stop inference requested")
            self.inference_timer.stop()
            if self.vis_worker:
                self.vis_worker.stop()
                self.vis_worker = None
                
            # Stop the robot
            cmd = Twist()
            self.publish_cmd_vel(cmd)

            self.status_dot.setText("■ IDLE")
            self.status_dot.setStyleSheet("color: #FFA000; margin-left: 10px;")
            self.inf_btn.setText("Start Inference")
            self.inf_btn.setProperty("running", "false")
            self.instr_input.setEnabled(True)
            self.wp_input.setEnabled(True)

        self.inf_btn.style().unpolish(self.inf_btn)
        self.inf_btn.style().polish(self.inf_btn)

    def inference_step(self):
        with self.frame_lock:
            if self.latest_frame_bgr is None:
                debug_log("Inference step skipped: no frame")
                return
            frame = self.latest_frame_bgr.copy()

        try:
            debug_log(f"Inference step begin next_count={self.count_id + 1} frame_shape={frame.shape}")
            debug_log("Inference jpeg encode start via PIL")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            jpeg_buffer = io.BytesIO()
            Image.fromarray(frame_rgb).save(jpeg_buffer, format="JPEG", quality=90)
            jpeg_bytes = jpeg_buffer.getvalue()
            debug_log(f"Inference jpeg encode complete bytes={len(jpeg_bytes)}")
                
            files = {"image": ("robot_camera.jpg", jpeg_bytes, "image/jpeg")}
            data = {
                "instruction": self.instr_input.text(),
                "waypoint_select": str(self.wp_input.value())
            }
            
            response = requests.post(self.server_url, files=files, data=data, timeout=2.0)
            debug_log(f"Inference HTTP status={response.status_code}")
            if response.status_code == 200:
                resp_data = response.json()
                linear = resp_data.get("linear_velocity", 0.0)
                angular = resp_data.get("angular_velocity", 0.0)
                debug_log(f"Inference response parsed linear={linear} angular={angular}")

                cmd = Twist()
                if self.drive_enabled:
                    cmd.linear.x = float(linear)
                    cmd.angular.z = float(angular)
                    debug_log(f"Publishing cmd_vel linear.x={cmd.linear.x:.3f} angular.z={cmd.angular.z:.3f}")
                else:
                    debug_log("Drive disabled in GUI; publishing zero cmd_vel")
                self.publish_cmd_vel(cmd)
                
                self.count_id += 1
                
                # Push to visualization
                if self.vis_worker:
                    self.vis_worker.enqueue_task({
                        "count_id": self.count_id,
                        "frame_bgr": frame.copy(),
                        "linear": float(linear),
                        "angular": float(angular),
                        "waypoint_select": data["waypoint_select"],
                        "waypoints": resp_data.get("waypoints", []),
                        "goal_pose": resp_data.get("goal_pose", []),
                        "modality_id": resp_data.get("modality_id", [7]),
                    })
        except requests.exceptions.RequestException as e:
            debug_log(f"Inference request network failed: {e}")
            if self.ros_active and rclpy.ok():
                self.node.get_logger().error(f"Inference request network failed: {e}")
        except Exception as e:
            debug_log(f"Inference request error: {e}")
            if self.ros_active and rclpy.ok():
                self.node.get_logger().error(f"Inference request error: {e}")

    def on_visualization_ready(self, save_path, task):
        debug_log(f"Visualization ready count_id={task['count_id']} path={save_path}")
        try:
            debug_log("Visualization image load start via PIL")
            result_rgb = np.array(Image.open(save_path).convert("RGB"))
            debug_log(f"Visualization image load complete shape={result_rgb.shape}")

            result_rgb = np.ascontiguousarray(result_rgb)
            h, w, ch = result_rgb.shape
            bytes_per_line = ch * w
            q_img = QtGui.QImage(result_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
            pixmap = QtGui.QPixmap.fromImage(q_img)
            debug_log(f"Visualization pixmap created null={pixmap.isNull()}")

            if not pixmap.isNull():
                lbl_w = self.result_label.width()
                lbl_h = self.result_label.height()
                scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.result_label.setPixmap(scaled_pixmap)
                debug_log("Visualization pixmap applied to label")
                
                # Update info label
                folder_name = os.path.basename(self.save_dir)
                self.info_label.setText(f"Folder: {folder_name}\nStep: {task['count_id']}")
                debug_log("Visualization info label updated")
        except Exception as exc:
            debug_log(f"Visualization display failed: {exc}")

    def closeEvent(self, event):
        debug_log("Close event begin")
        self.inference_timer.stop()
        self.vis_result_timer.stop()
        self.display_timer.stop()
        self.ros_timer.stop()
        self.drive_sync_timer.stop()
        if self.vis_worker:
            self.vis_worker.stop()
        if self.camera_capture:
            self.camera_capture.stop()
            
        cmd = Twist()
        self.publish_cmd_vel(cmd)

        self.disable_ros()
        try:
            self.node.destroy_node()
        except Exception as exc:
            debug_log(f"destroy_node failed: {exc}")
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as exc:
                debug_log(f"rclpy.shutdown failed: {exc}")
        debug_log("Close event complete")
        super().closeEvent(event)

if __name__ == "__main__":
    debug_log(f"Process start pid={os.getpid()} log_path={DEBUG_LOG_PATH}")
    app = QtWidgets.QApplication(sys.argv)
    monitor = InferenceMonitorApp()
    monitor.show()
    sys.exit(app.exec_())
