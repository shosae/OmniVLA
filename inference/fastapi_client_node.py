import os
import queue
import time
import datetime
import threading
import requests
import math

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ROSImage
except ImportError:
    raise ImportError("ROS2 dependencies are required for this client")

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

class OmniVLAClientNode(Node):
    def __init__(self):
        super().__init__("omnivla_fastapi_client")
        self.bridge = CvBridge()
        self.server_ip = os.environ.get("OMNIVLA_SERVER_IP", "100.69.46.104")
        self.server_port = os.environ.get("OMNIVLA_SERVER_PORT", "8000")
        self.server_url = f"http://{self.server_ip}:{self.server_port}/predict"

        # Image source can be "topic" or "camera"
        self.image_source = os.environ.get("OMNIVLA_IMAGE_SOURCE", "camera").lower()
        self.image_topic = os.environ.get("OMNIVLA_IMAGE_TOPIC", "/camera/image_raw")
        self.cmd_vel_topic = os.environ.get("OMNIVLA_CMD_VEL_TOPIC", "/cmd_vel")
        self.tick_rate = float(os.environ.get("OMNIVLA_TICK_RATE", "3.0"))
        self.instruction = os.environ.get("OMNIVLA_INSTRUCTION", "move toward black office chair")
        self.waypoint_select = int(os.environ.get("OMNIVLA_WAYPOINT_SELECT", "4"))
        
        self.count_id = 0
        safe_instr = self.instruction.replace(" ", "_").replace("/", "")
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        self.save_dir = os.path.join(
            base_dir,
            f"{safe_instr}_{int(time.time())}"
        )
        os.makedirs(self.save_dir, exist_ok=True)        
        self.waiting_for_image_logged = False
        self.image_sub = None
        self.cap = None
        self.capture_thread = None
        self.visualization_thread = None
        self.stop_event = threading.Event()
        self.latest_frame_bgr = None
        self.visualization_queue = queue.Queue(
            maxsize=int(os.environ.get("OMNIVLA_VIS_QUEUE_SIZE", "2"))
        )

        if self.image_source == "topic":
            self.image_sub = self.create_subscription(
                ROSImage,
                self.image_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
            image_source_desc = f"topic `{self.image_topic}`"
        elif self.image_source == "camera":
            self._open_internal_camera()
            image_source_desc = "internal camera"
        else:
            raise ValueError(f"Unsupported OMNIVLA_IMAGE_SOURCE `{self.image_source}`. Use `camera` or `topic`.")

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(1.0 / self.tick_rate, self.timer_callback)
        self.visualization_thread = threading.Thread(
            target=self._visualization_loop,
            daemon=True,
        )
        self.visualization_thread.start()

        self.get_logger().info(
            f"Using {image_source_desc} and publishing `{self.cmd_vel_topic}` at {self.tick_rate} Hz"
        )
        self.get_logger().info(f"Target Server URL: {self.server_url}")
        self.get_logger().info(f"Instruction: '{self.instruction}'")
        self.get_logger().info(f"Saving visualization images to: {self.save_dir}")

    def _open_internal_camera(self) -> None:
        preloaded_jpeg_lib = preload_jpeg_compat()
        if preloaded_jpeg_lib is None:
            self.get_logger().warn(
                "Failed to preload a libjpeg compatibility library; nvarguscamerasrc may fail to load"
            )
        else:
            self.get_logger().info(f"Preloaded JPEG compatibility library: {preloaded_jpeg_lib}")

        width = int(os.environ.get("OMNIVLA_CAMERA_WIDTH", "1280"))
        height = int(os.environ.get("OMNIVLA_CAMERA_HEIGHT", "720"))
        fps = int(os.environ.get("OMNIVLA_CAMERA_FPS", "30"))
        rotate = os.environ.get("OMNIVLA_CAMERA_ROTATE", "1").lower() not in {"0", "false", "no"}
        flip_method = 2 if rotate else 0

        gst_str = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! video/x-raw, width=1280, height=720, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"videorate ! video/x-raw, framerate={fps}/1 ! "
            f"appsink max-buffers=1 drop=true sync=false"
        )

        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Internal camera not opened")

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame_bgr = self.cap.read()
            if not ret:
                self.get_logger().error("Internal camera frame read failed")
                time.sleep(0.05)
                continue

            self.latest_frame_bgr = frame_bgr
            self.waiting_for_image_logged = False

    def image_callback(self, msg: ROSImage) -> None:
        try:
            self.latest_frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.waiting_for_image_logged = False
        except Exception as e:
            self.get_logger().error(f"cv_bridge exception: {e}")

    def _enqueue_visualization_task(self, task) -> None:
        try:
            self.visualization_queue.put_nowait(task)
        except queue.Full:
            try:
                self.visualization_queue.get_nowait()
                self.visualization_queue.task_done()
            except queue.Empty:
                pass
            try:
                self.visualization_queue.put_nowait(task)
            except queue.Full:
                self.get_logger().warn("Visualization queue is full; dropping current frame")

    def _visualization_loop(self) -> None:
        while True:
            try:
                task = self.visualization_queue.get(timeout=0.1)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

            if task is None:
                self.visualization_queue.task_done()
                break

            try:
                self._save_visualization(task)
            except Exception as exc:
                self.get_logger().error(f"Failed to save visualization image: {exc}")
            finally:
                self.visualization_queue.task_done()

    def _save_visualization(self, task) -> None:
        fig = plt.figure(figsize=(34, 16), dpi=80)
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
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def timer_callback(self) -> None:
        if self.latest_frame_bgr is None:
            if not self.waiting_for_image_logged:
                if self.image_source == "topic":
                    self.get_logger().info(f"Waiting for image on `{self.image_topic}`")
                else:
                    self.get_logger().info("Waiting for internal camera frame")
                self.waiting_for_image_logged = True
            return

        frame = self.latest_frame_bgr.copy()
        
        # Encode image to jpg
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            self.get_logger().error("Failed to encode image to JPG")
            return
            
        files = {"image": ("robot_camera.jpg", buffer.tobytes(), "image/jpeg")}
        data = {
            "instruction": self.instruction,
            "waypoint_select": self.waypoint_select
        }
        
        try:
            # Send HTTP POST to FastAPI server
            response = requests.post(self.server_url, files=files, data=data, timeout=2.0)
            
            if response.status_code == 200:
                data = response.json()
                linear = data.get("linear_velocity", 0.0)
                angular = data.get("angular_velocity", 0.0)

                cmd = Twist()
                cmd.linear.x = float(linear)
                cmd.angular.z = float(angular)
                self.cmd_vel_pub.publish(cmd)
                self.get_logger().info(f"Published cmd_vel linear.x={cmd.linear.x:.3f} angular.z={cmd.angular.z:.3f}")
                self.count_id += 1
                self._enqueue_visualization_task({
                    "count_id": self.count_id,
                    "frame_bgr": frame.copy(),
                    "linear": float(linear),
                    "angular": float(angular),
                    "waypoint_select": data.get("waypoint_select", 4),
                    "waypoints": data.get("waypoints", []),
                    "goal_pose": data.get("goal_pose", []),
                    "modality_id": data.get("modality_id", [7]),
                })
                
            else:
                self.get_logger().error(f"Server returned status {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Failed to communicate with server: {e}")

    def destroy_node(self):
        self.stop_event.set()
        if self.visualization_thread is not None and self.visualization_thread.is_alive():
            try:
                self.visualization_queue.put(None, timeout=0.5)
            except queue.Full:
                pass
            self.visualization_thread.join(timeout=1.0)
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OmniVLAClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
