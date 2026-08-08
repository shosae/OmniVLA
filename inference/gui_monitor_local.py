#!/usr/bin/env python3
"""Integrated OmniVLA monitor that runs the TensorRT model in this process."""

import os
import sys
import importlib.util
from pathlib import Path

import cv2
from PIL import Image

INFERENCE_DIR = Path(__file__).resolve().parent
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

_INFERENCE_SPEC = importlib.util.spec_from_file_location(
    "omnivla_local_inference",
    INFERENCE_DIR / "inference.py",
)
if _INFERENCE_SPEC is None or _INFERENCE_SPEC.loader is None:
    raise ImportError(f"Unable to load local inference module from {INFERENCE_DIR}")
local_inference = importlib.util.module_from_spec(_INFERENCE_SPEC)
sys.modules[_INFERENCE_SPEC.name] = local_inference
_INFERENCE_SPEC.loader.exec_module(local_inference)

from gui_monitor_integrated import (
    DEBUG_LOG_PATH,
    InferenceMonitorApp,
    QtWidgets,
    Twist,
    debug_log,
    rclpy,
)


class LocalOmniVLARunner:
    """Owns the local model and exposes the response shape expected by the GUI."""

    def __init__(self) -> None:
        # The integrated local GUI is language-plus-current-image only.
        local_inference.pose_goal = False
        local_inference.satellite = False
        local_inference.image_goal = False
        local_inference.lan_prompt = True

        cfg = local_inference.InferenceConfig()
        workspace_root = Path(__file__).resolve().parents[2]
        cfg.vla_path = os.environ.get(
            "OMNIVLA_VLA_PATH",
            str(workspace_root / "omnivla-original"),
        )
        if not Path(cfg.vla_path).is_dir():
            raise FileNotFoundError(
                f"OmniVLA checkpoint directory does not exist: {cfg.vla_path}. "
                "Set OMNIVLA_VLA_PATH to the original FP16 checkpoint directory."
            )
        cfg.llm_backend = "tensorrt_llm"
        cfg.enable_llm_profile = False
        cfg.enable_benchmark = False
        (
            local_inference.vla,
            local_inference.action_head,
            local_inference.pose_projector,
            local_inference.device_id,
            local_inference.NUM_PATCHES,
            action_tokenizer,
            processor,
            compute_dtype,
            llm_backend,
        ) = local_inference.define_model(cfg)

        self._inference = local_inference.Inference(
            save_dir=str(Path(__file__).resolve().parent / "runs_local_model"),
            lan_inst_prompt="",
            goal_utm=None,
            goal_compass=0.0,
            goal_image_PIL=None,
            action_tokenizer=action_tokenizer,
            processor=processor,
            compute_dtype=compute_dtype,
            llm_backend=llm_backend,
        )

    def predict(self, frame_bgr, instruction: str, waypoint_select: int) -> dict:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._inference.set_current_image_from_pil(Image.fromarray(frame_rgb))
        self._inference.lan_inst_prompt = instruction
        linear, angular = self._inference.run_omnivla(
            save_behavior=False,
            profile_language_model=False,
            log_output=False,
            waypoint_select=waypoint_select,
        )
        return {
            "linear_velocity": float(linear),
            "angular_velocity": float(angular),
            "waypoint_select": self._inference.last_waypoint_select,
            "waypoints": self._inference.last_waypoints,
            "goal_pose": self._inference.last_goal_pose,
            "modality_id": self._inference.last_modality_id,
        }


class LocalInferenceMonitorApp(InferenceMonitorApp):
    def __init__(self) -> None:
        debug_log("Loading local OmniVLA TensorRT model")
        self.local_model = LocalOmniVLARunner()
        debug_log("Local OmniVLA TensorRT model loaded")
        super().__init__()
        self.setWindowTitle("OmniVLA Real-Time Inference Monitor (Local)")
        self.title_label.setText("OmniVLA Real-Time Monitor (Local TensorRT)")

    def inference_step(self) -> None:
        with self.frame_lock:
            if self.latest_frame_bgr is None:
                debug_log("Local inference step skipped: no frame")
                return
            frame = self.latest_frame_bgr.copy()

        try:
            result = self.local_model.predict(
                frame,
                self.instr_input.text(),
                self.wp_input.value(),
            )
            linear = result["linear_velocity"]
            angular = result["angular_velocity"]
            debug_log(f"Local inference result linear={linear:.3f} angular={angular:.3f}")

            cmd = Twist()
            if self.drive_enabled:
                cmd.linear.x = linear
                cmd.angular.z = angular
            self.publish_cmd_vel(cmd)

            self.count_id += 1
            if self.vis_worker:
                self.vis_worker.enqueue_task(
                    {
                        "count_id": self.count_id,
                        "frame_bgr": frame,
                        "linear": linear,
                        "angular": angular,
                        "waypoint_select": result["waypoint_select"],
                        "waypoints": result["waypoints"],
                        "goal_pose": result["goal_pose"],
                        "modality_id": result["modality_id"],
                    }
                )
        except Exception as exc:
            debug_log(f"Local model inference failed: {exc}")
            if self.ros_active and rclpy.ok():
                self.node.get_logger().error(f"Local model inference failed: {exc}")


if __name__ == "__main__":
    debug_log(f"Local GUI process start pid={os.getpid()} log_path={DEBUG_LOG_PATH}")
    app = QtWidgets.QApplication(sys.argv)
    monitor = LocalInferenceMonitorApp()
    monitor.show()
    sys.exit(app.exec_())
