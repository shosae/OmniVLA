# ===============================================================
# OmniVLA Inference
# ===============================================================
# 
# Sample inference code for OmniVLA
# if you want to control the robot, you need to update the current state such as pose and image in "run_omnivla" and comment out "break" in "run".
#
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
import re
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import ctypes
import time, math, json
import threading
from typing import Any, Optional, Tuple, Type, Dict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utm

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from awq_loader import load_awq_openvla, load_embedding_only_openvla
from llm_backends import (
    BaseLLMBackend,
    LLMBackendInputs,
    build_llm_backend,
    validate_trtllm_engine_version,
)
from time_checker import (
    LLMProfileResult,
    LLMProfiler,
    ModuleTimer,
    configure_module_timer,
    create_llm_profiler,
    run_benchmark,
    run_llm_profile,
    synchronize_if_cuda,
)

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ROSImage
except ImportError:
    rclpy = None
    CvBridge = None
    Twist = None
    Node = object
    qos_profile_sensor_data = None
    ROSImage = None

# ===============================================================
# Utility Functions
# ===============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def sanitize_instruction_for_path(instruction: str) -> str:
    folder_name = re.sub(r"\s+", "_", instruction.strip())
    folder_name = re.sub(r"[^A-Za-z0-9_.-]", "", folder_name)
    folder_name = folder_name.strip("._-")
    return folder_name or "instruction"


def make_unique_instruction_save_dir(save_root: str, instruction: str) -> str:
    root = Path(save_root).expanduser()
    folder_stem = sanitize_instruction_for_path(instruction)
    for index in range(1, 10000):
        candidate = root / f"{folder_stem}_{index}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return str(candidate)
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate unique save directory under `{root}`")


def preload_jpeg_compat() -> Optional[str]:
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


def resolve_compute_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized = dtype_name.lower()
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported compute dtype `{dtype_name}`. Use one of: {sorted(dtype_map)}")
    return dtype_map[normalized]


def _debug_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _tensor_stats_line(name: str, tensor: torch.Tensor) -> str:
    tensor_f32 = tensor.detach().float()
    if tensor_f32.numel() == 0:
        return f"{name}: shape={tuple(tensor.shape)} empty"
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"mean_abs={tensor_f32.abs().mean().item():.6f} "
        f"max_abs={tensor_f32.abs().max().item():.6f} "
        f"min={tensor_f32.min().item():.6f} "
        f"max={tensor_f32.max().item():.6f}"
    )


def _format_index_ranges(indices: list[int], limit: int = 12) -> str:
    if not indices:
        return "[]"

    ranges: list[str] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = idx
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")

    if len(ranges) > limit:
        hidden = len(ranges) - limit
        ranges = ranges[:limit] + [f"...(+{hidden} ranges)"]
    return "[" + ", ".join(ranges) + "]"


def _debug_action_summary(
    backend_name: str,
    text_hidden_states: Optional[torch.Tensor],
    actions_hidden_states: torch.Tensor,
    predicted_actions: torch.Tensor,
    current_action_mask: torch.Tensor,
    next_actions_mask: torch.Tensor,
) -> None:
    if not _debug_flag("OMNIVLA_ACTION_DEBUG"):
        return

    token_mask = current_action_mask | next_actions_mask
    if token_mask.ndim == 2:
        token_indices = token_mask[0].nonzero(as_tuple=False).flatten().tolist()
    else:
        token_indices = token_mask.nonzero(as_tuple=False).flatten().tolist()

    print(
        f"[OmniVLA Action Debug] backend={backend_name} "
        f"action_token_count={len(token_indices)} "
        f"action_token_ranges={_format_index_ranges(token_indices)}",
        file=sys.stderr,
    )
    if text_hidden_states is not None:
        print(f"[OmniVLA Action Debug] {_tensor_stats_line('text_hidden_states', text_hidden_states)}", file=sys.stderr)
    print(f"[OmniVLA Action Debug] {_tensor_stats_line('actions_hidden_states', actions_hidden_states)}", file=sys.stderr)
    print(f"[OmniVLA Action Debug] {_tensor_stats_line('predicted_actions', predicted_actions)}", file=sys.stderr)

    sample_tokens = (
        actions_hidden_states[0, : min(4, actions_hidden_states.shape[1]), : min(8, actions_hidden_states.shape[2])]
        .detach()
        .float()
        .cpu()
        .numpy()
        .round(4)
        .tolist()
    )
    sample_actions = (
        predicted_actions[0, : min(3, predicted_actions.shape[1])]
        .detach()
        .float()
        .cpu()
        .numpy()
        .round(4)
        .tolist()
    )
    print(f"[OmniVLA Action Debug] actions_hidden_sample={sample_tokens}", file=sys.stderr)
    print(f"[OmniVLA Action Debug] predicted_actions_sample={sample_actions}", file=sys.stderr)

def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    module_dtype: Optional[torch.dtype] = None,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if module_dtype is not None:
        module = module.to(module_dtype)
    module = module.to(device_id)
    return module

# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(
        self,
        save_dir,
        lan_inst_prompt,
        goal_utm,
        goal_compass,
        goal_image_PIL,
        action_tokenizer,
        processor,
        compute_dtype: torch.dtype,
        llm_backend: BaseLLMBackend,
        llm_profiler: Optional[LLMProfiler] = None,
        module_timer: Optional[ModuleTimer] = None,
    ):
        self.tick_rate = int(os.environ.get("OMNIVLA_TICK_RATE", "10"))
        if self.tick_rate <= 0:
            raise ValueError("OMNIVLA_TICK_RATE must be positive")
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        self.goal_image_PIL = goal_image_PIL
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.compute_dtype = compute_dtype
        self.llm_backend = llm_backend
        self.llm_profiler = llm_profiler
        self.module_timer = module_timer
        self.last_llm_profile: Optional[LLMProfileResult] = None
        self.last_profile_context: Optional[Dict[str, Any]] = None
        self.last_waypoints: list[list[float]] = []
        self.last_goal_pose: list[float] = []
        self.last_modality_id: list[float] = []
        self.last_waypoint_select = 0
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0
        self.current_image_PIL: Optional[Image.Image] = None
        self._current_image_path: Optional[str] = None
        self._current_image_cache: Optional[Image.Image] = None
        self.datastore_path_image = save_dir
        os.makedirs(self.datastore_path_image, exist_ok=True)

    @staticmethod
    def make_dummy_goal_image(reference_image: Image.Image) -> Image.Image:
        return Image.new("RGB", reference_image.size, color=(0, 0, 0))
    # ----------------------------
    # Static Utility Methods
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return default
        return float(value)

    def _load_current_image(self) -> Image.Image:
        if self.current_image_PIL is not None:
            return self.current_image_PIL.copy()

        current_image_path = os.environ.get(
            "OMNIVLA_CURRENT_IMAGE",
            str(SCRIPT_DIR / "current_img.jpg"),
        )
        if self._current_image_path != current_image_path or self._current_image_cache is None:
            self._current_image_cache = Image.open(current_image_path).convert("RGB")
            self._current_image_path = current_image_path
        return self._current_image_cache.copy()

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.monotonic()
        while True:
            if time.monotonic() - start_time > loop_time:
                self.tick()
                start_time = time.monotonic()
            time.sleep(0.001)

    def tick(self):
        self.linear, self.angular = self.run_omnivla()

    def set_current_image_from_pil(self, current_image: Image.Image) -> None:
        self.current_image_PIL = current_image.copy()

    @staticmethod
    def get_modality_id() -> torch.Tensor:
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            return torch.as_tensor([0], dtype=torch.float32)
        if satellite and not lan_prompt and pose_goal and not image_goal:
            return torch.as_tensor([1], dtype=torch.float32)
        if satellite and not lan_prompt and not pose_goal and image_goal:
            return torch.as_tensor([2], dtype=torch.float32)
        if satellite and not lan_prompt and pose_goal and image_goal:
            return torch.as_tensor([3], dtype=torch.float32)
        if not satellite and not lan_prompt and pose_goal and not image_goal:
            return torch.as_tensor([4], dtype=torch.float32)
        if not satellite and not lan_prompt and pose_goal and image_goal:
            return torch.as_tensor([5], dtype=torch.float32)
        if not satellite and not lan_prompt and not pose_goal and image_goal:
            return torch.as_tensor([6], dtype=torch.float32)
        if not satellite and lan_prompt and not pose_goal and not image_goal:
            return torch.as_tensor([7], dtype=torch.float32)
        if not satellite and lan_prompt and pose_goal and not image_goal:
            return torch.as_tensor([8], dtype=torch.float32)
        raise ValueError("Unsupported modality configuration")

    # ----------------------------
    # OmniVLA Inference
    # ----------------------------
    def run_omnivla(
        self,
        save_behavior: bool = False,
        profile_language_model: bool = False,
        log_output: bool = False,
        waypoint_select: int = 4,
    ):
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1
        self.last_llm_profile = None

        # Load current image
        current_image_PIL = self._load_current_image()

        if pose_goal:
            current_lat = self._env_float("OMNIVLA_CURRENT_LAT", 37.87371258374039)
            current_lon = self._env_float("OMNIVLA_CURRENT_LON", -122.26729417226024)
            current_compass = self._env_float("OMNIVLA_CURRENT_COMPASS", 270.0)
            cur_utm = utm.from_latlon(current_lat, current_lon)
            cur_compass = -float(current_compass) / 180.0 * math.pi  # inverted compass

            delta_x, delta_y = self.calculate_relative_position(
                cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
            )
            relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
            radius = np.sqrt(relative_x**2 + relative_y**2)
            if radius > thres_dist:
                relative_x *= thres_dist / radius
                relative_y *= thres_dist / radius

            goal_pose_loc_norm = np.array([
                relative_y / metric_waypoint_spacing,
                -relative_x / metric_waypoint_spacing,
                np.cos(self.goal_compass - cur_compass),
                np.sin(self.goal_compass - cur_compass),
            ])
        else:
            goal_pose_loc_norm = np.zeros(POSE_DIM, dtype=np.float32)

        if image_goal and self.goal_image_PIL is not None:
            goal_image_PIL = self.goal_image_PIL.copy()
        else:
            goal_image_PIL = self.make_dummy_goal_image(current_image_PIL)

        # Language instruction
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"

        # Prepare batch
        batch = self.data_transformer_omnivla(
            current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )
        modality_id = self.get_modality_id()
        self.last_profile_context = {
            "batch": batch,
            "modality_id": modality_id,
            "instruction": lan_inst,
        }

        if self.llm_profiler is not None and profile_language_model:
            run_llm_profile(
                self,
                vla,
                pose_projector,
                device_id,
                self.module_timer,
                print_result=save_behavior,
            )

        # Run forward pass
        actions, modality_id = self.run_forward_pass(
            vla=vla.eval(),
            action_head=action_head.eval(),
            noisy_action_projector=None,
            pose_projector=pose_projector.eval(),
            batch=batch,
            device_id=device_id,
            use_diffusion=False,
            use_film=False,
            num_patches=NUM_PATCHES,
            modality_id=modality_id,
        )
        self.count_id += 1

        waypoints = actions.float().cpu().numpy()
        self.last_waypoints = waypoints[0].tolist()
        self.last_goal_pose = goal_pose_loc_norm.tolist()
        self.last_modality_id = modality_id.detach().cpu().tolist()

        # Select waypoint
        self.last_waypoint_select = min(max(int(waypoint_select), 0), waypoints.shape[1] - 1)
        chosen_waypoint = waypoints[0][self.last_waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        # PD controller
        EPS = 1e-8
        DT = 1 / self.tick_rate
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        # Velocity limitation
        maxv = self._env_float("OMNIVLA_MAX_LINEAR", 0.2)
        maxw = self._env_float("OMNIVLA_MAX_ANGULAR", 0.5)
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        # Save behavior
        if save_behavior:
            self.save_robot_behavior(
                current_image_PIL, goal_image_PIL, goal_pose_loc_norm, waypoints[0],
                linear_vel_value_limit, angular_vel_value_limit, metric_waypoint_spacing, modality_id.cpu().numpy()
            )

        if log_output:
            print("linear angular", linear_vel_value_limit, angular_vel_value_limit)
        return linear_vel_value_limit, angular_vel_value_limit

    # ----------------------------
    # Save Robot Behavior Visualization
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq = waypoints[:, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv = -waypoints[:, 1]           
        ax_graph_pos.plot(np.insert(y_seq_inv, 0, 0.0), np.insert(x_seq, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue')

        # Mask annotation
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("Egocentric current image", fontsize=18)
        ax_goal.set_title("Egocentric goal image", fontsize=18)
        ax_graph_pos.tick_params(axis='x', labelsize=15) 
        ax_graph_pos.tick_params(axis='y', labelsize=15) 
        
        if int(mask_number[0]) == 1 or int(mask_number[0]) == 3 or int(mask_number[0]) == 4 or int(mask_number[0]) == 5 or int(mask_number[0]) == 8:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker = '*', color='red', markersize=15)  
        else:                           
            ax_graph_pos.set_xlim(-3.0, 3.0)
            ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_xlim(-3.0, 3.0)
        ax_graph_pos.set_ylim(-0.1, 10.0)
                        
        ax_graph_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)
        
        save_path = os.path.join(self.datastore_path_image, f"{self.count_id}_ex.jpg")
        plt.savefig(save_path)
        plt.close(fig)


    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL,
            inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer for OmniVLA
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 prompt_builder, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions
        goal_pose_cos_sin = goal_pose_loc_norm

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(
        self,
        vla,
        action_head,
        noisy_action_projector,
        pose_projector,
        batch,
        device_id,
        use_diffusion,
        use_film,
        num_patches,
        modality_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        if modality_id is None:
            modality_id = self.get_modality_id()

        with torch.inference_mode(), torch.autocast("cuda", dtype=self.compute_dtype):
            last_hidden_states = self.llm_backend.forward_action_hidden_states(
                vla,
                LLMBackendInputs(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(dtype=self.compute_dtype, device=device_id),
                    modality_id=modality_id.to(dtype=self.compute_dtype, device=device_id),
                    labels=batch["labels"].to(device_id),
                    proprio=batch["goal_pose"].to(dtype=self.compute_dtype, device=device_id),
                    proprio_projector=pose_projector,
                    noisy_actions=noisy_actions if use_diffusion else None,
                    noisy_action_projector=noisy_action_projector if use_diffusion else None,
                    diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                    use_film=use_film,
                ),
            )

        # Prepare data for metrics
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
        batch_size = batch["input_ids"].shape[0]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(self.compute_dtype)
        )  # (B, act_chunk_len, D)

        with torch.inference_mode():
            predicted_actions = action_head.predict_action(
                actions_hidden_states,
                modality_id.to(dtype=self.compute_dtype, device=device_id),
            )

        _debug_action_summary(
            backend_name=getattr(self.llm_backend, "name", type(self.llm_backend).__name__),
            text_hidden_states=text_hidden_states,
            actions_hidden_states=actions_hidden_states,
            predicted_actions=predicted_actions,
            current_action_mask=current_action_mask,
            next_actions_mask=next_actions_mask,
        )

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions, modality_id


class OmniVLAInferenceNode(Node):
    def __init__(self, inference: Inference):
        super().__init__("omnivla_inference")
        self.inference = inference
        self.bridge = CvBridge()
        self.image_source = os.environ.get("OMNIVLA_IMAGE_SOURCE", "camera").lower()
        self.image_topic = os.environ.get("OMNIVLA_IMAGE_TOPIC", "/camera/image_raw")
        self.cmd_vel_topic = os.environ.get("OMNIVLA_CMD_VEL_TOPIC", "/cmd_vel")
        self.waiting_for_image_logged = False
        self.image_sub = None
        self.cap = None
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.last_image_time: Optional[float] = None
        self.image_timeout = float(os.environ.get("OMNIVLA_IMAGE_TIMEOUT", "1.0"))
        if self.image_timeout <= 0:
            raise ValueError("OMNIVLA_IMAGE_TIMEOUT must be positive")
        self.stale_image_logged = False
        self.log_output = os.environ.get("OMNIVLA_LOG_OUTPUT", "0").lower() not in {"0", "false", "no"}

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
            raise ValueError(
                f"Unsupported OMNIVLA_IMAGE_SOURCE `{self.image_source}`. Use `camera` or `topic`."
            )

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(1.0 / self.inference.tick_rate, self.timer_callback)

        self.get_logger().info(
            f"Using {image_source_desc} and publishing `{self.cmd_vel_topic}` at {self.inference.tick_rate} Hz"
        )

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
            f"nvvidconv flip-method={flip_method} ! video/x-raw, width=960, height=544, format=BGRx ! "
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

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._set_current_image(Image.fromarray(frame_rgb))

    def image_callback(self, msg: ROSImage) -> None:
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._set_current_image(Image.fromarray(frame_rgb))

    def _set_current_image(self, image: Image.Image) -> None:
        self.inference.set_current_image_from_pil(image)
        self.last_image_time = time.monotonic()
        self.waiting_for_image_logged = False
        self.stale_image_logged = False

    def _publish_stop(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def timer_callback(self) -> None:
        if self.inference.current_image_PIL is None:
            if not self.waiting_for_image_logged:
                if self.image_source == "topic":
                    self.get_logger().info(f"Waiting for image on `{self.image_topic}`")
                else:
                    self.get_logger().info("Waiting for internal camera frame")
                self.waiting_for_image_logged = True
            return

        if self.last_image_time is None or time.monotonic() - self.last_image_time > self.image_timeout:
            if not self.stale_image_logged:
                self.get_logger().warn("Image stream timed out; publishing zero velocity")
                self.stale_image_logged = True
            self._publish_stop()
            return

        linear, angular = self.inference.run_omnivla(log_output=self.log_output)

        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_vel_pub.publish(cmd)
        if self.log_output:
            self.get_logger().info(f"Published cmd_vel linear.x={cmd.linear.x:.3f} angular.z={cmd.angular.z:.3f}")

    def destroy_node(self):
        self.stop_event.set()
        self._publish_stop()
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

                
# ===============================================================
# Inference Configuration
# ===============================================================
class InferenceConfig:
    resume: bool = True
    vla_path: str = os.environ.get(
        "OMNIVLA_VLA_PATH",
        str(PROJECT_DIR / "omnivla-original"),
    )
    awq_llm_path: Optional[str] = os.environ.get(
        "OMNIVLA_AWQ_LLM_PATH",
        str(PROJECT_DIR / "models" / "llama2_awq_packed"),
    )
    resume_step: Optional[int] = (
        int(os.environ.get("OMNIVLA_RESUME_STEP"))
        if os.environ.get("OMNIVLA_RESUME_STEP") is not None
        else 120000
    )
    #vla_path: str = "./omnivla-finetuned-cast"    
    #resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    num_images_in_input: int = 2
    compute_dtype: str = "bfloat16"
    awq_fuse_layers: bool = True
    llm_backend: str = os.environ.get("OMNIVLA_LLM_BACKEND", "tensorrt_llm")
    trtllm_engine_dir: Optional[str] = os.environ.get("OMNIVLA_TRTLLM_ENGINE_DIR")
    trtllm_runner_path: Optional[str] = os.environ.get(
        "OMNIVLA_LLM_EMBED_RUNNER",
        str(PROJECT_DIR / "runtime" / "trtllm" / "bin" / "llm_embed_inference"),
    )
    trtllm_embedding_path: Optional[str] = os.environ.get("OMNIVLA_TRTLLM_EMBEDDING_PATH")
    benchmark_warmup_iters: int = int(os.environ.get("OMNIVLA_BENCHMARK_WARMUP_ITERS", "3"))
    benchmark_timed_iters: int = int(os.environ.get("OMNIVLA_BENCHMARK_TIMED_ITERS", "10"))
    enable_llm_profile: bool = os.environ.get("OMNIVLA_ENABLE_LLM_PROFILE", "0").lower() not in {"0", "false", "no"}
    enable_benchmark: bool = os.environ.get("OMNIVLA_ENABLE_BENCHMARK", "0").lower() not in {"0", "false", "no"}

def define_model(cfg: InferenceConfig) -> None:
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")
    compute_dtype = resolve_compute_dtype(cfg.compute_dtype)

    # GPU setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OmniVLA inference; CPU inference is unsupported.")
    device_id = torch.device("cuda:0")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}\n"
        f"\tCOMPUTE_DTYPE: {compute_dtype}\n"
        f"\tAWQ_FUSE_LAYERS: {cfg.awq_fuse_layers}\n"
        f"\tLLM_BACKEND: {cfg.llm_backend}\n"
        f"\tTRTLLM_ENGINE_DIR: {cfg.trtllm_engine_dir}\n"
        f"\tTRTLLM_RUNNER_PATH: {cfg.trtllm_runner_path}\n"
        f"\tTRTLLM_EMBEDDING_PATH: {cfg.trtllm_embedding_path}"
    )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    
    if cfg.llm_backend.lower() == "tensorrt_llm":
        if not cfg.trtllm_engine_dir or not cfg.trtllm_embedding_path:
            raise ValueError(
                "Set `OMNIVLA_TRTLLM_ENGINE_DIR` and `OMNIVLA_TRTLLM_EMBEDDING_PATH` "
                "when using the TensorRT LLM backend."
            )
        validate_trtllm_engine_version(cfg.trtllm_engine_dir)
        print(f"Loading VLA non-language weights and TRT embedding `{cfg.trtllm_embedding_path}`")
        vla, processor = load_embedding_only_openvla(
            cfg.vla_path,
            cfg.trtllm_embedding_path,
            device_id,
            cfg.num_images_in_input,
            compute_dtype,
        )
    elif cfg.awq_llm_path:
        cfg.awq_llm_path = cfg.awq_llm_path.rstrip("/")
        print(f"Loading AWQ LLM `{cfg.awq_llm_path}` without loading original LLM weights")
        vla, processor = load_awq_openvla(
            cfg.vla_path,
            cfg.awq_llm_path,
            device_id,
            cfg.num_images_in_input,
            compute_dtype,
            cfg.awq_fuse_layers,
        )
    else:
        # Load processor and VLA
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
        ).to(device_id) #            trust_remote_code=True,

        vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
        vla.to(dtype=compute_dtype, device=device_id)
    
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            
        module_dtype=compute_dtype,
    )
    
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead_idcat,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            
            module_dtype=compute_dtype,
        )            
 
    # Get number of vision patches
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    llm_backend = build_llm_backend(cfg.llm_backend, cfg.trtllm_engine_dir, cfg.trtllm_runner_path)

    return vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor, compute_dtype, llm_backend

# ===============================================================
# Main Entry
# ===============================================================
if __name__ == "__main__":
    # select modality
    pose_goal = False
    satellite = False
    image_goal = False
    lan_prompt = True

    # Goal definitions
    lan_inst_prompt = os.environ.get("OMNIVLA_INSTRUCTION", "move toward black office chair")
    if pose_goal:
        goal_lat = float(os.environ.get("OMNIVLA_GOAL_LAT", "37.8738930785863"))
        goal_lon = float(os.environ.get("OMNIVLA_GOAL_LON", "-122.26746181032362"))
        goal_compass = float(os.environ.get("OMNIVLA_GOAL_COMPASS", "0.0"))
        goal_utm = utm.from_latlon(goal_lat, goal_lon)
        goal_compass = -float(goal_compass) / 180.0 * math.pi
    else:
        goal_utm = None
        goal_compass = 0.0

    if image_goal:
        goal_image_path = os.environ.get(
            "OMNIVLA_GOAL_IMAGE",
            str(SCRIPT_DIR / "goal_img.jpg"),
        )
        goal_image_PIL = Image.open(goal_image_path).convert("RGB")
    else:
        goal_image_PIL = None
    save_root = os.environ.get("OMNIVLA_SAVE_DIR", str(SCRIPT_DIR))
    save_dir = make_unique_instruction_save_dir(save_root, lan_inst_prompt)
    print(f"Saving inference visualizations to: {save_dir}")

    # Define models (VLA, action_head, pose_projector, processor, etc.)
    cfg = InferenceConfig()
    model_load_t0 = time.perf_counter()
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor, compute_dtype, llm_backend = define_model(cfg)
    synchronize_if_cuda(device_id)
    model_load_elapsed = time.perf_counter() - model_load_t0
    print(f"Checkpoint load time: {model_load_elapsed:.4f} s")

    module_timer = configure_module_timer(vla, pose_projector, action_head, llm_backend, device_id)
    llm_profiler = create_llm_profiler(vla, llm_backend, device_id, cfg.enable_llm_profile)

    # Run inference
    inference = Inference(
        save_dir=save_dir,
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
        compute_dtype=compute_dtype,
        llm_backend=llm_backend,
        llm_profiler=llm_profiler,
        module_timer=module_timer,
    )
    if cfg.enable_benchmark:
        run_benchmark(
            inference,
            vla,
            pose_projector,
            device_id,
            module_timer,
            num_warmup=cfg.benchmark_warmup_iters,
            num_iters=cfg.benchmark_timed_iters,
        )

    use_ros = os.environ.get("OMNIVLA_USE_ROS", "1").lower() not in {"0", "false", "no"}
    if use_ros:
        if rclpy is None or CvBridge is None or Twist is None or ROSImage is None:
            raise ImportError("ROS2 dependencies are required for OMNIVLA_USE_ROS=1")
        rclpy.init(args=None)
        node = OmniVLAInferenceNode(inference)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    else:
        image_source = os.environ.get("OMNIVLA_IMAGE_SOURCE", "").lower()
        if image_source == "camera":
            preloaded_jpeg_lib = preload_jpeg_compat()
            if preloaded_jpeg_lib is None:
                print("Warning: Failed to preload a libjpeg compatibility library; nvarguscamerasrc may fail to load")
            else:
                print(f"Preloaded JPEG compatibility library: {preloaded_jpeg_lib}")

            width = int(os.environ.get("OMNIVLA_CAMERA_WIDTH", "1280"))
            height = int(os.environ.get("OMNIVLA_CAMERA_HEIGHT", "720"))
            fps = int(os.environ.get("OMNIVLA_CAMERA_FPS", "30"))
            rotate = os.environ.get("OMNIVLA_CAMERA_ROTATE", "1").lower() not in {"0", "false", "no"}
            flip_method = 2 if rotate else 0

            gst_str = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
                f"nvvidconv flip-method={flip_method} ! video/x-raw, width=960, height=544, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"videorate ! video/x-raw, framerate={fps}/1 ! "
                f"appsink max-buffers=1 drop=true sync=false"
            )

            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError("Internal camera not opened")

            def capture_loop():
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        print("Internal camera frame read failed")
                        time.sleep(0.05)
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    inference.set_current_image_from_pil(Image.fromarray(frame_rgb))

            capture_thread = threading.Thread(target=capture_loop, daemon=True)
            capture_thread.start()
            print("Started internal camera thread without ROS")

        inference.run()
