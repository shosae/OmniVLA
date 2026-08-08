"""Small helpers shared by OmniVLA inference."""

import json
import os
import textwrap

import torch
from PIL import Image


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


def resolve_modality_flags() -> tuple[bool, bool, bool, bool]:
    modality = os.environ.get("OMNIVLA_MODALITY", "language_only").strip().lower()
    mapping = {
        "language_only": (False, False, False, True),
        "image_only": (False, False, True, False),
        "language_and_pose": (True, False, False, True),
        "pose_only": (True, False, False, False),
    }
    if modality not in mapping:
        raise ValueError(f"Unsupported OMNIVLA_MODALITY `{modality}`. Use one of: {', '.join(sorted(mapping))}")
    return mapping[modality]


def get_modality_id(pose_goal: bool, satellite: bool, image_goal: bool, lan_prompt: bool) -> torch.Tensor:
    mapping = {
        (False, True, False, False): 0,
        (True, True, False, False): 1,
        (False, True, True, False): 2,
        (True, True, True, False): 3,
        (True, False, False, False): 4,
        (True, False, True, False): 5,
        (False, False, True, False): 6,
        (False, False, False, True): 7,
        (True, False, False, True): 8,
    }
    try:
        return torch.as_tensor([mapping[(pose_goal, satellite, image_goal, lan_prompt)]], dtype=torch.float32)
    except KeyError as exc:
        raise ValueError("Unsupported modality configuration") from exc


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_cached_rgb_image(path: str, cached_path: str | None, cached_image: Image.Image | None) -> tuple[str, Image.Image, Image.Image]:
    if path != cached_path or cached_image is None:
        cached_path = path
        cached_image = load_rgb_image(path)
    return cached_path, cached_image, cached_image.copy()


def save_prediction_record(actions: torch.Tensor, modality: torch.Tensor, num_actions: int, action_dim: int) -> None:
    prediction_path = os.environ.get("OMNIVLA_PREDICTIONS_PATH")
    if not prediction_path:
        return

    predicted_action = actions[0].detach().float().cpu()
    if tuple(predicted_action.shape) != (num_actions, action_dim):
        raise ValueError(f"Expected predicted action shape ({num_actions}, {action_dim}), got {tuple(predicted_action.shape)}")
    record = {
        "cache_file": os.environ.get("OMNIVLA_CACHE_FILE"),
        "sample_id": os.environ.get("OMNIVLA_SAMPLE_ID"),
        "dataset": os.environ.get("OMNIVLA_DATASET"),
        "modality_id": int(modality.detach().float().reshape(-1)[0].item()),
        "predicted_action": predicted_action.tolist(),
    }
    output_path = os.path.abspath(prediction_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def add_visualization_metadata(fig, environment: str, model: str, prompt: str) -> None:
    metadata = f"Env: {environment}\nModel: {model}\nPrompt: {textwrap.fill(str(prompt).strip() or 'x', width=42)}"
    fig.text(
        0.985, 0.975, metadata,
        ha="right", va="top", fontsize=14,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "black", "alpha": 0.88},
    )
