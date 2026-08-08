"""Optional inference debug output."""

import os
import sys
from typing import Optional

import torch


def _enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in {"0", "false", "no", "off", ""}


def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
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
        ranges = ranges[:limit] + [f"...(+{len(ranges) - limit} ranges)"]
    return "[" + ", ".join(ranges) + "]"


def debug_action_summary(
    backend_name: str,
    text_hidden_states: Optional[torch.Tensor],
    actions_hidden_states: torch.Tensor,
    predicted_actions: torch.Tensor,
    current_action_mask: torch.Tensor,
    next_actions_mask: torch.Tensor,
) -> None:
    if not _enabled("OMNIVLA_ACTION_DEBUG"):
        return

    token_mask = current_action_mask | next_actions_mask
    token_indices = token_mask[0].nonzero(as_tuple=False).flatten().tolist() if token_mask.ndim == 2 else token_mask.nonzero(as_tuple=False).flatten().tolist()
    print(
        f"[OmniVLA Action Debug] backend={backend_name} "
        f"action_token_count={len(token_indices)} "
        f"action_token_ranges={_format_index_ranges(token_indices)}",
        file=sys.stderr,
    )
    if text_hidden_states is not None:
        print(f"[OmniVLA Action Debug] {_tensor_stats('text_hidden_states', text_hidden_states)}", file=sys.stderr)
    print(f"[OmniVLA Action Debug] {_tensor_stats('actions_hidden_states', actions_hidden_states)}", file=sys.stderr)
    print(f"[OmniVLA Action Debug] {_tensor_stats('predicted_actions', predicted_actions)}", file=sys.stderr)

    sample_tokens = actions_hidden_states[0, : min(4, actions_hidden_states.shape[1]), : min(8, actions_hidden_states.shape[2])].detach().float().cpu().numpy().round(4).tolist()
    sample_actions = predicted_actions[0, : min(3, predicted_actions.shape[1])].detach().float().cpu().numpy().round(4).tolist()
    print(f"[OmniVLA Action Debug] actions_hidden_sample={sample_tokens}", file=sys.stderr)
    print(f"[OmniVLA Action Debug] predicted_actions_sample={sample_actions}", file=sys.stderr)
