#!/usr/bin/env python3
"""Replay a shared OmniVLA decoder-input cache through fp16, AWQ, or TensorRT."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import inference as omnivla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive")
    paths = sorted(args.cache_dir.glob("*.pt"))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"No .pt cache files under {args.cache_dir}")
    runner, config = omnivla.create_inference()
    device = runner.device_id

    def run(payload: dict) -> tuple[torch.Tensor, float, float]:
        embeds = payload["inputs_embeds"].to(device=device, dtype=torch.float16, non_blocking=True)
        mask = payload["attention_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
        if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[2] != runner.vla.llm_dim:
            raise ValueError(f"Unexpected inputs_embeds shape: {tuple(embeds.shape)}")
        if mask.shape != embeds.shape[:2]:
            raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match inputs_embeds")
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            hidden = runner.llm_backend.forward_cached_hidden_states(embeds, mask, runner.vla)
            actions = runner.action_head.eval().predict_action(
                hidden[:, -33:-1, :].to(dtype=runner.compute_dtype),
                torch.tensor([int(payload["modality_id"])], dtype=runner.compute_dtype, device=device),
            )
        if tuple(actions.shape) != (1, 8, 4):
            raise ValueError(f"Unexpected predicted_action shape: {tuple(actions.shape)}")
        torch.cuda.synchronize(device)
        return actions, (time.perf_counter() - started) * 1000.0, runner.llm_backend.last_llm_inference_ms

    for index in range(args.warmup):
        payload = torch.load(paths[index % len(paths)], map_location="cpu", weights_only=False)
        run(payload)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    records = []
    for repeat in range(args.repeats):
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            actions, latency_ms, llm_ms = run(payload)
            records.append({
                "backend": config.backend,
                "cache_file": str(path),
                "sample_id": path.stem,
                "repeat_index": repeat,
                "dataset": payload["dataset"],
                "modality_id": int(payload["modality_id"]),
                "sequence_length": int(payload["inputs_embeds"].shape[1]),
                "valid_sequence_length": int(payload["attention_mask"].sum().item()),
                "latency_ms": latency_ms,
                "llm_inference_ms": llm_ms,
                "llm_memory_mib": runner.llm_backend.last_llm_memory_mib,
                "predicted_action": actions[0].detach().float().cpu().tolist(),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    values = [record["latency_ms"] for record in records]
    print(
        f"saved {len(records)} {config.backend} records to {args.output}\n"
        f"latency mean={statistics.fmean(values):.3f}ms median={statistics.median(values):.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
