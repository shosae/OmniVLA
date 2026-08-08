"""Lightweight CUDA-aware inference timing helpers."""

import time
import statistics
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from llm_backends import LLMBackendInputs


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iters: int = 3
    timed_iters: int = 10
    enable_llm_profile: bool = True
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "BenchmarkConfig":
        enabled = os.environ.get("OMNIVLA_ENABLE_BENCHMARK", "1").lower() not in {"0", "false", "no"}
        return cls(
            warmup_iters=int(os.environ.get("OMNIVLA_BENCHMARK_WARMUP_ITERS", "3")),
            timed_iters=int(os.environ.get("OMNIVLA_BENCHMARK_TIMED_ITERS", "10")),
            enabled=enabled,
        )


def synchronize_if_cuda(device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None or device.type == "cuda":
        torch.cuda.synchronize(device=device)


class ModuleTimer:
    """Forward-hook based timer for coarse module-level profiling."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.handles = []
        self.active = False
        self._start_times = {}
        self.total_times = defaultdict(float)
        self.call_counts = defaultdict(int)

    def reset(self):
        self._start_times.clear()
        self.total_times.clear()
        self.call_counts.clear()

    def register(self, name: str, module: nn.Module):
        def pre_hook(_module, _inputs):
            if not self.active:
                return
            synchronize_if_cuda(self.device)
            self._start_times[name] = time.perf_counter()

        def post_hook(_module, _inputs, _output):
            if not self.active:
                return
            start_time = self._start_times.pop(name, None)
            if start_time is None:
                return
            synchronize_if_cuda(self.device)
            elapsed = time.perf_counter() - start_time
            self.total_times[name] += elapsed
            self.call_counts[name] += 1

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def summary_lines(self, divisor: int) -> list[str]:
        lines = []
        for name in sorted(self.total_times.keys()):
            total = self.total_times[name]
            count = self.call_counts[name]
            mean_per_iter = total / divisor if divisor > 0 else 0.0
            mean_per_call = total / count if count > 0 else 0.0
            lines.append(
                f"  {name:<16} iter_mean={mean_per_iter:.4f} s "
                f"call_mean={mean_per_call:.4f} s calls={count}"
            )
        return lines

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass
class LLMProfileResult:
    embed_tokens: float
    layers_total: float
    self_attn_total: float
    mlp_total: float
    input_layernorm_total: float
    post_attention_layernorm_total: float
    final_norm: float
    lm_head: float
    layer_times: List[float]
    self_attn_times: List[float]
    mlp_times: List[float]
    input_layernorm_times: List[float]
    post_attention_layernorm_times: List[float]
    call_counts: Dict[str, int]

    @property
    def layernorm_total(self) -> float:
        return self.input_layernorm_total + self.post_attention_layernorm_total

    @property
    def other_layer_ops_total(self) -> float:
        return max(
            self.layers_total - self.self_attn_total - self.mlp_total - self.layernorm_total,
            0.0,
        )

    def top_slowest_layers(self, top_k: int = 5) -> List[Tuple[int, float]]:
        ranked = sorted(enumerate(self.layer_times), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]


class LLMProfiler:
    def __init__(
        self,
        language_model: nn.Module,
        device: torch.device,
        top_k_layers: int = 5,
    ):
        self.language_model = language_model
        self.device = device
        self.top_k_layers = top_k_layers
        self.handles = []
        self.active = False
        self._start_times = {}
        self.total_times = defaultdict(float)
        self.call_counts = defaultdict(int)

        base_model = getattr(self.language_model, "model", None)
        self.embed_tokens = getattr(base_model, "embed_tokens", None)
        self.layers = list(getattr(base_model, "layers", []))
        self.final_norm = getattr(base_model, "norm", None)
        self.lm_head = getattr(self.language_model, "lm_head", None)
        self.num_layers = len(self.layers)

        self._register_hooks()

    def _register(self, name: str, module: Optional[nn.Module]) -> None:
        if module is None:
            return

        def pre_hook(_module, _inputs):
            if not self.active:
                return
            synchronize_if_cuda(self.device)
            self._start_times[name] = time.perf_counter()

        def post_hook(_module, _inputs, _output):
            if not self.active:
                return
            start_time = self._start_times.pop(name, None)
            if start_time is None:
                return
            synchronize_if_cuda(self.device)
            elapsed = time.perf_counter() - start_time
            self.total_times[name] += elapsed
            self.call_counts[name] += 1

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def _register_hooks(self) -> None:
        self._register("embed_tokens", self.embed_tokens)
        for layer_idx, layer in enumerate(self.layers):
            self._register(f"layer_{layer_idx}", layer)
            self._register(f"self_attn_{layer_idx}", getattr(layer, "self_attn", None))
            self._register(f"mlp_{layer_idx}", getattr(layer, "mlp", None))
            self._register(f"input_layernorm_{layer_idx}", getattr(layer, "input_layernorm", None))
            self._register(
                f"post_attention_layernorm_{layer_idx}",
                getattr(layer, "post_attention_layernorm", None),
            )
        self._register("final_norm", self.final_norm)
        self._register("lm_head", self.lm_head)

    def reset(self) -> None:
        self._start_times.clear()
        self.total_times.clear()
        self.call_counts.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _collect_series(self, prefix: str) -> List[float]:
        return [self.total_times.get(f"{prefix}_{layer_idx}", 0.0) for layer_idx in range(self.num_layers)]

    def profile(self, run_fn) -> LLMProfileResult:
        self.reset()
        self.active = True
        try:
            run_fn()
        finally:
            self.active = False
        return self.snapshot()

    def snapshot(self) -> LLMProfileResult:
        layer_times = self._collect_series("layer")
        self_attn_times = self._collect_series("self_attn")
        mlp_times = self._collect_series("mlp")
        input_layernorm_times = self._collect_series("input_layernorm")
        post_attention_layernorm_times = self._collect_series("post_attention_layernorm")

        return LLMProfileResult(
            embed_tokens=self.total_times.get("embed_tokens", 0.0),
            layers_total=sum(layer_times),
            self_attn_total=sum(self_attn_times),
            mlp_total=sum(mlp_times),
            input_layernorm_total=sum(input_layernorm_times),
            post_attention_layernorm_total=sum(post_attention_layernorm_times),
            final_norm=self.total_times.get("final_norm", 0.0),
            lm_head=self.total_times.get("lm_head", 0.0),
            layer_times=layer_times,
            self_attn_times=self_attn_times,
            mlp_times=mlp_times,
            input_layernorm_times=input_layernorm_times,
            post_attention_layernorm_times=post_attention_layernorm_times,
            call_counts=dict(self.call_counts),
        )

    @staticmethod
    def print_result(result: LLMProfileResult, title: str = "LLM INTERNAL PROFILE") -> None:
        top_layers = [f"{idx}:{elapsed:.4f}s" for idx, elapsed in result.top_slowest_layers()]
        print(f"================ {title} ================")
        print(f"embed_tokens: {result.embed_tokens:.4f}s calls={result.call_counts.get('embed_tokens', 0)}")
        print(f"layers_total: {result.layers_total:.4f}s")
        print(f"self_attn_total: {result.self_attn_total:.4f}s")
        print(f"mlp_total: {result.mlp_total:.4f}s")
        print(f"input_layernorm_total: {result.input_layernorm_total:.4f}s")
        print(f"post_attention_layernorm_total: {result.post_attention_layernorm_total:.4f}s")
        print(f"other_layer_ops_total: {result.other_layer_ops_total:.4f}s")
        print(f"final_norm: {result.final_norm:.4f}s")
        print(f"lm_head: {result.lm_head:.4f}s")
        print(f"top_slowest_layers: {top_layers}")
        print(f"layer_times: {[round(v, 4) for v in result.layer_times]}")
        print(f"self_attn_times: {[round(v, 4) for v in result.self_attn_times]}")
        print(f"mlp_times: {[round(v, 4) for v in result.mlp_times]}")
        if result.call_counts.get("embed_tokens", 0) == 0:
            print("note: embed_tokens not called because language_model received inputs_embeds")
        print("================================================")

    @staticmethod
    def print_aggregate(results: List[LLMProfileResult]) -> None:
        if not results:
            return

        num_results = len(results)
        num_layers = len(results[0].layer_times)
        mean_layer_times = [
            sum(result.layer_times[layer_idx] for result in results) / num_results
            for layer_idx in range(num_layers)
        ]
        mean_self_attn_times = [
            sum(result.self_attn_times[layer_idx] for result in results) / num_results
            for layer_idx in range(num_layers)
        ]
        mean_mlp_times = [
            sum(result.mlp_times[layer_idx] for result in results) / num_results
            for layer_idx in range(num_layers)
        ]

        aggregate = LLMProfileResult(
            embed_tokens=sum(result.embed_tokens for result in results) / num_results,
            layers_total=sum(result.layers_total for result in results) / num_results,
            self_attn_total=sum(result.self_attn_total for result in results) / num_results,
            mlp_total=sum(result.mlp_total for result in results) / num_results,
            input_layernorm_total=sum(result.input_layernorm_total for result in results) / num_results,
            post_attention_layernorm_total=sum(result.post_attention_layernorm_total for result in results) / num_results,
            final_norm=sum(result.final_norm for result in results) / num_results,
            lm_head=sum(result.lm_head for result in results) / num_results,
            layer_times=mean_layer_times,
            self_attn_times=mean_self_attn_times,
            mlp_times=mean_mlp_times,
            input_layernorm_times=[
                sum(result.input_layernorm_times[layer_idx] for result in results) / num_results
                for layer_idx in range(num_layers)
            ],
            post_attention_layernorm_times=[
                sum(result.post_attention_layernorm_times[layer_idx] for result in results) / num_results
                for layer_idx in range(num_layers)
            ],
            call_counts={},
        )

        top_layers = [f"{idx}:{elapsed:.4f}s" for idx, elapsed in aggregate.top_slowest_layers()]
        print(f"================ LLM INTERNAL PROFILE SUMMARY (n={num_results}) ================")
        print(f"embed_tokens: {aggregate.embed_tokens:.4f}s")
        print(f"layers_total: {aggregate.layers_total:.4f}s")
        print(f"self_attn_total: {aggregate.self_attn_total:.4f}s")
        print(f"mlp_total: {aggregate.mlp_total:.4f}s")
        print(f"input_layernorm_total: {aggregate.input_layernorm_total:.4f}s")
        print(f"post_attention_layernorm_total: {aggregate.post_attention_layernorm_total:.4f}s")
        print(f"other_layer_ops_total: {aggregate.other_layer_ops_total:.4f}s")
        print(f"final_norm: {aggregate.final_norm:.4f}s")
        print(f"lm_head: {aggregate.lm_head:.4f}s")
        print(f"top_slowest_layers: {top_layers}")
        print(f"mean_layer_times: {[round(v, 4) for v in aggregate.layer_times]}")
        print(f"mean_self_attn_times: {[round(v, 4) for v in aggregate.self_attn_times]}")
        print(f"mean_mlp_times: {[round(v, 4) for v in aggregate.mlp_times]}")
        print("=========================================================")


def configure_module_timer(vla, pose_projector, action_head, llm_backend, device: torch.device) -> ModuleTimer:
    timer = ModuleTimer(device=device)
    timer.register("vision_backbone", vla.vision_backbone)
    timer.register("projector", vla.projector)
    language_model = llm_backend.profile_target(vla)
    if language_model is not None:
        timer.register("language_model_full_forward", language_model)
    timer.register("pose_projector", pose_projector)
    timer.register("action_head", action_head)
    return timer


def create_llm_profiler(vla, llm_backend, device: torch.device, enabled: bool) -> Optional[LLMProfiler]:
    if not enabled or not llm_backend.supports_llm_profiler:
        return None
    return LLMProfiler(language_model=vla.language_model, device=device)


def run_llm_profile(inference, vla, pose_projector, device: torch.device, module_timer: Optional[ModuleTimer], print_result: bool = False) -> Optional[LLMProfileResult]:
    if inference.llm_profiler is None or inference.last_profile_context is None:
        return None

    timer_was_active = module_timer.active if module_timer is not None else False
    if module_timer is not None:
        module_timer.active = False

    batch = inference.last_profile_context["batch"]
    modality_id = inference.last_profile_context["modality_id"]

    def forward_once():
        with torch.inference_mode(), torch.autocast("cuda", dtype=inference.compute_dtype):
            inference.llm_backend.forward_action_hidden_states(
                vla.eval(),
                LLMBackendInputs(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(dtype=inference.compute_dtype, device=device),
                    modality_id=modality_id.to(dtype=inference.compute_dtype, device=device),
                    labels=batch["labels"].to(device),
                    proprio=batch["goal_pose"].to(dtype=inference.compute_dtype, device=device),
                    proprio_projector=pose_projector.eval(),
                    noisy_actions=None,
                    noisy_action_projector=None,
                    diffusion_timestep_embeddings=None,
                    use_film=False,
                ),
            )

    inference.last_llm_profile = inference.llm_profiler.profile(forward_once)
    if module_timer is not None:
        module_timer.active = timer_was_active
    if print_result:
        LLMProfiler.print_result(inference.last_llm_profile)
    return inference.last_llm_profile


def run_benchmark(inference, vla, pose_projector, device: torch.device, module_timer: Optional[ModuleTimer], num_warmup: int = 3, num_iters: int = 10):
    print(f"Running inference benchmark: warmup={num_warmup}, timed_iters={num_iters}")
    if module_timer is not None:
        module_timer.active = False
        module_timer.reset()
    if hasattr(inference.llm_backend, "reset_timing_stats"):
        inference.llm_backend.reset_timing_stats()

    for idx in range(num_warmup):
        inference.run_omnivla(save_behavior=False, profile_language_model=False, log_output=False)
        print(f"Warmup {idx + 1}/{num_warmup} complete")

    timings = []
    llm_profiles = []
    last_output = None
    if module_timer is not None:
        module_timer.reset()
        module_timer.active = True
    if hasattr(inference.llm_backend, "reset_timing_stats"):
        inference.llm_backend.reset_timing_stats()
    for idx in range(num_iters):
        synchronize_if_cuda(device)
        started = time.perf_counter()
        last_output = inference.run_omnivla(save_behavior=False, profile_language_model=False, log_output=False)
        synchronize_if_cuda(device)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        profile = run_llm_profile(inference, vla, pose_projector, device, module_timer)
        if profile is not None:
            llm_profiles.append(profile)
        print(f"Timed run {idx + 1}/{num_iters}: {elapsed:.4f} s")
    if module_timer is not None:
        module_timer.active = False

    print("Inference benchmark summary")
    print(f"  mean   : {sum(timings) / len(timings):.4f} s")
    print(f"  median : {statistics.median(timings):.4f} s")
    print(f"  min    : {min(timings):.4f} s")
    print(f"  max    : {max(timings):.4f} s")
    if module_timer is not None:
        print("Module timing summary")
        for line in module_timer.summary_lines(num_iters):
            print(line)
    if hasattr(inference.llm_backend, "timing_summary_lines"):
        for line in inference.llm_backend.timing_summary_lines():
            print(line)
    if inference.llm_profiler is not None:
        LLMProfiler.print_aggregate(llm_profiles)
    return last_output, timings
