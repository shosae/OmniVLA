from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
import tempfile
import statistics
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


OMNIVLA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDGELLM_PLUGIN_PATH = OMNIVLA_ROOT / "runtime" / "trtllm" / "lib" / "libNvInfer_edgellm_plugin.so"


@dataclass
class LLMBackendInputs:
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    pixel_values: torch.FloatTensor
    labels: torch.LongTensor
    modality_id: Optional[torch.FloatTensor] = None
    proprio: Optional[torch.Tensor] = None
    proprio_projector: Optional[nn.Module] = None
    noisy_actions: Optional[torch.Tensor] = None
    noisy_action_projector: Optional[nn.Module] = None
    diffusion_timestep_embeddings: Optional[torch.Tensor] = None
    use_film: bool = False


class BaseLLMBackend:
    name = "base"
    last_llm_inference_ms: Optional[float] = None
    last_llm_memory_mib: dict[str, float] = {}

    def forward_action_hidden_states(self, vla: nn.Module, inputs: LLMBackendInputs) -> torch.Tensor:
        raise NotImplementedError

    def profile_target(self, vla: nn.Module) -> Optional[nn.Module]:
        return getattr(vla, "language_model", None)

    @property
    def supports_llm_profiler(self) -> bool:
        return False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _nvml_process_memory_mib() -> Optional[float]:
    """Current process GPU occupancy, as reported by nvidia-smi/NVML."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in output.splitlines():
        pid, memory = (part.strip() for part in line.split(",", 1))
        if pid == str(os.getpid()):
            try:
                return float(memory)
            except ValueError:
                return None
    return None


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


def _build_external_attention_inputs(attention_mask_1d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask_1d.ndim != 2:
        raise ValueError(f"Expected rank-2 attention_mask, got shape {tuple(attention_mask_1d.shape)}")

    batch_size, seq_len = attention_mask_1d.shape
    token_mask = attention_mask_1d.to(dtype=torch.bool)
    device = attention_mask_1d.device
    causal = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    ).unsqueeze(0).expand(batch_size, -1, -1)
    query_valid = token_mask.unsqueeze(2)
    key_valid = token_mask.unsqueeze(1)
    attention_mask_2d = (causal & query_valid & key_valid).to(dtype=torch.int32).contiguous()
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch_size, -1)
    return position_ids.contiguous(), attention_mask_2d


def _handle_version_mismatch(
    engine_config: dict,
    engine_version: str,
    runtime_version: str,
) -> None:
    if engine_version == runtime_version:
        return

    message = (
        "omni-TRT-Edge-LLM engine/runtime version mismatch: "
        f"engine config expects `{engine_version}` but `_omnivla_trtllm` is `{runtime_version}`."
    )
    raise RuntimeError(message)


def validate_trtllm_engine_version(
    engine_dir: str,
    pybind_module_dir: Optional[str] = None,
    plugin_path: Optional[str] = None,
) -> None:
    if os.environ.get("OMNIVLA_TRTLLM_USE_SUBPROCESS", "0") == "1":
        return

    runtime_dir = OMNIVLA_ROOT / "runtime" / "trtllm"
    pybind_dir = pybind_module_dir or os.environ.get(
        "OMNIVLA_TRTLLM_PYBIND_DIR",
        str(runtime_dir / "python"),
    )
    plugin = plugin_path or os.environ.get(
        "EDGELLM_PLUGIN_PATH",
        str(DEFAULT_EDGELLM_PLUGIN_PATH),
    )

    with open(os.path.join(engine_dir, "config.json"), "r", encoding="utf-8") as f:
        engine_config = json.load(f)
    engine_version = str(engine_config.get("edgellm_version", "unknown"))

    if pybind_dir not in sys.path:
        sys.path.insert(0, pybind_dir)
    os.environ["EDGELLM_PLUGIN_PATH"] = plugin
    import _omnivla_trtllm

    runtime_version = getattr(_omnivla_trtllm, "runtime_version", None)
    if runtime_version:
        _handle_version_mismatch(engine_config, engine_version, runtime_version)


class PyTorchFastLLMBackend(BaseLLMBackend):
    name = "pytorch_fast"

    def __init__(self):
        self.last_llm_inference_ms: Optional[float] = None
        self.last_llm_memory_mib: dict[str, float] = {}

    def forward_action_hidden_states(self, vla: nn.Module, inputs: LLMBackendInputs) -> torch.Tensor:
        llm_inputs = vla.prepare_action_forward_inputs(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            labels=inputs.labels,
            modality_id=inputs.modality_id,
            proprio=inputs.proprio,
            proprio_projector=inputs.proprio_projector,
            noisy_actions=inputs.noisy_actions,
            noisy_action_projector=inputs.noisy_action_projector,
            diffusion_timestep_embeddings=inputs.diffusion_timestep_embeddings,
            use_film=inputs.use_film,
        )
        language_model = getattr(vla.language_model, "model", None)
        if language_model is None:
            raise RuntimeError("language_model does not expose a `.model` backbone for fast inference")
        allocated_before = torch.cuda.memory_allocated() / 2**20
        reserved_before = torch.cuda.memory_reserved() / 2**20
        nvml_before = _nvml_process_memory_mib()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = language_model(
            input_ids=None,
            attention_mask=llm_inputs["attention_mask"],
            position_ids=None,
            past_key_values=None,
            inputs_embeds=llm_inputs["inputs_embeds"],
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        end.record()
        end.synchronize()
        self.last_llm_inference_ms = start.elapsed_time(end)
        self.last_llm_memory_mib = {
            "torch_allocated_before": allocated_before,
            "torch_allocated_after": torch.cuda.memory_allocated() / 2**20,
            "torch_peak_allocated": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_allocated_delta": max(torch.cuda.max_memory_allocated() / 2**20 - allocated_before, 0.0),
            "torch_reserved_before": reserved_before,
            "torch_peak_reserved": torch.cuda.max_memory_reserved() / 2**20,
        }
        if nvml_before is not None:
            self.last_llm_memory_mib["nvml_process_before"] = nvml_before
        nvml_after = _nvml_process_memory_mib()
        if nvml_after is not None:
            self.last_llm_memory_mib["nvml_process_after"] = nvml_after
        return output.last_hidden_state

    def profile_target(self, vla: nn.Module) -> Optional[nn.Module]:
        language_model = getattr(vla, "language_model", None)
        return getattr(language_model, "model", language_model)

    @property
    def supports_llm_profiler(self) -> bool:
        return True


class TensorRTLLMBackend(BaseLLMBackend):
    name = "tensorrt_llm"

    def __init__(self, engine_dir: str, runner_path: Optional[str] = None):
        runtime_dir = OMNIVLA_ROOT / "runtime" / "trtllm"
        self.engine_dir = engine_dir
        self.runner_path = runner_path or os.environ.get(
            "OMNIVLA_LLM_EMBED_RUNNER",
            str(runtime_dir / "bin" / "llm_embed_inference"),
        )
        self.plugin_path = os.environ.get(
            "EDGELLM_PLUGIN_PATH",
            str(DEFAULT_EDGELLM_PLUGIN_PATH),
        )
        self.pybind_module_dir = os.environ.get(
            "OMNIVLA_TRTLLM_PYBIND_DIR",
            str(runtime_dir / "python"),
        )
        self.use_subprocess = os.environ.get("OMNIVLA_TRTLLM_USE_SUBPROCESS", "0") == "1"
        self.force_host_copy = os.environ.get("OMNIVLA_TRTLLM_FORCE_HOST_COPY", "0") == "1"
        self._engine_config = self._load_engine_config()
        builder_config = self._engine_config.get("builder_config", {})
        self.engine_version = str(self._engine_config.get("edgellm_version", "unknown"))
        self.external_attention_inputs = bool(
            builder_config.get("external_attention_inputs", self._engine_config.get("external_attention_inputs", False))
        ) or _env_flag("OMNIVLA_TRTLLM_EXTERNAL_ATTN", False)
        self._runner = None
        self.runtime_version: Optional[str] = None
        self._timings: list[dict[str, float]] = []
        self.debug_enabled = _env_flag("OMNIVLA_TRTLLM_DEBUG", False)
        self._validate_paths()
        if not self.use_subprocess:
            self._runner = self._load_pybind_runner()

    def _debug_log(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[TensorRTLLMBackend] {message}", file=sys.stderr)

    def _validate_paths(self) -> None:
        engine_file = os.path.join(self.engine_dir, "llm.engine")
        config_file = os.path.join(self.engine_dir, "config.json")
        required = [self.plugin_path, engine_file, config_file]
        if self.use_subprocess:
            required.append(self.runner_path)
        else:
            required.append(self.pybind_module_dir)
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "TensorRT LLM backend is missing required file(s): " + ", ".join(missing)
            )

    def _load_engine_config(self) -> dict:
        config_file = os.path.join(self.engine_dir, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pybind_runner(self):
        if self.pybind_module_dir not in sys.path:
            sys.path.insert(0, self.pybind_module_dir)
        os.environ["EDGELLM_PLUGIN_PATH"] = self.plugin_path
        try:
            import _omnivla_trtllm
        except ImportError as exc:
            raise ImportError(
                "Failed to import `_omnivla_trtllm`. Build it with "
                "the OmniVLA TensorRT adapter, or set "
                f"OMNIVLA_TRTLLM_USE_SUBPROCESS=1. Original error: {exc}"
            ) from exc
        self.runtime_version = getattr(_omnivla_trtllm, "runtime_version", None)
        if self.runtime_version:
            _handle_version_mismatch(self._engine_config, self.engine_version, self.runtime_version)
        return _omnivla_trtllm.OmniVLAEmbedRunner(self.engine_dir)

    def _prepare_action_forward_inputs(self, vla: nn.Module, inputs: LLMBackendInputs) -> dict[str, torch.Tensor]:
        if hasattr(vla, "prepare_action_forward_inputs"):
            return vla.prepare_action_forward_inputs(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                labels=inputs.labels,
                modality_id=inputs.modality_id,
                proprio=inputs.proprio,
                proprio_projector=inputs.proprio_projector,
                noisy_actions=inputs.noisy_actions,
                noisy_action_projector=inputs.noisy_action_projector,
                diffusion_timestep_embeddings=inputs.diffusion_timestep_embeddings,
                use_film=inputs.use_film,
            )

        input_embeddings = vla.get_input_embeddings()(inputs.input_ids)
        all_actions_mask = vla._process_action_masks(inputs.labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected_patch_embeddings = vla._process_vision_features(
            inputs.pixel_values, language_embeddings, inputs.use_film
        )
        projected_patch_embeddings = vla._process_proprio_features(
            projected_patch_embeddings, inputs.proprio, inputs.proprio_projector
        )

        if inputs.diffusion_timestep_embeddings is not None:
            projected_patch_embeddings = torch.cat(
                (projected_patch_embeddings, inputs.diffusion_timestep_embeddings), dim=1
            )

        if inputs.noisy_actions is not None:
            batch_size = inputs.noisy_actions.shape[0]
            noisy_actions = inputs.noisy_actions.reshape(batch_size, -1).unsqueeze(-1)
            noisy_action_features = inputs.noisy_action_projector(noisy_actions)
            input_embeddings = vla._replace_input_embeddings(
                input_embeddings, all_actions_mask, noisy_action_features
            )
        else:
            input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)

        if inputs.modality_id is not None:
            multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention_MMN(
                input_embeddings, projected_patch_embeddings, inputs.attention_mask, None, inputs.modality_id
            )
        else:
            multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, inputs.attention_mask
            )

        return {
            "inputs_embeds": multimodal_embeddings,
            "attention_mask": multimodal_attention_mask,
        }

    def forward_action_hidden_states(self, vla: nn.Module, inputs: LLMBackendInputs) -> torch.Tensor:
        llm_inputs = self._prepare_action_forward_inputs(vla, inputs)

        inputs_embeds = llm_inputs["inputs_embeds"].detach()
        attention_mask = llm_inputs["attention_mask"].detach()
        if inputs_embeds.ndim != 3:
            raise ValueError(f"Expected inputs_embeds rank 3, got shape {tuple(inputs_embeds.shape)}")

        batch_size, seq_len, hidden_size = inputs_embeds.shape
        if batch_size != 1:
            raise ValueError(f"TensorRT LLM backend currently supports batch_size=1, got {batch_size}")

        token_mask = attention_mask.to(dtype=torch.bool)
        run_inputs_embeds = inputs_embeds
        scatter_mask = None
        removed_indices: list[int] = []
        attention_pos_ids: Optional[torch.Tensor] = None
        attention_mask_2d: Optional[torch.Tensor] = None
        if self.external_attention_inputs:
            attention_pos_ids, attention_mask_2d = _build_external_attention_inputs(attention_mask)
        elif not torch.all(token_mask):
            scatter_mask = token_mask[0]
            removed_indices = (~scatter_mask).nonzero(as_tuple=False).flatten().tolist()
            run_inputs_embeds = inputs_embeds[:, scatter_mask, :]

        _, run_seq_len, _ = run_inputs_embeds.shape
        device = inputs_embeds.device
        run_inputs_embeds_fp16 = run_inputs_embeds.to(dtype=torch.float16).contiguous()
        output_seq_len = run_seq_len

        if self.debug_enabled:
            input_seq_len = int(inputs.input_ids.shape[1])
            projected_seq_len = seq_len - input_seq_len
            projected_removed = [idx for idx in removed_indices if idx < 1 + projected_seq_len]
            text_removed = [idx for idx in removed_indices if idx >= 1 + projected_seq_len]
            modality_value = None
            if inputs.modality_id is not None:
                modality_value = float(inputs.modality_id.detach().float().reshape(-1)[0].item())
            self._debug_log(
                "forward_action_hidden_states "
                f"modality_id={modality_value} "
                f"external_attention_inputs={self.external_attention_inputs} "
                f"full_seq_len={seq_len} "
                f"input_seq_len={input_seq_len} "
                f"projected_seq_len={projected_seq_len} "
                f"pruned_seq_len={run_seq_len} "
                f"removed_tokens={len(removed_indices)} "
                f"output_seq_len={output_seq_len}"
            )
            if removed_indices:
                self._debug_log(f"removed_token_ranges={_format_index_ranges(removed_indices)}")
                self._debug_log(
                    "removed_token_ranges_split "
                    f"projected={_format_index_ranges(projected_removed)} "
                    f"text={_format_index_ranges(text_removed)}"
                )
            self._debug_log(_tensor_stats_line("inputs_embeds_full", inputs_embeds))
            if run_seq_len != seq_len:
                self._debug_log(_tensor_stats_line("inputs_embeds_pruned", run_inputs_embeds_fp16))
            if attention_pos_ids is not None and attention_mask_2d is not None:
                masked_indices = (~token_mask[0]).nonzero(as_tuple=False).flatten().tolist()
                self._debug_log(f"masked_token_ranges={_format_index_ranges(masked_indices)}")
                self._debug_log(
                    "external_attention_shapes "
                    f"position_ids={tuple(attention_pos_ids.shape)} "
                    f"attention_mask={tuple(attention_mask_2d.shape)}"
                )

        if self._runner is not None:
            expected_shape = (batch_size, output_seq_len, hidden_size)
            if (
                not self.force_host_copy
                and hasattr(self._runner, "forward_cuda")
                and run_inputs_embeds_fp16.is_cuda
            ):
                hidden_states = torch.empty(expected_shape, dtype=torch.float16, device=device)
                stream = torch.cuda.current_stream(device=device)
                try:
                    self._runner.forward_cuda(
                        int(run_inputs_embeds_fp16.data_ptr()),
                        int(hidden_states.data_ptr()),
                        batch_size,
                        run_seq_len,
                        hidden_size,
                        int(attention_pos_ids.data_ptr()) if attention_pos_ids is not None else 0,
                        int(attention_mask_2d.data_ptr()) if attention_mask_2d is not None else 0,
                        int(attention_mask_2d.shape[2]) if attention_mask_2d is not None else 0,
                        int(stream.cuda_stream),
                    )
                except TypeError:
                    self._runner.forward_cuda(
                        int(run_inputs_embeds_fp16.data_ptr()),
                        int(hidden_states.data_ptr()),
                        batch_size,
                        run_seq_len,
                        hidden_size,
                        int(stream.cuda_stream),
                    )
                self._timings.append({k: float(v) for k, v in self._runner.last_timing().items()})
            else:
                embeds_np = run_inputs_embeds_fp16.cpu().numpy()
                hidden_u16 = self._runner.forward(
                    embeds_np.view(np.uint16),
                    attention_pos_ids.cpu().numpy() if attention_pos_ids is not None else None,
                    attention_mask_2d.cpu().numpy() if attention_mask_2d is not None else None,
                )
                self._timings.append({k: float(v) for k, v in self._runner.last_timing().items()})
                hidden_np = np.asarray(hidden_u16).view(np.float16)
                expected_elems = batch_size * output_seq_len * hidden_size
                if hidden_np.size != expected_elems:
                    raise RuntimeError(
                        f"TensorRT hidden_states size mismatch: expected {expected_elems}, got {hidden_np.size}"
                    )
                hidden_states = torch.from_numpy(hidden_np.reshape(expected_shape)).to(device=device)
        else:
            embeds_np = run_inputs_embeds_fp16.cpu().numpy()
            with tempfile.TemporaryDirectory(prefix="omnivla_trtllm_") as tmpdir:
                embeds_path = os.path.join(tmpdir, "inputs_embeds.fp16.bin")
                hidden_path = os.path.join(tmpdir, "hidden_states.fp16.bin")
                embeds_np.tofile(embeds_path)
                attention_pos_ids_path = None
                attention_mask_path = None
                if attention_pos_ids is not None and attention_mask_2d is not None:
                    attention_pos_ids_path = os.path.join(tmpdir, "attention_pos_id.int32.bin")
                    attention_mask_path = os.path.join(tmpdir, "attention_mask.int32.bin")
                    attention_pos_ids.cpu().numpy().astype(np.int32, copy=False).tofile(attention_pos_ids_path)
                    attention_mask_2d.cpu().numpy().astype(np.int32, copy=False).tofile(attention_mask_path)
                env = os.environ.copy()
                env["EDGELLM_PLUGIN_PATH"] = self.plugin_path

                cmd = [
                    self.runner_path,
                    "--engineDir", self.engine_dir,
                    "--embedsFile", embeds_path,
                    "--seqLen", str(run_seq_len),
                    "--batchSize", str(batch_size),
                    "--outFile", hidden_path,
                ]
                if attention_pos_ids_path is not None and attention_mask_path is not None:
                    cmd.extend(
                        [
                            "--attentionPosIdFile", attention_pos_ids_path,
                            "--attentionMaskFile", attention_mask_path,
                            "--attentionMaskWidth", str(int(attention_mask_2d.shape[2])),
                        ]
                    )
                subprocess.run(cmd, check=True, env=env)

                expected_elems = batch_size * output_seq_len * hidden_size
                hidden_np = np.fromfile(hidden_path, dtype=np.float16)
                if hidden_np.size != expected_elems:
                    raise RuntimeError(
                        f"TensorRT hidden_states size mismatch: expected {expected_elems}, got {hidden_np.size}"
                    )
                hidden_states = torch.from_numpy(hidden_np.reshape(batch_size, output_seq_len, hidden_size)).to(
                    device=device
                )

        if self.debug_enabled:
            self._debug_log(_tensor_stats_line("hidden_states_engine", hidden_states))

        if scatter_mask is None or self.external_attention_inputs:
            return hidden_states

        full_hidden_states = torch.zeros(
            (batch_size, seq_len, hidden_size),
            dtype=hidden_states.dtype,
            device=device,
        )
        full_hidden_states[:, scatter_mask, :] = hidden_states
        if self.debug_enabled:
            self._debug_log(_tensor_stats_line("hidden_states_scattered", full_hidden_states))
        return full_hidden_states

    def profile_target(self, vla: nn.Module) -> Optional[nn.Module]:
        return None

    @property
    def supports_llm_profiler(self) -> bool:
        return False

    def reset_timing_stats(self) -> None:
        self._timings.clear()

    def timing_summary_lines(self) -> list[str]:
        if not self._timings:
            return []
        lines = ["TensorRT LLM timing summary"]
        for key in ("h2d", "execute", "d2h", "total"):
            values = [entry[key] for entry in self._timings if key in entry]
            if not values:
                continue
            lines.append(
                f"  trtllm_{key:<7} mean={sum(values) / len(values):.4f} s "
                f"median={statistics.median(values):.4f} s "
                f"min={min(values):.4f} s max={max(values):.4f} s calls={len(values)}"
            )
        return lines


class TensorRTEdgeBackend(TensorRTLLMBackend):
    """Run the validated OmniVLA TensorRT-Edge-LLM engine subprocess."""

    name = "tensorrt_edge"

    def __init__(self, engine_dir: str, plugin_path: str, runner_path: str, kv_cache_capacity: Optional[int] = None):
        self.engine_dir = engine_dir
        self.plugin_path = plugin_path
        self.runner_path = runner_path
        self._engine_config = self._load_engine_config()
        builder_config = self._engine_config.get("builder_config", {})
        self.kv_cache_capacity = kv_cache_capacity or int(
            builder_config.get("max_kv_cache_capacity", builder_config.get("max_input_len", 320))
        )
        self.last_llm_inference_ms: Optional[float] = None
        self.last_llm_memory_mib: dict[str, float] = {}
        required = [
            os.path.join(self.engine_dir, "llm.engine"),
            os.path.join(self.engine_dir, "config.json"),
            self.plugin_path,
            self.runner_path,
        ]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("TensorRT engine backend is missing required file(s): " + ", ".join(missing))

    def forward_action_hidden_states(self, vla: nn.Module, inputs: LLMBackendInputs) -> torch.Tensor:
        llm_inputs = self._prepare_action_forward_inputs(vla, inputs)
        full_embeds = llm_inputs["inputs_embeds"].detach()
        attention_mask = llm_inputs["attention_mask"].detach().to(torch.bool)
        if full_embeds.shape[0] != 1:
            raise ValueError(f"TensorRT engine backend supports batch_size=1, got {full_embeds.shape[0]}")

        valid_indices = attention_mask.nonzero(as_tuple=False)[:, 1]
        compact_embeds = full_embeds.index_select(1, valid_indices).cpu().to(torch.float16).contiguous()
        sequence_length = int(compact_embeds.shape[1])
        if sequence_length > self.kv_cache_capacity:
            raise ValueError(
                f"Engine sequence length {sequence_length} exceeds KV cache capacity {self.kv_cache_capacity}"
            )

        half_dim = 64
        inv_freq = 1.0 / (10000.0 ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))
        phases = valid_indices.cpu().to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
        rope = torch.cat((phases.cos(), phases.sin()), dim=-1).unsqueeze(0)

        with tempfile.TemporaryDirectory(prefix="omnivla_trt_engine_") as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            compact_embeds.numpy().tofile(input_dir / "inputs_embeds_fp16.bin")
            padded_rope = torch.zeros((1, self.kv_cache_capacity, rope.shape[-1]), dtype=torch.float32)
            padded_rope[:, :sequence_length] = rope
            padded_rope.numpy().tofile(input_dir / "rope_rotary_cos_sin_fp32.bin")
            completed = subprocess.run(
                [
                    self.runner_path, os.path.join(self.engine_dir, "llm.engine"), self.plugin_path,
                    str(input_dir), str(output_dir), str(sequence_length), str(self.kv_cache_capacity),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"TensorRT engine failed: {completed.stderr[-2000:]}")
            self.last_llm_inference_ms = next(
                (float(line.split("=", 1)[1]) for line in completed.stdout.splitlines() if line.startswith("engine_enqueue_ms=")),
                None,
            )
            if self.last_llm_inference_ms is None:
                raise RuntimeError("TensorRT engine did not report engine_enqueue_ms")
            self.last_llm_memory_mib = {
                key: float(line.split("=", 1)[1])
                for line in completed.stdout.splitlines()
                if line.startswith("engine_cuda_used_mib_")
                for key in [line.split("=", 1)[0]]
            }
            hidden_np = np.fromfile(output_dir / "hidden_states_fp16.bin", dtype=np.float16)

        expected_shape = (1, sequence_length, full_embeds.shape[-1])
        if hidden_np.size != int(np.prod(expected_shape)):
            raise RuntimeError(f"TensorRT hidden_states size mismatch: expected {expected_shape}, got {hidden_np.size} values")
        compact_hidden = torch.from_numpy(hidden_np.reshape(expected_shape)).to(device=full_embeds.device)
        full_hidden = torch.zeros_like(full_embeds, dtype=compact_hidden.dtype)
        full_hidden[:, valid_indices, :] = compact_hidden
        return full_hidden

    def profile_target(self, vla: nn.Module) -> Optional[nn.Module]:
        return None


def build_llm_backend(
    backend_name: str,
    trtllm_engine_dir: Optional[str] = None,
    trtllm_runner_path: Optional[str] = None,
    trt_plugin_path: Optional[str] = None,
    trt_kv_cache_capacity: Optional[int] = None,
) -> BaseLLMBackend:
    normalized = backend_name.lower()
    if normalized == "pytorch_fast":
        return PyTorchFastLLMBackend()
    if normalized == "tensorrt_llm":
        if not trtllm_engine_dir:
            raise ValueError("`trtllm_engine_dir` must be set when `llm_backend=tensorrt_llm`.")
        return TensorRTLLMBackend(trtllm_engine_dir, trtllm_runner_path)
    if normalized == "tensorrt_edge":
        if not trtllm_engine_dir or not trtllm_runner_path or not trt_plugin_path:
            raise ValueError("engine, runner, and plugin paths are required for `tensorrt_edge`.")
        return TensorRTEdgeBackend(
            trtllm_engine_dir, trt_plugin_path, trtllm_runner_path, trt_kv_cache_capacity
        )
    raise ValueError(f"Unsupported llm backend `{backend_name}`.")
