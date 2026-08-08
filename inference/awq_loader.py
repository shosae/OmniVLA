import json
import os
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1


class EmbeddingOnlyLanguageModel(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embed_tokens = embedding

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value


def _iter_non_language_safetensor_paths(model_path: str) -> Iterable[str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)

        seen = set()
        # AWQ inference reuses only the non-language weights from the original
        # OpenVLA checkpoint, so skip shards that contain language-model weights only.
        for key, filename in index["weight_map"].items():
            if key.startswith("language_model."):
                continue
            if filename in seen:
                continue
            seen.add(filename)
            shard_path = os.path.join(model_path, filename)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Missing non-language checkpoint shard required for AWQ inference: {shard_path}"
                )
            yield shard_path
        return

    for filename in sorted(os.listdir(model_path)):
        if filename.endswith(".safetensors"):
            yield os.path.join(model_path, filename)


def _load_non_language_weights(model: OpenVLAForActionPrediction_MMNv1, model_path: str) -> None:
    loaded_keys = 0

    for shard_path in _iter_non_language_safetensor_paths(model_path):
        shard_state = {}
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key.startswith("language_model."):
                    continue
                shard_state[key] = shard.get_tensor(key)

        if not shard_state:
            continue

        incompatible = model.load_state_dict(shard_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected keys while loading {shard_path}: {incompatible.unexpected_keys}")

        loaded_keys += len(shard_state)

    if loaded_keys == 0:
        raise RuntimeError(f"No non-language weights were loaded from {model_path}")


def load_awq_openvla(
    vla_path: str,
    awq_llm_path: str,
    device: torch.device,
    num_images_in_input: int,
    compute_dtype: torch.dtype,
    fuse_layers: bool,
) -> Tuple[OpenVLAForActionPrediction_MMNv1, object]:
    from awq import AutoAWQForCausalLM

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(vla_path, trust_remote_code=True)
    config.defer_language_model_init = True

    vla = OpenVLAForActionPrediction_MMNv1(config)
    _load_non_language_weights(vla, vla_path)

    awq_model = AutoAWQForCausalLM.from_quantized(
        awq_llm_path,
        trust_remote_code=True,
        fuse_layers=fuse_layers,
        safetensors=True,
    )
    vla.language_model = awq_model.model

    module_dtype = compute_dtype if device.type == "cuda" else torch.float32
    vla.vision_backbone.set_num_images_in_input(num_images_in_input)
    vla.vision_backbone.to(device=device, dtype=module_dtype)
    vla.projector.to(device=device, dtype=module_dtype)
    vla.eval()

    return vla, processor


def load_embedding_only_openvla(
    vla_path: str,
    embedding_path: str,
    device: torch.device,
    num_images_in_input: int,
    compute_dtype: torch.dtype,
) -> Tuple[OpenVLAForActionPrediction_MMNv1, object]:
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(vla_path, trust_remote_code=True)
    config.defer_language_model_init = True

    vla = OpenVLAForActionPrediction_MMNv1(config)
    _load_non_language_weights(vla, vla_path)

    with safe_open(embedding_path, framework="pt", device="cpu") as f:
        if "embedding" not in f.keys():
            raise KeyError(f"`embedding` tensor not found in {embedding_path}")
        embedding_weight = f.get_tensor("embedding")

    module_dtype = compute_dtype if device.type == "cuda" else torch.float32
    embedding = nn.Embedding(
        num_embeddings=embedding_weight.shape[0],
        embedding_dim=embedding_weight.shape[1],
        _weight=embedding_weight.to(dtype=module_dtype),
    )
    vla.language_model = EmbeddingOnlyLanguageModel(embedding)

    vla.vision_backbone.set_num_images_in_input(num_images_in_input)
    vla.vision_backbone.to(device=device, dtype=module_dtype)
    vla.projector.to(device=device, dtype=module_dtype)
    vla.language_model.to(device=device, dtype=module_dtype)
    vla.eval()

    return vla, processor
