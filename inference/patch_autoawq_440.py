#!/usr/bin/env python3
"""Patch AutoAWQ 0.2.8 for OmniVLA's Transformers 4.40 Llama backend."""

import importlib.util
from importlib.metadata import version
from pathlib import Path
import re


def update(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Unexpected AutoAWQ source: {path}")
    path.write_text(text.replace(old, new, 1))
    return True


spec = importlib.util.find_spec("awq")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Install autoawq==0.2.8 before running this script")
if version("autoawq") != "0.2.8":
    raise RuntimeError("This patch supports autoawq==0.2.8 only")
root = Path(next(iter(spec.submodule_search_locations)))

changed = []
base = root / "models/base.py"
text = base.read_text()
if "from transformers.image_processing_utils import BaseImageProcessor" not in text:
    text = text.replace("    BaseImageProcessor,\n", "", 1)
    anchor = ")\nfrom accelerate.big_modeling"
    if anchor not in text:
        raise RuntimeError(f"Unexpected AutoAWQ source: {base}")
    text = text.replace(
        anchor,
        ")\ntry:\n    from transformers import BaseImageProcessor\nexcept ImportError:\n"
        "    from transformers.image_processing_utils import BaseImageProcessor\nfrom accelerate.big_modeling",
        1,
    )
    base.write_text(text)
    changed.append(base)

models_init = root / "models/__init__.py"
llama_import = "from .llama import LlamaAWQForCausalLM\n"
if models_init.read_text() != llama_import:
    models_init.write_text(llama_import)
    changed.append(models_init)

auto = root / "models/auto.py"
text = auto.read_text()
llama_map = 'AWQ_CAUSAL_LM_MODEL_MAP = {"llama": LlamaAWQForCausalLM}\n'
if llama_map not in text:
    text, count = re.subn(
        r"AWQ_CAUSAL_LM_MODEL_MAP = \{.*?\n\}\n",
        llama_map,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Unexpected AutoAWQ source: {auto}")
    auto.write_text(text)
    changed.append(auto)

scale = root / "quantize/scale.py"
update(
    scale,
    "from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm\n"
    "from transformers.models.cohere.modeling_cohere import CohereLayerNorm\n",
    "try:\n    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm\n"
    "except ImportError:\n    Gemma2RMSNorm = type(\"Gemma2RMSNorm\", (), {})\n"
    "try:\n    from transformers.models.cohere.modeling_cohere import CohereLayerNorm\n"
    "except ImportError:\n    CohereLayerNorm = type(\"CohereLayerNorm\", (), {})\n",
) and changed.append(scale)

llama = root / "models/llama.py"
update(
    llama,
    "        model.model.rotary_emb = model.model.rotary_emb.to(device)\n",
    "        if hasattr(model.model, \"rotary_emb\"):\n"
    "            model.model.rotary_emb = model.model.rotary_emb.to(device)\n"
    "        else:\n"
    "            for layer in model.model.layers:\n"
    "                layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)\n",
) and changed.append(llama)

from awq import AutoAWQForCausalLM  # noqa: E402
from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP  # noqa: E402

assert set(AWQ_CAUSAL_LM_MODEL_MAP) == {"llama"}
print(f"AutoAWQ 0.2.8 patched for Transformers 4.40 ({len(changed)} files changed)")
