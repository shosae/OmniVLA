"""Lightweight VLA package init.

Inference code imports ``prismatic.vla.action_tokenizer`` directly and should not
require RLDS/tensorflow dataset dependencies at import time.
"""

__all__ = ["get_vla_dataset_and_collator"]


def __getattr__(name):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator

        return get_vla_dataset_and_collator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
