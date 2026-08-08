"""Lightweight package init for inference.

Avoid eager imports here so importing ``prismatic.vla.action_tokenizer`` does not
pull in the full training/data stack.
"""

__all__ = ["available_model_names", "available_models", "get_model_description", "load"]


def __getattr__(name):
    if name in __all__:
        from .models import available_model_names, available_models, get_model_description, load

        return {
            "available_model_names": available_model_names,
            "available_models": available_models,
            "get_model_description": get_model_description,
            "load": load,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
