"""Model-aware reasoning cues used during latent rollouts and decoding."""

from typing import Optional


# Store canonical Hugging Face model IDs in lowercase for case-insensitive lookup.
# Add a model here only when its chat template already emits the reasoning prefix.
REASONING_MODEL_NAMES = frozenset(
    {
        "deepseek-ai/deepseek-r1",
        "deepseek-ai/deepseek-r1-distill-llama-8b",
    }
)

# Mistral-NeMo is not trained to use Qwen/DeepSeek's ``<think>`` marker. It
# receives an explicit natural-language chain-of-thought cue at the same
# boundary where the other models receive their reasoning prefix.
STEP_BY_STEP_MODEL_NAMES = frozenset(
    {
        "mistralai/mistral-nemo-instruct-2407",
    }
)


def _matches_model(model_name: str, candidates) -> bool:
    """Match either a canonical Hub ID or a local checkpoint basename."""
    normalized = str(model_name).replace("\\", "/").rstrip("/").lower()
    if normalized in candidates:
        return True
    basename = normalized.rsplit("/", 1)[-1]
    return any(name.rsplit("/", 1)[-1] == basename for name in candidates)


def model_adds_think_tag(model_name: str) -> bool:
    """Return whether the model's own chat template supplies ``<think>``."""
    return _matches_model(model_name, REASONING_MODEL_NAMES)


def resolve_manual_think(model_name: str, requested: Optional[bool]) -> bool:
    """Resolve framework-added reasoning while preserving explicit overrides."""
    if requested is not None:
        return requested
    return not model_adds_think_tag(model_name)


def manual_reasoning_cue(model_name: str) -> str:
    """Return the model-appropriate cue used when manual reasoning is enabled."""
    if _matches_model(model_name, STEP_BY_STEP_MODEL_NAMES):
        return "Let's think step by step."
    return "<think>"


def append_manual_reasoning_cue(
    prompt: str, model_name: str, enabled: bool
) -> str:
    """Append the appropriate reasoning cue to a rendered assistant prompt."""
    if not enabled:
        return prompt
    return f"{prompt}{manual_reasoning_cue(model_name)}"
