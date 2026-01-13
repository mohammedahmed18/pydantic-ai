from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer

# Cache a single ModelProfile instance since the construction is stateless and identical every call.
# This avoids creating a new instance on every call, significantly improving performance for repeated accesses.
_qwen_model_profile_instance: ModelProfile = ModelProfile(
    json_schema_transformer=InlineDefsJsonSchemaTransformer
)


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    return _qwen_model_profile_instance
