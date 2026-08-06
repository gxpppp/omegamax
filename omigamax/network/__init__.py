"""Neural network: ResNet policy-value model and 17-plane feature encoding (todos 6-8)."""

from omigamax.network.model import (
    INPUT_PLANES,
    PolicyValueNetwork,
    create_model,
    create_model_from_config,
)

__all__ = [
    "INPUT_PLANES",
    "PolicyValueNetwork",
    "create_model",
    "create_model_from_config",
]
