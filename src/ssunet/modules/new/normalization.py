# src/ssunet/modules/normalization.py
"""Normalization layers for 3D convolution outputs."""

from logging import getLogger

import torch
from torch import nn

from ...configs.options import NormTypeOptions

LOGGER = getLogger(__name__)


class LayerNorm3D(nn.Module):
    """LayerNorm for 3D convolution outputs (B, C, D, H, W)."""

    def __init__(self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, D, H, W), got {x.ndim}D.")
        if x.shape[1] != self.num_features:
            LOGGER.warning(
                f"LayerNorm3D expected {self.num_features} channels, got {x.shape[1]}. "
                "Mismatch may lead to errors if not intended."
            )

        # Permute C to last dim for LayerNorm, then permute back
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        x = nn.functional.layer_norm(x, (self.num_features,), self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        return x

    def __repr__(self) -> str:
        return (
            f"LayerNorm3D(num_features={self.num_features}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine})"
        )


def get_norm_layer(norm_type: NormTypeOptions, num_features: int, **kwargs: dict) -> nn.Module:
    """Factory function to get normalization layer."""
    norm_type_lower = norm_type.lower()

    if num_features <= 0 and norm_type_lower not in ["none", ""]:  # Cannot normalize 0 features
        LOGGER.warning(
            f"Cannot create norm layer for {num_features} features. Returning nn.Identity()."
        )
        return nn.Identity()

    if norm_type_lower == "layer":
        return LayerNorm3D(num_features, **kwargs)
    if norm_type_lower == "batch":
        return nn.BatchNorm3d(num_features, **kwargs)
    if norm_type_lower == "group":
        num_groups = kwargs.get("num_groups", 32)
        # Ensure num_features is divisible by num_groups if num_features > 0
        if num_features > 0 and num_features % num_groups != 0:
            # Attempt to find a valid number of groups
            found_valid_group = False
            for ng_candidate in range(min(num_groups, num_features), 0, -1):  # Iterate downwards
                if num_features % ng_candidate == 0:
                    num_groups = ng_candidate
                    found_valid_group = True
                    LOGGER.info(
                        f"GroupNorm: Adjusted num_groups to {num_groups}"
                        f"for {num_features} features."
                    )
                    break
            if not found_valid_group:  # Should ideally not happen if num_features > 0
                num_groups = 1  # Fallback makes it like LayerNorm
                LOGGER.warning(
                    f"GroupNorm: Could not find suitable num_groups for {num_features} features. "
                    "Using num_groups=1."
                )
        return nn.GroupNorm(num_groups, num_features, **kwargs)
    if norm_type_lower == "instance":
        return nn.InstanceNorm3d(num_features, affine=kwargs.get("affine", True), **kwargs)
    if norm_type_lower in ["none", ""]:
        return nn.Identity()

    raise ValueError(f"Unknown normalization type: {norm_type}")
