# src/ssunet/modules/__init__.py
"""SSUnet modules for 3D medical image processing."""

from .blocks import (
    BLOCK_REGISTRY,
    ConvNeXtDownBlock3D,
    ConvNeXtUpBlock3D,
    DownBlock3D,
    DownConvTri3D,
    EfficientDownBlock3D,
    EfficientUpBlock3D,
    NAFDownBlock3D,
    NAFUpBlock3D,  # Added
    UpBlock3D,  # Aliases
    UpConvTri3D,  # Legacy
)

# Alias for backward compatibility
from .layers import (
    ConvNeXtBlock3D,
    NAFBlock3D,
    SimpleGate3D,
    build_conv_unit3d,
    conv111,
    conv333,
    conv777,
    create_conv3d,  # Conv primitives
    depthwise_conv333,
    merge,
    pool,
    upconv222,
)
from .normalization import LayerNorm3D, get_norm_layer
from .special_layers import (
    GatedReLUMix,
    PartialConv2d,
    PartialConv3d,
    PixelShuffle2d,
    PixelShuffle3d,
    PixelUnshuffle2d,
    PixelUnshuffle3d,
    SinGatedMix,
)

BLOCK = BLOCK_REGISTRY

__all__ = [
    "BLOCK",  # Backward compatibility alias
    # Blocks & Registry
    "BLOCK_REGISTRY",
    "ConvNeXtBlock3D",
    "ConvNeXtDownBlock3D",
    "ConvNeXtUpBlock3D",
    "DownBlock3D",
    "DownConvTri3D",
    "EfficientDownBlock3D",
    "EfficientUpBlock3D",
    "GatedReLUMix",
    # Normalization
    "LayerNorm3D",
    "NAFBlock3D",
    "NAFDownBlock3D",
    "NAFUpBlock3D",
    # Special layers
    "PartialConv2d",
    "PartialConv3d",
    "PixelShuffle2d",
    "PixelShuffle3d",
    "PixelUnshuffle2d",
    "PixelUnshuffle3d",
    "SimpleGate3D",
    "SinGatedMix",
    "UpBlock3D",
    "UpConvTri3D",
    "activation_function",
    "build_conv_unit3d",
    # Layers & Helpers
    "conv111",
    "conv333",
    "conv777",
    "create_conv3d",
    "depthwise_conv333",
    "get_norm_layer",
    "merge",
    "pool",
    "upconv222",
]
