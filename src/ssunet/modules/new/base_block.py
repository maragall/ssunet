from abc import ABC, abstractmethod
from functools import partial

import torch
from torch import nn

from ...configs.options import (
    ActivationNameOptions,
    DownModeOptions,
    MergeModeOptions,
    UpModeOptions,
)
from ...constants import LOGGER
from .layers import (
    activation_function,
    merge,
)


class BaseUnetBlock3D(nn.Module, ABC):
    """Simplified base class for UNet blocks with flexible channel configuration."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        post_op: DownModeOptions | UpModeOptions | None = None,
        merge_mode: MergeModeOptions = "concat",
        z_conv: bool = True,
        block_config: dict | None = None,
        activation: ActivationNameOptions | None = None,
        dropout_p: float = 0.0,
        expected_dimensions: list[tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__()

        # Channel configuration
        self.main_in_channels = in_channels[0] if in_channels else 0
        self.skip_in_channels_list = in_channels[1:] if len(in_channels) > 1 else []

        self.main_out_channels = out_channels[0] if out_channels else 0
        self.skip_out_channels_list = out_channels[1:] if len(out_channels) > 1 else []

        # Operation configuration
        self.post_op_mode = post_op
        self.merge_mode = merge_mode
        self.z_conv = z_conv
        self.block_config = block_config or {}

        # Store new parameters
        self.activation_name = activation
        self.dropout_p = dropout_p if dropout_p > 0 and dropout_p < 1 else 0.0
        self.expected_dimensions = expected_dimensions

        self.activation_module = activation_function(activation) if activation else nn.Identity()
        self.dropout_module = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self._setup_shared_ops()

    def _setup_shared_ops(self):
        """Setup shared operations: merge, pooling, and upsampling."""
        self.merge_op = partial(merge, merge_mode=self.merge_mode)
        self.post_op_module = self._setup_post_op()

    def _setup_post_op(self) -> nn.Module:
        """Setup post operation module using match/case.

        Returns:
            nn.Module: Post operation (pooling, conv, upsample, etc.) or identity
        """
        if self.post_op_mode is None:
            return nn.Identity()

        # Common parameters
        kernel_size = 2 if self.z_conv else (1, 2, 2)
        stride = 2 if self.z_conv else (1, 2, 2)
        scale_factor = 2 if self.z_conv else (1, 2, 2)

        # Calculate output channels for up operations
        divisor = self.block_config.get("post_up_op_channel_divisor", 2)
        up_op_out_channels = (
            max(1, self.main_out_channels // divisor) if self.main_out_channels > 0 else 1
        )

        match self.post_op_mode:
            # Downsampling operations
            case "maxpool":
                return nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

            case "avgpool":
                return nn.AvgPool3d(kernel_size=kernel_size, stride=stride)

            case "conv":
                return nn.Conv3d(
                    self.main_out_channels,
                    self.main_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )

            case "unshuffle":
                from .layers import PixelUnshuffle2d, PixelUnshuffle3d

                return PixelUnshuffle3d(scale=2) if self.z_conv else PixelUnshuffle2d(scale=2)

            # Upsampling operations
            case "transpose":
                return nn.ConvTranspose3d(
                    self.main_out_channels,
                    up_op_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )

            case "upsample":
                return nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode="nearest"),
                    nn.Conv3d(self.main_out_channels, up_op_out_channels, kernel_size=1),
                )

            case "pixelshuffle":
                from .layers import PixelShuffle2d, PixelShuffle3d

                multiplier = 8 if self.z_conv else 4
                shuffle_layer = PixelShuffle3d(scale=2) if self.z_conv else PixelShuffle2d(scale=2)

                return nn.Sequential(
                    nn.Conv3d(
                        self.main_out_channels, up_op_out_channels * multiplier, kernel_size=1
                    ),
                    shuffle_layer,
                )

            case _:
                LOGGER.warning(f"Unknown post operation: {self.post_op_mode}")
                return nn.Identity()

    def _validate_dimensions(self, tensor: torch.Tensor, is_input: bool = True) -> None:
        """Validate tensor dimensions against expected dimensions.

        Args:
            tensor: Input or output tensor to validate
            is_input: Whether this is an input tensor (True) or output tensor (False)
        """
        if self.expected_dimensions is None:
            return

        # Get tensor shape excluding batch and channel dimensions
        # Assumes tensor shape is [batch, channels, *spatial_dims]
        tensor_spatial_dims = tensor.shape[2:]

        for expected_dims in self.expected_dimensions:
            if tensor_spatial_dims == expected_dims:
                return  # Found matching dimensions

        # No matching dimensions found, log warning
        tensor_type = "input" if is_input else "output"
        LOGGER.warning(
            f"Dimension mismatch for {tensor_type} tensor in {self.__class__.__name__}. "
            f"Expected one of {self.expected_dimensions}, got {tensor_spatial_dims}"
        )

    @abstractmethod
    def _build_layers(self):
        """Build the actual layers for the block. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_layers method")

    @abstractmethod
    def forward(self, x):
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")
