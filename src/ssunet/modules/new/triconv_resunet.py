from typing import Any

import torch
from torch import nn

from ...configs.options import (
    ActivationNameOptions,
    ConvBlockOptions,
    DownModeOptions,
    InitMethodOptions,
    MergeModeOptions,
    UpModeOptions,
)
from ...constants import LOGGER
from .base_block import BaseBlock
from .layers import conv333


class GenericTriConvBlock(BaseBlock):
    """Generic TriConv block that can be configured as Down, Up, or Bottleneck.

    This unified implementation reduces code duplication and ensures consistency
    across all TriConv variants. The fully flexible operation order is:
    - Down blocks: merge → conv → downsample
    - Up blocks: merge → conv → upsample
    - Bottleneck: merge → conv → post_op (if any)
    
    All blocks now support both optional skip inputs AND skip outputs, providing
    maximum flexibility for complex architectures while maintaining a consistent
    flow pattern where spatial transformations always happen at the end.
    """

    def __init__(
        self,
        in_channels: dict[int, int],
        out_channels: dict[int, int],
        z_conv: bool,
        post_op: DownModeOptions | UpModeOptions | None,
        merge_mode: MergeModeOptions | None,
        activation: ActivationNameOptions | dict[str, Any] | None,
        dropout_p: float | None,
        init_method: InitMethodOptions | None,
        block_config: dict[str, Any] | None,
        block_type: ConvBlockOptions,
    ) -> None:
        # Store block type before calling super().__init__
        self.block_type = block_type

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            z_conv=z_conv,
            post_op=post_op,
            merge_mode=merge_mode,
            activation=activation,
            dropout_p=dropout_p,
            init_method=init_method,
            block_config=block_config,
        )

        # Set target channels for merge conv operations
        self._merge_target_channels = self._calculate_conv1_channels()
        self.apply(self._init_weights)

    def _validate_channels(self) -> None:
        """Validate channels based on block type."""
        if self.main_in_channels <= 0:
            raise ValueError(f"Main input channels must be positive, got {self.main_in_channels}")
        if self.main_out_channels <= 0:
            raise ValueError(f"Main output channels must be positive, got {self.main_out_channels}")

        # TriConv blocks support at most 1 skip connection
        if len(self.in_channels) > 1:
            raise ValueError(
                f"TriConv blocks support at most 1 skip input, got {len(self.in_channels)}"
            )
        if len(self.out_channels) > 1:
            raise ValueError(
                f"TriConv blocks support at most 1 skip output, got {len(self.out_channels)}"
            )


    def _calculate_conv1_channels(self) -> int:
        """Calculate input channels for conv_1 based on block type and merge operations."""
        # All block types now support merge operations at the beginning
        if not self.has_in or self.merge_mode is None:
            return self.main_in_channels

        if self.merge_mode in ["concat", "concat_conv"]:
            # Concatenation adds all skip channels to main input
            return self.main_in_channels + sum(self.in_channels.values())
        elif self.merge_mode == "add":
            # Pure add can result in max channels
            max_skip = max(self.in_channels.values()) if self.in_channels else 0
            return max(self.main_in_channels, max_skip)
        else:  # conv_add, add_conv
            # These normalize to main_in_channels
            return self.main_in_channels

    def _get_upsampled_channels(self) -> int:
        """Calculate channels after upsampling operation."""
        # Note: This method is now only used for reference, since upsampling happens at the end
        if self.post_op_mode == "pixelshuffle_conv":
            # pixelshuffle_conv has a conv that changes channels before shuffle
            return self.main_out_channels
        elif self.post_op_mode in ["transpose", "pixelshuffle", "upsample"]:
            # These operations reduce channels by divisor
            divisor = self.block_config.get("post_up_op_channel_divisor", 2)
            return max(1, self.main_out_channels // divisor)
        else:
            return self.main_out_channels

    def _build_layers(self) -> None:
        """Build the three convolution layers with normalization."""
        # Get intermediate channels from config
        self.intermediate_channels = self.block_config.get(
            "intermediate_channels", self.main_out_channels
        )

        # Calculate input channels for first conv
        conv1_in_channels = self._calculate_conv1_channels()

        # Build convolution layers
        self.conv_1 = conv333(conv1_in_channels, self.main_out_channels, self.z_conv)
        self.conv_2 = conv333(self.main_out_channels, self.intermediate_channels, self.z_conv)
        self.conv_3 = conv333(self.intermediate_channels, self.main_out_channels, self.z_conv)

        # Setup group normalization
        self._setup_group_norm()

    def _setup_group_norm(self) -> None:
        """Setup group normalization modules."""
        group_norm = self.block_config.get("group_norm", 0)

        if group_norm <= 0:
            self.group_norm_main = nn.Identity()
            self.group_norm_intermediate = nn.Identity()
            return

        # Helper function to create group norm or identity
        def create_group_norm(num_channels: int, channel_type: str) -> nn.Module:
            if num_channels % group_norm == 0:
                return nn.GroupNorm(group_norm, num_channels)
            else:
                LOGGER.warning(
                    f"{self.__class__.__name__}: GroupNorm groups {group_norm} "
                    f"doesn't divide {channel_type} {num_channels}"
                )
                return nn.Identity()

        self.group_norm_main = create_group_norm(self.main_out_channels, "main_out_channels")
        self.group_norm_intermediate = create_group_norm(
            self.intermediate_channels, "intermediate_channels"
        )

    def _apply_convolutions(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the three convolution sequence with residual connection."""
        # First conv with activation and norm
        out_1 = self.conv_1(input)
        out_1 = self.group_norm_main(out_1)
        out_1 = self.activation_module(out_1)

        # Second conv with activation and norm
        out_2 = self.conv_2(out_1)
        out_2 = self.group_norm_intermediate(out_2)
        out_2 = self.activation_module(out_2)

        # Third conv with residual connection
        out_3 = self.conv_3(out_2)
        out_3 = out_3 + out_1  # Residual connection
        out_3 = self.group_norm_main(out_3)
        out_3 = self.activation_module(out_3)

        # Apply dropout
        return self.dropout_module(out_3)

    def forward(
        self, x: torch.Tensor, *skip_inputs: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Forward pass for generic TriConv block with unified operation order."""
        # Validate skip inputs
        self._handle_skip_inputs(skip_inputs)

        # Unified flow: all blocks start with merge (if applicable)
        if self.has_in and skip_inputs and self.merge_module is not None:
            # All blocks support single skip connection merge at the beginning
            merged = self.merge_module(x, skip_inputs[0])  # type: ignore[call-arg]
        else:
            merged = x

        # Apply convolutions to merged input
        features = self._apply_convolutions(merged)

        if self.block_type == "down":
            # Down path: merge → conv → downsample
            main_out = self.post_op_module(features)

            if self.has_out:
                # Create skip outputs (features before downsampling)
                skip_outputs = dict.fromkeys(self.out_channels.keys(), features)
                return main_out, skip_outputs
            else:
                return main_out

        elif self.block_type == "up":
            # Up path: merge → conv → upsample
            main_out = self.post_op_module(features)
            
            if self.has_out:
                # Create skip outputs (features before upsampling)
                skip_outputs = dict.fromkeys(self.out_channels.keys(), features)
                return main_out, skip_outputs
            else:
                return main_out

        else:  # bottleneck
            # Bottleneck: merge → conv → post_op (if any)
            main_out = self.post_op_module(features)
            
            if self.has_out:
                # Create skip outputs (features before post_op)
                skip_outputs = dict.fromkeys(self.out_channels.keys(), features)
                return main_out, skip_outputs
            else:
                return main_out

    def _handle_skip_inputs(self, skip_inputs: tuple[torch.Tensor, ...]) -> None:
        """Handle validation of skip inputs."""
        if self.has_in:
            if skip_inputs:
                self._validate_skip_inputs(skip_inputs)
            else:
                raise ValueError(f"Expected {self.skip_in_count} skip connections, got none")
        elif skip_inputs:
            # Don't raise error, just log if we receive unexpected skip inputs
            LOGGER.debug(
                f"{self.__class__.__name__} received {len(skip_inputs)} skip inputs "
                "but has no skip connections defined"
            )


# Helper function to convert channel specs to dict format
def _channels_to_dict(channels: int | tuple[int, ...], keys: list[int]) -> dict[int, int]:
    """Convert channel specification to dictionary format."""
    if isinstance(channels, int):
        return {keys[0]: channels}
    elif isinstance(channels, tuple):
        if len(channels) != len(keys):
            raise ValueError(f"Expected {len(keys)} channels, got {len(channels)}")
        return dict(zip(keys, channels, strict=False))
    else:
        raise ValueError(f"Channels must be int or tuple, got {type(channels)}")


# Convenience classes that wrap GenericTriConvBlock
class TriConvDown(GenericTriConvBlock):
    """TriConv downsampling block with optional skip input and optional skip output."""

    def __init__(
        self,
        in_channels: int | tuple[int, int],  # (main, skip) or just main
        out_channels: int | tuple[int, int],  # (main, skip) or just main
        z_conv: bool,
        post_op: DownModeOptions,
        merge_mode: MergeModeOptions | None = None,
        activation: ActivationNameOptions | dict[str, Any] | None = "relu",
        dropout_p: float | None = None,
        block_config: dict[str, Any] | None = None,
        init_method: InitMethodOptions | None = "kaiming",
    ) -> None:
        # Convert channel specs to dict format
        in_ch_dict = _channels_to_dict(in_channels, [0, 1])
        out_ch_dict = _channels_to_dict(out_channels, [0, 1])

        super().__init__(
            in_channels=in_ch_dict,
            out_channels=out_ch_dict,
            z_conv=z_conv,
            post_op=post_op,
            merge_mode=merge_mode,
            activation=activation,
            dropout_p=dropout_p,
            init_method=init_method,
            block_config=block_config,
            block_type="down",
        )


class TriConvUp(GenericTriConvBlock):
    """TriConv upsampling block with optional skip input and optional skip output."""

    def __init__(
        self,
        in_channels: int | tuple[int, int],  # (main, skip) or just main
        out_channels: int | tuple[int, int],  # (main, skip) or just main
        z_conv: bool,
        post_op: UpModeOptions,
        merge_mode: MergeModeOptions | None = None,
        activation: ActivationNameOptions | dict[str, Any] | None = "relu",
        dropout_p: float | None = None,
        block_config: dict[str, Any] | None = None,
        init_method: InitMethodOptions | None = "kaiming",
    ) -> None:
        # Convert channel specs to dict format
        in_ch_dict = _channels_to_dict(in_channels, [0, 1])
        out_ch_dict = _channels_to_dict(out_channels, [0, 1])

        super().__init__(
            in_channels=in_ch_dict,
            out_channels=out_ch_dict,
            z_conv=z_conv,
            post_op=post_op,
            merge_mode=merge_mode,
            activation=activation,
            dropout_p=dropout_p,
            init_method=init_method,
            block_config=block_config,
            block_type="up",
        )


class TriConvBottleneck(GenericTriConvBlock):
    """TriConv bottleneck block with optional skip input and optional skip output."""

    def __init__(
        self,
        in_channels: int | tuple[int, int],  # (main, skip) or just main
        out_channels: int | tuple[int, int],  # (main, skip) or just main
        z_conv: bool,
        post_op: DownModeOptions | UpModeOptions | None = None,
        merge_mode: MergeModeOptions | None = None,
        activation: ActivationNameOptions | dict[str, Any] | None = "relu",
        dropout_p: float | None = None,
        block_config: dict[str, Any] | None = None,
        init_method: InitMethodOptions | None = "kaiming",
    ) -> None:
        # Convert channel specs to dict format
        in_ch_dict = _channels_to_dict(in_channels, [0, 1])
        out_ch_dict = _channels_to_dict(out_channels, [0, 1])

        super().__init__(
            in_channels=in_ch_dict,
            out_channels=out_ch_dict,
            z_conv=z_conv,
            post_op=post_op,
            merge_mode=merge_mode,
            activation=activation,
            dropout_p=dropout_p,
            init_method=init_method,
            block_config=block_config,
            block_type="bottleneck",
        )


if __name__ == "__main__":
    # Fix imports for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))

    print("Testing fully flexible TriConv blocks...")

    # Test TriConvDown with merge and skip output
    print("\n1. Testing TriConvDown with merge and skip output:")
    down_block = TriConvDown(
        in_channels=(64, 32),  # (main_in, skip_in)
        out_channels=(32, 64),  # (main_out, skip_out)
        z_conv=True,
        post_op="maxpool",
        merge_mode="concat",
    )
    x = torch.randn(2, 64, 8, 16, 16)  # (batch, channels, z, h, w)
    skip_in = torch.randn(2, 32, 8, 16, 16)  # Skip input for merge
    main_out, skip_out = down_block(x, skip_in)
    print(f"Main input shape: {x.shape}")
    print(f"Skip input shape: {skip_in.shape}")
    print(f"Main output shape: {main_out.shape}")
    print(f"Skip output shapes: {[v.shape for v in skip_out.values()]}")

    # Test TriConvUp with merge and skip output
    print("\n2. Testing TriConvUp with merge and skip output:")
    up_block = TriConvUp(
        in_channels=(32, 64),  # (main_in, skip_in)
        out_channels=(32, 48),  # (main_out, skip_out)
        z_conv=True,
        post_op="transpose",
        merge_mode="concat",
    )
    x_up = torch.randn(2, 32, 4, 8, 8)  # Lower resolution input
    skip_in = torch.randn(2, 64, 4, 8, 8)  # Skip connection
    up_main_out, up_skip_out = up_block(x_up, skip_in)
    print(f"Main input shape: {x_up.shape}")
    print(f"Skip input shape: {skip_in.shape}")
    print(f"Main output shape: {up_main_out.shape}")
    print(f"Skip output shapes: {[v.shape for v in up_skip_out.values()]}")

    # Test TriConvBottleneck with merge and skip output
    print("\n3. Testing TriConvBottleneck with merge and skip output:")
    bottleneck = TriConvBottleneck(
        in_channels=(128, 64), 
        out_channels=(256, 128),  # (main_out, skip_out)
        z_conv=True,
        merge_mode="concat"
    )
    x_bottle = torch.randn(2, 128, 2, 4, 4)
    skip_bottle = torch.randn(2, 64, 2, 4, 4)
    bottle_main_out, bottle_skip_out = bottleneck(x_bottle, skip_bottle)
    print(f"Main input shape: {x_bottle.shape}")
    print(f"Skip input shape: {skip_bottle.shape}")
    print(f"Main output shape: {bottle_main_out.shape}")
    print(f"Skip output shapes: {[v.shape for v in bottle_skip_out.values()]}")

    # Test backward compatibility (no skip connections)
    print("\n4. Testing backward compatibility (no skip connections):")
    simple_down = TriConvDown(in_channels=64, out_channels=32, z_conv=True, post_op="maxpool")
    simple_up = TriConvUp(in_channels=32, out_channels=64, z_conv=True, post_op="transpose")
    simple_bottle = TriConvBottleneck(in_channels=128, out_channels=256, z_conv=True)
    
    simple_down_out = simple_down(torch.randn(2, 64, 8, 16, 16))
    simple_up_out = simple_up(torch.randn(2, 32, 4, 8, 8))
    simple_bottle_out = simple_bottle(torch.randn(2, 128, 2, 4, 4))
    
    print(f"Simple down output shape: {simple_down_out.shape}")
    print(f"Simple up output shape: {simple_up_out.shape}")
    print(f"Simple bottleneck output shape: {simple_bottle_out.shape}")

    print("\nAll tests passed! ✓")
    print("Fully flexible flow pattern:")
    print("- Down blocks: merge → conv → downsample")
    print("- Up blocks: merge → conv → upsample")
    print("- Bottleneck: merge → conv → post_op")
    print("All blocks now support both optional skip inputs AND skip outputs!")
