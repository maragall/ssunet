from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from ...configs.options import (
    ActivationNameOptions,
    DownModeOptions,
    InitMethodOptions,
    MergeModeOptions,
    UpModeOptions,
)
from ...constants import LOGGER
from ...exceptions import InvalidPValueError
from .layers import activation_function, conv111


class BaseBlock(nn.Module, ABC):
    """Simplified base class for UNet blocks with flexible channel configuration."""

    def __init__(
        self,
        in_channels: dict[int, int],
        out_channels: dict[int, int],
        z_conv: bool,
        post_op: DownModeOptions | UpModeOptions | None,
        merge_mode: MergeModeOptions | None,
        activation: ActivationNameOptions | dict | None,
        dropout_p: float | None,
        init_method: InitMethodOptions | None,
        block_config: dict | None,
    ) -> None:
        super().__init__()

        self.in_channels = {k: v for k, v in in_channels.items() if k != 0}
        self.out_channels = {k: v for k, v in out_channels.items() if k != 0}
        self.main_in_channels = in_channels.get(0, 0)
        self.main_out_channels = out_channels.get(0, 0)
        if self.main_in_channels == 0 or self.main_out_channels == 0:
            raise InvalidPValueError("in_channels and out_channels must have a main channel.")
        self._validate_channels()

        # Operation configuration
        self.post_op_mode = post_op
        self.merge_mode = merge_mode
        self.z_conv = z_conv
        self.block_config = block_config or {}
        self.init_method = init_method

        # Setup activation
        self.activation_module = self._setup_activation(activation)

        # Setup dropout with strict validation
        self.dropout_module = self._setup_dropout(dropout_p)

        # Lazy-initialized merge conv layers
        self.merge_conv_layers = nn.ModuleDict()

        # Setup merge module based on number of skip connections
        if self.has_in:
            self.merge_module = (
                self._create_multi_merge() if len(self.in_channels) > 1 else self._create_merge()
            )
        else:
            self.merge_module = None

        # Setup post operation
        self.post_op_module = self._setup_post_op()

        self._build_layers()

    @property
    def has_in(self) -> bool:
        """Check if there are skip input connections."""
        return len(self.in_channels) > 0

    @property
    def has_out(self) -> bool:
        """Check if there are skip output connections."""
        return len(self.out_channels) > 0

    @property
    def skip_in_count(self) -> int:
        """Get the number of skip input connections."""
        return len(self.in_channels)

    @property
    def skip_out_count(self) -> int:
        """Get the number of skip output connections."""
        return len(self.out_channels)

    def _setup_activation(self, activation: ActivationNameOptions | dict | None) -> nn.Module:
        """Setup activation module."""
        if isinstance(activation, str):
            return activation_function(activation_name=activation)
        elif isinstance(activation, dict):
            return activation_function(**activation)
        else:
            return nn.Identity()

    def _setup_dropout(self, dropout_p: float | None) -> nn.Module:
        """Setup dropout module with validation."""
        if dropout_p is None or dropout_p == 0:
            return nn.Identity()

        if not 0 <= dropout_p < 1:
            raise InvalidPValueError(f"Dropout probability must be in [0, 1), got {dropout_p}")

        return nn.Dropout3d(p=dropout_p)

    def _get_lazy_conv(self, key: str, in_channels: int, out_channels: int) -> nn.Module:
        """Get or create a lazy-initialized conv layer."""
        if key not in self.merge_conv_layers:
            self.merge_conv_layers[key] = conv111(in_channels, out_channels)
        return self.merge_conv_layers[key]

    def _create_merge(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None:
        """Create merge function for single skip connection."""
        merge_funcs = {
            "concat": self._concat,
            "concat_conv": self._concat_conv,
            "add": self._add,
            "conv_add": self._conv_add,
            "add_conv": self._add_conv,
            None: None,
        }

        if self.merge_mode not in merge_funcs:
            raise InvalidPValueError(f"Unknown merge mode: {self.merge_mode}")

        return merge_funcs[self.merge_mode]

    def _create_multi_merge(self) -> Callable[[list[torch.Tensor]], torch.Tensor] | None:
        """Create merge function for multiple skip connections."""
        # Map single merge functions to multi versions
        merge_funcs = {
            "concat": lambda inputs: torch.cat(inputs, dim=1),
            "concat_conv": self._multi_concat_conv,
            "add": self._multi_add,
            "add_conv": self._multi_add_conv,
            None: None,
        }

        if self.merge_mode == "conv_add":
            raise NotImplementedError("conv_add merge mode is not supported for multi merge.")

        if self.merge_mode not in merge_funcs:
            raise InvalidPValueError(f"Unsupported merge mode: {self.merge_mode}")

        return merge_funcs[self.merge_mode]

    # Single merge operations
    def _concat(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """Concatenate two tensors along the channel dim."""
        return torch.cat((input_a, input_b), dim=1)

    def _concat_conv(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """Concatenate two tensors and apply a 1x1 convolution."""
        concat = self._concat(input_a, input_b)
        out_channels = getattr(self, "_merge_target_channels", concat.shape[1])
        conv = self._get_lazy_conv("concat", concat.shape[1], out_channels)
        return conv(concat)

    def _add(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """Add two tensors with channel alignment."""
        a_channels = input_a.shape[1]
        b_channels = input_b.shape[1]

        if a_channels == b_channels:
            return input_a + input_b

        min_channels = min(a_channels, b_channels)
        sum_part = input_a[:, :min_channels] + input_b[:, :min_channels]
        remainder = (
            input_a[:, min_channels:] if a_channels > b_channels else input_b[:, min_channels:]
        )
        return torch.cat([sum_part, remainder], dim=1)

    def _conv_add(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """Add two tensors with projection if needed."""
        a_channels = input_a.shape[1]
        b_channels = input_b.shape[1]

        if a_channels != b_channels:
            proj_key = f"proj_{b_channels}_to_{a_channels}"
            proj = self._get_lazy_conv(proj_key, b_channels, a_channels)
            return input_a + proj(input_b)
        return input_a + input_b

    def _add_conv(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        """Add two tensors and apply a 1x1 convolution."""
        output = self._add(input_a, input_b)
        out_channels = getattr(self, "_merge_target_channels", output.shape[1])
        conv = self._get_lazy_conv("add", output.shape[1], out_channels)
        return conv(output)

    # Multi merge operations
    def _multi_concat_conv(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate multiple tensors and apply a 1x1 convolution."""
        concat = torch.cat(inputs, dim=1)
        conv = self._get_lazy_conv("multi_concat", concat.shape[1], concat.shape[1])
        return conv(concat)

    def _multi_add(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Add multiple tensors with optional channel matching."""
        if self.block_config.get("strict_channel_matching", True):
            if not all(t.shape[1] == inputs[0].shape[1] for t in inputs):
                shapes = [t.shape for t in inputs]
                raise ValueError(
                    f"All tensors must have same number of channels for multi_add. "
                    f"Got shapes: {shapes}"
                )
            return torch.stack(inputs).sum(dim=0)
        else:
            # Flexible mode with warning
            min_channels = min(t.shape[1] for t in inputs)
            if not all(t.shape[1] == min_channels for t in inputs):
                LOGGER.warning("Channel mismatch in multi_add, truncating to min channels")
            aligned = [t[:, :min_channels] for t in inputs]
            return torch.stack(aligned).sum(dim=0)

    def _multi_add_conv(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Add multiple tensors and apply a 1x1 convolution."""
        output = self._multi_add(inputs)
        conv = self._get_lazy_conv("multi_add", output.shape[1], output.shape[1])
        return conv(output)

    def _get_pixel_shuffle_layer(self, scale: int, unshuffle: bool = False):
        """Get appropriate pixel shuffle/unshuffle layer based on z_conv."""
        from .layers import PixelShuffle2d, PixelShuffle3d, PixelUnshuffle2d, PixelUnshuffle3d

        if unshuffle:
            return PixelUnshuffle3d(scale=scale) if self.z_conv else PixelUnshuffle2d(scale=scale)
        else:
            return PixelShuffle3d(scale=scale) if self.z_conv else PixelShuffle2d(scale=scale)

    def _setup_post_op(self) -> nn.Module:
        """Setup post operation module."""
        if self.post_op_mode is None or self.post_op_mode == "identity":
            return nn.Identity()

        # Common parameters
        kernel_size = 2 if self.z_conv else (1, 2, 2)
        stride = 2 if self.z_conv else (1, 2, 2)
        scale_factor = 2 if self.z_conv else (1, 2, 2)

        # Calculate output channels for up operations
        divisor = self.block_config.get("post_up_op_channel_divisor", 2)
        up_op_out_channels = max(1, self.main_out_channels // divisor)

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

            case "pixelunshuffle":
                return self._get_pixel_shuffle_layer(scale=2, unshuffle=True)

            case "pixelunshuffle_conv":
                divisor = 8 if self.z_conv else 4
                return nn.Sequential(
                    self._get_pixel_shuffle_layer(scale=2, unshuffle=True),
                    nn.Conv3d(
                        self.main_out_channels * divisor,
                        self.main_out_channels,
                        kernel_size=1,
                    ),
                )

            # Upsampling operations - now use main_out_channels since they happen after convolutions
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
                return self._get_pixel_shuffle_layer(scale=2, unshuffle=False)

            case "pixelshuffle_conv":
                multiplier = 8 if self.z_conv else 4
                return nn.Sequential(
                    nn.Conv3d(
                        self.main_out_channels, up_op_out_channels * multiplier, kernel_size=1
                    ),
                    self._get_pixel_shuffle_layer(scale=2, unshuffle=False),
                )

            case _:
                LOGGER.warning(f"Unknown post operation: {self.post_op_mode}")
                return nn.Identity()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using specified initialization method."""
        if self.init_method is None:
            return

        if isinstance(module, nn.Conv3d | nn.ConvTranspose3d):
            if self.init_method == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif self.init_method == "xavier":
                nn.init.xavier_normal_(module.weight)
            else:
                raise InvalidPValueError(f"Unknown initialization method: {self.init_method}")

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _validate_skip_inputs(self, skip_inputs: tuple[torch.Tensor, ...]) -> None:
        """Validate skip connections match expected configuration."""
        if len(skip_inputs) != self.skip_in_count:
            raise ValueError(
                f"Expected {self.skip_in_count} skip connections, got {len(skip_inputs)}"
            )

        # Validate channels for each skip connection
        sorted_indices = sorted(self.in_channels.keys())
        for i, (idx, expected_channels) in enumerate(
            zip(sorted_indices, self.in_channels.values(), strict=False)
        ):
            actual_channels = skip_inputs[i].shape[1]
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Skip connection at index {idx} expected {expected_channels} channels, "
                    f"got {actual_channels}"
                )

    @abstractmethod
    def _validate_channels(self):
        """Validate the channels for the block."""
        raise NotImplementedError("Subclasses must implement _validate_channels method")

    @abstractmethod
    def _build_layers(self):
        """Build the actual layers for the block. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_layers method")

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *skip_inputs: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: Main input tensor
            *skip_inputs: Skip connections in order of sorted self.in_channels keys

        Returns:
            If self.has_out is False: returns output tensor
            If self.has_out is True: returns (main_output, skip_dict) where skip_dict
                maps skip indices to tensors
        """
        raise NotImplementedError("Subclasses must implement forward method")


@dataclass
class BlockNode:
    block: BaseBlock
    nodes_before: dict[int, "BlockNode"] = field(default_factory=dict)
    nodes_after: dict[int, "BlockNode"] = field(default_factory=dict)

    def add_node_before(self, node: "BlockNode", index: int):
        self.nodes_before[index] = node
        node.nodes_after[index] = self

    def add_node_after(self, node: "BlockNode", index: int):
        self.nodes_after[index] = node
        node.nodes_before[index] = self
