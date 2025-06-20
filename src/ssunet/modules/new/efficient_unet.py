# src/ssunet/modules/blocks.py

from typing import ClassVar

import torch
from torch import nn

from ...configs.options import (
    ActivationNameOptions,
    DownModeOptions,
    MergeModeOptions,
    NormTypeOptions,
    UpModeOptions,
)
from ...constants import LOGGER
from .base_block import BaseBlock
from .layers import (
    activation_function,
    build_conv_unit3d,
    conv111,
)


class EfficientDownBlock3D(BaseUnetBlock3D):
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {"residual_activation_name": "none"}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_out: bool = True,
        down_mode: DownModeOptions = "maxpool",
        z_conv: bool = True,
        activation_name: ActivationNameOptions = "gelu",
        activation_params: dict | None = None,
        dropout_p: float = 0.0,
        norm_type: NormTypeOptions = "layer",
        norm_params: dict | None = None,
        num_blocks: int = 2,
        intermediate_channels_list: list[int] | None = None,
        use_depthwise_separable_conv: bool = False,
        block_config: dict | None = None,
    ):
        self.skip_out = skip_out
        self.norm_type = norm_type
        self.norm_params = norm_params
        self.num_blocks = num_blocks
        self.intermediate_channels_list = intermediate_channels_list
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}

        super().__init__(
            in_channels=in_channels,
            out_channels=[out_channels, out_channels] if skip_out else out_channels,
            post_op=down_mode,
            z_conv=z_conv,
            block_config=self.config,
            activation={
                "activation_name": activation_name,
                "activation_params": activation_params,
            },
            dropout_p=dropout_p,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using He initialization."""
        if isinstance(module, nn.Conv3d | nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _build_layers(self) -> None:
        self.main_path_units = nn.ModuleList()
        current_c = self.main_in_channels
        for i in range(self.num_blocks):
            unit_out_ch = self.main_out_channels
            if (
                i < self.num_blocks - 1
                and self.intermediate_channels_list
                and i < len(self.intermediate_channels_list)
            ):
                unit_out_ch = self.intermediate_channels_list[i]
            self.main_path_units.append(
                build_conv_unit3d(
                    current_c,
                    unit_out_ch,
                    3,
                    self.z_conv,
                    self.activation_module,
                    self.norm_type,  # type: ignore
                    self.norm_params,
                    self.use_depthwise_separable_conv,
                )
            )
            current_c = unit_out_ch
        if current_c != self.main_out_channels:
            LOGGER.error(f"EffDown: path ch {current_c} != out_ch {self.main_out_channels}.")

        res_act_fn = activation_function(self.config["residual_activation_name"], None)
        self.residual_proj = (
            nn.Sequential(
                conv111(self.main_in_channels, self.main_out_channels, bias=True), res_act_fn
            )
            if self.main_in_channels != self.main_out_channels
            else res_act_fn
        )

    def _module_logic(self):
        # The logic is handled in the forward pass for this specific block.
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        res_x = self.residual_proj(x)
        main_x = x
        for unit in self.main_path_units:
            main_x = unit(main_x)
        out_s = main_x + res_x
        bp = self.dropout_module(out_s)
        p = self.post_op_module(bp)
        return (p, bp) if self.skip_out else (p, None)


class EfficientUpBlock3D(BaseUnetBlock3D):
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {}  # Can add specific params if needed

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_out: bool = False,
        dropout_p: float = 0.0,
        up_mode: UpModeOptions = "transpose",
        merge_mode: MergeModeOptions = "concat",
        z_conv: bool = True,
        activation_name: ActivationNameOptions = "gelu",
        activation_params: dict | None = None,
        norm_type: NormTypeOptions = "layer",
        norm_params: dict | None = None,
        num_blocks: int = 2,
        intermediate_channels_list: list[int] | None = None,
        use_depthwise_separable_conv: bool = False,
        block_config: dict | None = None,
    ):
        self.skip_out = skip_out
        self.norm_type = norm_type
        self.norm_params = norm_params
        self.num_blocks = num_blocks
        self.intermediate_channels_list = intermediate_channels_list
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}

        # The up-sampled input from the previous layer will have out_channels / 2 channels
        # if up_mode is 'transpose' or 'upsample'.
        # The skip connection will have `out_channels` channels.
        # So in_channels for this block will be [out_channels/2, out_channels]
        # But the `in_channels` parameter is the output of post_op of the previous block.
        # So main_in_channels = in_channels.
        # skip_in_channels will be out_channels from the corresponding down block.
        # This is tricky. Let's assume the U-Net architecture handles the channel matching.
        # The `in_channels` to this block is from the previous (lower) up-block.
        # The `skip` connection comes from the corresponding down-block.
        # The output of a down-block is `out_channels`. So the skip connection has `out_channels` from there.
        # The `in_channels` to `__init__` is the main path.
        # So `in_channels` for `BaseUnetBlock3D` is `[in_channels, out_channels]`.
        # However, `in_channels` to this `__init__` should be the channels from previous layer, which is not `out_channels/2` but `in_channels`.
        # The `out_channels` argument to this block is the number of output channels for THIS block.

        skip_in_channels = (
            out_channels  # From corresponding encoder, typically same as this block's out_channels.
        )

        super().__init__(
            in_channels=[in_channels, skip_in_channels],
            out_channels=out_channels,
            post_op=up_mode,
            merge_mode=merge_mode,
            z_conv=z_conv,
            block_config=self.config,
            activation={
                "activation_name": activation_name,
                "activation_params": activation_params,
            },
            dropout_p=dropout_p,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using He initialization."""
        if isinstance(module, nn.Conv3d | nn.ConvTranspose3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _build_layers(self):
        # After upsampling and merge, the number of channels will be adjusted before the main path.
        if self.merge_mode == "concat":
            merged_ch = self.main_out_channels + self.skip_in_channels[0]
        else:
            merged_ch = self.main_out_channels
        # After upsampling post_op, the main path from previous layer has main_out_channels.
        # The skip connection comes with skip_in_channels[0].
        # up_op from base class handles channel changes.
        # Calculate output channels based on post operation type
        if self.post_op_mode in ["transpose", "upsample"]:
            divisor = self.block_config.get("post_up_op_channel_divisor", 2)
            up_op_out_channels = max(1, self.main_in_channels // divisor)
        else:
            up_op_out_channels = self.main_in_channels

        if self.merge_mode == "concat":
            merged_ch = up_op_out_channels + self.skip_in_channels[0]
        else:
            merged_ch = (
                up_op_out_channels  # Assumes skip and up_op_out have same channels for 'add'
            )

        self.proj_after_merge = (
            conv111(merged_ch, self.main_out_channels, bias=True)
            if merged_ch != self.main_out_channels
            else nn.Identity()
        )

        self.main_path_units = nn.ModuleList()
        current_c = self.main_out_channels  # After proj, current_c is self.out_channels
        for i in range(self.num_blocks):
            unit_out_ch = self.main_out_channels
            if (
                i < self.num_blocks - 1
                and self.intermediate_channels_list
                and i < len(self.intermediate_channels_list)
            ):
                unit_out_ch = self.intermediate_channels_list[i]
            self.main_path_units.append(
                build_conv_unit3d(
                    current_c,
                    unit_out_ch,
                    3,
                    self.z_conv,
                    self.activation_module,
                    self.norm_type,  # type: ignore
                    self.norm_params,
                    self.use_depthwise_separable_conv,
                )
            )
            current_c = unit_out_ch
        if current_c != self.main_out_channels:
            LOGGER.error(f"EffUp: path ch {current_c} != out_ch {self.main_out_channels}.")

    def _module_logic(self):
        # The logic is handled in the forward pass for this specific block.
        pass

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x_up = self.post_op_module(x)
        main_x: torch.Tensor
        if skip is not None:
            merged = self.merge_module(x_up, skip)
            main_x = self.proj_after_merge(merged)
        else:
            main_x = self.proj_after_merge(x_up)

        for unit in self.main_path_units:
            main_x = unit(main_x)
        return self.dropout_module(main_x)
