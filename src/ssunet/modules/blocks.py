# src/ssunet/modules/blocks.py

from functools import partial
from typing import ClassVar

import torch
from torch import nn

from ..configs.options import (
    ActivationNameOptions,
    DownModeOptions,
    MergeModeOptions,
    NormTypeOptions,
    UpModeOptions,
)
from ..constants import LOGGER
from .layers import (
    ConvNeXtBlock3D,
    NAFBlock3D,  # Internal building blocks
    SimpleGate3D,
    activation_function,
    build_conv_unit3d,
    conv111,
    conv333,
    merge,
    pool,
    upconv222,
)


class BaseUnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_out: bool = True,
        dropout_p: float = 0.0,
        down_mode: DownModeOptions = "maxpool",
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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_out = skip_out
        self.dropout_p = dropout_p
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.z_conv = z_conv
        self.stored_activation_name = activation_name
        self.norm_type = norm_type
        self.norm_params = norm_params if norm_params is not None else {}
        self.num_blocks = max(1, num_blocks)
        self.intermediate_channels_list = intermediate_channels_list
        if self.intermediate_channels_list:  # Validate list length against num_blocks
            expected_len = self.num_blocks - 1
            if self.num_blocks == 1 and len(self.intermediate_channels_list) > 0:
                LOGGER.info(
                    "intermediate_channels_list provided but num_blocks is 1. List will be ignored."
                )
                self.intermediate_channels_list = None
            elif self.num_blocks > 1 and len(self.intermediate_channels_list) != expected_len:
                LOGGER.warning(
                    f"Length of intermediate_channels_list ({len(self.intermediate_channels_list)})"
                    f" does not match num_blocks - 1 ({expected_len}). Truncating or ignoring."
                )
                self.intermediate_channels_list = (
                    self.intermediate_channels_list[:expected_len]
                    if self.intermediate_channels_list and expected_len > 0
                    else None
                )

        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.activation_fn = activation_function(activation_name, activation_params)

        if self.norm_type == "batch":
            LOGGER.warning("Batch norm. May perform poorly with small batch sizes.")
        if self.norm_type == "group" and "num_groups" not in self.norm_params:
            default_ng = 32
            if self.out_channels > 0:
                for ng_cand in range(min(default_ng, self.out_channels), 0, -1):
                    if self.out_channels % ng_cand == 0:
                        default_ng = ng_cand
                        break
                if self.out_channels % default_ng != 0:
                    default_ng = 1
            self.norm_params["num_groups"] = default_ng
            LOGGER.info(
                f"GroupNorm: num_groups defaulted to {default_ng} for out_channels={out_channels}."
            )

        self._setup_shared_ops(down_mode, up_mode, self.merge_mode)

    def _setup_shared_ops(
        self, down_mode: DownModeOptions, up_mode: UpModeOptions, merge_arg: MergeModeOptions
    ):
        self.merge_op = partial(merge, merge_mode=merge_arg)
        self.pool_op = partial(pool, down_mode=down_mode, z_conv=self.z_conv, is_last_stage=False)
        self.up_sample_op = upconv222(self.in_channels, self.out_channels, self.z_conv, up_mode)

    def _build_layers(self):
        raise NotImplementedError

    def _init_weights(self, mod: nn.Module):  # Same init_weights as before
        if isinstance(mod, nn.Conv3d | nn.ConvTranspose3d | nn.Linear):
            nl = "leaky_relu" if self.stored_activation_name == "leaky" else "relu"
            if self.stored_activation_name.lower() in ["relu", "leaky", "silu", "gelu"]:
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity=nl)
            else:
                nn.init.xavier_normal_(mod.weight)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)
        elif isinstance(mod, nn.LayerNorm | nn.BatchNorm3d | nn.GroupNorm | nn.InstanceNorm3d):
            if hasattr(mod, "weight") and mod.weight is not None:
                nn.init.constant_(mod.weight, 1.0)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)


class EfficientDownBlock3D(BaseUnetBlock3D):
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {"residual_activation_name": "none"}

    def __init(
        self,
        in_channels: int,
        out_channels: int,
        skip_out: bool = True,
        dropout_p: float = 0.0,
        down_mode: DownModeOptions = "maxpool",
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
        super().__init__(
            in_channels,
            out_channels,
            skip_out,
            dropout_p,
            down_mode,
            z_conv=z_conv,
            activation_name=activation_name,
            activation_params=activation_params,
            norm_type=norm_type,
            norm_params=norm_params,
            num_blocks=num_blocks,
            intermediate_channels_list=intermediate_channels_list,
            use_depthwise_separable_conv=use_depthwise_separable_conv,
        )
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        self.main_path_units = nn.ModuleList()
        current_c = self.in_channels
        for i in range(self.num_blocks):
            unit_out_ch = self.out_channels
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
                    self.activation_fn,
                    self.norm_type,
                    self.norm_params,
                    self.use_depthwise_separable_conv,
                )
            )
            current_c = unit_out_ch
        if current_c != self.out_channels:
            LOGGER.error(f"EffDown: path ch {current_c} != out_ch {self.out_channels}.")

        res_act_fn = activation_function(self.config["residual_activation_name"], None)
        self.residual_proj = (
            nn.Sequential(conv111(self.in_channels, self.out_channels, bias=True), res_act_fn)
            if self.in_channels != self.out_channels
            else res_act_fn
        )
        self.pool = self.pool_op(self.out_channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        res_x = self.residual_proj(x)
        main_x = x
        for unit in self.main_path_units:
            main_x = unit(main_x)
        out_s = main_x + res_x
        bp = self.dropout(out_s)
        p = self.pool(bp)
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
        super().__init__(
            in_channels,
            out_channels,
            skip_out,
            dropout_p,
            up_mode=up_mode,
            merge_mode=merge_mode,
            z_conv=z_conv,
            activation_name=activation_name,
            activation_params=activation_params,
            norm_type=norm_type,
            norm_params=norm_params,
            num_blocks=num_blocks,
            intermediate_channels_list=intermediate_channels_list,
            use_depthwise_separable_conv=use_depthwise_separable_conv,
        )
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        self.up_sample = self.up_sample_op
        # Initial projection to handle merged channels before main path
        # Max possible input channels to main path (if concat and skip is present)
        merged_ch_max = self.out_channels * 2 if self.merge_mode == "concat" else self.out_channels
        self.proj_after_merge = (
            conv111(merged_ch_max, self.out_channels, bias=True)
            if merged_ch_max != self.out_channels
            else nn.Identity()
        )
        self.proj_no_skip_concat_case = (
            conv111(self.out_channels, self.out_channels, bias=True)
            if self.merge_mode == "concat" and merged_ch_max != self.out_channels
            else nn.Identity()
        )

        self.main_path_units = nn.ModuleList()
        current_c = self.out_channels  # After proj, current_c is self.out_channels
        for i in range(self.num_blocks):
            unit_out_ch = self.out_channels
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
                    self.activation_fn,
                    self.norm_type,
                    self.norm_params,
                    self.use_depthwise_separable_conv,
                )
            )
            current_c = unit_out_ch
        if current_c != self.out_channels:
            LOGGER.error(f"EffUp: path ch {current_c} != out_ch {self.out_channels}.")

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x_up = self.up_sample(x)
        main_x: torch.Tensor
        if skip is not None:
            merged = self.merge_op(x_up, skip)
            main_x = self.proj_after_merge(merged)
        else:  # Skip is None
            if self.merge_mode == "concat":  # proj_after_merge might expect 2*C
                main_x = self.proj_no_skip_concat_case(
                    x_up
                )  # x_up (C) -> proj (C) -> main_path_units
            else:  # Add mode, proj_after_merge expects C
                main_x = self.proj_after_merge(x_up)
        for unit in self.main_path_units:
            main_x = unit(main_x)
        return self.dropout(main_x)


class ConvNeXtDownBlock3D(BaseUnetBlock3D):
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {
        "num_internal_cnext_units": 1,
        "expand_ratio": 4,
        "kernel_size": 7,
        "layer_scale_init": 1e-6,
        "internal_norm_type": "layer",
        "internal_activation_name": "gelu",
    }

    def __init__(
        self, in_channels: int, out_channels: int, **base_kwargs
    ):  # Pass all base kwargs through
        block_config = base_kwargs.pop("block_config", None)
        super().__init__(in_channels, out_channels, **base_kwargs)
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        cfg = self.config
        self.proj = (
            conv111(self.in_channels, self.out_channels, True)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.cnext_blocks = nn.ModuleList()
        cnext_cfg = {k: v for k, v in cfg.items() if k in ConvNeXtBlock3D.DEFAULT_BLOCK_CONFIG}
        cnext_cfg["norm_type"] = cfg["internal_norm_type"]
        cnext_cfg["activation_name"] = cfg["internal_activation_name"]
        for _ in range(max(1, cfg["num_internal_cnext_units"])):
            self.cnext_blocks.append(ConvNeXtBlock3D(self.out_channels, self.z_conv, cnext_cfg))
        self.pool = self.pool_op(self.out_channels, self.out_channels)
        if self.use_depthwise_separable_conv:
            LOGGER.info("ConvNeXt uses own DW-sep. Base flag ignored.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.proj(x)
        for blk in self.cnext_blocks:
            x = blk(x)
        bp = self.dropout(x)
        p = self.pool(bp)
        return (p, bp) if self.skip_out else (p, None)


class ConvNeXtUpBlock3D(BaseUnetBlock3D):
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {
        "num_internal_cnext_units": 1,
        "expand_ratio": 4,
        "kernel_size": 7,
        "layer_scale_init": 1e-6,
        "internal_norm_type": "layer",
        "internal_activation_name": "gelu",
    }

    def __init__(self, in_channels: int, out_channels: int, **base_kwargs):
        block_config = base_kwargs.pop("block_config", None)
        super().__init__(in_channels, out_channels, **base_kwargs)
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        cfg = self.config
        self.up_sample = self.up_sample_op
        merged_ch = self.out_channels * 2 if self.merge_mode == "concat" else self.out_channels
        self.proj_merged = (
            conv111(merged_ch, self.out_channels, True)
            if merged_ch != self.out_channels
            else nn.Identity()
        )
        self.proj_no_skip = (
            nn.Identity()
        )  # If skip is None, input to proj is self.out_channels. proj_merged might take 2*C
        if (
            self.merge_mode == "concat" and self.out_channels != self.out_channels
        ):  # This is for when merged_ch is diff than self.out_channels, only if skip is None
            self.proj_no_skip = conv111(self.out_channels, self.out_channels, True)

        self.cnext_blocks = nn.ModuleList()
        cnext_cfg = {k: v for k, v in cfg.items() if k in ConvNeXtBlock3D.DEFAULT_BLOCK_CONFIG}
        cnext_cfg["norm_type"] = cfg["internal_norm_type"]
        cnext_cfg["activation_name"] = cfg["internal_activation_name"]
        for _ in range(max(1, cfg["num_internal_cnext_units"])):
            self.cnext_blocks.append(ConvNeXtBlock3D(self.out_channels, self.z_conv, cnext_cfg))
        if self.use_depthwise_separable_conv:
            LOGGER.info("ConvNeXt uses own DW-sep. Base flag ignored.")

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x_up = self.up_sample(x)
        proj_x: torch.Tensor
        if skip is not None:
            proj_x = self.proj_merged(self.merge_op(x_up, skip))
        else:
            proj_x = (
                self.proj_no_skip(x_up) if self.merge_mode == "concat" else self.proj_merged(x_up)
            )
        for blk in self.cnext_blocks:
            proj_x = blk(proj_x)
        return self.dropout(proj_x)


class NAFDownBlock3D(BaseUnetBlock3D):
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {
        "num_internal_naf_units": 1,
        "dw_expand": 2,
        "ffn_expand": 2,
        "internal_dropout_p": 0.0,
        "internal_norm_type": "layer",
    }

    def __init__(self, in_channels: int, out_channels: int, **base_kwargs):
        block_config = base_kwargs.pop("block_config", None)
        # NAFNet has internal dropout, so base dropout_p could be 0 or passed to NAFBlock3D
        # Let's pass base dropout_p to NAFBlock3D via config if desired, or use internal_dropout_p
        base_dropout = base_kwargs.get("dropout_p", 0.0)
        super().__init__(in_channels, out_channels, **base_kwargs)
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        if "dropout_p" not in self.config:  # Prioritize specific config, else use base dropout
            self.config["internal_dropout_p"] = base_dropout

        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        cfg = self.config
        self.proj = (
            conv111(self.in_channels, self.out_channels, True)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.naf_blocks = nn.ModuleList()
        naf_cfg = {k: v for k, v in cfg.items() if k in NAFBlock3D.DEFAULT_BLOCK_CONFIG}
        naf_cfg["dropout_p"] = cfg["internal_dropout_p"]  # Ensure correct dropout is passed
        naf_cfg["norm_type"] = cfg["internal_norm_type"]
        for _ in range(max(1, cfg["num_internal_naf_units"])):
            self.naf_blocks.append(NAFBlock3D(self.out_channels, self.z_conv, naf_cfg))
        self.pool = self.pool_op(self.out_channels, self.out_channels)
        if self.use_depthwise_separable_conv:
            LOGGER.info("NAFNet uses own DW. Base flag ignored.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.proj(x)
        for blk in self.naf_blocks:
            x = blk(x)
        # NAFBlock3D has internal dropout, so base self.dropout might be redundant if applied here
        before_pool = x  # Base dropout not applied here, NAFBlock3D handles it.
        pooled = self.pool(before_pool)
        return (pooled, before_pool) if self.skip_out else (pooled, None)


class NAFUpBlock3D(BaseUnetBlock3D):
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {
        "num_internal_naf_units": 1,
        "dw_expand": 2,
        "ffn_expand": 2,
        "internal_dropout_p": 0.0,
        "internal_norm_type": "layer",
    }

    def __init__(self, in_channels: int, out_channels: int, **base_kwargs):
        block_config = base_kwargs.pop("block_config", None)
        base_dropout = base_kwargs.get("dropout_p", 0.0)
        super().__init__(in_channels, out_channels, **base_kwargs)
        self.config = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        if "dropout_p" not in self.config:
            self.config["internal_dropout_p"] = base_dropout
        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        cfg = self.config
        self.up_sample = self.up_sample_op
        merged_ch = self.out_channels * 2 if self.merge_mode == "concat" else self.out_channels
        self.proj_merged = (
            conv111(merged_ch, self.out_channels, True)
            if merged_ch != self.out_channels
            else nn.Identity()
        )
        self.proj_no_skip = nn.Identity()
        if self.merge_mode == "concat" and self.out_channels != self.out_channels:
            self.proj_no_skip = conv111(self.out_channels, self.out_channels, True)

        self.naf_blocks = nn.ModuleList()
        naf_cfg = {k: v for k, v in cfg.items() if k in NAFBlock3D.DEFAULT_BLOCK_CONFIG}
        naf_cfg["dropout_p"] = cfg["internal_dropout_p"]
        naf_cfg["norm_type"] = cfg["internal_norm_type"]
        for _ in range(max(1, cfg["num_internal_naf_units"])):
            self.naf_blocks.append(NAFBlock3D(self.out_channels, self.z_conv, naf_cfg))
        if self.use_depthwise_separable_conv:
            LOGGER.info("NAFNet uses own DW. Base flag ignored.")

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x_up = self.up_sample(x)
        proj_x: torch.Tensor
        if skip is not None:
            proj_x = self.proj_merged(self.merge_op(x_up, skip))
        else:
            proj_x = (
                self.proj_no_skip(x_up) if self.merge_mode == "concat" else self.proj_merged(x_up)
            )
        for blk in self.naf_blocks:
            proj_x = blk(proj_x)
        # NAFBlock3D handles internal dropout. Base dropout not applied again.
        return proj_x


# Legacy blocks (content from previous response, assumed correct)
class DownConvTri3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        z_conv: bool = True,
        skip_out: bool = True,
        dropout_p: float = 0.0,
        group_norm_legacy: int = 0,
        down_mode: DownModeOptions = "maxpool",
        activation: ActivationNameOptions = "relu",
        activation_params: dict | None = None,
    ):
        super().__init__()
        self.skip_out = skip_out
        self.rsconv = conv333(in_channels, out_channels, z_conv, True)
        self.cv2 = conv333(out_channels, out_channels, z_conv, True)
        self.cv3 = conv333(out_channels, out_channels, z_conv, True)
        self.pool = pool(out_channels, out_channels, down_mode, z_conv, False)
        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.gn = (
            nn.GroupNorm(group_norm_legacy, out_channels)
            if group_norm_legacy > 0 and out_channels > 0 and out_channels % group_norm_legacy == 0
            else nn.Identity()
        )
        if group_norm_legacy > 0 and (out_channels == 0 or out_channels % group_norm_legacy != 0):
            LOGGER.warning(f"LegacyDCT3D: GN issue {out_channels},{group_norm_legacy}")
        self.act = activation_function(activation, activation_params)
        self.apply(self._iwl)

    def _iwl(self, m: nn.Module):
        if isinstance(m, nn.Conv3d | nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0) if m.bias is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        r = self.act(self.gn(self.rsconv(x)))
        o = self.act(self.gn(self.cv2(r)))
        o = self.act(self.gn(self.cv3(o) + r))
        bp = self.drop(o)
        p = self.pool(bp)
        return (p, bp) if self.skip_out else (p, None)


class UpConvTri3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        z_conv: bool = True,
        skip_out: bool = True,
        dropout_p: float = 0.0,
        group_norm_legacy: int = 0,
        up_mode: UpModeOptions = "transpose",
        merge_mode: MergeModeOptions = "concat",
        activation: ActivationNameOptions = "relu",
        activation_params: dict | None = None,
    ):
        super().__init__()
        self.up = upconv222(in_channels, out_channels, z_conv, up_mode)
        self.mrg = partial(merge, merge_mode=merge_mode)
        m_ch = out_channels * 2 if merge_mode == "concat" else out_channels
        self.cv1m = conv333(m_ch, out_channels, z_conv, True)
        self.cv1ns = conv333(out_channels, out_channels, z_conv, True)
        self.cv2 = conv333(out_channels, out_channels, z_conv, True)
        self.cv3 = conv333(out_channels, out_channels, z_conv, True)
        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.gn = (
            nn.GroupNorm(group_norm_legacy, out_channels)
            if group_norm_legacy > 0 and out_channels > 0 and out_channels % group_norm_legacy == 0
            else nn.Identity()
        )
        if group_norm_legacy > 0 and (out_channels == 0 or out_channels % group_norm_legacy != 0):
            LOGGER.warning(f"LegacyUCT3D: GN issue {out_channels},{group_norm_legacy}")
        self.act = activation_function(activation, activation_params)
        self.apply(self._iwL)

    def _iwL(self, m: nn.Module):  # noqa: N802
        if isinstance(m, nn.Conv3d | nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0) if m.bias is not None else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x_u = self.up(x)
        m = self.mrg(x_u, skip)
        r_x = self.act(self.gn(self.cv1m(m) if skip is not None else self.cv1ns(m)))
        o = self.act(self.gn(self.cv2(r_x)))
        o = self.act(self.gn(self.cv3(o) + r_x))
        return self.drop(o)


DownBlock3D = EfficientDownBlock3D
UpBlock3D = EfficientUpBlock3D
BLOCK_REGISTRY = {
    "efficient": {"down": EfficientDownBlock3D, "up": EfficientUpBlock3D},
    "convnext": {"down": ConvNeXtDownBlock3D, "up": ConvNeXtUpBlock3D},
    "nafnet": {"down": NAFDownBlock3D, "up": NAFUpBlock3D},
    "tri": {"down": DownConvTri3D, "up": UpConvTri3D},
}

# Export list for test compatibility
__all__ = [
    "BLOCK_REGISTRY",
    "BaseUnetBlock3D",
    "ConvNeXtDownBlock3D",
    "ConvNeXtUpBlock3D",
    "DownBlock3D",
    "DownConvTri3D",
    "EfficientDownBlock3D",
    "EfficientUpBlock3D",
    "NAFDownBlock3D",
    "NAFUpBlock3D",
    "SimpleGate3D",
    "UpBlock3D",
    "UpConvTri3D",
]
