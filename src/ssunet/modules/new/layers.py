from logging import getLogger
from typing import ClassVar

import torch
from torch import nn

from ...configs.options import (
    ActivationNameOptions,
    DownModeOptions,
    NormTypeOptions,
    UpModeOptions,
)
from .normalization import get_norm_layer
from .special_layers import (  # Ensure all used special layers are imported
    GatedReLUMix,
    PixelShuffle2d,
    PixelShuffle3d,
    PixelUnshuffle2d,
    PixelUnshuffle3d,
    SimpleGate3D,  # SimpleGate3D moved from blocks.py
    SinGatedMix,
)

LOGGER = getLogger(__name__)
KernelSize3D = int | tuple[int, int, int]
Stride3D = int | tuple[int, int, int]


# --- Refined Convolution Creation ---
def _calculate_padding(kernel_size: KernelSize3D, z_conv: bool) -> int | tuple[int, int, int]:
    actual_kernel_tuple: tuple[int, int, int]
    if isinstance(kernel_size, int):
        actual_kernel_tuple = (
            (kernel_size, kernel_size, kernel_size) if z_conv else (1, kernel_size, kernel_size)
        )
    else:
        actual_kernel_tuple = kernel_size
    if not all(k > 0 for k in actual_kernel_tuple):
        raise ValueError(f"Kernel dimensions must be positive, got {actual_kernel_tuple}")
    padding = tuple(k // 2 for k in actual_kernel_tuple)
    if len(padding) == 3 and actual_kernel_tuple[0] == 1 and not z_conv:
        return (0, padding[1], padding[2])
    if (
        all(p == padding[0] for p in padding) and isinstance(kernel_size, int) and len(padding) == 3
    ):  # e.g. kernel_size = 3 -> padding = 1
        return padding[0]
    return padding


def create_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: KernelSize3D,
    z_conv: bool = True,
    stride: Stride3D = 1,
    is_depthwise: bool = False,
    bias: bool = True,
) -> nn.Conv3d:
    actual_kernel: KernelSize3D = kernel_size
    if isinstance(kernel_size, int) and not z_conv and kernel_size > 1:
        actual_kernel = (1, kernel_size, kernel_size)

    padding = _calculate_padding(actual_kernel, z_conv)
    groups = in_channels if is_depthwise else 1
    _out_channels = out_channels
    if is_depthwise and out_channels != in_channels:
        LOGGER.debug(
            f"Depthwise conv: in_ch({in_channels})!=out_ch({out_channels}). Forcing out_ch=in_ch."
        )
        _out_channels = in_channels

    return nn.Conv3d(
        in_channels,
        _out_channels,
        actual_kernel,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
    )


def conv111(in_channels: int, out_channels: int, bias: bool = True) -> nn.Conv3d:
    return create_conv3d(in_channels, out_channels, 1, z_conv=True, bias=bias)


def conv333(
    in_channels: int, out_channels: int, z_conv: bool = True, bias: bool = True
) -> nn.Conv3d:
    return create_conv3d(in_channels, out_channels, 3, z_conv=z_conv, bias=bias)


def depthwise_conv333(channels: int, z_conv: bool = True, bias: bool = True) -> nn.Conv3d:
    return create_conv3d(channels, channels, 3, z_conv=z_conv, is_depthwise=True, bias=bias)


def conv777(
    in_channels: int, out_channels: int, z_conv: bool = True, bias: bool = True
) -> nn.Conv3d:
    return create_conv3d(in_channels, out_channels, 7, z_conv=z_conv, bias=bias)


# --- Versatile Convolution Unit Builder ---
def build_conv_unit3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    z_conv: bool,
    activation_fn: nn.Module,
    norm_type: NormTypeOptions = "layer",
    norm_params: dict | None = None,
    use_depthwise_separable: bool = False,
    stride: Stride3D = 1,
) -> nn.Sequential:
    if norm_params is None:
        norm_params = {}
    layers_unit = []

    bias_if_norm_follows = False
    bias_if_no_norm = True

    if use_depthwise_separable and kernel_size > 1:
        dw_actual_bias = bias_if_norm_follows if norm_type != "none" else bias_if_no_norm
        layers_unit.append(
            create_conv3d(
                in_channels,
                in_channels,
                kernel_size,
                z_conv,
                stride=stride,
                is_depthwise=True,
                bias=dw_actual_bias,
            )
        )
        if norm_type != "none":
            layers_unit.append(get_norm_layer(norm_type, in_channels, **norm_params))
        layers_unit.append(activation_fn)

        pw_actual_bias = bias_if_norm_follows if norm_type != "none" else bias_if_no_norm
        layers_unit.append(
            create_conv3d(
                in_channels,
                out_channels,
                1,
                z_conv=True,
                stride=1,
                is_depthwise=False,
                bias=pw_actual_bias,
            )
        )
        current_c_for_final_norm_act = out_channels
    else:
        std_actual_bias = bias_if_norm_follows if norm_type != "none" else bias_if_no_norm
        layers_unit.append(
            create_conv3d(
                in_channels,
                out_channels,
                kernel_size,
                z_conv,
                stride=stride,
                is_depthwise=False,
                bias=std_actual_bias,
            )
        )
        current_c_for_final_norm_act = out_channels

    if norm_type != "none":
        layers_unit.append(get_norm_layer(norm_type, current_c_for_final_norm_act, **norm_params))
    layers_unit.append(activation_fn)
    return nn.Sequential(*layers_unit)


# --- Activation Function Factory ---
def activation_function(
    activation_name: ActivationNameOptions, activation_params: dict | None = None
) -> nn.Module:
    """Helper function to create activation layers.

    Args:
        activation_name: Name of the activation function
        activation_params: Optional parameters for the activation function

    Returns:
        Activation layer module
    """
    if activation_params is None:
        activation_params = {}

    name_lower = activation_name.lower()

    match name_lower:
        case "relu":
            return nn.ReLU(inplace=activation_params.get("inplace", True))
        case "leakyrelu":
            return nn.LeakyReLU(
                negative_slope=activation_params.get("negative_slope", 0.01),
                inplace=activation_params.get("inplace", True),
            )
        case "prelu":
            return nn.PReLU(
                num_parameters=activation_params.get("num_parameters", 1),
                init=activation_params.get("init", 0.25),
            )
        case "gelu":
            return nn.GELU(approximate=activation_params.get("approximate", "none"))
        case "silu":
            return nn.SiLU(inplace=activation_params.get("inplace", True))
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "softmax":
            return nn.Softmax(dim=activation_params.get("dim", None))
        case "logsoftmax":
            return nn.LogSoftmax(dim=activation_params.get("dim", None))
        case "gated_relu_mix":
            return GatedReLUMix()
        case "sin_gated_mix":
            return SinGatedMix()
        case "none":
            return nn.Identity()
        case _:
            LOGGER.warning(f"Unknown activation: {activation_name}. Using ReLU instead.")
            return nn.ReLU(inplace=activation_params.get("inplace", True))


# --- Pooling and Upsampling ---
def pool(
    in_channels: int,
    out_channels: int,
    down_mode: DownModeOptions,
    z_conv: bool,
    is_last_stage: bool = False,
) -> nn.Module:
    # (Content from previous response, bias decisions are now internal to create_conv3d/conv111)
    if is_last_stage:
        return nn.Identity()
    kernel_size_pool: KernelSize3D = 2 if z_conv else (1, 2, 2)
    stride_pool: Stride3D = 2 if z_conv else (1, 2, 2)
    if down_mode == "maxpool":
        return nn.MaxPool3d(kernel_size_pool, stride=stride_pool)
    if down_mode == "avgpool":
        return nn.AvgPool3d(kernel_size_pool, stride=stride_pool)
    if down_mode == "conv":
        return create_conv3d(
            in_channels, out_channels, kernel_size_pool, z_conv, stride_pool, bias=True
        )
    if down_mode == "unshuffle":
        mod: nn.Module
        exp_ch: int
        if z_conv:
            mod = PixelUnshuffle3d(2)
            exp_ch = in_channels * 8
        else:
            mod = PixelUnshuffle2d(2)
            exp_ch = in_channels * 4
        if out_channels == exp_ch:
            return mod
        return nn.Sequential(mod, conv111(exp_ch, out_channels, bias=True))
    LOGGER.warning(f"Unknown down_mode: {down_mode}. Using maxpool.")
    return nn.MaxPool3d(kernel_size_pool, stride_pool)


def upconv222(
    in_channels: int, out_channels: int, z_conv: bool, up_mode: UpModeOptions = "transpose"
) -> nn.Module:
    kernel_t: KernelSize3D = 2 if z_conv else (1, 2, 2)
    stride_t: Stride3D = 2 if z_conv else (1, 2, 2)
    s_factor: tuple[float, ...] = (2.0, 2.0, 2.0) if z_conv else (1.0, 2.0, 2.0)
    if up_mode == "transpose":
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_t, stride_t, bias=True)
    if up_mode == "pixelshuffle":
        mod: nn.Module
        div = 8 if z_conv else 4
        if in_channels % div == 0:
            sh_ch = in_channels // div
            mod = PixelShuffle3d(2) if z_conv else PixelShuffle2d(2)
            if sh_ch == out_channels:
                return mod
            return nn.Sequential(mod, conv111(sh_ch, out_channels, bias=True))
        LOGGER.warning("Shuffle fail. Fallback Transpose.")
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_t, stride_t, bias=True)

    umode = up_mode if up_mode in ["trilinear", "nearest", "bilinear"] else "trilinear"
    ac = False if umode != "linear" else None  # align_corners for trilinear/bilinear
    return nn.Sequential(
        nn.Upsample(scale_factor=s_factor, mode=umode, align_corners=ac),
        conv333(in_channels, out_channels, z_conv, bias=True),
    )


# --- Merge Operation ---
def merge(
    input_a: torch.Tensor, input_b: torch.Tensor | None, merge_mode: str = "concat"
) -> torch.Tensor:
    # (Content from previous response, bias decisions for projection internal to conv111)
    if input_b is None:
        return input_a
    if merge_mode == "concat":
        return torch.cat((input_a, input_b), dim=1)
    if merge_mode == "add":
        if input_a.shape != input_b.shape:
            LOGGER.warning(f"Add merge: shapes {input_a.shape} vs {input_b.shape}. Proj skip.")
            if input_a.shape[1] != input_b.shape[1]:  # Project channels of skip
                proj = conv111(input_b.shape[1], input_a.shape[1], bias=True).to(
                    input_b.device, input_b.dtype
                )
                input_b = proj(input_b)
            # Add spatial proj/padding if needed
        return input_a + input_b
    LOGGER.warning(f"Unknown merge: {merge_mode}. Concat.")
    return torch.cat((input_a, input_b), dim=1)


# --- Specialized Blocks (Internal Units) ---
class ConvNeXtBlock3D(nn.Module):  # Internal unit for ConvNeXt stages
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {
        "expand_ratio": 4,
        "kernel_size": 7,
        "layer_scale_init": 1e-6,
        "norm_type": "layer",
        "activation_name": "gelu",
    }

    def __init__(self, channels: int, z_conv: bool = True, block_config: dict | None = None):
        super().__init__()
        cfg = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        norm_params_internal = {}  # Pass if specific norm_params needed, e.g. for group norm

        # DW Conv is followed by Norm, so bias=False
        self.dwconv = create_conv3d(
            channels, channels, cfg["kernel_size"], z_conv, is_depthwise=True, bias=False
        )
        self.norm = get_norm_layer(cfg["norm_type"], channels, **norm_params_internal)

        hidden_dim = channels * cfg["expand_ratio"]
        # Linear layers are not typically followed by spatial norm, so bias=True
        self.pwconv1 = nn.Linear(channels, hidden_dim, bias=True)
        self.act = activation_function(cfg["activation_name"], None)
        self.pwconv2 = nn.Linear(hidden_dim, channels, bias=True)

        if cfg["layer_scale_init"] > 0:
            self.layer_scale = nn.Parameter(
                cfg["layer_scale_init"] * torch.ones((channels, 1, 1, 1)), requires_grad=True
            )
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Same forward as before
        inp = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return inp + self.layer_scale(x) if isinstance(self.layer_scale, nn.Parameter) else inp + x


class NAFBlock3D(nn.Module):  # Internal unit for NAFNet stages
    DEFAULT_BLOCK_CONFIG: ClassVar[dict] = {
        "dw_expand": 2,
        "ffn_expand": 2,
        "dropout_p": 0.0,
        "norm_type": "layer",
    }

    def __init__(self, channels: int, z_conv: bool = True, block_config: dict | None = None):
        super().__init__()
        cfg = {**self.DEFAULT_BLOCK_CONFIG, **(block_config or {})}
        norm_params_internal = {}
        dw_ch, ffn_ch = channels * cfg["dw_expand"], channels * cfg["ffn_expand"]

        # NAFNet specific structure, bias decisions:
        # Conv1 is input to norm1 in forward, so bias can be True.
        # Conv2 (DW) is not immediately followed by its own norm before SG1, so bias=True.
        self.norm1 = get_norm_layer(cfg["norm_type"], channels, **norm_params_internal)
        self.conv1 = conv111(channels, dw_ch, bias=True)
        k_dw: KernelSize3D = 3 if z_conv else (1, 3, 3)
        p_dw = _calculate_padding(k_dw, z_conv)
        self.conv2 = nn.Conv3d(dw_ch, dw_ch, k_dw, 1, p_dw, groups=dw_ch, bias=True)
        self.sg1 = SimpleGate3D()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), conv111(dw_ch // 2, dw_ch // 2, bias=True)
        )  # SCA internal conv
        self.conv3 = conv111(dw_ch // 2, channels, bias=True)
        self.dropout1 = nn.Dropout3d(cfg["dropout_p"]) if cfg["dropout_p"] > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

        self.norm2 = get_norm_layer(cfg["norm_type"], channels, **norm_params_internal)
        self.conv4 = conv111(channels, ffn_ch, bias=True)
        self.sg2 = SimpleGate3D()
        self.conv5 = conv111(ffn_ch // 2, channels, bias=True)
        self.dropout2 = nn.Dropout3d(cfg["dropout_p"]) if cfg["dropout_p"] > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Same forward as before
        identity = x
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.sg1(out)
        out = out * self.sca(out)
        out = self.conv3(out)
        out = self.dropout1(out)
        x = identity + out * self.beta
        identity = x
        out = self.norm2(x)
        out = self.conv4(out)
        out = self.sg2(out)
        out = self.conv5(out)
        out = self.dropout2(out)
        return identity + out * self.gamma
