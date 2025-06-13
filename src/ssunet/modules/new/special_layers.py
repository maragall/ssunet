# src/ssunet/modules/special_layers.py
"""Special convolution layers and custom components for SSUnet."""

import torch
import torch.nn.functional as tnf
from torch import nn

EPSILON = 1e-6  # Small constant to prevent division by zero


# Custom Activation Functions
class GatedReLUMix(nn.Module):
    """Mixes two parallel processing paths using ReLU and gating mechanism."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels % 2 != 0:
            raise ValueError(f"Input channels must be even for GatedReLUMix, but got {channels}")
        x1, x2 = torch.chunk(x, 2, dim=1)
        out1 = tnf.relu(x1)
        out2 = x1 * x2
        return torch.cat((out1, out2), dim=1)


class SinGatedMix(nn.Module):
    """Mixes a direct sinusoidal activation with sinusoidally gated second half."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels % 2 != 0:
            raise ValueError(f"Input channels must be even for SinGatedMix, but got {channels}")
        x1, x2 = torch.chunk(x, 2, dim=1)
        sin_gate_x1 = torch.sin(x1)
        out1 = sin_gate_x1
        out2 = sin_gate_x1 * x2
        return torch.cat((out1, out2), dim=1)


# Partial Convolution Layers
class PartialConv3d(nn.Conv3d):
    """
    3D Partial Convolutional Layer.
    Adapted from NVIDIA's public implementation: https://github.com/NVIDIA/partialconv
    and https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py (for PConvNd)
    """

    def __init__(self, *args, multi_channel: bool = True, return_mask: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        if (
            self.bias is not None and self.bias.requires_grad
        ):  # PartialConv typically updates bias manually
            self.update_bias = True
        else:
            self.update_bias = False

        self.multi_channel = multi_channel
        self.return_mask = return_mask

        mask_kernel_shape: tuple[int, ...]
        if multi_channel:
            mask_kernel_shape = (
                self.out_channels,
                self.in_channels // self.groups,
                *self.kernel_size,
            )
        else:  # Mask is single channel
            mask_kernel_shape = (1, 1, *self.kernel_size)

        self.register_buffer("weight_mask_updater", torch.ones(mask_kernel_shape))

        # Calculate slide_winsize based on the mask_kernel_shape, not self.weight
        # It's the number of elements in the mask kernel used for convolution
        self.slide_winsize = float(self.weight_mask_updater.shape[1:].numel())

    def forward(
        self, input_tensor: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mask is not None and input_tensor.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Input batch size ({input_tensor.shape[0]}) and "
                f"mask batch size ({mask.shape[0]}) mismatch."
            )
        if mask is not None and input_tensor.shape[2:] != mask.shape[2:]:  # Check D, H, W
            raise ValueError(
                f"Input spatial dims ({input_tensor.shape[2:]}) and "
                f"mask spatial dims ({mask.shape[2:]}) mismatch."
            )

        if mask is None:  # If no mask provided, assume all valid
            mask = torch.ones(
                input_tensor.shape[0],
                1,
                *input_tensor.shape[2:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        if (
            not self.multi_channel and mask.shape[1] != 1
        ):  # Ensure single channel mask if not multi_channel
            mask = mask[:, 0:1, ...]
        elif (
            self.multi_channel and mask.shape[1] != input_tensor.shape[1]
        ):  # Ensure mask channels match input if multi_channel
            if mask.shape[1] == 1:  # If single channel mask given for multi-channel mode, expand it
                mask = mask.expand(-1, input_tensor.shape[1], -1, -1, -1)
            else:
                raise ValueError(
                    f"Mask channels ({mask.shape[1]}) must be 1 "
                    f"or match input channels ({input_tensor.shape[1]}) for multi_channel=True."
                )

        # Mask update (convolution on the mask)
        with torch.no_grad():
            update_mask = tnf.conv3d(
                mask,
                self.weight_mask_updater,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,  # Mask updater is always groups=1 or specialized
            )

        # Mask ratio is sum_1 / sum_M (convention from paper for normalization)
        # Add EPSILON to prevent division by zero
        mask_ratio = self.slide_winsize / (update_mask + EPSILON)

        # Clamp update_mask to [0,1] after calculating ratio, then use for masking output
        update_mask = torch.clamp(update_mask, 0, 1)
        mask_ratio = mask_ratio * update_mask  # Ensure ratio is zero where mask is zero

        # Apply mask to input
        masked_input = input_tensor * mask

        # Perform convolution
        output = super().forward(masked_input)

        # Normalize output and update bias (if applicable)
        if self.update_bias and self.bias is not None:
            # More robust bias handling for partial conv
            bias_view = self.bias.view(1, -1, 1, 1, 1)
            output = (output - bias_view) * mask_ratio + (
                bias_view * update_mask
            )  # Scale bias by updated mask region
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, update_mask
        return output


class PartialConv2d(nn.Conv2d):  # Simplified 2D version for completeness if needed
    """2D Partial Convolutional Layer."""

    def __init__(self, *args, multi_channel: bool = True, return_mask: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_bias = self.bias is not None and self.bias.requires_grad
        self.multi_channel = multi_channel
        self.return_mask = return_mask

        mask_kernel_shape = (
            (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
            if multi_channel
            else (1, 1, *self.kernel_size)
        )
        self.register_buffer("weight_mask_updater", torch.ones(mask_kernel_shape))
        self.slide_winsize = float(self.weight_mask_updater.shape[1:].numel())

    def forward(
        self, input_tensor: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(
                input_tensor.shape[0],
                1,
                *input_tensor.shape[2:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
        if not self.multi_channel and mask.shape[1] != 1:
            mask = mask[:, 0:1, ...]
        elif self.multi_channel and mask.shape[1] != input_tensor.shape[1]:
            if mask.shape[1] == 1:
                mask = mask.expand(-1, input_tensor.shape[1], -1, -1)
            else:
                raise ValueError("Mask channels mismatch for multi_channel=True.")

        with torch.no_grad():
            update_mask = tnf.conv2d(
                mask, self.weight_mask_updater, None, self.stride, self.padding, self.dilation, 1
            )
        mask_ratio = self.slide_winsize / (update_mask + EPSILON)
        update_mask = torch.clamp(update_mask, 0, 1)
        mask_ratio *= update_mask

        output = super().forward(input_tensor * mask)
        if self.update_bias and self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            output = (output - bias_view) * mask_ratio + (bias_view * update_mask)
        else:
            output *= mask_ratio
        return (output, update_mask) if self.return_mask else output


# Gating Mechanism (used by NAFNet)
class SimpleGate3D(nn.Module):
    """3D Simple Gate mechanism from NAFNet (element-wise multiply after channel chunk)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Input channels ({x.shape[1]}) must be even for SimpleGate3D.")
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# Pixel Shuffle/Unshuffle Layers (from previous response, assumed correct)
class PixelShuffle3d(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        s = self.scale
        c_out = c // (s**3)
        return (
            x.contiguous()
            .view(b, c_out, s, s, s, d, h, w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .contiguous()
            .view(b, c_out, d * s, h * s, w * s)
        )


class PixelUnshuffle3d(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        s = self.scale
        c_out = c * (s**3)
        return (
            x.contiguous()
            .view(b, c, d // s, s, h // s, s, w // s, s)
            .permute(0, 1, 3, 5, 7, 2, 4, 6)
            .contiguous()
            .view(b, c_out, d // s, h // s, w // s)
        )


class PixelShuffle2d(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, z, h, w = x.shape
        s = self.scale
        c_out = c // (s**2)
        return (
            x.contiguous()
            .view(b, c_out, s, s, z, h, w)
            .permute(0, 1, 4, 5, 2, 6, 3)
            .contiguous()
            .view(b, c_out, z, h * s, w * s)
        )


class PixelUnshuffle2d(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, z, h, w = x.shape
        s = self.scale
        c_out = c * (s**2)
        return (
            x.contiguous()
            .view(b, c, z, h // s, s, w // s, s)
            .permute(0, 1, 4, 6, 2, 3, 5)
            .contiguous()
            .view(b, c_out, z, h // s, w // s)
        )
