# src/ssunet/modules/special_layers.py
"""Special convolution layers and custom components for SSUnet."""

import torch
import torch.nn.functional as tnf
from torch import nn

EPSILON = 1e-6  # Small constant to prevent division by zero


class PartialConv3d(nn.Conv3d):
    def __init__(
        self,
        *args,
        multi_channel: bool = False,
        cache_masks: bool = True,
        return_mask: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.cache_masks = cache_masks
        self.return_mask = return_mask

        # Calculate sliding window size
        kernel_elements = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        if self.multi_channel:
            self.slide_winsize = float(kernel_elements * (self.in_channels // self.groups))
        else:
            self.slide_winsize = float(kernel_elements)

        # Initialize cache
        if self.cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None

        # DON'T register bias_view as a buffer - compute it dynamically!

    def _compute_mask_updates(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return from cache if possible
        if (
            self.cache_masks
            and self._last_mask_shape == mask.shape
            and self._last_mask_ptr == mask.data_ptr()
            and self._last_result is not None
        ):
            return self._last_result

        with torch.no_grad():
            # Create weight for sum pooling on-the-fly
            if not self.multi_channel or mask.shape[1] == 1:
                mask_for_sum = mask if mask.shape[1] == 1 else mask[:, 0:1, ...]
                conv_weight = torch.ones(
                    1, 1, *self.kernel_size, device=mask.device, dtype=mask.dtype
                )
                groups_for_mask_conv = 1
            else:
                mask_for_sum = mask
                if self.groups == 1:
                    conv_weight = torch.ones(
                        1,
                        self.in_channels,
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = 1
                else:
                    channels_per_group = self.in_channels // self.groups
                    conv_weight = torch.ones(
                        self.groups,
                        channels_per_group,
                        *self.kernel_size,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    groups_for_mask_conv = self.groups

            # Perform sum pooling
            update_mask = tnf.conv3d(
                mask_for_sum,
                conv_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=groups_for_mask_conv,
            )

            # Calculate ratio and clamp mask
            # For mixed precision training, consider using 1e-6 instead of 1e-6
            mask_ratio = self.slide_winsize / (update_mask + 1e-6)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = mask_ratio * update_mask

        # Update cache
        if self.cache_masks:
            self._last_mask_shape = mask.shape
            self._last_mask_ptr = mask.data_ptr()
            self._last_result = (update_mask, mask_ratio)

        return update_mask, mask_ratio

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

        # Handle mask expansion for input multiplication
        current_mask_for_input_mult = mask
        if self.multi_channel and mask.shape[1] == 1 and input_tensor.shape[1] != 1:
            current_mask_for_input_mult = mask.expand(-1, input_tensor.shape[1], -1, -1, -1)
        elif not self.multi_channel and mask.shape[1] != 1:
            current_mask_for_input_mult = mask[:, 0:1, ...].expand(
                -1, input_tensor.shape[1], -1, -1, -1
            )

        update_mask, mask_ratio = self._compute_mask_updates(mask)

        # Main convolution without bias
        output = tnf.conv3d(
            input_tensor * current_mask_for_input_mult,
            self.weight,
            None,  # Bias is None
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Apply partial conv formula with proper gradient flow
        if self.bias is not None:
            # CRITICAL FIX: Compute bias view dynamically for proper gradient flow
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = output * mask_ratio + bias_view
            output = output * update_mask
        else:
            output = output * mask_ratio

        if self.return_mask:
            return output, update_mask

        return output

    def clear_cache(self):
        """Clear the mask cache. Useful when switching between different mask patterns."""
        if self.cache_masks:
            self._last_mask_shape = None
            self._last_mask_ptr = None
            self._last_result = None


class PixelShuffle3d(nn.Module):
    """Optimized 3D PixelShuffle."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.scale_cubed = scale**3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        s = self.scale
        c_out = c // self.scale_cubed

        return (
            x.reshape(b, c_out, s, s, s, d, h, w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .reshape(b, c_out, d * s, h * s, w * s)
        )

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class PixelUnshuffle3d(nn.Module):
    """Optimized 3D PixelUnshuffle."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.scale_cubed = scale**3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        s = self.scale
        c_out = c * self.scale_cubed

        return (
            x.reshape(b, c, d // s, s, h // s, s, w // s, s)
            .permute(0, 1, 3, 5, 7, 2, 4, 6)
            .reshape(b, c_out, d // s, h // s, w // s)
        )

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class PixelShuffle2d(nn.Module):
    """Optimized 2D PixelShuffle for 5D tensors - preserves depth dimension."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.scale_squared = scale**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, z, h, w = x.shape
        s = self.scale
        c_out = c // self.scale_squared

        return (
            x.reshape(b, c_out, s, s, z, h, w)
            .permute(0, 1, 4, 5, 2, 6, 3)
            .reshape(b, c_out, z, h * s, w * s)
        )

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class PixelUnshuffle2d(nn.Module):
    """Optimized 2D PixelUnshuffle for 5D tensors - preserves depth dimension."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.scale_squared = scale**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, z, h, w = x.shape
        s = self.scale
        c_out = c * self.scale_squared

        return (
            x.reshape(b, c, z, h // s, s, w // s, s)
            .permute(0, 1, 4, 6, 2, 3, 5)
            .reshape(b, c_out, z, h // s, w // s)
        )

    def extra_repr(self) -> str:
        return f"scale={self.scale}"
