"""Single volume dataset."""

import traceback
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as tnf

from ..constants import LOGGER
from ..exceptions import ShapeMismatchError, UnsupportedInputModeError
from ..utils import to_tensor

ArrayOrTensor = np.ndarray | torch.Tensor


@dataclass
class SSUnetData:
    """
    Dataclass for the input data of a single volume dataset.

    Manages primary data and optional secondary/reference data,
    providing utilities for common operations like binning.
    All input data (np.ndarray or torch.Tensor) is converted to torch.Tensor
    internally upon initialization.
    """

    primary_data: ArrayOrTensor
    secondary_data: ArrayOrTensor | None = None
    allow_dimensionality_mismatch: bool = False

    _primary_original_ndim_on_init: int = field(init=False, repr=False)
    _secondary_original_ndim_on_init: int | None = field(init=False, repr=False, default=None)

    _internal_primary_data: torch.Tensor = field(init=False, repr=False)
    _internal_secondary_data: torch.Tensor | None = field(init=False, repr=False)

    def _check_spatial_dims_match(
        self, primary_shape: tuple[int, ...], secondary_shape: tuple[int, ...]
    ) -> bool:
        """
        Checks if trailing spatial dimensions match if allow_dimensionality_mismatch is True.
        Assumes dimensions like (T, C, Z, Y, X) or (C, Z, Y, X).
        """
        if not primary_shape or not secondary_shape:
            return False  # Should not be called with empty shapes

        # Case 1: Match from the second dimension onwards (e.g., C, Z, Y, X)
        if (
            len(primary_shape) > 1
            and len(secondary_shape) > 1
            and primary_shape[1:] == secondary_shape[1:]
        ):
            LOGGER.info(
                f"SSUnetData: Primary shape {primary_shape} and secondary shape {secondary_shape} "
                "mismatch in the first dimension, but subsequent dimensions match. Allowed."
            )
            return True
        # Case 2: Match from the third dimension onwards (e.g., Z, Y, X)
        elif (
            len(primary_shape) > 2
            and len(secondary_shape) > 2
            and primary_shape[2:] == secondary_shape[2:]
        ):
            LOGGER.info(
                f"SSUnetData: Primary shape {primary_shape} and secondary shape {secondary_shape} "
                "mismatch in leading dimensions, but deeper spatial dimensions match. Allowed."
            )
            return True
        return False

    def _validate_shapes(self):
        """Validates shapes of internal primary and secondary data."""
        primary_shape_tuple = tuple(self._internal_primary_data.shape)
        secondary_shape_tuple: tuple[int, ...] | None = None
        if self._internal_secondary_data is not None:
            secondary_shape_tuple = tuple(self._internal_secondary_data.shape)

        if self._internal_secondary_data is not None:
            shape_match = primary_shape_tuple == secondary_shape_tuple

            if not shape_match and self.allow_dimensionality_mismatch:
                shape_match = self._check_spatial_dims_match(
                    primary_shape_tuple, secondary_shape_tuple
                )

            if not shape_match:
                secondary_shape_str = (
                    str(secondary_shape_tuple) if secondary_shape_tuple else "None"
                )
                LOGGER.error("Problematic SSUnetData instance state:")
                LOGGER.error(f"  Primary shape: {primary_shape_tuple}")
                LOGGER.error(f"  Secondary shape: {secondary_shape_str}")
                # Consider making traceback logging conditional
                # (e.g., on a debug flag or higher log level)
                # if it's too verbose for standard errors.
                for line in traceback.format_stack()[:-3]:  # Adjust depth as needed
                    LOGGER.error(f"    {line.strip()}")
                raise ShapeMismatchError(
                    f"Primary data shape {primary_shape_tuple} and secondary data shape "
                    f"{secondary_shape_str} do not match. "
                    "If mismatch is allowed for leading dimensions (e.g., time, channel) "
                    "while spatial dimensions match, set `allow_dimensionality_mismatch=True` "
                    "and ensure relevant trailing dimensions are identical."
                )

    def __post_init__(self):
        """
        Post-initialization:
        - Stores original ndims for context.
        - Converts input arrays/tensors to PyTorch tensors for internal use.
        - Validates data shapes.
        """
        # We can skip the original ndim tracking for the internal tensors,
        # as this is primarily for user-facing APIs and debugging.
        self._internal_primary_data = to_tensor(self.primary_data)

        if self.secondary_data is not None:
            self._internal_secondary_data = to_tensor(self.secondary_data)
        else:
            self._internal_secondary_data = None

        self._validate_shapes()

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the primary data tensor."""
        return tuple(self._internal_primary_data.shape)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the primary data tensor."""
        return self._internal_primary_data.ndim

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the primary data tensor."""
        return self._internal_primary_data.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the primary data tensor."""
        return self._internal_primary_data.device

    @property
    def data(self) -> torch.Tensor:
        """Access the primary data as a PyTorch Tensor."""
        return self._internal_primary_data

    @data.setter
    def data(self, value: ArrayOrTensor):
        """Set the primary data, converting to tensor and re-validating shapes."""
        self._primary_original_ndim_on_init = value.ndim
        self._internal_primary_data = to_tensor(value)
        self._validate_shapes()  # Re-validate

    @property
    def reference(self) -> torch.Tensor | None:
        """Access the secondary data as a PyTorch Tensor, or None."""
        return self._internal_secondary_data

    @reference.setter
    def reference(self, value: ArrayOrTensor | None):
        """Set the secondary data, converting to tensor if not None, and re-validating shapes."""
        if value is not None:
            self._secondary_original_ndim_on_init = value.ndim
            self._internal_secondary_data = to_tensor(value)
        else:
            self._secondary_original_ndim_on_init = None
            self._internal_secondary_data = None
        self._validate_shapes()  # Re-validate

    def to_device(self, device: str | torch.device) -> "SSUnetData":
        """Moves both primary and secondary data to the specified device. Returns self."""
        self._internal_primary_data = self._internal_primary_data.to(device)
        if self._internal_secondary_data is not None:
            self._internal_secondary_data = self._internal_secondary_data.to(device)
        return self

    @staticmethod
    def _apply_binning_2d_hw(
        input_tensor: torch.Tensor, bin_factor: int, mode: str
    ) -> torch.Tensor:
        """
        Apply 2D binning to the H, W dimensions (last two) of the input tensor.
        Input tensor is expected to be at least 2D.
        It will be temporarily reshaped to 4D (B, C, H, W) for PyTorch pooling operations.
        """
        if bin_factor <= 1:
            return input_tensor

        if input_tensor.ndim < 2:
            raise ValueError(
                f"Input tensor must be at least 2D for HW binning, got {input_tensor.ndim}D."
            )

        lead_dims = input_tensor.shape[:-2]
        h, w = input_tensor.shape[-2:]

        # Reshape to 4D: (BatchCombined, Channels=1, H, W)
        if len(lead_dims) > 0:
            # Product of leading dimensions becomes the combined batch size
            combined_batch_size = np.prod(lead_dims).item()
            current_data = input_tensor.reshape(combined_batch_size, 1, h, w)
        else:  # Original was 2D (H,W)
            current_data = input_tensor.reshape(1, 1, h, w)

        if h % bin_factor != 0 or w % bin_factor != 0:
            if mode == "sum":
                raise ValueError(
                    f"Input HW dimensions ({h}, {w}) must be divisible by bin size "
                    f"({bin_factor}) for 'sum' binning. Padding not implemented for sum mode."
                )
            # For 'max', tnf.max_pool2d handles non-divisible sizes by default (no padding).

        binned_4d_tensor: torch.Tensor
        if mode == "sum":
            N, C, H_in, W_in = current_data.shape  # noqa: N806
            binned_4d_tensor = (
                current_data.reshape(
                    N, C, H_in // bin_factor, bin_factor, W_in // bin_factor, bin_factor
                )
                .permute(0, 1, 2, 4, 3, 5)  # N, C, H_new, W_new, bin, bin
                .sum(dim=(-1, -2))  # Sum over the two bin dimensions
            )
        elif mode == "max":
            binned_4d_tensor = tnf.max_pool2d(
                current_data, kernel_size=bin_factor, stride=bin_factor, padding=0
            )
        else:
            raise UnsupportedInputModeError(mode)

        # Restore original leading dimensions
        new_h, new_w = binned_4d_tensor.shape[-2:]
        # binned_4d_tensor shape is (CombinedBatch, 1, new_h, new_w)
        # Reshape back to (*lead_dims, new_h, new_w)
        final_binned_tensor = binned_4d_tensor.reshape(*lead_dims, new_h, new_w)

        return final_binned_tensor

    def bin_hw(self, bin_factor: int = 2, mode: str = "sum") -> "SSUnetData":
        """
        Apply binning to the input data in the H, W dimensions (last two).
        Modifies the data in-place.

        :param bin_factor: Binning factor for HW dimensions.
        :param mode: Binning mode ("sum" or "max").
        :return: self (for chaining)
        """
        if bin_factor <= 1:
            return self

        self._internal_primary_data = self._apply_binning_2d_hw(
            self._internal_primary_data, bin_factor, mode=mode
        )
        if self._internal_secondary_data is not None:
            self._internal_secondary_data = self._apply_binning_2d_hw(
                self._internal_secondary_data, bin_factor, mode=mode
            )
        return self

    def copy(self) -> "SSUnetData":
        """Creates a deep copy of this SSUnetData instance."""
        # We instantiate with the internal tensors, which are already torch.Tensor
        # The __post_init__ of the new instance will handle them correctly.
        # The `_primary_original_ndim_on_init` will be set based on the ndim of these tensors.
        new_instance = SSUnetData(
            primary_data=self._internal_primary_data.clone(),
            secondary_data=self._internal_secondary_data.clone()
            if self._internal_secondary_data is not None
            else None,
            allow_dimensionality_mismatch=self.allow_dimensionality_mismatch,
        )
        return new_instance
