"""Single volume dataset."""

import traceback  # Add this import
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as tnf

from ..constants import LOGGER  # Assuming LOGGER is defined here or imported from constants
from ..exceptions import (
    ShapeMismatchError,
    UnsupportedInputModeError,
)
from ..utils import _to_tensor


@dataclass
class DataConfig:
    """Data class for the input data of a single volume dataset."""

    xy_size: int = 256
    z_size: int = 32
    virtual_size: int = 0
    augments: bool = False
    rotation: float = 0
    random_crop: bool = True
    skip_frames: int = 1
    normalize_target: bool = True
    note: str = ""
    seed: int | None = None

    @property
    def name(self) -> str:
        """Get the name of the dataset."""
        return (
            f"{self.note}_{self.virtual_size}x{self.z_size}x{self.xy_size}x{self.xy_size}_skip"
            f"={self.skip_frames}"
        )

    @property
    def validation_config(self) -> "DataConfig":
        """Get the validation configuration."""
        return DataConfig(
            xy_size=self.xy_size,
            z_size=self.z_size,
            virtual_size=0,
            augments=False,
            rotation=0,
            random_crop=False,
            skip_frames=1,
            normalize_target=False,
            note=self.note + "_validation",
        )

    @property
    def as_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class SSUnetData:
    """Dataclass for the input data of a single volume dataset."""

    primary_data: np.ndarray | torch.Tensor
    secondary_data: np.ndarray | torch.Tensor | None = None
    allow_dimensionality_mismatch_for_temporal_spatial: bool = False

    def __post_init__(self):
        """Post initialization function."""
        primary_shape_str = (
            str(self.primary_data.shape) if hasattr(self.primary_data, "shape") else "N/A"
        )
        secondary_shape_str = "None"
        if self.secondary_data is not None:
            secondary_shape_str = (
                str(self.secondary_data.shape) if hasattr(self.secondary_data, "shape") else "N/A"
            )

        # Check if the primary and secondary data shapes match, unless mismatch is allowed
        if (
            self.secondary_data is not None
            and not self.allow_dimensionality_mismatch_for_temporal_spatial
        ):
            data_shape_tuple = (
                self.primary_data.shape
                if isinstance(self.primary_data, np.ndarray)
                else tuple(self.primary_data.size())
            )
            secondary_shape_tuple = (
                self.secondary_data.shape
                if isinstance(self.secondary_data, np.ndarray)
                else tuple(self.secondary_data.size())
            )

            if data_shape_tuple != secondary_shape_tuple:
                LOGGER.error("Problematic SSUnetData instance creation:")
                LOGGER.error(f"  Primary shape: {primary_shape_str}")
                LOGGER.error(f"  Secondary shape: {secondary_shape_str}")
                LOGGER.error("  Call stack leading to this SSUnetData instantiation:")
                # Print stack, excluding this __post_init__ and the raise call itself
                for line in traceback.format_stack()[:-2]:
                    LOGGER.error(f"    {line.strip()}")
                raise ShapeMismatchError(
                    f"Data shape {primary_shape_str} and reference shape "
                    f"{secondary_shape_str} do not match. "
                    "Set allow_dimensionality_mismatch_for_temporal_spatial=True "
                    "if this is intended."
                )

    @property
    def shape(self) -> tuple:
        """Get the shape of the data."""
        return self.primary_data.shape

    @property
    def data(self) -> np.ndarray | torch.Tensor:
        """Alias for the primary data."""
        return self.primary_data

    @property
    def reference(self) -> np.ndarray | torch.Tensor | None:
        """Alias for the secondary data."""
        return self.secondary_data

    @staticmethod
    def _apply_binning(input: np.ndarray | torch.Tensor, bin: int, mode: str) -> torch.Tensor:
        """Apply binning to the input data.

        :param input: Input data (np.ndarray or torch.Tensor)
        :param bin: Binning factor
        :param mode: Binning mode ("sum" or "max")
        :return: Binned data as a torch.Tensor
        """
        if isinstance(input, np.ndarray):
            input = _to_tensor(input)
        # Ensure input is 4D (N, C, H, W) for conv2d/max_pool2d
        if input.ndim == 3:  # Assume D, H, W -> 1, D, H, W
            input = input.unsqueeze(0)
        elif input.ndim == 2:  # Assume H, W -> 1, 1, H, W
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.ndim != 4:
            # Handle other dimensions if necessary, or raise error
            raise ValueError(f"Unsupported input dimension for binning: {input.ndim}")

        if mode == "sum":
            # Use unfold/fold or reshape for sum binning in 2D
            # This is a common way to implement sum pooling
            # Input shape: (N, C, H, W)
            # Output shape: (N, C, H/bin, W/bin)
            N, C, H, W = input.shape  # noqa: N806
            if H % bin != 0 or W % bin != 0:
                raise ValueError(
                    f"Input dimensions ({H}, {W}) must be divisible by bin size "
                    f"({bin}) for sum binning."
                )

            # Reshape to (N, C, H/bin, bin, W/bin, bin)
            # Permute to (N, C, H/bin, W/bin, bin, bin)
            # Reshape to (N, C, H/bin, W/bin, bin*bin)
            # Sum over the last dimension
            binned_input = (
                input.reshape(N, C, H // bin, bin, W // bin, bin)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(N, C, H // bin, W // bin, bin * bin)
                .sum(dim=-1)
            )
            return binned_input

        elif mode == "max":
            # Max pool expects (N, C, H, W) or (N, C, D, H, W)
            # Assuming 2D binning (H, W)
            return tnf.max_pool2d(input, kernel_size=bin, stride=bin)
        else:
            raise UnsupportedInputModeError(
                f"Unsupported binning mode: {mode}. Supported modes are 'sum' and 'max'."
            )

    def binxy(self, bin: int = 2, mode: str = "sum") -> None:
        """Apply binning to the input data in the XY dimensions.

        :param bin: Binning factor for XY dimensions.
        :param mode: Binning mode ("sum" or "max").
        """
        self.primary_data = self._apply_binning(self.primary_data, bin, mode=mode)
        if self.secondary_data is not None:
            self.secondary_data = self._apply_binning(self.secondary_data, bin, mode=mode)
