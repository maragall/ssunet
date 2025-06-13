"""Single volume dataset."""

from dataclasses import dataclass, field
from typing import Literal  # Ensure Literal and Any are imported if used

import numpy as np
import torch
import torch.nn.functional as tnf

from ..constants import LOGGER
from ..exceptions import ShapeMismatchError, UnsupportedInputModeError
from ..utils import to_tensor


@dataclass
class SSUnetData:
    """
    Dataclass for the input data of a single volume dataset.

    :param primary_data: Primary input data (np.ndarray or torch.Tensor)
    :param secondary_data: Optional secondary/reference data (np.ndarray or torch.Tensor)
    :param allow_dimensionality_mismatch_for_temporal_spatial: Allow mismatch in leading dimensions
            if spatial dims match
    """

    primary_data: np.ndarray | torch.Tensor
    _internal_primary_data: torch.Tensor = field(init=False, repr=False)
    _internal_secondary_data: torch.Tensor | None = field(init=False, repr=False, default=None)
    secondary_data: np.ndarray | torch.Tensor | None = None
    allow_dimensionality_mismatch_for_temporal_spatial: bool = False

    def __post_init__(self) -> None:
        """
        Post-initialization:
        - Converts input arrays/tensors to PyTorch tensors for internal use.
        - Validates data shapes.
        """
        self._internal_primary_data = to_tensor(self.primary_data)
        self._internal_secondary_data = (
            to_tensor(self.secondary_data) if self.secondary_data is not None else None
        )

        primary_shape_str = (
            str(self._internal_primary_data.shape)
            if hasattr(self._internal_primary_data, "shape")
            else "N/A"
        )
        secondary_shape_str = "None"
        if self._internal_secondary_data is not None:
            secondary_shape_str = (
                str(self._internal_secondary_data.shape)
                if hasattr(self._internal_secondary_data, "shape")
                else "N/A"
            )

        if (
            self._internal_secondary_data is not None
            and not self.allow_dimensionality_mismatch_for_temporal_spatial
        ):
            data_shape_tuple = tuple(self._internal_primary_data.shape)
            secondary_shape_tuple = tuple(self._internal_secondary_data.shape)
            if data_shape_tuple != secondary_shape_tuple:
                LOGGER.error("Problematic SSUnetData instance creation:")
                LOGGER.error(f"  Primary shape: {primary_shape_str}")
                LOGGER.error(f"  Secondary shape: {secondary_shape_str}")
                LOGGER.error("  Call stack leading to this SSUnetData instantiation:")
                import traceback

                for line in traceback.format_stack()[:-2]:
                    LOGGER.error(f"    {line.strip()}")
                raise ShapeMismatchError(
                    f"Data shape {primary_shape_str} and reference shape "
                    f"{secondary_shape_str} do not match. "
                    "Set allow_dimensionality_mismatch_for_temporal_spatial=True "
                    "if this is intended."
                )

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the primary data."""
        return tuple(self._internal_primary_data.shape)

    @property
    def data(self) -> torch.Tensor:
        """Access the primary data as a torch.Tensor."""
        return self._internal_primary_data

    @property
    def reference(self) -> torch.Tensor | None:
        """Access the secondary data as a torch.Tensor, or None."""
        return self._internal_secondary_data

    @staticmethod
    def _apply_binning_2d_hw(
        input_tensor: torch.Tensor, bin_factor: int, mode: str
    ) -> torch.Tensor:
        """
        Apply 2D binning to the H, W dimensions (last two) of the input tensor.
        Input tensor is expected to be at least 2D.
        It will be temporarily reshaped to 4D (B, C, H, W) for PyTorch pooling operations.

        :param input_tensor: Input tensor to bin (at least 2D)
        :param bin_factor: Binning factor for HW dimensions
        :param mode: Binning mode ("sum" or "max")
        :return: Binned tensor
        """
        if bin_factor <= 1:
            return input_tensor

        if input_tensor.ndim < 2:
            raise ValueError(
                f"Input tensor must be at least 2D for HW binning, got {input_tensor.ndim}D."
            )

        lead_dims = input_tensor.shape[:-2]
        h, w = input_tensor.shape[-2:]

        if len(lead_dims) > 0:
            combined_batch_size = int(np.prod(lead_dims))
            current_data = input_tensor.reshape(combined_batch_size, 1, h, w)
        else:
            current_data = input_tensor.reshape(1, 1, h, w)

        current_data_float = current_data.float()

        if h % bin_factor != 0 or w % bin_factor != 0:
            if mode == "sum":
                raise ValueError(
                    f"Input HW dimensions ({h}, {w}) must be divisible by bin size "
                    f"({bin_factor}) for 'sum' binning. Padding not implemented for sum mode."
                )

        binned_4d_tensor: torch.Tensor
        if mode == "sum":
            N, C, H_in, W_in = current_data_float.shape  # noqa: N806
            binned_4d_tensor = (
                current_data_float.reshape(
                    N, C, H_in // bin_factor, bin_factor, W_in // bin_factor, bin_factor
                )
                .permute(0, 1, 2, 4, 3, 5)
                .sum(dim=(-1, -2))
            )
        elif mode == "max":
            binned_4d_tensor = tnf.max_pool2d(
                current_data_float,
                kernel_size=(bin_factor, bin_factor),
                stride=(bin_factor, bin_factor),
                padding=0,
            )
        else:
            raise UnsupportedInputModeError(mode)

        new_h, new_w = binned_4d_tensor.shape[-2:]
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

        if not hasattr(self, "_internal_primary_data"):
            raise AttributeError(
                "SSUnetData object missing '_internal_primary_data'. Make sure it's initialized."
            )

        self._internal_primary_data = self._apply_binning_2d_hw(
            self._internal_primary_data, bin_factor, mode=mode
        )
        if self._internal_secondary_data is not None:
            self._internal_secondary_data = self._apply_binning_2d_hw(
                self._internal_secondary_data, bin_factor, mode=mode
            )
        return self


@dataclass
class SplitConfig:
    """Configuration for splitting data in BinomDataset."""

    method: Literal["signal", "fixed", "list"] = "signal"
    min_p: float = 1e-6
    max_p: float = 1.0 - 1e-6
    p_list: list[float] | None = None
    seed: int | None = None


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
    normalize_target: bool = True  # This is for the main dataset's target
    note: str = ""
    seed: int | None = None

    # New fields for deterministic validation patching
    fixed_z_start: int | None = None
    fixed_t_idx: int | None = None

    @property
    def name(self) -> str:
        """Get the name of the dataset."""
        return (
            f"{self.note}_{self.virtual_size}x{self.z_size}x{self.xy_size}x{self.xy_size}_skip"
            f"={self.skip_frames}"
        )

    @property
    def validation_config(self) -> "DataConfig":
        """General validation configuration (center crop, no augments)."""
        return DataConfig(
            xy_size=self.xy_size,
            z_size=self.z_size,
            virtual_size=0,
            augments=False,
            rotation=0,
            random_crop=False,  # Key for deterministic behavior in BasePatchDataset
            skip_frames=1,
            normalize_target=False,  # ValidationDataset often handles its own target normalization
            note=self.note + "_validation_general",
            seed=None,  # Avoid interference for deterministic parts
            # fixed_z_start and fixed_t_idx will use defaults (None) unless overridden
        )

    @property
    def deterministic_validation_config(self) -> "DataConfig":
        """
        Configuration for highly deterministic validation:
        - Top-Z, Center-XY crop.
        - First T-slice.
        - No augmentations, no rotation.
        - Target normalization is off by default for this config (validation set can decide).
        """
        return DataConfig(
            xy_size=self.xy_size,
            z_size=self.z_size,
            virtual_size=0,
            augments=False,
            rotation=0,
            random_crop=False,  # Crucial: signals non-random patching
            skip_frames=1,
            normalize_target=False,
            note=self.note + "_validation_deterministic",
            seed=None,
            fixed_z_start=0,  # Specific: Start at the first Z-slice
            fixed_t_idx=0,  # Specific: Use the first T-slice
        )

    @property
    def as_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
