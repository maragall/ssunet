# src/ssunet/datasets/paired.py
import numpy as np
import torch

from ..configs import DataConfig  # For type hinting
from ..exceptions import MissingReferenceError, ShapeMismatchError
from ..utils import LOGGER
from .base_patch import BasePatchDataset
from .ssunet_data import SSUnetData  # For type hinting


class PairedDataset(BasePatchDataset):
    """
    Dataset for supervised training with paired data (input and target).

    It expects the input SSUnetData object to contain both primary_data (input)
    and secondary_data (target/ground truth). It yields spatially aligned
    and identically transformed patches for both.
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,
        **kwargs,
    ):
        """
        Initialize the PairedDataset.

        :param input_data: SSUnetData object containing both primary (input)
                           and secondary (target) data.
        :param config: DataConfig object for patch extraction and augmentation.
        :param kwargs: Additional keyword arguments passed to BasePatchDataset.
        """
        super().__init__(input_data, config, **kwargs)
        # BasePatchDataset's __post_init__ will run, then this one.

    def __post_init__(self):
        """
        Post-initialization validation.
        Ensures that secondary_data (target) is present in the raw SSUnetData.
        """
        super().__post_init__()  # Call BasePatchDataset's __post_init__
        if self.input_data_raw.secondary_data is None:
            raise MissingReferenceError(
                "PairedDataset requires secondary_data (target/ground truth) "
                "to be present in the input SSUnetData instance."
            )
        # Ensure spatial dimensions match if secondary data is present
        # This check is also in SSUnetData, but good to have a dataset-level one too.
        if self.input_data_raw.secondary_data is not None:
            primary_shape = self.input_data_raw.primary_data.shape
            secondary_shape = self.input_data_raw.secondary_data.shape
            # Compare spatial dimensions (last 3 for 5D/4D, all for 3D)
            primary_spatial = primary_shape[-3:]
            secondary_spatial = secondary_shape[-3:]
            if primary_spatial != secondary_spatial:
                raise ShapeMismatchError(
                    f"Spatial dimensions of primary ({primary_spatial}) and "
                    f"secondary ({secondary_spatial}) data must match in PairedDataset."
                )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single pair of (input_patch, target_patch) from the dataset.

        The patches are spatially aligned and have undergone the same transformations
        (e.g., cropping, rotation, flips) because BasePatchDataset processes them
        as a list.

        :param index: Index of the item to fetch (often ignored if virtual_size is used).
        :return: A tuple containing the input patch and the target patch.
        """
        # _get_transformed_patches from BasePatchDataset is expected to return
        # a list: [primary_patch_tensor, secondary_patch_tensor]
        # because self.input_data_raw.secondary_data is guaranteed to be present
        # due to the __post_init__ check.
        transformed_patches = self._get_transformed_patches()

        if len(transformed_patches) != 2:
            # This check is a safeguard. If __post_init__ passes and BasePatchDataset
            # works as intended, this condition should not be met.
            raise RuntimeError(
                f"PairedDataset expected 2 patches (input, target) "
                f"from _get_transformed_patches, but got {len(transformed_patches)}. "
                "This may indicate an issue with how SSUnetData or BasePatchDataset "
                "handles secondary data."
            )

        input_patch = transformed_patches[0]
        target_patch = transformed_patches[1]

        return input_patch, target_patch

    def create_validation_dataset(self) -> "PairedValidationDataset":
        """
        Creates a validation dataset for paired data.

        This validation dataset uses the same raw input and target data
        but extracts patches deterministically (e.g., center crop, first time slice)
        and without random augmentations.

        :returns: An instance of PairedValidationDataset.
        """
        LOGGER.info(
            "Creating PairedValidationDataset, sharing tensor data from parent PairedDataset."
        )

        # Create a DataConfig for validation: no random crop, no augmentations
        validation_config = DataConfig(
            xy_size=self.config.xy_size,
            z_size=self.config.z_size,
            virtual_size=self.config.virtual_size,  # Or a specific validation virtual size
            random_crop=False,  # Deterministic crop
            augments=False,  # No random augmentations
            rotation=0,  # No rotation
            normalize_target=self.config.normalize_target,  # Keep same normalization
            skip_frames=self.config.skip_frames,
            seed=self.config.seed,  # Can keep seed or set to None for validation
            # Add any other relevant fields from DataConfig
        )

        # Share the raw SSUnetData (which holds torch.Tensors internally)
        # This avoids reloading or deep copying large data volumes.
        shared_ssunet_data = SSUnetData(
            primary_data=self.input_data_raw._internal_primary_data,
            secondary_data=self.input_data_raw._internal_secondary_data,
            allow_dimensionality_mismatch=self.input_data_raw.allow_dimensionality_mismatch,
        )

        return PairedValidationDataset(
            input_data=shared_ssunet_data,
            config=validation_config,
            **self.kwargs,  # Pass along other relevant kwargs if any
        )


class PairedValidationDataset(BasePatchDataset):
    """
    A dataset for validation with paired data (input and target).

    It yields spatially aligned, identically transformed, and deterministically
    selected patches (e.g., center crop, first time slice) from both
    primary_data (input) and secondary_data (target/ground truth).
    It expects the input SSUnetData object to contain both.
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,  # This config should have random_crop=False, augments=False
        **kwargs,
    ):
        """
        Initialize the PairedValidationDataset.

        :param input_data: SSUnetData object containing both primary (input)
                           and secondary (target) data.
        :param config: DataConfig object. For validation, this should typically
                       have random_crop=False and augments=False.
        :param kwargs: Additional keyword arguments passed to BasePatchDataset.
        """
        super().__init__(input_data, config, **kwargs)
        LOGGER.info("PairedValidationDataset initialized.")
        # Ensure config is set for deterministic validation
        if self.config.random_crop:
            LOGGER.warning(
                "PairedValidationDataset created with random_crop=True. "
                "For deterministic validation, random_crop should be False."
            )
        if self.config.augments:
            LOGGER.warning(
                "PairedValidationDataset created with augments=True. "
                "For deterministic validation, augments should be False."
            )

    def __post_init__(self):
        """
        Post-initialization validation.
        Ensures that secondary_data (target) is present.
        """
        super().__post_init__()
        if self.input_data_raw.secondary_data is None:
            raise MissingReferenceError(
                "PairedValidationDataset requires secondary_data (target/ground truth) "
                "to be present in the input SSUnetData instance."
            )
        # Ensure spatial dimensions match
        primary_shape = self.input_data_raw.primary_data.shape
        secondary_shape = (
            self.input_data_raw.secondary_data.shape
        )  # secondary_data is not None here
        primary_spatial = primary_shape[-3:]
        secondary_spatial = secondary_shape[-3:]
        if primary_spatial != secondary_spatial:
            raise ShapeMismatchError(
                f"Spatial dimensions of primary ({primary_spatial}) and "
                f"secondary ({secondary_spatial}) data must match in PairedValidationDataset."
            )

    def _get_volume_for_patching(self) -> SSUnetData:
        """
        Overrides BasePatchDataset._get_volume_for_patching.
        If raw data is 5D, always selects the first time frame (t_idx=0).
        Otherwise, behaves like the parent but ensures both primary and secondary are processed.
        """
        primary_vol_raw = self.input_data_raw.primary_data
        # secondary_vol_raw should not be None due to __post_init__ check
        secondary_vol_raw = self.input_data_raw.secondary_data

        primary_tensor_full: torch.Tensor
        if isinstance(primary_vol_raw, np.ndarray):
            primary_tensor_full = torch.from_numpy(primary_vol_raw).float()
        elif isinstance(primary_vol_raw, torch.Tensor):
            primary_tensor_full = primary_vol_raw.float()  # Ensure it's float
        else:
            raise TypeError(
                f"Expected primary_vol_raw to be np.ndarray or torch.Tensor, "
                f"got {type(primary_vol_raw)}"
            )

        if secondary_vol_raw is None:
            # This should ideally be caught by __post_init__, but as a safeguard:
            raise MissingReferenceError(
                "Secondary data is None in _get_volume_for_patching for PairedValidationDataset."
            )

        secondary_tensor_full: torch.Tensor
        if isinstance(secondary_vol_raw, np.ndarray):
            secondary_tensor_full = torch.from_numpy(secondary_vol_raw).float()
        elif isinstance(secondary_vol_raw, torch.Tensor):
            secondary_tensor_full = secondary_vol_raw.float()  # Ensure it's float
        else:
            raise TypeError(
                f"Expected secondary_vol_raw to be np.ndarray or torch.Tensor, "
                f"got {type(secondary_vol_raw)}"
            )

        primary_volume_final: torch.Tensor
        secondary_volume_final: torch.Tensor

        if self.source_ndim_raw == 5:
            if self.total_time_frames is None:
                raise RuntimeError("total_time_frames not set for 5D data in validation.")
            t_idx = 0  # Always use the first time frame for validation
            LOGGER.debug(f"PairedValidationDataset: Using t_idx={t_idx} for 5D data.")
            primary_volume_final = primary_tensor_full[t_idx]
            secondary_volume_final = secondary_tensor_full[t_idx]  # type: ignore
        else:
            primary_volume_final = primary_tensor_full
            secondary_volume_final = secondary_tensor_full

        return SSUnetData(primary_data=primary_volume_final, secondary_data=secondary_volume_final)

    def _get_random_3d_patch_coordinates(self) -> tuple[int, int, int]:
        """
        Overrides BasePatchDataset._get_random_3d_patch_coordinates.
        Always returns coordinates for a center crop, as random_crop in config should be False.
        """
        if self.config.random_crop:
            # This case should ideally not be hit if config is set correctly for validation
            LOGGER.warning(
                "PairedValidationDataset is using random_crop despite being a validation set."
            )
            return super()._get_random_3d_patch_coordinates()

        d_start = (self.effective_depth - self.config.z_size) // 2
        h_start = (self.effective_height - self.config.xy_size) // 2
        w_start = (self.effective_width - self.config.xy_size) // 2
        LOGGER.debug(f"PairedValidationDataset: Center crop: d={d_start}, h={h_start}, w={w_start}")
        return d_start, h_start, w_start

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single pair of (input_patch, target_patch) from the dataset.
        Patches are from the first time slice (if 5D) and center-cropped.
        """
        # _get_transformed_patches will use the overridden methods for
        # time slicing and coordinate generation.
        transformed_patches = self._get_transformed_patches()

        if len(transformed_patches) != 2:
            raise RuntimeError(
                f"PairedValidationDataset expected 2 patches (input, target) "
                f"from _get_transformed_patches, but got {len(transformed_patches)}."
            )

        input_patch = transformed_patches[0]
        target_patch = transformed_patches[1]

        return input_patch, target_patch
