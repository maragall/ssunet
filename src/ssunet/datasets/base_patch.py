# src/ssunet/datasets/singlevolume.py
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.transforms.v2.functional as tf
from numpy.random import rand, randint, seed
from torch.utils.data import Dataset

from ..configs import DataConfig
from ..exceptions import InvalidDataDimensionError
from ..utils import LOGGER, _lucky, to_tensor
from .ssunet_data import SSUnetData


class BasePatchDataset(Dataset, ABC):
    """Base dataset for extracting random 3D patches.

    Handles input data of shape:
    - (D, H, W) -> processed as (1, D, H, W) for internal consistency
    - (C, D, H, W) -> processed as (C, D, H, W)
    - (T, C, D, H, W) -> a time slice is taken, resulting in (C, D, H, W)
    - (T, D, H, W) -> a time slice is taken, resulting in (D,H,W), then (1,D,H,W)

    Output patches from _get_transformed_patches are always in
    (C_out, patch_D, patch_H, patch_W) format.
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_data_raw = input_data
        self.config = config
        self.kwargs = kwargs

        self.source_ndim_raw: int
        self.total_time_frames: int | None = None
        self.effective_channels: int
        self.effective_depth: int
        self.effective_height: int
        self.effective_width: int

        self._initial_data_setup()
        self._validate_patch_vs_effective_dims()
        self.__post_init__()

    def _initial_data_setup(self) -> None:
        primary_data_raw = self.input_data_raw.primary_data
        current_data_shape = (
            primary_data_raw.shape
            if isinstance(primary_data_raw, np.ndarray)
            else primary_data_raw.size()
        )

        self.source_ndim_raw = len(current_data_shape)
        self.total_time_frames = None

        if self.source_ndim_raw == 5:  # Strictly (T, C, D, H, W)
            self.total_time_frames = current_data_shape[0]
            self.effective_channels = current_data_shape[1]
            self.effective_depth = current_data_shape[2]
            self.effective_height = current_data_shape[3]
            self.effective_width = current_data_shape[4]
            LOGGER.info(
                f"5D raw data (shape {current_data_shape}) interpreted as (T, C, D, H, W). "
                f"T={self.total_time_frames}, C={self.effective_channels}, "
                f"D={self.effective_depth}, H={self.effective_height}, W={self.effective_width}."
            )
        elif self.source_ndim_raw == 4:  # Strictly (C, D, H, W)
            self.effective_channels = current_data_shape[0]
            self.effective_depth = current_data_shape[1]
            self.effective_height = current_data_shape[2]
            self.effective_width = current_data_shape[3]
            LOGGER.info(
                f"4D raw data (shape {current_data_shape}) interpreted as (C, D, H, W). "
                f"C={self.effective_channels}, D={self.effective_depth}, "
                f"H={self.effective_height}, W={self.effective_width}."
            )
        elif self.source_ndim_raw == 3:  # Strictly (D, H, W)
            self.effective_channels = 1
            self.effective_depth = current_data_shape[0]
            self.effective_height = current_data_shape[1]
            self.effective_width = current_data_shape[2]
            LOGGER.info(
                f"3D raw data (shape {current_data_shape}) interpreted as (D, H, W). "
                f"Effective C=1, D={self.effective_depth}, "
                f"H={self.effective_height}, W={self.effective_width}."
            )
        else:
            raise InvalidDataDimensionError(
                f"Raw primary data must be 3D, 4D, or 5D. Got {self.source_ndim_raw}D "
                f"with shape {current_data_shape}"
            )

        if self.input_data_raw.secondary_data is not None:
            secondary_data_raw = self.input_data_raw.secondary_data
            secondary_shape = (
                secondary_data_raw.shape
                if isinstance(secondary_data_raw, np.ndarray)
                else secondary_data_raw.size()
            )
            if current_data_shape != secondary_shape:
                raise ValueError(
                    f"Shape mismatch: raw primary data {current_data_shape}, "
                    f"raw secondary data {secondary_shape}"
                )

    def _get_volume_for_patching(self) -> SSUnetData:
        primary_vol_raw = self.input_data_raw.primary_data
        secondary_vol_raw = self.input_data_raw.secondary_data

        primary_tensor_full = (
            to_tensor(primary_vol_raw)
            if isinstance(primary_vol_raw, np.ndarray)
            else primary_vol_raw
        )
        secondary_tensor_full: torch.Tensor | None = None
        if secondary_vol_raw is not None:
            secondary_tensor_full = (
                to_tensor(secondary_vol_raw)
                if isinstance(secondary_vol_raw, np.ndarray)
                else secondary_vol_raw
            )

        primary_volume_final: torch.Tensor
        secondary_volume_final: torch.Tensor | None = None

        if self.source_ndim_raw == 5:
            if self.total_time_frames is None:
                raise RuntimeError("total_time_frames not set for 5D data.")
            t_idx = randint(self.total_time_frames)
            primary_volume_final = primary_tensor_full[t_idx]
            if secondary_tensor_full is not None:
                secondary_volume_final = secondary_tensor_full[t_idx]
        else:
            primary_volume_final = primary_tensor_full
            secondary_volume_final = secondary_tensor_full

        return SSUnetData(primary_data=primary_volume_final, secondary_data=secondary_volume_final)

    def _validate_patch_vs_effective_dims(self) -> None:
        if self.effective_depth < self.config.z_size:
            raise ValueError(
                f"Effective data depth ({self.effective_depth}) is smaller than "
                f"patch z_size ({self.config.z_size})."
            )
        if self.effective_height < self.config.xy_size:
            raise ValueError(
                f"Effective data height ({self.effective_height}) is smaller than "
                f"patch xy_size ({self.config.xy_size})."
            )
        if self.effective_width < self.config.xy_size:
            raise ValueError(
                f"Effective data width ({self.effective_width}) is smaller than "
                f"patch xy_size ({self.config.xy_size})."
            )

    def __post_init__(self):
        """Post initialization function."""
        if self.config.seed is not None:
            seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            LOGGER.info(f"Random seed set to {self.config.seed} for dataset.")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.config.virtual_size == 0:
            LOGGER.debug(
                "virtual_size is 0. Length based on effective_depth: %s, z_size: %s",
                self.effective_depth,
                self.config.z_size,
            )
            base_len = self.effective_depth - self.config.z_size + 1
        else:
            base_len = self.config.virtual_size
        return max(1, base_len // self.config.skip_frames)

    def _get_random_3d_patch_coordinates(self) -> tuple[int, int, int]:
        if self.config.random_crop:
            d_start = randint(self.effective_depth - self.config.z_size + 1)
            h_start = randint(self.effective_height - self.config.xy_size + 1)
            w_start = randint(self.effective_width - self.config.xy_size + 1)
        else:
            d_start = (self.effective_depth - self.config.z_size) // 2
            h_start = (self.effective_height - self.config.xy_size) // 2
            w_start = (self.effective_width - self.config.xy_size) // 2
        return d_start, h_start, w_start

    def _extract_spatial_patch(
        self, volume_after_t_slice: torch.Tensor, d_start: int, h_start: int, w_start: int
    ) -> torch.Tensor:
        d_end = d_start + self.config.z_size
        h_end = h_start + self.config.xy_size
        w_end = w_start + self.config.xy_size

        if volume_after_t_slice.ndim == 4:
            return volume_after_t_slice[:, d_start:d_end, h_start:h_end, w_start:w_end]
        elif volume_after_t_slice.ndim == 3:
            return volume_after_t_slice[d_start:d_end, h_start:h_end, w_start:w_end]
        else:
            raise InvalidDataDimensionError(
                f"Volume for patch extraction has unsupported shape {volume_after_t_slice.shape}. "
                f"Expected 3D or 4D."
            )

    def _apply_sin_gate_transform(self, patch_list_cdhw: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.kwargs.get("apply_custom_sin_gate", False):
            LOGGER.debug("Applying custom 'sin_gate' transform (currently placeholder).")
            return patch_list_cdhw
        return patch_list_cdhw

    def _get_transformed_patches(self) -> list[torch.Tensor]:
        """Get transformed patches."""
        data_for_patching = self._get_volume_for_patching()
        current_primary_vol = data_for_patching.primary_data
        current_secondary_vol = data_for_patching.secondary_data

        d_start, h_start, w_start = self._get_random_3d_patch_coordinates()

        raw_spatial_patches = [
            self._extract_spatial_patch(current_primary_vol, d_start, h_start, w_start)
        ]
        if current_secondary_vol is not None:
            raw_spatial_patches.append(
                self._extract_spatial_patch(current_secondary_vol, d_start, h_start, w_start)
            )

        cdhw_patches = self._ensure_cdhw_format(raw_spatial_patches)

        transformed_list = self._rotate_list_cdhw(cdhw_patches)
        transformed_list = self._augment_list_cdhw(transformed_list)

        final_patches = self._apply_sin_gate_transform(transformed_list)
        return final_patches

    @staticmethod
    def _ensure_cdhw_format(patch_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return [patch.unsqueeze(0) if patch.ndim == 3 else patch for patch in patch_list]

    def _rotate_list_cdhw(self, item_list_cdhw: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.config.rotation != 0:
            angle = (
                (rand() * 2 - 1) * self.config.rotation
                if self.config.augments
                else self.config.rotation
            )
            if angle != 0:
                return [tf.rotate(item, angle) for item in item_list_cdhw]
        return item_list_cdhw

    def _augment_list_cdhw(self, item_list_cdhw: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.config.augments:
            if _lucky():
                item_list_cdhw = [torch.transpose(item, -1, -2) for item in item_list_cdhw]
            if _lucky():
                item_list_cdhw = [torch.flip(item, [-1]) for item in item_list_cdhw]
            if _lucky():
                item_list_cdhw = [torch.flip(item, [-2]) for item in item_list_cdhw]
        return item_list_cdhw

    @abstractmethod
    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a single patch from the dataset.

        :param index: Index of the patch to get
        :type index: int
        :return: List of patches
        :rtype: list[torch.Tensor]
        """
        pass
