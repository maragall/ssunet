# src/ssunet/datasets/temporal_halves_validation.py (or your actual validation dataset file name)
import numpy as np
import torch
import torchvision.transforms.v2.functional as tf
from numpy.random import rand, randint
from torch.utils.data import Dataset

from ..configs import DataConfig  # Specific config for patch size, augments etc.
from ..utils import LOGGER, _lucky, _to_tensor  # Your project's helpers


class TemporalHalvesValidationDataset(Dataset):
    def __init__(
        self,
        primary_timeseries_data: np.ndarray | torch.Tensor,
        config: DataConfig,
        spatial_ground_truth_for_metrics: np.ndarray | torch.Tensor | None = None,
        aggregation_halves: str = "mean",  # "sum" or "mean"
        **kwargs,
    ):  # kwargs can be used for other options if needed
        super().__init__()

        # 1. Process primary_timeseries_data (ensure it's a 5D float tensor [T,C,D,H,W])
        if not isinstance(primary_timeseries_data, torch.Tensor):
            self.primary_ts_data = torch.from_numpy(primary_timeseries_data.astype(np.float32))
        else:
            self.primary_ts_data = primary_timeseries_data.float()

        if self.primary_ts_data.ndim == 4:  # Auto-unsqueeze C for (T, D, H, W)
            self.primary_ts_data = self.primary_ts_data.unsqueeze(1)
            LOGGER.info(
                f"Primary timeseries data was 4D, unsqueezed to {self.primary_ts_data.shape}"
            )
        elif self.primary_ts_data.ndim != 5:
            raise ValueError(
                f"Primary timeseries data must be 4D (TDHW) or 5D (TCDHW), "
                f"got {self.primary_ts_data.ndim}D with shape {self.primary_ts_data.shape}"
            )

        self.num_time_frames = self.primary_ts_data.shape[0]
        if self.num_time_frames < 2:
            raise ValueError(
                f"TemporalHalvesValidationDataset requires at least 2 time frames for splitting, "
                f"got {self.num_time_frames}"
            )

        self.num_channels = self.primary_ts_data.shape[1]
        self.full_depth_spatial = self.primary_ts_data.shape[2]
        self.full_height = self.primary_ts_data.shape[3]
        self.full_width = self.primary_ts_data.shape[4]

        self.config = config
        # Validate patch size against full spatial dimensions from primary_ts_data
        if not (
            0 < self.config.z_size <= self.full_depth_spatial
            and 0 < self.config.xy_size <= self.full_height
            and 0 < self.config.xy_size <= self.full_width
        ):
            raise ValueError(
                f"Patch dimensions z:{config.z_size}, xy:{config.xy_size} are invalid or "
                f"exceed data spatial dimensions D:{self.full_depth_spatial}, "
                f"H:{self.full_height}, W:{self.full_width}"
            )

        # 2. Process optional spatial_ground_truth_for_metrics
        self.spatial_gt_for_metrics = None
        if spatial_ground_truth_for_metrics is not None:
            temp_gt = _to_tensor(
                spatial_ground_truth_for_metrics
            ).float()  # Ensure tensor and float

            if temp_gt.ndim == 3:  # (D,H,W) - this is the spatial GT
                self.spatial_gt_for_metrics = temp_gt.unsqueeze(0)  # Convert to (1,D,H,W)
                LOGGER.info(
                    f"Spatial GT (3D) processed to shape: {self.spatial_gt_for_metrics.shape}"
                )
            elif temp_gt.ndim == 4:  # (C,D,H,W) - this is the spatial GT
                self.spatial_gt_for_metrics = temp_gt
                LOGGER.info(
                    f"Spatial GT (4D) processed to shape: {self.spatial_gt_for_metrics.shape}"
                )
            else:
                # This is the error source from your traceback
                raise ValueError(
                    f"spatial_ground_truth_for_metrics must be 3D (DHW) or 4D (CDHW), "
                    f"but got {temp_gt.ndim}D with shape {temp_gt.shape}"
                )

            # Consistency checks for the processed spatial GT
            if self.spatial_gt_for_metrics.shape[0] != self.num_channels:
                LOGGER.warning(
                    f"Spatial GT channel count ({self.spatial_gt_for_metrics.shape[0]}) "
                    f"does not match primary data's effective channels ({self.num_channels}). "
                    "Metrics might be problematic if C dimension is not aligned."
                )

            # Check if full spatial dimensions of GT match primary data's full spatial dimensions
            # This ensures that patch coordinates derived from primary can be applied to GT
            if (
                self.spatial_gt_for_metrics.shape[1] != self.full_depth_spatial
                or self.spatial_gt_for_metrics.shape[2] != self.full_height
                or self.spatial_gt_for_metrics.shape[3] != self.full_width
            ):
                LOGGER.warning(
                    f"Full spatial dimensions of GT ({self.spatial_gt_for_metrics.shape[1:]}) "
                    f"do not match primary data's full spatial dimensions "
                    f"({self.full_depth_spatial},{self.full_height},{self.full_width}). "
                    "Patch extraction assumes spatial alignment."
                )

        self.aggregation_halves = aggregation_halves
        if self.aggregation_halves not in ["sum", "mean"]:
            raise ValueError(
                "aggregation_halves in TemporalHalvesValidationDataset must be 'sum' or 'mean'"
            )

        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)  # Also seed torch for any torch-based random ops

    def __len__(self):
        # For validation, often a fixed number of diverse patches is sufficient.
        # If virtual_size is 0, default to a small number (e.g., 1 or a few).
        # Or, could be based on non-overlapping patches if random_crop is False.
        return (
            self.config.virtual_size if self.config.virtual_size > 0 else 20
        )  # Example: default to 20 samples for validation

    def _get_random_spatial_patch_coords(self) -> tuple[int, int, int]:
        # Uses spatial dimensions of the primary time series
        if self.config.random_crop:
            d_start = randint(self.full_depth_spatial - self.config.z_size + 1)
            h_start = randint(self.full_height - self.config.xy_size + 1)
            w_start = randint(self.full_width - self.config.xy_size + 1)
        else:
            d_start = (self.full_depth_spatial - self.config.z_size) // 2
            h_start = (self.full_height - self.config.xy_size) // 2
            w_start = (self.full_width - self.config.xy_size) // 2
        return d_start, h_start, w_start

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        d_start, h_start, w_start = self._get_random_spatial_patch_coords()

        d_end = d_start + self.config.z_size
        h_end = h_start + self.config.xy_size
        w_end = w_start + self.config.xy_size

        # Extract spatial patch across all time frames
        spatial_patch_all_t = self.primary_ts_data[
            :, :, d_start:d_end, h_start:h_end, w_start:w_end
        ]
        # Resulting shape: (T, C, D_patch, H_patch, W_patch)

        mid_t = self.num_time_frames // 2  # Integer division

        # Ensure both halves are non-empty
        # If T=2, mid_t=1. first_half is [0:1] (1 frame), second_half is [1:2] (1 frame)
        # If T=3, mid_t=1. first_half is [0:1] (1 frame), second_half is [1:3] (2 frames)
        # - adjust if strict equal halves needed
        if mid_t == 0:  # Should be caught by T<2 check in __init__
            raise RuntimeError(
                f"Cannot split T={self.num_time_frames} into two halves for patch processing."
            )

        first_half_t_frames = spatial_patch_all_t[0:mid_t, ...]
        second_half_t_frames = spatial_patch_all_t[mid_t : self.num_time_frames, ...]

        if self.aggregation_halves == "sum":
            model_input_cdhw = torch.sum(first_half_t_frames, dim=0).float()
            target_for_loss_cdhw = torch.sum(second_half_t_frames, dim=0).float()
        elif self.aggregation_halves == "mean":
            model_input_cdhw = torch.mean(first_half_t_frames, dim=0).float()
            target_for_loss_cdhw = torch.mean(second_half_t_frames, dim=0).float()
        else:  # Should have been caught in __init__
            raise RuntimeError(f"Invalid aggregation_halves: {self.aggregation_halves}")

        # List of patches for transformation (model input, model target)
        # These are now (C, D_patch, H_patch, W_patch)
        patches_to_transform = [model_input_cdhw, target_for_loss_cdhw]

        # Augmentations are typically OFF for validation for consistency.
        # If on, they apply to the (C,D_patch,H_patch,W_patch) volumes.
        if self.config.augments:
            if _lucky():
                patches_to_transform = [
                    torch.transpose(p, -1, -2) for p in patches_to_transform
                ]  # Transpose H,W
            if _lucky():
                patches_to_transform = [torch.flip(p, [-1]) for p in patches_to_transform]  # Flip W
            if _lucky():
                patches_to_transform = [torch.flip(p, [-2]) for p in patches_to_transform]  # Flip H

        if self.config.rotation != 0:
            # For validation, rotation should ideally be fixed or off.
            # If augments is True, apply random rotation, else fixed rotation.
            angle = (
                (rand() * 2 - 1) * self.config.rotation
                if self.config.augments
                else self.config.rotation
            )
            if angle != 0:  # Avoid rotating by 0 degrees
                patches_to_transform = [tf.rotate(p, angle) for p in patches_to_transform]

        output_list = patches_to_transform  # Contains [transformed_input, transformed_target]

        if self.spatial_gt_for_metrics is not None:
            # Extract corresponding spatial patch from the external GT volume
            gt_spatial_patch_cdhw = self.spatial_gt_for_metrics[
                :, d_start:d_end, h_start:h_end, w_start:w_end
            ]

            if self.config.augments or self.config.rotation != 0:
                LOGGER.warning(
                    "Augmentations/rotations on validation GT patch not synchronized with "
                    "input/target halves. "
                    "Metrics might be affected if validation augmentations are enabled."
                )
            output_list.append(gt_spatial_patch_cdhw)

        return output_list
