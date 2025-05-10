import numpy as np
import torch
from numpy.random import choice, permutation, rand

from ..configs import DataConfig, SplitParams, SSUnetData
from ..configs.file_config import ConfigError
from ..constants import LOGGER
from ..exceptions import DataError, InvalidPValueError
from ..utils import _normalize_by_mean, _to_tensor
from .base_patch import BasePatchDataset  # Assuming BasePatchDataset is in base_patch.py


class TemporalSumSplitDataset(BasePatchDataset):
    """
    Dataset that takes a 4D (T,D,H,W) or 5D (T,C,D,H,W) input.
    It first extracts a spatial 3D patch (C, D_patch, H_patch, W_patch)
    that exists across all T time frames.
    Then, it randomly splits the T dimension into two counts, m and n (m+n=T, m>=1, n>=1).
    It sums m randomly selected frames for the input and uses n randomly selected (and
    mean-normalized) frames as the target.
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,
        split_params: SplitParams,  # Re-purposing for temporal split fraction
        **kwargs,
    ):
        super().__init__(input_data, config, **kwargs)
        self.split_params = split_params

        if self.total_time_frames is None or self.total_time_frames < 2:
            raise DataError(
                f"TemporalSumSplitDataset requires input data with "
                f"at least 2 time frames (T >= 2). "
                f"Detected T={self.total_time_frames}."
            )

        # Seed for temporal splitting randomness if provided in split_params
        if self.split_params.seed is not None:
            # This seed is distinct from self.config.seed for spatial patch/augmentation
            np.random.seed(self.split_params.seed)
            LOGGER.info(
                f"NumPy random seed set to {self.split_params.seed} for temporal splitting."
            )

    def _get_full_time_spatial_patch(self) -> list[torch.Tensor]:
        """
        Overrides base method to get the *entire time series* for a selected
        spatial patch before transformations.
        Returns list: [primary_TCDHW_patch, (optional) secondary_TCDHW_patch]
        where patch is (T, C_eff, patch_D, patch_H, patch_W) or (T, patch_D, patch_H, patch_W)
        """
        # Get raw data (could be NP array or Tensor)
        primary_vol_raw = self.input_data_raw.primary_data
        secondary_vol_raw = self.input_data_raw.secondary_data

        primary_tensor_full = (
            _to_tensor(primary_vol_raw)
            if isinstance(primary_vol_raw, np.ndarray)
            else primary_vol_raw
        )

        secondary_tensor_full: torch.Tensor | None = None
        if secondary_vol_raw is not None:
            secondary_tensor_full = (
                _to_tensor(secondary_vol_raw)
                if isinstance(secondary_vol_raw, np.ndarray)
                else secondary_vol_raw
            )

        # Get coordinates for a spatial patch (D,H,W)
        # These coordinates will be applied across all time frames T.
        d_patch_start, h_patch_start, w_patch_start = self._get_random_3d_patch_coordinates()

        d_patch_end = d_patch_start + self.config.z_size
        h_patch_end = h_patch_start + self.config.xy_size
        w_patch_end = w_patch_start + self.config.xy_size

        # Extract spatial patch across all time frames
        # Input shapes to this function are (T,C,D,H,W) or (T,D,H,W) or (C,D,H,W) or (D,H,W)

        def extract_spatial_across_time(volume_nd: torch.Tensor) -> torch.Tensor:
            if volume_nd.ndim == 5:  # (T, C, D_eff, H_eff, W_eff)
                return volume_nd[
                    :,
                    :,
                    d_patch_start:d_patch_end,
                    h_patch_start:h_patch_end,
                    w_patch_start:w_patch_end,
                ]
            elif (
                volume_nd.ndim == 4 and self.total_time_frames is not None
            ):  # (T, D_eff, H_eff, W_eff) assumed C=1
                return volume_nd[
                    :,
                    d_patch_start:d_patch_end,
                    h_patch_start:h_patch_end,
                    w_patch_start:w_patch_end,
                ]
            # These cases should not be hit if T >= 2 for this dataset
            elif volume_nd.ndim == 4 and self.total_time_frames is None:  # (C,D,H,W)
                raise DataError(
                    "TemporalSumSplitDataset expects time dimension (T) "
                    "as the first dimension for 4D/5D data."
                )
            elif volume_nd.ndim == 3 and self.total_time_frames is None:  # (D,H,W)
                raise DataError("TemporalSumSplitDataset expects time dimension (T).")
            else:
                raise DataError(
                    f"Unsupported input shape {volume_nd.shape} for temporal processing."
                )

        # Patches will be (T, C_eff, patch_D, patch_H, patch_W) or (T, patch_D, patch_H, patch_W)
        time_stacked_primary_patch = extract_spatial_across_time(primary_tensor_full)

        output_patches_t_first = [time_stacked_primary_patch]

        if secondary_tensor_full is not None:
            time_stacked_secondary_patch = extract_spatial_across_time(secondary_tensor_full)
            output_patches_t_first.append(time_stacked_secondary_patch)

        return output_patches_t_first

    def _split_temporal_frames(
        self, patch_t_first_dims: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the T dimension of the patch into m and n frames.
        Sums m frames for input, takes the mean of n frames for target, then normalizes target.

        Args:
            patch_t_first_dims: Tensor of shape (T, C, D, H, W) or (T, D, H, W).

        Returns:
            Tuple of (summed_m_frames, normalized_target_mean_n_frames),
            where both tensors have spatial dimensions (C,D,H,W) or (D,H,W).
        """
        t_total = patch_t_first_dims.shape[0]  # Renamed t to T_total for clarity
        if t_total < 2:
            raise DataError(f"Temporal splitting requires at least T=2 frames. Got T={t_total}.")

        # Determine p_for_n (fraction for n_count)
        if self.split_params.method == "signal":
            p_for_n = (
                rand() * (self.split_params.max_p - self.split_params.min_p)
                + self.split_params.min_p
            )
        elif self.split_params.method == "fixed":
            p_for_n = (
                self.split_params.p_list[0] if self.split_params.p_list else self.split_params.min_p
            )
        elif self.split_params.method == "list":
            if not self.split_params.p_list:  # Ensure p_list exists
                raise InvalidPValueError("p_list is required for 'list' method in temporal split.")
            p_for_n = choice(self.split_params.p_list)
        else:
            # This should ideally be caught by SplitParams validation earlier
            raise ConfigError(f"Unsupported split_params.method: {self.split_params.method}")

        # Validate p_for_n itself (should be probability-like for fraction)
        if not (0 < p_for_n < 1):  # Strict (0,1) ensures both m and n can be > 0 from fraction
            raise InvalidPValueError(
                f"Derived p_for_n ({p_for_n:.4f}) "
                f"for temporal split must be strictly between 0 and 1."
            )

        # Calculate n_count (number of frames for target)
        n_count_float = t_total * p_for_n
        n_count = round(n_count_float)  # Convert to int

        # Ensure n_count is at least 1, and m_count (T_total - n_count) is also at least 1.
        # This means 1 <= n_count <= T_total - 1.
        n_count = max(1, min(n_count, t_total - 1))

        m_count = t_total - n_count
        # Due to the clamping above, m_count will also be at least 1.

        # Randomly select distinct indices for m and n frames
        all_indices = permutation(t_total)  # Shuffled indices [0, ..., T_total-1]

        m_indices_np = all_indices[:m_count]
        n_indices_np = all_indices[m_count:]  # The rest are for n

        m_indices_torch = torch.from_numpy(m_indices_np).to(patch_t_first_dims.device)
        n_indices_torch = torch.from_numpy(n_indices_np).to(patch_t_first_dims.device)

        m_frames = patch_t_first_dims.index_select(0, m_indices_torch)
        n_frames_for_target = patch_t_first_dims.index_select(0, n_indices_torch)

        # Input is sum of m_frames
        input_summed = torch.sum(m_frames, dim=0)  # Sum along T dim

        # Target is mean of n_frames_for_target (if n_count > 1), then normalized
        if n_frames_for_target.shape[0] > 1:  # i.e., n_count > 1
            target_frames_mean = torch.mean(n_frames_for_target.float(), dim=0)
        else:  # n_count == 1
            target_frames_mean = n_frames_for_target.squeeze(0).float()

        target_normalized = _normalize_by_mean(target_frames_mean)

        return input_summed.float(), target_normalized

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """
        Returns: [input_sum_m_frames, target_normalized_n_frames, (optional) gt_patch_processed]
        All output tensors are (C_out, D_patch, H_patch, W_patch).
        """
        # 1. Get full time series for a chosen spatial patch.
        # List of [primary_T_first, (optional) secondary_T_first]
        # Shapes are (T, C_eff, D_patch, H_patch, W_patch) or (T, D_patch, H_patch, W_patch)
        time_stacked_patches = self._get_full_time_spatial_patch()

        primary_t_first = time_stacked_patches[0]

        # 2. Split primary data temporally
        input_sum_m, target_norm_n = self._split_temporal_frames(primary_t_first)
        # input_sum_m and target_norm_n are (C_eff, D_patch, H, W) or (D_patch, H, W)

        # 3. Apply spatial transformations (rotation, augmentation) to these processed frames
        # And ensure final C,D,H,W format
        # _ensure_cdhw_format adds channel dim if input is 3D (D,H,W) -> (1,D,H,W)
        # Here, C_eff might already be the channel dim.

        processed_patches_for_transform = [input_sum_m, target_norm_n]

        if len(time_stacked_patches) > 1:
            secondary_t_first = time_stacked_patches[1]
            # A simpler GT for now: mean of all T frames of secondary data's patch.
            gt_candidate = torch.mean(secondary_t_first.float(), dim=0)  # (C_eff, D,H,W) or (D,H,W)
            gt_normalized = _normalize_by_mean(gt_candidate)
            processed_patches_for_transform.append(gt_normalized)

        cdhw_list = self._ensure_cdhw_format(processed_patches_for_transform)
        transformed_list = self._rotate_list_cdhw(cdhw_list)
        transformed_list = self._augment_list_cdhw(transformed_list)
        final_patches = self._apply_sin_gate_transform(transformed_list)  # Placeholder

        return final_patches
