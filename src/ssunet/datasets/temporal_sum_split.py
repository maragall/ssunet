import numpy as np
import torch
from numpy.random import choice, permutation, rand  # Keep existing numpy.random imports

from ..configs import DataConfig, SplitParams
from ..constants import LOGGER
from ..exceptions import ConfigError, DataError, InvalidPValueError
from ..utils import _normalize_by_mean, to_tensor  # Ensure _normalize_by_mean is available
from .base_patch import BasePatchDataset
from .ssunet_data import SSUnetData


class TemporalSumSplitValidationDataset(BasePatchDataset):
    """
    Validation dataset for TemporalSumSplit.
    It extracts a center spatial patch across all time frames.
    Then, it deterministically splits the T dimension into two groups based on
    x_offset_input and y_offset_target parameters, sums frames in each group.
    The 'target' group (sum from y_offset_target to T-1) can be normalized.
    No random augmentations are applied.
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,  # Should be configured for validation
        split_params: SplitParams,  # Kept for API consistency, but not used for splitting logic
        x_offset_input: int = 0,
        y_offset_target: int = 0,
        **kwargs,
    ):
        super().__init__(input_data, config, **kwargs)
        self.split_params = split_params  # Retained for potential other uses or consistency
        self.x_offset_input = x_offset_input
        self.y_offset_target = y_offset_target

        if self.total_time_frames is None or self.total_time_frames < 2:
            raise DataError(
                f"TemporalSumSplitValidationDataset requires input data with "
                f"at least 2 time frames (T >= 2). "
                f"Detected T={self.total_time_frames}."
            )

        # Validate x_offset_input and y_offset_target
        if not (0 <= self.x_offset_input < self.total_time_frames):
            raise ValueError(
                f"x_offset_input ({self.x_offset_input}) must be in range [0, T-1), "
                f"where T={self.total_time_frames}."
            )
        if not (0 <= self.y_offset_target < self.total_time_frames):
            raise ValueError(
                f"y_offset_target ({self.y_offset_target}) must be in range [0, T-1), "
                f"where T={self.total_time_frames}."
            )

        # Ensure at least one frame is selected for each sum
        num_input_frames = self.total_time_frames - self.x_offset_input
        num_target_frames = self.total_time_frames - self.y_offset_target
        if num_input_frames < 1:
            raise ValueError(
                f"x_offset_input ({self.x_offset_input}) results in < 1 frame for input sum "
                f"(T={self.total_time_frames}). Must be T-x_offset_input >= 1."
            )
        if num_target_frames < 1:
            raise ValueError(
                f"y_offset_target ({self.y_offset_target}) results in < 1 frame for target sum "
                f"(T={self.total_time_frames}). Must be T-y_offset_target >= 1."
            )

        LOGGER.info(
            f"TemporalSumSplitValidationDataset initialized with "
            f"x_offset_input={self.x_offset_input}, "
            f"y_offset_target={self.y_offset_target}."
        )
        if self.config.random_crop:
            LOGGER.warning(
                "TemporalSumSplitValidationDataset created with random_crop=True. "
                "This should be False for deterministic validation."
            )
        if self.config.augments:
            LOGGER.warning(
                "TemporalSumSplitValidationDataset created with augments=True. "
                "This should be False for deterministic validation."
            )

    def _get_full_time_spatial_patch(self) -> list[torch.Tensor]:
        """
        Gets the *entire time series* for a selected *center* spatial patch.
        Leverages the overridden _get_random_3d_patch_coordinates for center crop.
        Returns list: [primary_TCDHW_patch, (optional) secondary_TCDHW_patch]
        """
        # This method is identical to TemporalSumSplitDataset._get_full_time_spatial_patch
        # It will use the overridden _get_random_3d_patch_coordinates from this class.
        primary_vol_raw = self.input_data_raw.primary_data
        secondary_vol_raw = self.input_data_raw.secondary_data

        primary_tensor_full = to_tensor(primary_vol_raw)
        secondary_tensor_full: torch.Tensor | None = None
        if secondary_vol_raw is not None:
            secondary_tensor_full = to_tensor(secondary_vol_raw)

        d_patch_start, h_patch_start, w_patch_start = self._get_random_3d_patch_coordinates()
        d_patch_end = d_patch_start + self.config.z_size
        h_patch_end = h_patch_start + self.config.xy_size
        w_patch_end = w_patch_start + self.config.xy_size

        def extract_spatial_across_time(volume_nd: torch.Tensor) -> torch.Tensor:
            if volume_nd.ndim == 5:
                return volume_nd[
                    :,
                    :,
                    d_patch_start:d_patch_end,
                    h_patch_start:h_patch_end,
                    w_patch_start:w_patch_end,
                ]
            elif volume_nd.ndim == 4 and self.total_time_frames is not None:
                return volume_nd[
                    :,
                    d_patch_start:d_patch_end,
                    h_patch_start:h_patch_end,
                    w_patch_start:w_patch_end,
                ]
            else:
                raise DataError(
                    f"Unsupported input shape {volume_nd.shape} for temporal validation processing."
                )

        time_stacked_primary_patch = extract_spatial_across_time(primary_tensor_full)
        output_patches_t_first = [time_stacked_primary_patch]
        if secondary_tensor_full is not None:
            time_stacked_secondary_patch = extract_spatial_across_time(secondary_tensor_full)
            output_patches_t_first.append(time_stacked_secondary_patch)
        return output_patches_t_first

    def _get_random_3d_patch_coordinates(self) -> tuple[int, int, int]:
        """
        Overrides BasePatchDataset._get_random_3d_patch_coordinates.
        Always returns coordinates for a center crop.
        """
        d_start = (self.effective_depth - self.config.z_size) // 2
        h_start = (self.effective_height - self.config.xy_size) // 2
        w_start = (self.effective_width - self.config.xy_size) // 2
        LOGGER.debug(
            f"TemporalSumSplitValidationDataset: Using center crop: "
            f"d={d_start}, h={h_start}, w={w_start}"
        )
        return d_start, h_start, w_start

    def _split_temporal_frames_deterministic(
        self, patch_t_first_dims: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministically splits the T dimension based on
        self.x_offset_input and self.y_offset_target.
        Input sum: frames from 0 to T-1-x_offset_input.
        Target sum: frames from y_offset_target to T-1.
        The target sum is normalized if self.config.normalize_target is True.
        """
        t_total = patch_t_first_dims.shape[0]
        if t_total < 1:  # Should be caught by __init__ (T>=2) but good to be safe
            raise DataError(f"Temporal splitting requires at least T=1 frame. Got T={t_total}.")

        # Define frame ranges
        # Input frames: 0, 1, ..., T-1-x_offset_input
        input_end_idx_exclusive = t_total - self.x_offset_input
        # Target frames: y_offset_target, ..., T-1
        target_start_idx_inclusive = self.y_offset_target

        if input_end_idx_exclusive <= 0:
            raise DataError(
                f"Calculated input end index {input_end_idx_exclusive} is not valid. "
                f"T={t_total}, x_offset_input={self.x_offset_input}"
            )
        if target_start_idx_inclusive >= t_total:
            raise DataError(
                f"Calculated target start index {target_start_idx_inclusive} is not valid. "
                f"T={t_total}, y_offset_target={self.y_offset_target}"
            )

        input_indices = torch.arange(0, input_end_idx_exclusive, device=patch_t_first_dims.device)
        target_indices = torch.arange(
            target_start_idx_inclusive, t_total, device=patch_t_first_dims.device
        )

        if len(input_indices) == 0:
            raise DataError(
                f"No frames selected for input sum. T={t_total}, "
                f"x_offset_input={self.x_offset_input}"
            )
        if len(target_indices) == 0:
            raise DataError(
                f"No frames selected for target sum. T={t_total}, "
                f"y_offset_target={self.y_offset_target}"
            )

        input_frames_selected = patch_t_first_dims.index_select(0, input_indices)
        target_frames_selected = patch_t_first_dims.index_select(0, target_indices)

        sum_input_frames = torch.sum(input_frames_selected.float(), dim=0)
        sum_target_frames = torch.sum(target_frames_selected.float(), dim=0)

        if self.config.normalize_target:  # From DataConfig
            sum_target_frames = _normalize_by_mean(sum_target_frames)

        # Output order: [target_sum (sum_n), input_sum (sum_m)]
        return sum_target_frames.float(), sum_input_frames.float()

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """
        Returns: [sum_target_frames, sum_input_frames, (optional) gt_patch_processed]
        Patches are center-cropped and deterministically split temporally.
        """
        time_stacked_patches = self._get_full_time_spatial_patch()  # Uses center crop
        primary_t_first = time_stacked_patches[0]

        # sum_target_frames is analogous to sum_n_frames, sum_input_frames to sum_m_frames
        sum_target_frames, sum_input_frames = self._split_temporal_frames_deterministic(
            primary_t_first
        )

        processed_patches_for_transform = [sum_target_frames, sum_input_frames]

        if len(time_stacked_patches) > 1:
            secondary_t_first = time_stacked_patches[1]
            gt_candidate = torch.mean(secondary_t_first.float(), dim=0)
            gt_normalized = _normalize_by_mean(gt_candidate)
            processed_patches_for_transform.append(gt_normalized)

        # Apply formatting. Augmentations/rotations should be disabled by validation DataConfig.
        cdhw_list = self._ensure_cdhw_format(processed_patches_for_transform)
        transformed_list = self._rotate_list_cdhw(cdhw_list)  # No-op if config.rotation=0
        final_patches = self._augment_list_cdhw(transformed_list)  # No-op if config.augments=False

        return final_patches


class TemporalSumSplitDataset(BasePatchDataset):
    """
    Dataset that takes a 4D (T,D,H,W) or 5D (T,C,D,H,W) input.
    It first extracts a random spatial 3D patch (C, D_patch, H_patch, W_patch)
    that exists across all T time frames.
    Then, it randomly splits the T dimension into two counts, m and n (m+n=T, m>=1, n>=1).
    It sums m randomly selected frames for one output component and sums n randomly
    selected frames for the other output component. One component (analogous to target)
    can be normalized based on DataConfig.normalize_target.
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,
        split_params: SplitParams,
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

        if self.split_params.seed is not None:
            # Consider using np.random.default_rng(self.split_params.seed) for isolated randomness
            # For now, sticking to global seed as per existing code.
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
        Splits the T dimension of the patch into two groups of frames (n and m).
        Sums frames in each group. The 'n_frames' group (analogous to target)
        can be normalized.

        Args:
            patch_t_first_dims: Tensor of shape (T, C, D, H, W) or (T, D, H, W).

        Returns:
            Tuple of (sum_n_frames, sum_m_frames),
            where sum_n_frames might be normalized. Both tensors have spatial
            dimensions (C,D,H,W) or (D,H,W).
        """
        t_total = patch_t_first_dims.shape[0]
        if t_total < 2:
            # This check is also in __init__, but good for robustness if method is called directly
            raise DataError(f"Temporal splitting requires at least T=2 frames. Got T={t_total}.")

        # Determine p_for_n (fraction for n_count, the "target-analogous" group)
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
            if not self.split_params.p_list:
                raise InvalidPValueError("p_list is required for 'list' method in temporal split.")
            p_for_n = choice(self.split_params.p_list)
        else:
            raise ConfigError(f"Unsupported split_params.method: {self.split_params.method}")

        if not (0 < p_for_n < 1):
            raise InvalidPValueError(
                f"Derived p_for_n ({p_for_n:.4f}) for temporal split "
                "must be strictly between 0 and 1 "
                "to ensure both groups can have at least one frame."
            )

        # Calculate n_count (number of frames for the first group, analogous to target)
        n_count = round(t_total * p_for_n)

        # Ensure n_count is at least 1, and m_count (T_total - n_count) is also at least 1.
        n_count = max(1, min(int(n_count), t_total - 1))

        all_indices = permutation(t_total)  # Shuffled indices [0, ..., T_total-1]

        # Assign indices for each group
        n_indices_np = all_indices[:n_count]
        m_indices_np = all_indices[n_count:]  # The remaining m_count indices

        n_indices_torch = torch.from_numpy(n_indices_np).to(patch_t_first_dims.device)
        m_indices_torch = torch.from_numpy(m_indices_np).to(patch_t_first_dims.device)

        n_frames = patch_t_first_dims.index_select(0, n_indices_torch)
        m_frames = patch_t_first_dims.index_select(0, m_indices_torch)

        # Sum frames for each group
        sum_n_frames = torch.sum(n_frames.float(), dim=0)
        sum_m_frames = torch.sum(m_frames.float(), dim=0)

        # Normalize the "target-analogous" group if configured
        if self.config.normalize_target:  # From DataConfig
            sum_n_frames = _normalize_by_mean(sum_n_frames)

        return sum_n_frames.float(), sum_m_frames.float()

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """
        Returns: [sum_n_frames, sum_m_frames, (optional) gt_patch_processed]
        sum_n_frames is analogous to 'target' and sum_m_frames to 'noise'.
        All output tensors are (C_out, D_patch, H_patch, W_patch).
        """
        time_stacked_patches = self._get_full_time_spatial_patch()
        primary_t_first = time_stacked_patches[0]

        sum_n_frames, sum_m_frames = self._split_temporal_frames(primary_t_first)

        processed_patches_for_transform = [sum_n_frames, sum_m_frames]

        if len(time_stacked_patches) > 1:
            secondary_t_first = time_stacked_patches[1]
            # Process secondary data: mean of all T frames, then normalize
            gt_candidate = torch.mean(secondary_t_first.float(), dim=0)
            gt_normalized = _normalize_by_mean(gt_candidate)
            processed_patches_for_transform.append(gt_normalized)

        cdhw_list = self._ensure_cdhw_format(processed_patches_for_transform)
        transformed_list = self._rotate_list_cdhw(cdhw_list)
        final_patches = self._augment_list_cdhw(transformed_list)
        # Removed call to _apply_sin_gate_transform as it's a placeholder

        return final_patches

    def create_validation_dataset(self) -> TemporalSumSplitValidationDataset:
        """
        Creates a validation dataset for temporal sum splitting.

        This validation dataset uses the same raw input data but extracts
        a center spatial patch and performs a deterministic temporal split.
        No random augmentations are applied.

        :returns: An instance of TemporalSumSplitValidationDataset.
        """
        LOGGER.info("Creating TemporalSumSplitValidationDataset, sharing tensor data from parent.")

        validation_data_config = DataConfig(
            xy_size=self.config.xy_size,
            z_size=self.config.z_size,
            virtual_size=self.config.virtual_size,  # Or a specific validation virtual size
            random_crop=False,  # Deterministic center crop
            augments=False,  # No random augmentations
            rotation=0,  # No rotation
            normalize_target=self.config.normalize_target,  # Keep same normalization for target sum
            skip_frames=self.config.skip_frames,  # Should be 0 for this dataset type
            seed=None,  # Validation does not need random seed for patch selection
        )

        # Share the raw SSUnetData (which holds torch.Tensors internally)
        shared_ssunet_data = SSUnetData(
            primary_data=self.input_data_raw._internal_primary_data,
            secondary_data=self.input_data_raw._internal_secondary_data,
            allow_dimensionality_mismatch=self.input_data_raw.allow_dimensionality_mismatch,
        )

        # Pass the same split_params; the validation dataset will use it deterministically
        return TemporalSumSplitValidationDataset(
            input_data=shared_ssunet_data,
            config=validation_data_config,
            split_params=self.split_params,  # Validation dataset will use min_p for fixed split
            **self.kwargs,
        )
