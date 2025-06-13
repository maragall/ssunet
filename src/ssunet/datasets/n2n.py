# src/ssunet/datasets/n2n.py
import torch
from numpy.random import randint

from ..configs import DataConfig
from ..exceptions import InvalidDataDimensionError
from ..utils import LOGGER, to_tensor
from .base_patch import BasePatchDataset
from .ssunet_data import SSUnetData


class N2NValidationDataset(BasePatchDataset):
    """
    Validation dataset for N2N (Noise2Noise) skip-frame strategy.

    It extracts a deterministic (center-cropped) deep spatial patch of depth
    (config.z_size * 2) from the first time slice (if data is 5D).
    This deep patch is then split into two sub-patches by taking odd and even
    slices along the depth dimension. No random augmentations are applied.
    Output: [odd_frames_patch, even_frames_patch, (optional) gt_odd_frames_patch]
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,  # Should be configured for validation
        **kwargs,
    ):
        super().__init__(input_data, config, **kwargs)
        self.required_source_patch_depth = self.config.z_size * 2
        if self.effective_depth < self.required_source_patch_depth:
            raise ValueError(
                f"N2NValidationDataset: Effective data depth ({self.effective_depth}) "
                f"is insufficient. Requires {self.required_source_patch_depth} (z_size * 2)."
            )
        LOGGER.info("N2NValidationDataset initialized.")
        if self.config.random_crop:
            LOGGER.warning(
                "N2NValidationDataset created with random_crop=True. "
                "This should be False for deterministic validation."
            )
        if self.config.augments:
            LOGGER.warning(
                "N2NValidationDataset created with augments=True. "
                "This should be False for deterministic validation."
            )

    def _get_volume_for_patching(self) -> SSUnetData:
        """
        Overrides BasePatchDataset._get_volume_for_patching.
        If raw data is 5D, always selects the first time frame (t_idx=0).
        """
        primary_vol_raw = self.input_data_raw.primary_data
        secondary_vol_raw = self.input_data_raw.secondary_data

        primary_tensor_full = to_tensor(primary_vol_raw)
        secondary_tensor_full: torch.Tensor | None = None
        if secondary_vol_raw is not None:
            secondary_tensor_full = to_tensor(secondary_vol_raw)

        primary_volume_final: torch.Tensor
        secondary_volume_final: torch.Tensor | None = None

        if self.source_ndim_raw == 5:
            if self.total_time_frames is None:
                raise RuntimeError("total_time_frames not set for 5D data in N2NValidationDataset.")
            t_idx = 0  # Always use the first time frame
            primary_volume_final = primary_tensor_full[t_idx]
            if secondary_tensor_full is not None:
                secondary_volume_final = secondary_tensor_full[t_idx]
        else:
            primary_volume_final = primary_tensor_full
            secondary_volume_final = secondary_tensor_full
        return SSUnetData(primary_data=primary_volume_final, secondary_data=secondary_volume_final)

    def _get_random_n2n_source_coordinates(self) -> tuple[int, int, int]:
        """
        Gets coordinates for a *center-cropped* deep source patch.
        Overrides the random version in N2NSkipFrameDataset.
        """
        # d_start is for the deeper patch of depth self.required_source_patch_depth
        d_start_source = (self.effective_depth - self.required_source_patch_depth) // 2
        h_start = (self.effective_height - self.config.xy_size) // 2
        w_start = (self.effective_width - self.config.xy_size) // 2
        LOGGER.debug(
            "N2NValidationDataset: Center crop for deep patch: "
            f"d_src={d_start_source}, h={h_start}, w={w_start}"
        )
        return d_start_source, h_start, w_start

    def _extract_deep_spatial_patch(
        self, volume_cdhw_or_dhw: torch.Tensor, d_start_source: int, h_start: int, w_start: int
    ) -> torch.Tensor:
        """Extracts a spatially deeper patch for N2N. Copied from N2NSkipFrameDataset."""
        d_end_source = d_start_source + self.required_source_patch_depth
        h_end = h_start + self.config.xy_size
        w_end = w_start + self.config.xy_size

        if volume_cdhw_or_dhw.ndim == 4:  # (C, D_eff, H_eff, W_eff)
            return volume_cdhw_or_dhw[:, d_start_source:d_end_source, h_start:h_end, w_start:w_end]
        elif volume_cdhw_or_dhw.ndim == 3:  # (D_eff, H_eff, W_eff)
            return volume_cdhw_or_dhw[d_start_source:d_end_source, h_start:h_end, w_start:w_end]
        else:
            raise InvalidDataDimensionError(
                f"Volume for N2N deep patch has shape {volume_cdhw_or_dhw.shape}"
            )

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """
        Output: [odd_frames_patch, even_frames_patch, (optional) gt_odd_frames_patch]
        All patches are (C, config.z_size, H, W).
        """
        data_for_patching = self._get_volume_for_patching()  # Uses first time slice if 5D
        current_primary_vol = data_for_patching.primary_data
        current_secondary_vol = data_for_patching.secondary_data

        d_start_source, h_start, w_start = self._get_random_n2n_source_coordinates()  # Center crop

        primary_deep_patch_raw = self._extract_deep_spatial_patch(
            current_primary_vol, d_start_source, h_start, w_start
        )
        source_patches_for_n2n_logic = [primary_deep_patch_raw]
        if current_secondary_vol is not None:
            secondary_deep_patch_raw = self._extract_deep_spatial_patch(
                current_secondary_vol, d_start_source, h_start, w_start
            )
            source_patches_for_n2n_logic.append(secondary_deep_patch_raw)

        source_patches_cdhw_deep = self._ensure_cdhw_format(source_patches_for_n2n_logic)
        primary_deep_patch_cdhw = source_patches_cdhw_deep[0]

        depth_dim_in_cdhw = 1
        odd_indices = torch.arange(
            0, self.required_source_patch_depth, 2, device=primary_deep_patch_cdhw.device
        )
        even_indices = torch.arange(
            1, self.required_source_patch_depth, 2, device=primary_deep_patch_cdhw.device
        )

        odd_frames_cdhw = primary_deep_patch_cdhw.index_select(depth_dim_in_cdhw, odd_indices)
        even_frames_cdhw = primary_deep_patch_cdhw.index_select(depth_dim_in_cdhw, even_indices)

        final_patches_to_transform = [odd_frames_cdhw, even_frames_cdhw]

        if len(source_patches_cdhw_deep) > 1:
            secondary_deep_patch_cdhw = source_patches_cdhw_deep[1]
            ground_truth_cdhw = secondary_deep_patch_cdhw.index_select(
                depth_dim_in_cdhw, odd_indices
            )
            final_patches_to_transform.append(ground_truth_cdhw)

        # Augmentations/rotations should be disabled by validation DataConfig
        transformed_output_list = self._rotate_list_cdhw(
            final_patches_to_transform
        )  # No-op if config.rotation=0
        transformed_output_list = self._augment_list_cdhw(
            transformed_output_list
        )  # No-op if config.augments=False

        # Assuming _apply_sin_gate_transform is a placeholder or specific to training
        # For validation, it's often omitted unless it's a fixed, non-random transform.
        # If it's a no-op or random, it should not be here for validation.
        # We use transformed_output_list = self._apply_sin_gate_transform(transformed_output_list)

        return transformed_output_list


class N2NSkipFrameDataset(BasePatchDataset):
    """
    Dataset for N2N (Noise2Noise) skip-frame strategy.
    Extracts a deep spatial patch of depth (config.z_size * 2).
    This deep patch is then split into two sub-patches by taking odd and even
    slices along the depth dimension.
    Output: [odd_frames_patch, even_frames_patch, (optional) gt_odd_frames_patch]
    """

    def __post_init__(self):
        super().__post_init__()
        self.required_source_patch_depth = self.config.z_size * 2
        if self.effective_depth < self.required_source_patch_depth:
            raise ValueError(
                f"N2NSkipFrameDataset: Effective data depth ({self.effective_depth}) "
                f"is insufficient. Requires {self.required_source_patch_depth} (z_size * 2)."
            )

    def _get_random_n2n_source_coordinates(self) -> tuple[int, int, int]:
        """Gets random start coordinates for the *deeper* source patch."""
        if self.config.random_crop:
            d_start_source = randint(self.effective_depth - self.required_source_patch_depth + 1)
            h_start = randint(self.effective_height - self.config.xy_size + 1)
            w_start = randint(self.effective_width - self.config.xy_size + 1)
        else:  # Should not happen if validation uses its own coordinate generation
            d_start_source = (self.effective_depth - self.required_source_patch_depth) // 2
            h_start = (self.effective_height - self.config.xy_size) // 2
            w_start = (self.effective_width - self.config.xy_size) // 2
        return d_start_source, h_start, w_start

    def _extract_deep_spatial_patch(
        self, volume_cdhw_or_dhw: torch.Tensor, d_start_source: int, h_start: int, w_start: int
    ) -> torch.Tensor:
        """Extracts a spatially deeper patch for N2N."""
        d_end_source = d_start_source + self.required_source_patch_depth
        h_end = h_start + self.config.xy_size
        w_end = w_start + self.config.xy_size

        if volume_cdhw_or_dhw.ndim == 4:  # (C, D_eff, H_eff, W_eff)
            return volume_cdhw_or_dhw[:, d_start_source:d_end_source, h_start:h_end, w_start:w_end]
        elif volume_cdhw_or_dhw.ndim == 3:  # (D_eff, H_eff, W_eff)
            return volume_cdhw_or_dhw[d_start_source:d_end_source, h_start:h_end, w_start:w_end]
        else:
            raise InvalidDataDimensionError(
                f"Volume for N2N deep patch has shape {volume_cdhw_or_dhw.shape}"
            )

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """
        Output: [odd_frames_patch, even_frames_patch, (optional) gt_odd_frames_patch]
        All patches are (C, config.z_size, H, W).
        """
        # 1. Get the (time-sliced) volume to work on
        data_for_patching = self._get_volume_for_patching()
        current_primary_vol = data_for_patching.primary_data
        current_secondary_vol = data_for_patching.secondary_data

        # 2. Get coordinates for the deep source patch
        d_start_source, h_start, w_start = self._get_random_n2n_source_coordinates()

        # 3. Extract the deep source patch(es)
        primary_deep_patch_raw = self._extract_deep_spatial_patch(
            current_primary_vol, d_start_source, h_start, w_start
        )
        source_patches_for_n2n_logic = [primary_deep_patch_raw]
        if current_secondary_vol is not None:
            secondary_deep_patch_raw = self._extract_deep_spatial_patch(
                current_secondary_vol, d_start_source, h_start, w_start
            )
            source_patches_for_n2n_logic.append(secondary_deep_patch_raw)

        # 4. Ensure they are (C, required_depth, H, W)
        source_patches_cdhw_deep = self._ensure_cdhw_format(source_patches_for_n2n_logic)
        primary_deep_patch_cdhw = source_patches_cdhw_deep[0]

        # 5. Perform N2N even/odd frame splitting
        depth_dim_in_cdhw = 1
        odd_indices = torch.arange(
            0, self.required_source_patch_depth, 2, device=primary_deep_patch_cdhw.device
        )
        even_indices = torch.arange(
            1, self.required_source_patch_depth, 2, device=primary_deep_patch_cdhw.device
        )

        odd_frames_cdhw = primary_deep_patch_cdhw.index_select(depth_dim_in_cdhw, odd_indices)
        even_frames_cdhw = primary_deep_patch_cdhw.index_select(depth_dim_in_cdhw, even_indices)

        final_patches_to_transform = [odd_frames_cdhw, even_frames_cdhw]

        if len(source_patches_cdhw_deep) > 1:
            secondary_deep_patch_cdhw = source_patches_cdhw_deep[1]
            ground_truth_cdhw = secondary_deep_patch_cdhw.index_select(
                depth_dim_in_cdhw, odd_indices
            )
            final_patches_to_transform.append(ground_truth_cdhw)

        # 6. Apply standard transformations
        transformed_output_list = self._rotate_list_cdhw(final_patches_to_transform)
        transformed_output_list = self._augment_list_cdhw(transformed_output_list)

        # 7. Apply SinGate (placeholder or specific transform)
        # If this is a random or training-specific transform, it might be omitted for validation.
        transformed_output_list = self._apply_sin_gate_transform(transformed_output_list)

        return transformed_output_list

    def create_validation_dataset(self) -> N2NValidationDataset:
        """
        Creates a validation dataset for N2N skip-frame.
        Uses the same raw input data but extracts patches deterministically.
        """
        LOGGER.info(
            "Creating N2NValidationDataset, sharing tensor data from parent N2NSkipFrameDataset."
        )

        validation_data_config = DataConfig(
            xy_size=self.config.xy_size,
            z_size=self.config.z_size,  # z_size for the final patches, not the deep one
            virtual_size=self.config.virtual_size,
            random_crop=False,  # Deterministic crop handled by N2NValidationDataset
            augments=False,  # No random augmentations
            rotation=0,  # No rotation
            normalize_target=self.config.normalize_target,  # If applicable to N2N outputs
            skip_frames=self.config.skip_frames,
            seed=None,
        )

        shared_ssunet_data = SSUnetData(
            primary_data=self.input_data_raw._internal_primary_data,
            secondary_data=self.input_data_raw._internal_secondary_data,
            allow_dimensionality_mismatch=self.input_data_raw.allow_dimensionality_mismatch,
        )

        return N2NValidationDataset(
            input_data=shared_ssunet_data,
            config=validation_data_config,
            **self.kwargs,
        )
