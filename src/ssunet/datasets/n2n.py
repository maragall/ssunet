# src/ssunet/datasets/n2n.py
import torch
from numpy.random import randint

from ..exceptions import InvalidDataDimensionError
from .base_patch import BasePatchDataset


class N2NSkipFrameDataset(BasePatchDataset):
    def __post_init__(self):
        super().__post_init__()  # Sets up effective_depth, etc.
        self.required_source_patch_depth = self.config.z_size * 2
        # The patch extraction must be from a D-dim of at least required_source_patch_depth
        if self.effective_depth < self.required_source_patch_depth:
            raise ValueError(
                f"Effective data depth ({self.effective_depth}) is insufficient for N2N skip frame "
                f"which requires source patch depth of "
                f"{self.required_source_patch_depth} (z_size * 2)."
            )

    def _get_random_n2n_source_coordinates(self) -> tuple[int, int, int]:
        """Gets random start coordinates for the *deeper* source patch."""
        if self.config.random_crop:
            # d_start is for the deeper patch
            d_start_source = randint(self.effective_depth - self.required_source_patch_depth + 1)
            h_start = randint(self.effective_height - self.config.xy_size + 1)
            w_start = randint(self.effective_width - self.config.xy_size + 1)
        else:
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
        # 1. Get the (time-sliced) volume to work on
        data_for_patching = self._get_volume_for_patching()  # SSUnetData with (C,D,H,W) or (D,H,W)
        current_primary_vol = data_for_patching.primary_data
        current_secondary_vol = data_for_patching.secondary_data

        # 2. Get coordinates for the deep source patch
        d_start_source, h_start, w_start = self._get_random_n2n_source_coordinates()

        # 3. Extract the deep source patch(es)
        # These are (C, required_depth, H, W) or (required_depth, H, W)
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
        primary_deep_patch_cdhw = source_patches_cdhw_deep[0]  # (C, req_D, H, W)

        # 5. Perform N2N even/odd frame splitting
        # Depth dimension is 1 in (C,D,H,W)
        depth_dim_in_cdhw = 1

        # Indices for selecting frames from the `required_source_patch_depth` dimension
        odd_indices = torch.arange(
            0, self.required_source_patch_depth, 2, device=primary_deep_patch_cdhw.device
        )
        even_indices = torch.arange(
            1, self.required_source_patch_depth, 2, device=primary_deep_patch_cdhw.device
        )

        odd_frames_cdhw = primary_deep_patch_cdhw.index_select(depth_dim_in_cdhw, odd_indices)
        even_frames_cdhw = primary_deep_patch_cdhw.index_select(depth_dim_in_cdhw, even_indices)
        # Now odd/even_frames_cdhw are (C, config.z_size, H, W)

        final_patches_to_transform = [odd_frames_cdhw, even_frames_cdhw]

        if len(source_patches_cdhw_deep) > 1:  # Ground truth was provided
            secondary_deep_patch_cdhw = source_patches_cdhw_deep[1]
            # Original N2N GT was odd frames from secondary
            ground_truth_cdhw = secondary_deep_patch_cdhw.index_select(
                depth_dim_in_cdhw, odd_indices
            )
            final_patches_to_transform.append(ground_truth_cdhw)

        # 6. Apply standard transformations (rotation, augmentation)
        # These operate on (C, config.z_size, H, W) tensors
        transformed_output_list = self._rotate_list_cdhw(final_patches_to_transform)
        transformed_output_list = self._augment_list_cdhw(transformed_output_list)

        # 7. Apply SinGate (placeholder)
        transformed_output_list = self._apply_sin_gate_transform(transformed_output_list)

        return transformed_output_list
