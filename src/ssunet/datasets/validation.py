# src/ssunet/datasets/validation.py
import torch

from ..utils import _normalize_by_mean
from .base_patch import BasePatchDataset


class ValidationDataset(BasePatchDataset):
    """Dataset for validation data."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a single patch from the dataset."""
        transformed_patches = self._get_transformed_patches()
        input_patch_processed_cdhw = transformed_patches[0]

        if self.config.normalize_target:
            target_for_loss_cdhw = _normalize_by_mean(input_patch_processed_cdhw.clone())
        else:
            target_for_loss_cdhw = input_patch_processed_cdhw.clone()

        output_list = [target_for_loss_cdhw, input_patch_processed_cdhw]

        if len(transformed_patches) > 1:  # Secondary data (ground truth) was provided
            ground_truth_processed_cdhw = transformed_patches[1]
            output_list.append(ground_truth_processed_cdhw)

        return output_list
