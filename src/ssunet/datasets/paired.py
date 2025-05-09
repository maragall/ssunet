# src/ssunet/datasets/paired.py
import torch

from ..exceptions import MissingReferenceError
from .base_patch import BasePatchDataset


class PairedDataset(BasePatchDataset):
    """Dataset for paired data."""

    def __post_init__(self):
        """Post initialization function."""
        super().__post_init__()
        if self.input_data_raw.secondary_data is None:  # Check raw SSUnetData
            raise MissingReferenceError("PairedDataset requires secondary_data.")

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a single patch from the dataset."""
        transformed_patches = self._get_transformed_patches()
        if len(transformed_patches) != 2:
            raise RuntimeError(
                f"PairedDataset expected 2 patches (primary, secondary) "
                f"from _get_transformed_patches, but got {len(transformed_patches)}."
            )
        return transformed_patches
