# src/ssunet/datasets/bernoulli.py
import torch

from .binomial import BinomDataset


class BernoulliDataset(BinomDataset):
    """Special case of the BinomDataset where the noise is sampled with a Bernoulli distribution."""

    @staticmethod
    def _sample_noise(input_patch_dhw: torch.Tensor, p_value: float) -> torch.Tensor:
        probabilities = torch.clamp(input_patch_dhw * p_value, 0, 1)
        return torch.bernoulli(probabilities)
