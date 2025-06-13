# src/ssunet/datasets/bernoulli.py
import torch

from ..configs import DataConfig, SplitConfig
from ..constants import LOGGER
from ..datasets.ssunet_data import SSUnetData
from .binomial import (  # Import BinomIdentityValidationDataset
    BinomDataset,
    BinomIdentityValidationDataset,
)

# Alias BinomIdentityValidationDataset for clarity in the Bernoulli context
BernoulliIdentityValidationDataset = BinomIdentityValidationDataset


class BernoulliDataset(BinomDataset):
    """
    Dataset that splits a signal into a target and noise component.
    The noise is sampled using a Bernoulli distribution where the probability
    of an event is derived from the input patch scaled by a p-value.
    This class inherits most of its functionality from BinomDataset,
    overriding only the noise sampling method.
    """

    def __init__(
        self,
        data_source: SSUnetData,
        config: DataConfig,
        split_params: SplitConfig,
        **kwargs,
    ) -> None:
        """
        Initialize the BernoulliDataset.

        :param data_source: SSUnetData object containing the raw data.
        :param config: DataConfig object with dataset parameters.
        :param split_params: SplitConfig object with parameters for p-value and splitting.
        :param kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(data_source, config, split_params, **kwargs)
        LOGGER.info(
            f"BernoulliDataset initialized. Noise will be sampled using Bernoulli distribution. "
            f"P-sampling method: '{self.split_params.method}'."
        )

    @staticmethod
    def _sample_noise(input_patch_cdhw: torch.Tensor, p_value: float) -> torch.Tensor:
        """Samples noise from a Bernoulli distribution.
           The probability for the Bernoulli distribution at each element is
           derived from `input_patch_cdhw * p_value`, clamped to [0, 1].

        :param input_patch_cdhw: The input patch tensor (C, D, H, W).
        :param p_value: The probability (used to scale input to get Bernoulli probabilities).
        :returns: The sampled noise tensor (C, D, H, W), with values 0 or 1.
        """
        if input_patch_cdhw.ndim != 4:
            raise ValueError(
                f"BernoulliDataset._sample_noise expects a 4D tensor (C,D,H,W), "
                f"got {input_patch_cdhw.shape}"
            )

        # Probabilities for Bernoulli are input_patch * p_value, clamped to [0,1]
        # This means where input_patch is high and p_value is high, probability of noise is high.
        probabilities = torch.clamp(input_patch_cdhw * p_value, 0.0, 1.0)
        return torch.bernoulli(probabilities).float()  # Ensure output is float

    def create_identity_validation_dataset(self) -> BernoulliIdentityValidationDataset:
        """
        Creates an identity validation dataset specific to BernoulliDataset context.
        This dataset returns the original (transformed) patch as both input and target.
        """
        # The actual instance created is BinomIdentityValidationDataset,
        # but we use the alias for the return type hint.
        return super().create_identity_validation_dataset()
