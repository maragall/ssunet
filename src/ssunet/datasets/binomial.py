# src/ssunet/datasets/binomial.py
import torch
from numpy.random import choice, rand, seed
from torch.distributions.binomial import Binomial

from ..configs import DataConfig, SplitParams, SSUnetData
from ..configs.file_config import ConfigError  # Import ConfigError
from ..constants import LOGGER
from ..exceptions import InvalidPValueError, MissingPListError
from ..utils import _normalize_by_mean
from .base_patch import BasePatchDataset


class BinomDataset(BasePatchDataset):
    """Dataset for binomial splitting."""

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,
        split_params: SplitParams,
        **kwargs,
    ) -> None:
        super().__init__(
            input_data, config, **kwargs
        )  # config.seed is handled in BasePatchDataset.__post_init__
        self.split_params = split_params
        self._validate_and_seed_split_params()  # split_params.seed is for p-sampling

    def _validate_and_seed_split_params(self):
        if self.split_params.seed is not None:  # Seed for p-sampling randomness
            seed(self.split_params.seed)
            LOGGER.info(f"Random seed set to {self.split_params.seed} for BinomDataset p-sampling.")

        # For now, basic check here:
        if self.split_params.min_p >= self.split_params.max_p and self.split_params.method not in [
            "fixed",
            "list",
        ]:
            LOGGER.warning(
                f"split_params.min_p ({self.split_params.min_p}) "
                f">= max_p ({self.split_params.max_p}). "
                f"Ensure this is intended for method '{self.split_params.method}'."
            )

        if self.split_params.method == "fixed":
            p_to_check = (
                self.split_params.p_list[0] if self.split_params.p_list else self.split_params.min_p
            )
            self._validate_p(p_to_check)
        elif self.split_params.method == "list":
            if not self.split_params.p_list:
                raise MissingPListError()
            for p_val in self.split_params.p_list:
                self._validate_p(p_val)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a single patch from the dataset.

        :param index: Index of the patch to retrieve.
        :returns: A list containing the target patch, noise patch,
                    and optionally the ground truth patch.
        """
        transformed_patches = self._get_transformed_patches()
        primary_patch_cdhw = transformed_patches[0]

        # _split expects a (D, H, W) tensor.
        # Handle multi-channel input for splitting:
        if primary_patch_cdhw.shape[0] > 1:  # C > 1
            LOGGER.debug(
                f"Binomial splitting received multi-channel patch "
                f"(shape {primary_patch_cdhw.shape}). "
                "Applying split to the first channel only."
            )
            image_for_splitting_dhw = primary_patch_cdhw[0]

        else:  # C == 1
            image_for_splitting_dhw = primary_patch_cdhw.squeeze(0)

        target_dhw, noise_dhw = self._split(image_for_splitting_dhw)

        output_list = [target_dhw.unsqueeze(0), noise_dhw.unsqueeze(0)]  # Add C=1 dim back

        if len(transformed_patches) > 1:
            output_list.append(transformed_patches[1])  # Append GT patch (C,D,H,W)

        return output_list

    def _split(self, input_patch_dhw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the input patch into target and noise components using binomial distribution.

        :param input_patch_dhw: The input patch tensor (D, H, W).
        :returns: A tuple containing the target patch and the noise patch.
        """
        p_value = self._sample_p(input_patch_dhw)
        noise = self._sample_noise(input_patch_dhw, p_value)
        target = (input_patch_dhw - noise).float()
        if self.split_params.normalize_target:
            target = _normalize_by_mean(target)
        return target, noise.float()

    def _sample_p(self, input_patch_dhw: torch.Tensor) -> float:
        """Samples a p-value based on the configured method.

        :param input_patch_dhw: The input patch tensor (D, H, W), used by some methods
                                (e.g., 'db' - though 'db' is removed).
        :returns: The sampled p-value.
        :raises ConfigError: If an unsupported p-sampling method is configured.
        """
        p_sampling_method_custom: callable | None = self.kwargs.get("p_sampling_method", None)
        if p_sampling_method_custom is not None and callable(p_sampling_method_custom):
            # Custom method must return a p_value that can be validated
            return self._validate_p(p_sampling_method_custom(input_patch_dhw, **self.kwargs))

        method = self.split_params.method
        # Removed "db" case
        if method == "signal":
            return self._sample_signal()  # _sample_signal will call _validate_p
        if method == "fixed":
            return self._sample_fixed()  # _sample_fixed will call _validate_p
        if method == "list":
            return self._sample_list()  # _sample_list will call _validate_p

        # Should be caught by SplitParams.__post_init__ ideally
        raise ConfigError(f"Unsupported p-sampling method '{method}' in BinomDataset.")

    @staticmethod
    def _validate_p(p_value: float) -> float:
        """Validates that the p-value is within the valid range (0, 1).

        :param p_value: The p-value to validate.
        :returns: The validated p-value.
        :raises InvalidPValueError: If the p-value is not between 0 and 1 (exclusive).
        """
        if not (0 < p_value < 1):
            raise InvalidPValueError(f"p-value must be between 0 and 1 (exclusive), got {p_value}")
        return p_value

    def _sample_signal(self) -> float:
        """Samples a p-value uniformly between min_p and max_p.

        :returns: The sampled p-value.
        """
        p_value = (
            rand() * (self.split_params.max_p - self.split_params.min_p) + self.split_params.min_p
        )
        return self._validate_p(p_value)

    def _sample_fixed(self) -> float:
        """Returns a fixed p-value from p_list[0] or min_p.

        :returns: The fixed p-value.
        """
        p_value = (
            self.split_params.p_list[0] if self.split_params.p_list else self.split_params.min_p
        )
        return self._validate_p(p_value)

    def _sample_list(self) -> float:
        """Samples a p-value randomly from the provided p_list.

        :returns: The sampled p-value from the list.
        :raises MissingPListError: If the p_list is empty when method is 'list'.
        """
        if not self.split_params.p_list:
            raise MissingPListError()
        p_value = choice(self.split_params.p_list)
        return self._validate_p(p_value)

    @staticmethod
    def _sample_noise(input_patch_dhw: torch.Tensor, p_value: float) -> torch.Tensor:
        """Samples noise from a binomial distribution based on the input patch and p-value.

        :param input_patch_dhw: The input patch tensor (D, H, W).
        :param p_value: The probability of success for the binomial distribution.
        :returns: The sampled noise tensor (D, H, W).
        """
        clamped_input = torch.clamp(input_patch_dhw, min=0).floor_()
        probs_tensor = torch.tensor(
            [p_value], device=clamped_input.device, dtype=clamped_input.dtype
        )
        binom_dist = Binomial(total_count=clamped_input, probs=probs_tensor)
        return binom_dist.sample()
