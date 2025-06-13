# src/ssunet/datasets/binomial.py
from collections.abc import Callable

import numpy as np
import torch
from numpy.random import choice, rand, seed

from ..configs import DataConfig, SplitConfig
from ..constants import LOGGER
from ..exceptions import ConfigError, InvalidPValueError, MissingPListError
from ..utils import _normalize_by_mean, to_tensor
from .base_patch import BasePatchDataset
from .ssunet_data import SSUnetData


class BinomIdentityValidationDataset(BasePatchDataset):
    """
    A dataset for validation that returns the original (transformed) patch
    as both the model input and the target for loss calculation.
    It shares the raw data and patch configuration with a parent BinomDataset.
    For 5D data, it always uses the first time slice (t=0).
    It always takes a center spatial crop.
    """

    def __init__(
        self,
        input_data: SSUnetData,
        config: DataConfig,
        **kwargs,
    ) -> None:
        super().__init__(input_data, config, **kwargs)
        LOGGER.info("BinomIdentityValidationDataset initialized.")

    def _get_volume_for_patching(self) -> SSUnetData:
        """
        Overrides BasePatchDataset._get_volume_for_patching.
        If raw data is 5D, always selects the first time frame (t_idx=0).
        Otherwise, behaves like the parent.
        """
        primary_vol_raw = self.input_data_raw.primary_data
        secondary_vol_raw = self.input_data_raw.secondary_data

        primary_tensor_full = (
            to_tensor(primary_vol_raw)
            if not isinstance(primary_vol_raw, torch.Tensor)
            else primary_vol_raw
        )
        secondary_tensor_full: torch.Tensor | None = None
        if secondary_vol_raw is not None:
            secondary_tensor_full = (
                to_tensor(secondary_vol_raw)
                if not isinstance(secondary_vol_raw, torch.Tensor)
                else secondary_vol_raw
            )

        primary_volume_final: torch.Tensor
        secondary_volume_final: torch.Tensor | None = None

        if self.source_ndim_raw == 5:
            if self.total_time_frames is None:  # Should be set by _initial_data_setup
                raise RuntimeError("total_time_frames not set for 5D data in validation.")
            t_idx = 0  # Always use the first time frame for validation
            LOGGER.debug(f"Validation dataset: Using t_idx={t_idx} for 5D data.")
            primary_volume_final = primary_tensor_full[t_idx]
            if secondary_tensor_full is not None:
                secondary_volume_final = secondary_tensor_full[t_idx]
        else:
            primary_volume_final = primary_tensor_full
            secondary_volume_final = secondary_tensor_full

        return SSUnetData(primary_data=primary_volume_final, secondary_data=secondary_volume_final)

    def _get_random_3d_patch_coordinates(self) -> tuple[int, int, int]:
        """
        Overrides BasePatchDataset._get_random_3d_patch_coordinates.
        Always returns coordinates for a center crop.
        """
        d_start = (self.effective_depth - self.config.z_size) // 2
        h_start = (self.effective_height - self.config.xy_size) // 2
        w_start = (self.effective_width - self.config.xy_size) // 2
        LOGGER.debug(
            "Validation dataset: Using center crop coordinates: "
            f"d={d_start}, h={h_start}, w={w_start}"
        )
        return d_start, h_start, w_start

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """
        Get a single patch from the dataset.

        :param index: Index of the patch to retrieve.
        :returns: A list containing [primary_patch, primary_patch, (optional)ground_truth_patch].
                  The first primary_patch is for the loss target, the second is for model input.
                  Patches are from the first time slice (if 5D) and center-cropped.
        """
        transformed_patches = (
            self._get_transformed_patches()
        )  # This will use the overridden methods
        primary_patch_cdhw = transformed_patches[0]  # Shape (C, D, H, W)

        # Order for Bit2Bit: [loss_target, model_input, optional_gt_for_metrics]
        output_list = [primary_patch_cdhw, primary_patch_cdhw]

        if len(transformed_patches) > 1:
            output_list.append(transformed_patches[1])  # Append GT patch (C_gt, D, H, W)

        return output_list


class BinomDataset(BasePatchDataset):
    """Dataset that splits a signal into a target and noise component using binomial splitting."""

    def __init__(
        self,
        data_source: SSUnetData,
        config: DataConfig,
        split_params: SplitConfig,  # Make sure SplitConfig type hint is correct
        **kwargs,
    ) -> None:
        super().__init__(data_source, config, **kwargs)
        self.split_params = split_params
        self._validate_and_seed_split_params()

    def _validate_and_seed_split_params(self):
        if self.split_params.seed is not None:
            seed(self.split_params.seed)
            LOGGER.info(f"Random seed set to {self.split_params.seed} for BinomDataset p-sampling.")

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
        :returns: A list containing [target_patch, noise_patch, (optional)ground_truth_patch].
                 Target and noise will have the same (C,D,H,W) shape as the primary input patch.
        """
        transformed_patches = self._get_transformed_patches()
        primary_patch_cdhw = transformed_patches[0]  # Shape (C, D, H, W)

        # _split now expects and processes a (C, D, H, W) tensor.
        target_cdhw, noise_cdhw = self._split(primary_patch_cdhw)
        output_list = [target_cdhw, noise_cdhw]

        if len(transformed_patches) > 1:
            output_list.append(transformed_patches[1])  # Append GT patch (C_gt, D, H, W)

        return output_list

    def _split(self, input_patch_cdhw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the input patch into target and noise components using binomial distribution.
           Operates on all channels independently if C > 1.

        :param input_patch_cdhw: The input patch tensor (C, D, H, W).
        :returns: A tuple containing the target patch (C,D,H,W) and the noise patch (C,D,H,W).
        """
        if input_patch_cdhw.ndim != 4:  # Expect C,D,H,W
            raise ValueError(
                f"BinomDataset._split expects a 4D tensor (C,D,H,W), got {input_patch_cdhw.shape}"
            )

        p_value = self._sample_p(input_patch_cdhw)  # p_value is sampled once for the patch

        # _sample_noise is now expected to handle (C,D,H,W)
        noise_cdhw = self._sample_noise(input_patch_cdhw, p_value)
        target_cdhw = (input_patch_cdhw - noise_cdhw).float()

        if self.split_params.normalize_target:
            # If _normalize_by_mean calculates a global mean, it will apply to the C*D*H*W tensor.
            # For per-channel normalization:
            if target_cdhw.shape[0] > 1:  # C > 1
                normalized_channels = []
                for c_idx in range(target_cdhw.shape[0]):
                    channel_data = target_cdhw[c_idx]  # D, H, W
                    normalized_channels.append(_normalize_by_mean(channel_data))
                target_cdhw = torch.stack(normalized_channels, dim=0)
            else:
                target_cdhw = _normalize_by_mean(target_cdhw)

        return target_cdhw, noise_cdhw.float()

    def _sample_p(self, input_patch_data: torch.Tensor) -> float:  # input_patch_data can be CDHW
        """Samples a p-value based on the configured method."""
        p_sampling_method_custom: Callable[..., float] | None = self.kwargs.get(
            "p_sampling_method", None
        )
        if p_sampling_method_custom is not None and callable(p_sampling_method_custom):
            return self._validate_p(p_sampling_method_custom(input_patch_data, **self.kwargs))

        method = self.split_params.method
        if method == "signal":
            return self._sample_signal()
        if method == "fixed":
            return self._sample_fixed()
        if method == "list":
            return self._sample_list()
        raise ConfigError(f"Unsupported p-sampling method '{method}' in BinomDataset.")

    @staticmethod
    def _validate_p(p_value: float) -> float:
        if not (0 < p_value < 1):
            raise InvalidPValueError(f"p-value must be between 0 and 1 (exclusive), got {p_value}")
        return p_value

    def _sample_signal(self) -> float:
        p_value = (
            rand() * (self.split_params.max_p - self.split_params.min_p) + self.split_params.min_p
        )
        return self._validate_p(p_value)

    def _sample_fixed(self) -> float:
        p_value = (
            self.split_params.p_list[0] if self.split_params.p_list else self.split_params.min_p
        )
        return self._validate_p(p_value)

    def _sample_list(self) -> float:
        if not self.split_params.p_list:
            raise MissingPListError()
        p_value = choice(self.split_params.p_list)
        return self._validate_p(p_value)

    @staticmethod
    def _sample_noise(input_patch_cdhw: torch.Tensor, p_value: float) -> torch.Tensor:
        """Sample binomial noise based on the input patch and p_value."""
        # Ensure input_patch_cdhw (total_count) is an integer tensor for Binomial distribution
        counts_int = input_patch_cdhw.long()  # Cast to integer type
        # Probs should be float, and match the shape of counts_int but not necessarily its dtype
        probs_tensor = torch.full_like(counts_int, p_value, dtype=torch.float)
        binom_dist = torch.distributions.Binomial(total_count=counts_int, probs=probs_tensor)
        return binom_dist.sample()

    def create_identity_validation_dataset(self) -> BinomIdentityValidationDataset:
        """
        Creates a validation dataset that uses the same raw input data and patch configuration
        but returns the original patch as both input and target (identity task).

        :returns: An instance of BinomIdentityValidationDataset.
        """
        LOGGER.info(
            "Creating BinomIdentityValidationDataset, with a new SSUnetData object "
            "SHARING TENSOR DATA from parent BinomDataset."
        )
        # Manually create a new SSUnetData that uses the original tensors
        shared_tensor_input_data = SSUnetData(
            primary_data=self.input_data_raw._internal_primary_data,  # Access internal tensor
            secondary_data=self.input_data_raw._internal_secondary_data  # Access internal tensor
            if self.input_data_raw._internal_secondary_data is not None
            else None,
            allow_dimensionality_mismatch=self.input_data_raw.allow_dimensionality_mismatch,
        )
        return BinomIdentityValidationDataset(
            input_data=shared_tensor_input_data,
            config=self.config,
            **self.kwargs,
        )

    def _split_signal_and_noise(
        self, patch: np.ndarray, p_value: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split the signal into target and noise components using binomial splitting."""
        if not (0 <= p_value <= 1):
            raise ValueError(f"p_value must be between 0 and 1, got {p_value}")

        # Ensure patch is float for calculations if it's not already
        patch_float = patch.astype(np.float32) if patch.dtype != np.float32 else patch

        if np.all(patch_float >= 0) and np.all(patch_float == np.round(patch_float)):
            target_component = np.random.binomial(patch_float.astype(np.int32), p_value).astype(
                np.float32
            )
        else:
            target_component = patch_float * p_value

        noise_component = patch_float - target_component
        target_component = self._normalize_target_component(target_component)

        return target_component, noise_component

    def _normalize_target_component(self, target_patch: np.ndarray) -> np.ndarray:
        """Normalize the target component if configured."""
        if self.config.normalize_target:  # This is DATA.normalize_target
            # Assuming min-max normalization to [0, 1]
            min_val = np.min(target_patch)
            max_val = np.max(target_patch)
            if max_val - min_val > self.epsilon:  # self.epsilon should be defined (e.g., 1e-8)
                normalized_patch = (target_patch - min_val) / (max_val - min_val)
                return normalized_patch.astype(np.float32)
            # If range is too small, return as is or a constant (e.g., all zeros or 0.5)
            # Returning as is for now, but ensure it's float32
            return target_patch.astype(np.float32)
        return target_patch  # Already float32 if processed by splitting logic
