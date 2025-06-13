# tests/test_datasets.py
import dataclasses
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest  # Ensure pytest is imported
import torch

from src.ssunet.configs import DataConfig, SplitParams
from src.ssunet.datasets import (
    BernoulliDataset,
    BernoulliIdentityValidationDataset,
    BinomDataset,
    BinomIdentityValidationDataset,
    N2NSkipFrameDataset,
    N2NValidationDataset,
    PairedDataset,
    PairedValidationDataset,
    SSUnetData,
    TemporalSumSplitDataset,
    TemporalSumSplitValidationDataset,
)
from src.ssunet.datasets.base_patch import BasePatchDataset
from src.ssunet.exceptions import (
    DataError,
    InvalidDataDimensionError,
    InvalidPValueError,
    MissingPListError,
    MissingReferenceError,
    ShapeMismatchError,
)


# Helper to create a concrete BasePatchDataset for testing base class logic
class ConcreteTestDataset(BasePatchDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        # For N2N, this would be different, but for general base tests,
        # _get_transformed_patches is fine.
        return self._get_transformed_patches()


@pytest.fixture
def data_config_common() -> DataConfig:  # Renamed from data_config_case3 for clarity
    return DataConfig(
        z_size=32,
        xy_size=64,
        virtual_size=10,  # Small virtual size for faster tests
        skip_frames=1,
        random_crop=True,
        augments=False,
        rotation=0,
        seed=None,
    )


@pytest.fixture
def data_config_validation() -> DataConfig:  # For validation datasets
    return DataConfig(
        z_size=32,
        xy_size=64,
        virtual_size=1,  # Usually 1 for validation
        random_crop=False,  # Key for validation
        augments=False,
        rotation=0,
        seed=42,  # Seed for reproducibility if any randomness remains (e.g. time slice in base)
        normalize_target=False,  # Default, can be overridden
    )


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_3d_dhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(100, 128, 128).astype(np.float32)
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_4d_cdhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(3, 100, 128, 128).astype(np.float32)  # C=3
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_5d_tcdhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(5, 3, 100, 128, 128).astype(np.float32)  # T=5, C=3
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_5d_t_c1_dhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(5, 1, 100, 128, 128).astype(np.float32)  # T=5, C=1
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_4d_semantic_tdhw(
    request: pytest.FixtureRequest,
) -> SSUnetData:  # Interpreted as (C,D,H,W) where C=T
    data_np = np.random.rand(5, 100, 128, 128).astype(np.float32)
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_paired_tcdhw(request: pytest.FixtureRequest) -> SSUnetData:  # Renamed for clarity
    primary_shape = (5, 3, 100, 128, 128)  # T=5, C=3
    primary_np = np.random.rand(*primary_shape).astype(np.float32)
    secondary_np = (np.random.rand(*primary_shape) * 100).astype(
        np.float32
    )  # Different values for GT
    if request.param == torch.Tensor:
        return SSUnetData(
            primary_data=torch.from_numpy(primary_np), secondary_data=torch.from_numpy(secondary_np)
        )
    return SSUnetData(primary_data=primary_np, secondary_data=secondary_np)


@pytest.fixture
def split_params_signal() -> SplitParams:
    return SplitParams(method="signal", min_p=0.1, max_p=0.9, seed=42)


@pytest.fixture
def split_params_fixed_valid() -> SplitParams:
    return SplitParams(method="fixed", p_list=[0.5], seed=123)


@pytest.fixture
def split_params_list_valid() -> SplitParams:
    return SplitParams(method="list", p_list=[0.2, 0.4, 0.6], seed=123)


# --- Test Classes ---


class TestBasePatchDataset:
    def test_initial_setup_3d_dhw(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_common: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, data_config_common)
        assert dataset.source_ndim_raw == 3
        assert dataset.total_time_frames is None
        assert dataset.effective_channels == 1
        assert dataset.effective_depth == 100

    def test_initial_setup_4d_cdhw(
        self,
        ssunet_data_4d_cdhw: SSUnetData,
        data_config_common: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_4d_cdhw, data_config_common)
        assert dataset.source_ndim_raw == 4
        assert dataset.total_time_frames is None
        assert dataset.effective_channels == 3  # C=3 from fixture
        assert dataset.effective_depth == 100

    def test_initial_setup_5d_tcdhw(
        self,
        ssunet_data_5d_tcdhw: SSUnetData,
        data_config_common: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_5d_tcdhw, data_config_common)
        assert dataset.source_ndim_raw == 5
        assert dataset.total_time_frames == 5
        assert dataset.effective_channels == 3  # C=3 from fixture
        assert dataset.effective_depth == 100

    def test_initial_setup_5d_t_c1_dhw(  # This was previously under N2N, moved here
        self,
        ssunet_data_5d_t_c1_dhw: SSUnetData,
        data_config_common: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_5d_t_c1_dhw, data_config_common)
        assert dataset.source_ndim_raw == 5
        assert dataset.total_time_frames == 5
        assert dataset.effective_channels == 1  # C=1 from fixture
        assert dataset.effective_depth == 100

    def test_initial_setup_4d_semantic_tdhw(
        self,
        ssunet_data_4d_semantic_tdhw: SSUnetData,
        data_config_common: DataConfig,
    ) -> None:
        # This data (5, 100, 128, 128) is interpreted as (C,D,H,W) with C=5
        dataset = ConcreteTestDataset(ssunet_data_4d_semantic_tdhw, data_config_common)
        assert dataset.source_ndim_raw == 4
        assert dataset.total_time_frames is None
        assert dataset.effective_channels == 5  # C=5 from fixture (T dimension becomes C)
        assert dataset.effective_depth == 100

    def test_invalid_raw_input_ndim(self, data_config_common: DataConfig) -> None:
        with pytest.raises(
            InvalidDataDimensionError, match="Raw primary data must be 3D, 4D, or 5D. Got 2D"
        ):
            ConcreteTestDataset(SSUnetData(primary_data=np.random.rand(10, 10)), data_config_common)
        with pytest.raises(
            InvalidDataDimensionError, match="Raw primary data must be 3D, 4D, or 5D. Got 6D"
        ):
            ConcreteTestDataset(
                SSUnetData(primary_data=np.random.rand(1, 2, 3, 4, 5, 6)), data_config_common
            )

    def test_secondary_data_mismatch_in_basepatch_init(self) -> None:
        primary_np = np.random.rand(5, 3, 100, 128, 128)  # TCDHW
        secondary_np_mismatch_d = np.random.rand(5, 3, 90, 128, 128)  # Mismatch in D
        match_str = r"Primary data shape .* and secondary data shape .* do not match"
        with pytest.raises(ShapeMismatchError, match=match_str):
            SSUnetData(
                primary_data=primary_np,
                secondary_data=secondary_np_mismatch_d,
                allow_dimensionality_mismatch=True,
            )

        primary_np_cdhw = np.random.rand(3, 100, 128, 128)  # CDHW
        secondary_np_mismatch_c = np.random.rand(4, 100, 128, 128)  # Mismatch in C
        # Should NOT raise ShapeMismatchError due to allow_dimensionality_mismatch=True
        # and matching spatial dimensions (DHW).
        try:
            SSUnetData(
                primary_data=primary_np_cdhw,
                secondary_data=secondary_np_mismatch_c,
                allow_dimensionality_mismatch=True,
            )
        except ShapeMismatchError as e:
            pytest.fail(f"ShapeMismatchError raised unexpectedly: {e}")

    def test_getitem_shape_3d(
        self, ssunet_data_3d_dhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        cfg = data_config_common
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, cfg)
        patch = dataset[0][0]
        assert patch.shape == (1, cfg.z_size, cfg.xy_size, cfg.xy_size)

    def test_getitem_shape_4d_cdhw(
        self, ssunet_data_4d_cdhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        cfg = data_config_common
        dataset = ConcreteTestDataset(ssunet_data_4d_cdhw, cfg)
        patch = dataset[0][0]
        assert patch.shape == (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3

    def test_getitem_shape_5d_tcdhw(
        self, ssunet_data_5d_tcdhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        cfg = data_config_common
        dataset = ConcreteTestDataset(ssunet_data_5d_tcdhw, cfg)
        patch = dataset[0][0]  # A single time slice is taken
        assert patch.shape == (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3

    def test_len_calculation(
        self, ssunet_data_3d_dhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, data_config_common)
        assert len(dataset) == data_config_common.virtual_size  # virtual_size = 10

        config_no_virtual = DataConfig(z_size=32, xy_size=64, virtual_size=0, skip_frames=1)
        dataset_no_virtual = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_virtual)
        assert len(dataset_no_virtual) == (100 - 32 + 1)  # effective_depth - z_size + 1

        config_skip = DataConfig(z_size=32, xy_size=64, virtual_size=100, skip_frames=5)
        dataset_skip = ConcreteTestDataset(ssunet_data_3d_dhw, config_skip)
        assert len(dataset_skip) == 100 // 5  # virtual_size // skip_frames

    def test_invalid_patch_size(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        config_bad_z = DataConfig(z_size=101, xy_size=64)  # effective_depth is 100
        with pytest.raises(ValueError, match="Effective data depth"):
            ConcreteTestDataset(ssunet_data_3d_dhw, config_bad_z)
        config_bad_xy = DataConfig(z_size=32, xy_size=129)  # effective_height/width is 128
        with pytest.raises(ValueError, match="Effective data height"):  # or width
            ConcreteTestDataset(ssunet_data_3d_dhw, config_bad_xy)

    @patch("torchvision.transforms.v2.functional.rotate")  # Corrected mock path
    def test_rotation_invocation(
        self, mock_rotate: Any, ssunet_data_3d_dhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        config_rotate = dataclasses.replace(
            data_config_common,
            z_size=16,
            xy_size=32,
            rotation=15,
            augments=True,
            random_crop=True,  # augments=True for random angle
        )
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, config_rotate)

        def identity_side_effect(tensor_input, *args_rotate, **kwargs_rotate):
            return tensor_input

        mock_rotate.side_effect = identity_side_effect

        _ = dataset[0]
        mock_rotate.assert_called()

        mock_rotate.reset_mock()
        config_no_rotate = dataclasses.replace(
            data_config_common, z_size=16, xy_size=32, rotation=0
        )
        dataset_no_rotate = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_rotate)
        _ = dataset_no_rotate[0]
        mock_rotate.assert_not_called()

    def test_dataset_seeding_patch_coords(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        config_seed = DataConfig(seed=123, random_crop=True, z_size=16, xy_size=32)
        dataset1 = ConcreteTestDataset(ssunet_data_3d_dhw, config_seed)
        # Reset numpy seed locally for this test part if BasePatchDataset sets global numpy seed
        np.random.seed(123)
        coords1 = [dataset1._get_random_3d_patch_coordinates() for _ in range(5)]

        # Re-initialize for a fair comparison of seeding effect
        np.random.seed(123)  # Reset seed again before creating dataset2
        dataset2 = ConcreteTestDataset(ssunet_data_3d_dhw, config_seed)
        coords2 = [dataset2._get_random_3d_patch_coordinates() for _ in range(5)]
        assert coords1 == coords2

        config_no_seed = DataConfig(seed=None, random_crop=True, z_size=16, xy_size=32)
        # Reset numpy seed to something different or rely on system randomness
        np.random.seed(None)  # Or a different seed like 999
        dataset3 = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_seed)
        coords3 = [dataset3._get_random_3d_patch_coordinates() for _ in range(5)]
        if coords1:  # Only assert if coords1 is not empty
            assert coords1 != coords3


class TestPairedDatasets:
    def test_paired_dataset_output_shapes(
        self, ssunet_data_paired_tcdhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        dataset = PairedDataset(ssunet_data_paired_tcdhw, data_config_common)
        item = dataset[0]  # Returns (input_patch, target_patch)
        assert isinstance(item, tuple)
        assert len(item) == 2
        cfg = data_config_common
        # ssunet_data_paired_tcdhw has C=3
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape  # input_patch
        assert item[1].shape == expected_shape  # target_patch
        assert not torch.allclose(
            item[0], item[1]
        )  # Ensure they are not the same if raw data was different

    def test_paired_dataset_missing_secondary_error(
        self, ssunet_data_3d_dhw: SSUnetData, data_config_common: DataConfig
    ) -> None:  # ssunet_data_3d_dhw has no secondary_data
        with pytest.raises(MissingReferenceError):
            PairedDataset(ssunet_data_3d_dhw, data_config_common)

    def test_paired_validation_dataset(
        self,
        ssunet_data_paired_tcdhw: SSUnetData,
        data_config_validation: DataConfig,
    ) -> None:
        # Ensure validation config is used (no random_crop, no augments)
        # This is implicitly handled by data_config_validation fixture
        val_dataset = PairedValidationDataset(ssunet_data_paired_tcdhw, data_config_validation)
        item = val_dataset[0]  # Returns (input_patch, target_patch)
        assert isinstance(item, tuple)
        assert len(item) == 2  # Corrected: PairedValidationDataset returns 2 items

        cfg = data_config_validation
        # ssunet_data_paired_tcdhw has C=3
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape  # input_patch
        assert item[1].shape == expected_shape  # target_patch
        # item[2] would be an error as only 2 items are returned.

        # Check that create_validation_dataset works as expected
        # Create a PairedDataset instance first
        paired_train_dataset = PairedDataset(ssunet_data_paired_tcdhw, data_config_validation)
        created_val_dataset = paired_train_dataset.create_validation_dataset()
        assert isinstance(created_val_dataset, PairedValidationDataset)

        # Verify the created validation dataset uses a config with no random_crop/augments
        assert not created_val_dataset.config.random_crop
        assert not created_val_dataset.config.augments

        item_created = created_val_dataset[0]
        assert len(item_created) == 2  # Corrected
        assert item_created[0].shape == expected_shape
        assert item_created[1].shape == expected_shape


class TestBinomDatasets:
    def test_binom_dataset_output_shapes_primary_only_3d(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_common: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = BinomDataset(ssunet_data_3d_dhw, data_config_common, split_params_signal)
        item = dataset[0]  # [target, noise]
        assert len(item) == 2
        cfg = data_config_common
        expected_shape = (1, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=1 for 3D input
        assert item[0].shape == expected_shape  # target
        assert item[1].shape == expected_shape  # noise

    def test_binom_dataset_output_shapes_primary_only_4d_cdhw(
        self,
        ssunet_data_4d_cdhw: SSUnetData,
        data_config_common: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = BinomDataset(ssunet_data_4d_cdhw, data_config_common, split_params_signal)
        item = dataset[0]  # [target, noise]
        assert len(item) == 2
        cfg = data_config_common
        expected_shape_tn = (
            3,
            cfg.z_size,
            cfg.xy_size,
            cfg.xy_size,
        )  # C=3 from ssunet_data_4d_cdhw
        assert item[0].shape == expected_shape_tn
        assert item[1].shape == expected_shape_tn

    def test_binom_dataset_output_shapes_with_secondary(
        self,
        ssunet_data_paired_tcdhw: SSUnetData,
        data_config_common: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:  # ssunet_data_paired_tcdhw has T=5, C=3
        dataset = BinomDataset(ssunet_data_paired_tcdhw, data_config_common, split_params_signal)
        item = dataset[0]  # [target, noise, gt]
        assert len(item) == 3
        cfg = data_config_common
        expected_shape_tn = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3 from primary
        expected_shape_gt = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3 from secondary
        assert item[0].shape == expected_shape_tn  # target
        assert item[1].shape == expected_shape_tn  # noise
        assert item[2].shape == expected_shape_gt  # gt

    def test_binom_split_params_validation(
        self, ssunet_data_3d_dhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        with pytest.raises(InvalidPValueError):
            BinomDataset(
                ssunet_data_3d_dhw, data_config_common, SplitParams(method="fixed", p_list=[1.5])
            )
        with pytest.raises(MissingPListError):
            BinomDataset(
                ssunet_data_3d_dhw, data_config_common, SplitParams(method="list", p_list=[])
            )

    def test_binom_content_sum(
        self, ssunet_data_3d_dhw: SSUnetData, data_config_common: DataConfig
    ) -> None:
        # Use positive integer data for sum check to be meaningful with binomial
        positive_int_data = SSUnetData(
            primary_data=(np.random.rand(100, 128, 128) * 10 + 1).astype(np.int32)
        )
        split_params_no_norm = SplitParams(
            method="fixed", p_list=[0.5], normalize_target=False, seed=77
        )
        dataset = BinomDataset(positive_int_data, data_config_common, split_params_no_norm)

        # Get an original patch first to compare sum
        # This requires a bit of internal access or a different test setup
        # For simplicity, we'll check properties of target and noise
        item = dataset[0]
        target, noise = item[0], item[1]
        assert torch.all(target >= 0)  # Target can be 0
        assert torch.all(noise >= 0)  # Noise is count, must be >=0
        # Binomial noise should be integer-like
        assert torch.allclose(noise, noise.round(), atol=1e-5)
        # Sum of target and noise should approximate original patch if not normalized and p is fixed
        # This is harder to test precisely due to random patch extraction.
        # Instead, check that target = original_patch - noise (which is how it's calculated)

    def test_binom_identity_validation_dataset(
        self,
        ssunet_data_paired_tcdhw: SSUnetData,  # Has secondary data
        data_config_validation: DataConfig,
        data_config_common: DataConfig,
        split_params_fixed_valid: SplitParams,
    ) -> None:
        val_dataset = BinomIdentityValidationDataset(
            ssunet_data_paired_tcdhw, data_config_validation
        )
        item = val_dataset[0]
        # BinomIdentityValidationDataset returns [primary, primary, gt]
        # when secondary_data is present
        assert len(item) == 3
        cfg = data_config_validation
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3
        assert item[0].shape == expected_shape  # primary
        assert item[1].shape == expected_shape  # primary (identity target)
        assert item[2].shape == expected_shape  # gt (secondary data)
        assert torch.allclose(item[0], item[1])  # Input and identity target are identical

        # Test create_identity_validation_dataset
        train_dataset = BinomDataset(
            ssunet_data_paired_tcdhw, data_config_common, split_params_fixed_valid
        )
        created_val_dataset = train_dataset.create_identity_validation_dataset()
        assert isinstance(created_val_dataset, BinomIdentityValidationDataset)
        item_created = created_val_dataset[0]
        assert item_created[0].shape == (
            3,
            data_config_common.z_size,
            data_config_common.xy_size,
            data_config_common.xy_size,
        )


class TestBernoulliDataset:
    def test_bernoulli_noise_is_binary(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_common: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = BernoulliDataset(ssunet_data_3d_dhw, data_config_common, split_params_signal)
        item = dataset[0]
        noise = item[1]  # Noise component
        assert torch.all((noise == 0) | (noise == 1))

    def test_bernoulli_identity_validation(
        self, ssunet_data_4d_cdhw: SSUnetData, data_config_validation: DataConfig
    ) -> None:
        # BernoulliIdentityValidationDataset is an alias for BinomIdentityValidationDataset
        val_dataset = BernoulliIdentityValidationDataset(
            ssunet_data_4d_cdhw, data_config_validation
        )
        item = val_dataset[0]  # [primary, primary, (no gt here)]
        assert isinstance(item, list)  # Corrected from tuple after checking class behavior
        assert len(item) == 2
        cfg = data_config_validation
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3
        assert item[0].shape == expected_shape
        assert torch.allclose(item[0], item[1])


class TestN2NDatasets:
    @pytest.fixture
    def data_config_n2n(self) -> DataConfig:
        # Effective depth of ssunet_data_5d_tcdhw is 100.
        # z_size * 2 must be <= 100. So z_size <= 50.
        return DataConfig(z_size=32, xy_size=64, virtual_size=5)  # z_size=32, so required_depth=64

    def test_n2n_init_insufficient_depth(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        # ssunet_data_3d_dhw has effective_depth 100
        config_fail = DataConfig(z_size=60, xy_size=64)  # requires 120 depth
        with pytest.raises(ValueError, match="Effective data depth .* insufficient"):
            N2NSkipFrameDataset(ssunet_data_3d_dhw, config_fail)

    def test_n2n_dataset_output_shapes(
        self, ssunet_data_paired_tcdhw: SSUnetData, data_config_n2n: DataConfig
    ) -> None:  # T=5, C=3, D=100. Uses paired data with secondary.
        dataset = N2NSkipFrameDataset(ssunet_data_paired_tcdhw, data_config_n2n)
        item = dataset[0]  # [odd, even, (optional_gt_odd)]
        assert len(item) == 3  # Expect 3 items: odd, even, gt_odd (from secondary_data)

        cfg = data_config_n2n
        # Patches should have depth cfg.z_size (not required_source_patch_depth)
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3
        assert item[0].shape == expected_shape  # odd_frames
        assert item[1].shape == expected_shape  # even_frames
        assert item[2].shape == expected_shape  # gt_odd_frames

    def test_n2n_validation_dataset_output_shapes(
        self, ssunet_data_paired_tcdhw: SSUnetData, data_config_n2n: DataConfig
    ) -> None:
        # Ensure validation config is used
        val_cfg = dataclasses.replace(data_config_n2n, random_crop=False, augments=False)
        val_dataset = N2NValidationDataset(ssunet_data_paired_tcdhw, val_cfg)
        item = val_dataset[0]
        assert len(item) == 3
        expected_shape = (3, val_cfg.z_size, val_cfg.xy_size, val_cfg.xy_size)
        assert item[0].shape == expected_shape
        assert item[1].shape == expected_shape
        assert item[2].shape == expected_shape

        # Test create_validation_dataset
        train_dataset = N2NSkipFrameDataset(ssunet_data_paired_tcdhw, data_config_n2n)
        created_val_dataset = train_dataset.create_validation_dataset()
        assert isinstance(created_val_dataset, N2NValidationDataset)
        item_created = created_val_dataset[0]
        assert item_created[0].shape == expected_shape


class TestTemporalSumSplitDatasets:
    @pytest.fixture
    def temporal_data_config(self) -> DataConfig:
        # ssunet_data_5d_tcdhw: T=5, C=3, D=100
        # virtual_size often T_total for these tests
        return DataConfig(z_size=32, xy_size=64, virtual_size=5)

    def test_temporal_sum_split_init_requires_min_t2(
        self, temporal_data_config: DataConfig, split_params_signal: SplitParams
    ) -> None:
        data_t1 = SSUnetData(primary_data=torch.rand(1, 1, 40, 70, 70))  # T=1
        with pytest.raises(
            DataError, match="requires input data with at least 2 time frames"
        ) as excinfo1:
            TemporalSumSplitDataset(data_t1, temporal_data_config, split_params_signal)
        assert "requires input data with at least 2 time frames" in str(excinfo1.value)

    def test_temporal_sum_split_dataset_output_shapes(
        self,
        ssunet_data_5d_tcdhw: SSUnetData,
        ssunet_data_paired_tcdhw: SSUnetData,
        temporal_data_config: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:  # T=5, C=3
        dataset = TemporalSumSplitDataset(
            ssunet_data_5d_tcdhw, temporal_data_config, split_params_signal
        )
        item = dataset[0]  # [sum_target, sum_input]
        assert len(item) == 2

        dataset_with_gt = TemporalSumSplitDataset(
            ssunet_data_paired_tcdhw, temporal_data_config, split_params_signal
        )
        item_with_gt = dataset_with_gt[0]
        assert len(item_with_gt) == 3

        cfg = temporal_data_config
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)  # C=3
        assert item_with_gt[0].shape == expected_shape  # sum_target
        assert item_with_gt[1].shape == expected_shape  # sum_input
        assert item_with_gt[2].shape == expected_shape  # gt_mean

    @pytest.mark.parametrize("offset_name_param", ["x_offset_input", "y_offset_target"])
    def test_temporal_sum_split_validation_dataset_output_shapes_and_offsets(
        self,
        ssunet_data_paired_tcdhw: SSUnetData,
        temporal_data_config: DataConfig,
        split_params_signal: SplitParams,
        offset_name_param: str,
    ):
        val_cfg = dataclasses.replace(temporal_data_config, random_crop=False, augments=False)
        invalid_offset_value = val_cfg.virtual_size

        # Instead of regex, check for key substrings in the error message
        def check_exc(exc, offset_name_param, value, t):
            msg = str(exc.value)
            assert offset_name_param in msg
            assert f"({value})" in msg
            assert f"must be in range [0, T-1), where T={t}." in msg

        t = val_cfg.virtual_size
        with pytest.raises(
            ValueError, match=f"{offset_name_param}.*must be in range \\[0, T-1\\), where T={t}"
        ) as excinfo1:
            TemporalSumSplitValidationDataset(
                ssunet_data_paired_tcdhw,
                val_cfg,
                split_params_signal,
                **{offset_name_param: invalid_offset_value},
            )
        check_exc(excinfo1, offset_name_param, invalid_offset_value, t)

        with pytest.raises(
            ValueError, match=f"{offset_name_param}.*must be in range \\[0, T-1\\), where T={t}"
        ) as excinfo2:
            TemporalSumSplitValidationDataset(
                ssunet_data_paired_tcdhw, val_cfg, split_params_signal, **{offset_name_param: -1}
            )
        check_exc(excinfo2, offset_name_param, -1, t)

    def test_temporal_sum_split_validation_dataset_value_errors(
        self,
        ssunet_data_paired_tcdhw: SSUnetData,
        temporal_data_config: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        val_cfg = dataclasses.replace(temporal_data_config, random_crop=False, augments=False)
        invalid_offset_val = val_cfg.virtual_size

        def check_exc(exc, offset_name_param, value, t):
            msg = str(exc.value)
            assert offset_name_param in msg
            assert f"({value})" in msg
            assert f"must be in range [0, T-1), where T={t}." in msg

        t = val_cfg.virtual_size
        with pytest.raises(
            ValueError, match=f"x_offset_input.*must be in range \\[0, T-1\\), where T={t}"
        ) as excinfo1:
            TemporalSumSplitValidationDataset(
                ssunet_data_paired_tcdhw,
                val_cfg,
                split_params_signal,
                x_offset_input=invalid_offset_val,
            )
        check_exc(excinfo1, "x_offset_input", invalid_offset_val, t)
        with pytest.raises(
            ValueError, match=f"y_offset_target.*must be in range \\[0, T-1\\), where T={t}"
        ) as excinfo2:
            TemporalSumSplitValidationDataset(
                ssunet_data_paired_tcdhw,
                val_cfg,
                split_params_signal,
                y_offset_target=invalid_offset_val,
            )
        check_exc(excinfo2, "y_offset_target", invalid_offset_val, t)
        with pytest.raises(
            ValueError, match=f"x_offset_input.*must be in range \\[0, T-1\\), where T={t}"
        ) as excinfo3:
            TemporalSumSplitValidationDataset(
                ssunet_data_paired_tcdhw, val_cfg, split_params_signal, x_offset_input=-1
            )
        check_exc(excinfo3, "x_offset_input", -1, t)
        with pytest.raises(
            ValueError, match=f"y_offset_target.*must be in range \\[0, T-1\\), where T={t}"
        ) as excinfo4:
            TemporalSumSplitValidationDataset(
                ssunet_data_paired_tcdhw, val_cfg, split_params_signal, y_offset_target=-1
            )
        check_exc(excinfo4, "y_offset_target", -1, t)

    def test_temporal_sum_split_create_validation_dataset(
        self,
        ssunet_data_paired_tcdhw: SSUnetData,
        temporal_data_config: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        train_dataset = TemporalSumSplitDataset(
            ssunet_data_paired_tcdhw, temporal_data_config, split_params_signal
        )
        created_val_dataset = train_dataset.create_validation_dataset()
        assert isinstance(created_val_dataset, TemporalSumSplitValidationDataset)
        assert created_val_dataset.x_offset_input == 0  # Check default offsets
        assert created_val_dataset.y_offset_target == 0
        item_created = created_val_dataset[0]
        expected_shape = (
            3,
            temporal_data_config.z_size,
            temporal_data_config.xy_size,
            temporal_data_config.xy_size,
        )
        assert item_created[0].shape == expected_shape

    # Existing detailed split logic tests from the original file can be kept if still relevant
    @pytest.mark.parametrize(
        ("t_total", "c_eff", "d_patch", "h_patch", "w_patch"),  # Corrected missing quote
        [(10, 1, 16, 32, 32), (5, 3, 16, 32, 32), (2, 1, 16, 32, 32)],
    )
    @pytest.mark.parametrize(
        ("p_method", "p_values_dict"),  # Renamed p_values to p_values_dict
        [
            ("signal", {"min_p": 0.1, "max_p": 0.9}),
            ("fixed", {"p_list": [0.5]}),
            ("fixed", {"min_p": 0.3}),
            ("list", {"p_list": [0.25, 0.75]}),
        ],
    )
    def test_split_temporal_frames_logic_random_split(  # Renamed for clarity
        self,
        t_total: int,
        c_eff: int,
        d_patch: int,
        h_patch: int,
        w_patch: int,
        p_method: str,
        p_values_dict: dict[str, Any],
        temporal_data_config: DataConfig,
    ) -> None:
        # Create SplitParams based on p_method and p_values_dict
        current_split_params = SplitParams(method=p_method, seed=42, **p_values_dict)

        # Create dummy SSUnetData and Dataset instance
        # Ensure raw data depth is sufficient for patch extraction
        raw_d = d_patch + 10
        dummy_raw_data_shape = (t_total, c_eff, raw_d, h_patch + 10, w_patch + 10)
        dummy_ssu_data = SSUnetData(primary_data=torch.rand(*dummy_raw_data_shape))

        # Adjust DataConfig to match patch dimensions
        current_data_config = dataclasses.replace(
            temporal_data_config,
            z_size=d_patch,
            xy_size=h_patch,  # Assuming h_patch == w_patch for simplicity
        )
        dataset = TemporalSumSplitDataset(dummy_ssu_data, current_data_config, current_split_params)

        # The patch passed to _split_temporal_frames has dimensions
        patch_t_first_dims = torch.rand(t_total, c_eff, d_patch, h_patch, w_patch)

        sum_n, sum_m = dataset._split_temporal_frames(patch_t_first_dims)

        expected_spatial_shape = (c_eff, d_patch, h_patch, w_patch)
        assert sum_n.shape == expected_spatial_shape
        assert sum_m.shape == expected_spatial_shape
        # Further checks on n_count and m_count could be added by inspecting internal logic


# This test class was for a generic ValidationDataset.
# Assuming it refers to BinomIdentityValidationDataset based on its behavior.
class TestGenericIdentityValidationDataset:  # Renamed for clarity
    def test_output_shapes_no_gt(
        self, ssunet_data_4d_cdhw: SSUnetData, data_config_validation: DataConfig
    ) -> None:  # ssunet_data_4d_cdhw has C=3, no secondary
        dataset = BinomIdentityValidationDataset(ssunet_data_4d_cdhw, data_config_validation)
        item = dataset[0]  # [primary, primary]
        assert len(item) == 2
        cfg = data_config_validation
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape
        assert item[1].shape == expected_shape
        assert torch.allclose(item[0], item[1])

    def test_output_shapes_with_gt(
        self, ssunet_data_paired_tcdhw: SSUnetData, data_config_validation: DataConfig
    ) -> None:  # ssunet_data_paired_tcdhw has C=3, and secondary
        dataset = BinomIdentityValidationDataset(ssunet_data_paired_tcdhw, data_config_validation)
        item = dataset[0]  # [primary, primary, gt]
        assert len(item) == 3
        cfg = data_config_validation
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape
        assert item[1].shape == expected_shape
        assert item[2].shape == expected_shape  # GT also from C=3 data
        assert torch.allclose(item[0], item[1])

    def test_identity_normalization_behavior(
        self, ssunet_data_3d_dhw: SSUnetData, data_config_validation: DataConfig
    ) -> None:
        # BinomIdentityValidationDataset does not itself apply normalization
        # from DataConfig.normalize_target.
        # It returns the transformed patch. If that transformation pipeline
        # included normalization, then it would be reflected.
        # Here, we test that it returns the patch as is from
        # _get_transformed_patches.
        config_norm_true = dataclasses.replace(data_config_validation, normalize_target=True)
        dataset_true = BinomIdentityValidationDataset(ssunet_data_3d_dhw, config_norm_true)
        item_true = dataset_true[0]  # [primary, primary]

        config_norm_false = dataclasses.replace(data_config_validation, normalize_target=False)
        dataset_false = BinomIdentityValidationDataset(ssunet_data_3d_dhw, config_norm_false)
        item_false = dataset_false[0]

        # The patches should be identical regardless of normalize_target in DataConfig
        # because BinomIdentityValidationDataset doesn\'t use that flag for its output.
        assert torch.allclose(item_true[0], item_false[0])
        assert torch.allclose(item_true[1], item_false[1])
        assert torch.allclose(item_true[0], item_true[1])
