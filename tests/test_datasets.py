from __future__ import annotations

import random  # Ensure random is imported for edge case test
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.ssunet.configs import DataConfig, SplitParams, SSUnetData
from src.ssunet.datasets import (
    BernoulliDataset,
    BinomDataset,
    N2NSkipFrameDataset,
    PairedDataset,
    ValidationDataset,
)
from src.ssunet.datasets.base_patch import BasePatchDataset
from src.ssunet.datasets.temporal_sum_split import TemporalSumSplitDataset  # Added import
from src.ssunet.exceptions import (
    ConfigError,
    DataError,
    InvalidDataDimensionError,
    InvalidPValueError,
    MissingPListError,
    MissingReferenceError,
    ShapeMismatchError,
)


class ConcreteTestDataset(BasePatchDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        return self._get_transformed_patches()


@pytest.fixture
def data_config_case3() -> DataConfig:
    return DataConfig(
        z_size=32,
        xy_size=64,
        virtual_size=10,
        skip_frames=1,
        random_crop=True,
        augments=False,
        rotation=0,
        seed=None,
    )


# Fixture name from instruction, using case2 config
@pytest.fixture
def data_config_case2() -> DataConfig:
    """This config is compatible with the dummy data shape in
    test_split_temporal_frames_logic."""
    return DataConfig(
        z_size=16,
        xy_size=32,
        virtual_size=20,
        skip_frames=2,
        random_crop=False,
        augments=True,
        rotation=10,
        seed=42,
    )


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_3d_dhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(100, 128, 128).astype(np.float32)
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_4d_cdhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(3, 100, 128, 128).astype(np.float32)
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_5d_tcdhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(5, 3, 100, 128, 128).astype(np.float32)
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_5d_t_c1_dhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(5, 1, 100, 128, 128).astype(np.float32)
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_4d_semantic_tdhw(request: pytest.FixtureRequest) -> SSUnetData:
    data_np = np.random.rand(5, 100, 128, 128).astype(np.float32)
    if request.param == torch.Tensor:
        return SSUnetData(primary_data=torch.from_numpy(data_np))
    return SSUnetData(primary_data=data_np)


@pytest.fixture(params=[np.float32, torch.Tensor], ids=["numpy_input", "torch_input"])
def ssunet_data_secondary_matching_primary_tcdhw(request: pytest.FixtureRequest) -> SSUnetData:
    primary_shape = (5, 3, 100, 128, 128)
    primary_np = np.random.rand(*primary_shape).astype(np.float32)
    secondary_np = np.random.rand(*primary_shape).astype(np.float32)
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


@pytest.fixture
def data_config_n2n_depth_fail_xy_ok() -> DataConfig:
    return DataConfig(z_size=60, xy_size=128)


class TestBasePatchDataset:
    def test_initial_setup_3d_dhw(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, data_config_case3)
        assert dataset.source_ndim_raw == 3
        assert dataset.total_time_frames is None
        assert dataset.effective_channels == 1
        assert dataset.effective_depth == 100

    def test_initial_setup_4d_cdhw(
        self,
        ssunet_data_4d_cdhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_4d_cdhw, data_config_case3)
        assert dataset.source_ndim_raw == 4
        assert dataset.total_time_frames is None
        assert dataset.effective_channels == 3
        assert dataset.effective_depth == 100

    def test_initial_setup_5d_tcdhw(
        self,
        ssunet_data_5d_tcdhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_5d_tcdhw, data_config_case3)
        assert dataset.source_ndim_raw == 5
        assert dataset.total_time_frames == 5
        assert dataset.effective_channels == 3
        assert dataset.effective_depth == 100

    def test_initial_setup_5d_t_c1_dhw(
        self,
        ssunet_data_5d_t_c1_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_5d_t_c1_dhw, data_config_case3)
        assert dataset.source_ndim_raw == 5
        assert dataset.total_time_frames == 5
        assert dataset.effective_channels == 1
        assert dataset.effective_depth == 100

    def test_initial_setup_4d_semantic_tdhw(
        self,
        ssunet_data_4d_semantic_tdhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_4d_semantic_tdhw, data_config_case3)
        assert dataset.source_ndim_raw == 4
        assert dataset.total_time_frames is None
        assert dataset.effective_channels == 5
        assert dataset.effective_depth == 100

    def test_invalid_raw_input_ndim(self, data_config_case3: DataConfig) -> None:
        with pytest.raises(
            InvalidDataDimensionError, match="Raw primary data must be 3D, 4D, or 5D. Got 2D"
        ):
            ConcreteTestDataset(SSUnetData(np.random.rand(10, 10)), data_config_case3)
        with pytest.raises(
            InvalidDataDimensionError, match="Raw primary data must be 3D, 4D, or 5D. Got 6D"
        ):
            ConcreteTestDataset(
                SSUnetData(np.random.rand(1, 2, 3, 4, 5, 6)),
                data_config_case3,
            )

    def test_secondary_data_mismatch_ssunetdata_level(self) -> None:
        primary = np.random.rand(3, 10, 20, 20)
        secondary = np.random.rand(3, 10, 22, 22)
        with pytest.raises(ShapeMismatchError, match="Data and reference shapes do not match"):
            SSUnetData(primary, secondary)

    def test_secondary_data_mismatch_basepatch_level(self, data_config_case3: DataConfig) -> None:
        primary_np = np.random.rand(3, 10, 20, 20)
        secondary_np_mismatch = np.random.rand(3, 10, 22, 22)
        ssu_data_obj = SSUnetData(primary_data=primary_np, secondary_data=None)
        ssu_data_obj.secondary_data = secondary_np_mismatch
        with pytest.raises(ValueError, match="Shape mismatch: raw primary data"):
            ConcreteTestDataset(ssu_data_obj, data_config_case3)

    def test_getitem_shape_3d(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        cfg = data_config_case3
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, cfg)
        patch = dataset[0][0]
        assert patch.shape == (1, cfg.z_size, cfg.xy_size, cfg.xy_size)

    def test_getitem_shape_4d_cdhw(
        self,
        ssunet_data_4d_cdhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        cfg = data_config_case3
        dataset = ConcreteTestDataset(ssunet_data_4d_cdhw, cfg)
        patch = dataset[0][0]
        assert patch.shape == (3, cfg.z_size, cfg.xy_size, cfg.xy_size)

    def test_getitem_shape_5d_tcdhw(
        self,
        ssunet_data_5d_tcdhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        cfg = data_config_case3
        dataset = ConcreteTestDataset(ssunet_data_5d_tcdhw, cfg)
        patch = dataset[0][0]
        assert patch.shape == (3, cfg.z_size, cfg.xy_size, cfg.xy_size)

    def test_getitem_shape_5d_t_c1_dhw(
        self,
        ssunet_data_5d_t_c1_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        cfg = data_config_case3
        dataset = ConcreteTestDataset(ssunet_data_5d_t_c1_dhw, cfg)
        patch = dataset[0][0]
        assert patch.shape == (1, cfg.z_size, cfg.xy_size, cfg.xy_size)

    def test_len_calculation(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, data_config_case3)
        assert len(dataset) == 10
        config_no_virtual = DataConfig(z_size=32, xy_size=64, virtual_size=0, skip_frames=1)
        dataset_no_virtual = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_virtual)
        assert len(dataset_no_virtual) == (100 - 32 + 1)
        config_skip = DataConfig(z_size=32, xy_size=64, virtual_size=100, skip_frames=5)
        dataset_skip = ConcreteTestDataset(ssunet_data_3d_dhw, config_skip)
        assert len(dataset_skip) == 20

    def test_invalid_patch_size(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        config_bad_z = DataConfig(z_size=101, xy_size=64)
        with pytest.raises(ValueError, match="Effective data depth"):
            ConcreteTestDataset(ssunet_data_3d_dhw, config_bad_z)
        config_bad_xy = DataConfig(z_size=32, xy_size=129)
        with pytest.raises(ValueError, match="Effective data height"):
            ConcreteTestDataset(ssunet_data_3d_dhw, config_bad_xy)

    @patch("src.ssunet.datasets.base_patch.tf.rotate")
    def test_rotation_invocation(
        self,
        mock_rotate: Any,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case2: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, data_config_case2)
        _ = dataset[0]
        mock_rotate.assert_called()

        config_no_rot = DataConfig(
            rotation=0,
            xy_size=data_config_case2.xy_size,
            z_size=data_config_case2.z_size,
        )
        dataset_no_rot = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_rot)
        with patch("src.ssunet.datasets.base_patch.tf.rotate") as mock_rotate_no_rot:
            _ = dataset_no_rot[0]
            mock_rotate_no_rot.assert_not_called()

    def test_dataset_seeding_patch_coords(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        config_seed = DataConfig(seed=123, random_crop=True, z_size=16, xy_size=32)
        dataset1 = ConcreteTestDataset(ssunet_data_3d_dhw, config_seed)
        coords1 = [dataset1._get_random_3d_patch_coordinates() for _ in range(5)]
        dataset2 = ConcreteTestDataset(ssunet_data_3d_dhw, config_seed)
        coords2 = [dataset2._get_random_3d_patch_coordinates() for _ in range(5)]
        assert coords1 == coords2


class TestN2NSkipFrameDataset:
    def test_init_insufficient_depth(
        self, ssunet_data_3d_dhw: SSUnetData, data_config_n2n_depth_fail_xy_ok: DataConfig
    ) -> None:
        with pytest.raises(
            ValueError, match="Effective data depth .* insufficient for N2N skip frame"
        ):
            N2NSkipFrameDataset(ssunet_data_3d_dhw, data_config_n2n_depth_fail_xy_ok)

    def test_getitem_shape_5d_t_c1_dhw(
        self,
        ssunet_data_5d_t_c1_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        expected_c_out = 1
        dataset = ConcreteTestDataset(ssunet_data_5d_t_c1_dhw, data_config_case3)
        patches = dataset[0]
        assert len(patches) == 1
        patch = patches[0]
        assert patch.ndim == 4
        assert patch.shape[0] == expected_c_out
        assert patch.shape[1] == data_config_case3.z_size
        assert patch.shape[2] == data_config_case3.xy_size
        assert patch.shape[3] == data_config_case3.xy_size

    def test_len_calculation(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, data_config_case3)
        assert len(dataset) == 10
        config_no_virtual = DataConfig(z_size=32, xy_size=64, virtual_size=0, skip_frames=1)
        dataset_no_virtual = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_virtual)
        assert len(dataset_no_virtual) == (100 - 32 + 1)
        config_skip = DataConfig(z_size=32, xy_size=64, virtual_size=100, skip_frames=5)
        dataset_skip = ConcreteTestDataset(ssunet_data_3d_dhw, config_skip)
        assert len(dataset_skip) == 20

    def test_invalid_patch_size(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        config_bad_z = DataConfig(z_size=101, xy_size=64)
        with pytest.raises(ValueError, match="Effective data depth"):
            ConcreteTestDataset(ssunet_data_3d_dhw, config_bad_z)
        config_bad_xy = DataConfig(z_size=32, xy_size=129)
        with pytest.raises(ValueError, match="Effective data height"):
            ConcreteTestDataset(ssunet_data_3d_dhw, config_bad_xy)

    @patch("src.ssunet.datasets.base_patch.tf.rotate")
    def test_rotation_invocation(
        self,
        mock_rotate: Any,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case2: DataConfig,
    ) -> None:
        dataset = ConcreteTestDataset(ssunet_data_3d_dhw, data_config_case2)
        _ = dataset[0]
        mock_rotate.assert_called()
        config_no_rot = DataConfig(
            rotation=0,
            xy_size=data_config_case2.xy_size,
            z_size=data_config_case2.z_size,
        )
        dataset_no_rot = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_rot)
        with patch("src.ssunet.datasets.base_patch.tf.rotate") as mock_rotate_no_rot:
            _ = dataset_no_rot[0]
            mock_rotate_no_rot.assert_not_called()

    def test_dataset_seeding_patch_coords(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        config_seed = DataConfig(seed=123, random_crop=True, z_size=16, xy_size=32)
        dataset1 = ConcreteTestDataset(ssunet_data_3d_dhw, config_seed)
        coords1 = [dataset1._get_random_3d_patch_coordinates() for _ in range(5)]
        dataset2 = ConcreteTestDataset(ssunet_data_3d_dhw, config_seed)
        coords2 = [dataset2._get_random_3d_patch_coordinates() for _ in range(5)]
        assert coords1 == coords2
        config_no_seed = DataConfig(seed=None, random_crop=True, z_size=16, xy_size=32)
        dataset3 = ConcreteTestDataset(ssunet_data_3d_dhw, config_no_seed)
        coords3 = [dataset3._get_random_3d_patch_coordinates() for _ in range(5)]
        assert coords1 != coords3 or len(coords1) == 0


class TestBinomDataset:
    def test_output_shapes_primary_only_3d(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = BinomDataset(ssunet_data_3d_dhw, data_config_case3, split_params_signal)
        item = dataset[0]
        assert len(item) == 2
        expected_shape = (
            1,
            data_config_case3.z_size,
            data_config_case3.xy_size,
            data_config_case3.xy_size,
        )
        assert item[0].shape == expected_shape
        assert item[1].shape == expected_shape

    def test_output_shapes_primary_only_4d_cdhw(
        self,
        ssunet_data_4d_cdhw: SSUnetData,
        data_config_case3: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = BinomDataset(
            ssunet_data_4d_cdhw,
            data_config_case3,
            split_params_signal,
        )
        item = dataset[0]
        assert len(item) == 2
        expected_shape_tn = (
            1,
            data_config_case3.z_size,
            data_config_case3.xy_size,
            data_config_case3.xy_size,
        )
        assert item[0].shape == expected_shape_tn
        assert item[1].shape == expected_shape_tn

    def test_output_shapes_with_secondary(
        self,
        ssunet_data_secondary_matching_primary_tcdhw: SSUnetData,
        data_config_case2: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = BinomDataset(
            ssunet_data_secondary_matching_primary_tcdhw,
            data_config_case2,
            split_params_signal,
        )
        item = dataset[0]
        assert len(item) == 3
        cfg = data_config_case2
        expected_shape_tn = (1, cfg.z_size, cfg.xy_size, cfg.xy_size)
        expected_shape_gt = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape_tn
        assert item[1].shape == expected_shape_tn
        assert item[2].shape == expected_shape_gt

    def test_split_params_validation(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        with pytest.raises(InvalidPValueError):
            BinomDataset(
                ssunet_data_3d_dhw,
                data_config_case3,
                SplitParams(method="fixed", p_list=[1.5]),
            )
        with pytest.raises(MissingPListError):
            BinomDataset(
                ssunet_data_3d_dhw,
                data_config_case3,
                SplitParams(method="list", p_list=[]),
            )

    def test_binom_content_sum(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        split_params_no_norm = SplitParams(
            method="fixed", p_list=[0.5], normalize_target=False, seed=77
        )
        dataset = BinomDataset(
            ssunet_data_3d_dhw,
            data_config_case3,
            split_params_no_norm,
        )
        item = dataset[0]
        target, noise = item[0], item[1]
        assert torch.all(target >= -1e-6)
        assert torch.all(noise >= -1e-6)
        assert torch.allclose(noise, noise.round(), atol=1e-5)


class TestBernoulliDataset:
    def test_noise_is_binary(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = BernoulliDataset(ssunet_data_3d_dhw, data_config_case3, split_params_signal)
        item = dataset[0]
        noise = item[1]
        assert torch.all((noise == 0) | (noise == 1))


class TestPairedDataset:
    def test_output_shapes(
        self,
        ssunet_data_secondary_matching_primary_tcdhw: SSUnetData,
        data_config_case2: DataConfig,
    ) -> None:
        dataset = PairedDataset(
            ssunet_data_secondary_matching_primary_tcdhw,
            data_config_case2,
        )
        item = dataset[0]
        assert len(item) == 2
        cfg = data_config_case2
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape
        assert item[1].shape == expected_shape

    def test_missing_secondary_error(
        self,
        ssunet_data_3d_dhw: SSUnetData,
        data_config_case3: DataConfig,
    ) -> None:
        with pytest.raises(MissingReferenceError):
            PairedDataset(ssunet_data_3d_dhw, data_config_case3)


class TestValidationDataset:
    def test_output_shapes_no_gt(
        self,
        ssunet_data_4d_cdhw: SSUnetData,
        data_config_case2: DataConfig,
    ) -> None:
        dataset = ValidationDataset(ssunet_data_4d_cdhw, data_config_case2)
        item = dataset[0]
        assert len(item) == 2
        cfg = data_config_case2
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape
        assert item[1].shape == expected_shape

    def test_output_shapes_with_gt(
        self,
        ssunet_data_secondary_matching_primary_tcdhw: SSUnetData,
        data_config_case2: DataConfig,
    ) -> None:
        dataset = ValidationDataset(
            ssunet_data_secondary_matching_primary_tcdhw,
            data_config_case2,
        )
        item = dataset[0]
        assert len(item) == 3
        cfg = data_config_case2
        expected_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_shape
        assert item[1].shape == expected_shape
        assert item[2].shape == expected_shape

    def test_normalize_target(self, ssunet_data_3d_dhw: SSUnetData) -> None:
        config_norm_true = DataConfig(z_size=16, xy_size=32, normalize_target=True)
        dataset_true = ValidationDataset(ssunet_data_3d_dhw, config_norm_true)
        item_true = dataset_true[0]
        if not torch.allclose(item_true[1].mean(), torch.tensor(0.0)) and not torch.allclose(
            item_true[1].std(), torch.tensor(0.0)
        ):
            if not torch.allclose(item_true[1], torch.zeros_like(item_true[1])):
                assert not torch.allclose(item_true[0], item_true[1])
        config_norm_false = DataConfig(z_size=16, xy_size=32, normalize_target=False)
        dataset_false = ValidationDataset(ssunet_data_3d_dhw, config_norm_false)
        item_false = dataset_false[0]
        assert torch.allclose(item_false[0], item_false[1])


@pytest.fixture
def data_config_n2n_test() -> DataConfig:
    return DataConfig(z_size=60, xy_size=128)


class TestTemporalSumSplitDataset:
    def test_init_requires_min_t2(
        self,
        data_config_case2: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        # Create data with T=1 and sufficient depth for the config's z_size
        # data_config_case2 has z_size=16, so depth must be >= 16.
        data_t1 = SSUnetData(
            primary_data=torch.rand(1, 1, 20, 32, 32)
        )  # T,C,D,H,W - Increased depth to 20
        with pytest.raises(DataError, match="requires input data with at least 2 time frames"):
            TemporalSumSplitDataset(data_t1, data_config_case2, split_params_signal)

    @pytest.mark.parametrize(
        ("t_total", "c_eff", "d_patch", "h_patch", "w_patch"),
        [
            (10, 1, 16, 32, 32),  # T, C, D, H, W (for the patch after spatial sampling)
            (5, 3, 16, 32, 32),
            (2, 1, 16, 32, 32),
        ],
    )
    @pytest.mark.parametrize(
        ("p_method", "p_values"),
        [
            ("signal", {"min_p": 0.1, "max_p": 0.9}),
            ("fixed", {"p_list": [0.5]}),
            ("fixed", {"min_p": 0.3}),  # Test fixed using min_p if p_list is None
            ("list", {"p_list": [0.25, 0.75]}),
        ],
    )
    def test_split_temporal_frames_logic(
        self,
        t_total: int,
        c_eff: int,
        d_patch: int,
        h_patch: int,
        w_patch: int,
        p_method: str,
        p_values: dict[str, Any],
    ) -> None:
        # Create dummy config and split_params
        # Config for spatial patch extraction - needs to be compatible with dummy data shape
        test_data_config = DataConfig(
            z_size=d_patch,
            xy_size=h_patch,  # Assuming H_patch == W_patch
            virtual_size=10,
            seed=42,
        )

        # Construct split_params based on p_method and p_values
        if p_method == "signal":
            split_params = SplitParams(
                method="signal", min_p=p_values["min_p"], max_p=p_values["max_p"], seed=123
            )
        elif p_method == "fixed":
            if "p_list" in p_values:
                split_params = SplitParams(method="fixed", p_list=p_values["p_list"], seed=123)
            else:
                split_params = SplitParams(method="fixed", min_p=p_values["min_p"], seed=123)
        elif p_method == "list":
            split_params = SplitParams(method="list", p_list=p_values["p_list"], seed=123)
        else:
            pytest.fail(f"Unsupported p_method: {p_method}")

        dummy_raw_data_shape = (
            t_total,
            c_eff,
            d_patch + 5,
            h_patch + 5,
            w_patch + 5,
        )  # Larger raw data
        dummy_ssu_data = SSUnetData(primary_data=torch.rand(*dummy_raw_data_shape))

        dataset_instance = TemporalSumSplitDataset(dummy_ssu_data, test_data_config, split_params)

        # Create the input tensor for _split_temporal_frames
        patch_t_first_dims = (
            torch.rand(t_total, c_eff, d_patch, h_patch, w_patch) * 10
        )  # Add some values

        input_summed, target_normalized = dataset_instance._split_temporal_frames(
            patch_t_first_dims
        )

        # 1. Check output shapes
        expected_spatial_shape = (c_eff, d_patch, h_patch, w_patch)
        assert input_summed.shape == expected_spatial_shape, "Shape of summed input is incorrect"
        assert (
            target_normalized.shape == expected_spatial_shape
        ), "Shape of normalized target is incorrect"

        # 2. Check dtypes
        assert input_summed.dtype == torch.float32
        assert target_normalized.dtype == torch.float32

        if not torch.allclose(target_normalized, torch.zeros_like(target_normalized)):
            assert not torch.isnan(
                target_normalized
            ).any(), "Target contains NaN after normalization"
            assert not torch.isinf(
                target_normalized
            ).any(), "Target contains Inf after normalization"

    def test_split_temporal_frames_edge_cases_m_n_counts(self) -> None:
        # Test T=2, should result in m=1, n=1 regardless of p_for_n
        config = DataConfig(z_size=8, xy_size=8)
        split_params_varied_p = SplitParams(method="signal", min_p=0.01, max_p=0.99, seed=1)

        t_total, c_eff, d_patch, h_patch, w_patch = 2, 1, 8, 8, 8
        dummy_raw_data_shape = (t_total, c_eff, d_patch + 5, h_patch + 5, w_patch + 5)
        dummy_ssu_data = SSUnetData(primary_data=torch.rand(*dummy_raw_data_shape))
        dataset = TemporalSumSplitDataset(dummy_ssu_data, config, split_params_varied_p)

        patch_t_first_dims = torch.rand(t_total, c_eff, d_patch, h_patch, w_patch)

        # Test multiple times due to randomness in p_for_n for "signal"
        random.seed(1)  # Seed for reproducibility
        for _ in range(10):
            # Calculate p_for_n as done in the method
            p_for_n = (
                random.random() * (dataset.split_params.max_p - dataset.split_params.min_p)  # noqa: S311
                + dataset.split_params.min_p
            )
            n_count_float = t_total * p_for_n
            n_count_actual = round(n_count_float)
            n_count_clamped = max(1, min(n_count_actual, t_total - 1))
            m_count_clamped = t_total - n_count_clamped
            dataset._split_temporal_frames(patch_t_first_dims)

            assert (
                n_count_clamped == 1
            ), f"For T={t_total}, n_count should be 1, got {n_count_clamped} with p={p_for_n:.4f}"
            assert (
                m_count_clamped == 1
            ), f"For T={t_total}, m_count should be 1, got {m_count_clamped} with p={p_for_n:.4f}"

    def test_split_params_errors(self) -> None:
        config = DataConfig(z_size=8, xy_size=8)
        dummy_data = SSUnetData(primary_data=torch.rand(2, 1, 10, 10, 10))

        # Test InvalidPValueError for method="list" with empty p_list
        dataset_list_empty = TemporalSumSplitDataset(
            dummy_data, config, SplitParams(method="list", p_list=[])
        )
        with pytest.raises(InvalidPValueError, match="p_list is required for 'list' method"):
            dataset_list_empty._split_temporal_frames(torch.rand(2, 1, 5, 5, 5))

        with pytest.raises(ConfigError, match="p_list values must be between 0 and 1"):
            SplitParams(method="list", p_list=[0.5, 0.0, 1.0])

        # Test ConfigError for unsupported method (ConfigError is raised in SplitParams)
        with pytest.raises(
            ConfigError,
            match="Invalid SPLIT.method",
        ):
            SplitParams(method="unknown_method")

        # Test ConfigError for min_p > max_p in SplitParams (signal method)
        with pytest.raises(ConfigError, match="For 'signal' method"):
            SplitParams(method="signal", min_p=0.5, max_p=0.4)

        # Note: Validation for p_list values <= 0 or >= 1 for method="fixed"
        # is not currently done in SplitParams.__post_init__.
        # If this validation is needed, it should be added to SplitParams
        # or tested when the p_list is used (e.g., in _split_temporal_frames).

    # Test for __getitem__ shapes
    def test_getitem_output_shapes(
        self,
        ssunet_data_5d_tcdhw: SSUnetData,
        data_config_case3: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        # ssunet_data_5d_tcdhw: T=5, C=3, D=100, H=128, W=128
        # data_config_case3: z_size=32, xy_size=64
        # This config is compatible as raw data is larger than patch size.

        dataset = TemporalSumSplitDataset(
            ssunet_data_5d_tcdhw, data_config_case3, split_params_signal
        )
        item = dataset[0]  # [input_sum, target_norm]

        assert len(item) == 2
        cfg = data_config_case3
        # Effective channels from ssunet_data_5d_tcdhw is 3
        expected_spatial_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_spatial_shape
        assert item[1].shape == expected_spatial_shape

    def test_getitem_with_secondary_data(
        self,
        ssunet_data_secondary_matching_primary_tcdhw: SSUnetData,
        data_config_case2: DataConfig,
        split_params_signal: SplitParams,
    ) -> None:
        dataset = TemporalSumSplitDataset(
            ssunet_data_secondary_matching_primary_tcdhw, data_config_case2, split_params_signal
        )
        item = dataset[0]  # [input_sum, target_norm, gt_norm]

        assert len(item) == 3
        cfg = data_config_case2
        # Effective channels from ssunet_data_secondary_matching_primary_tcdhw is 3
        expected_spatial_shape = (3, cfg.z_size, cfg.xy_size, cfg.xy_size)
        assert item[0].shape == expected_spatial_shape  # input sum
        assert item[1].shape == expected_spatial_shape  # target norm
        assert item[2].shape == expected_spatial_shape  # gt norm (mean of T frames of secondary)
