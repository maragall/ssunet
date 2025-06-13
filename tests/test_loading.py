from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

from src.ssunet.configs import FileSource, PathConfig
from src.ssunet.datasets import SSUnetData
from src.ssunet.exceptions import (
    ConfigError,
    DirectoryNotFoundError,
    FileIndexOutOfRangeError,
    FileNotFoundError,
    InvalidSliceRangeError,
)

D_IMGJ, C_IMGJ, T_IMGJ, H_IMGJ, W_IMGJ = 10, 3, 2, 32, 16
D_OME, C_OME, T_OME, H_OME, W_OME = 3, 2, 2, 32, 16
D_ZYX, H_ZYX, W_ZYX = 10, 64, 32


def write_dummy_tiff(filepath: Path, shape=(1, 10, 10), dtype=np.uint8) -> None:
    tifffile.imwrite(filepath, np.zeros(shape, dtype=dtype))


@pytest.fixture
def simple_zyx_tiff(tmp_path: Path) -> Path:
    filepath = tmp_path / "simple_zyx.tif"
    data = np.arange(D_ZYX * H_ZYX * W_ZYX, dtype=np.uint16).reshape(D_ZYX, H_ZYX, W_ZYX)
    tifffile.imwrite(filepath, data)
    return filepath


@pytest.fixture
def imagej_style_tczyx_tiff(tmp_path: Path) -> Path:
    filepath = tmp_path / "imagej_tczyx.tif"
    data_tczyx = np.arange(T_IMGJ * C_IMGJ * D_IMGJ * H_IMGJ * W_IMGJ, dtype=np.uint8).reshape(
        T_IMGJ, C_IMGJ, D_IMGJ, H_IMGJ, W_IMGJ
    )

    description = (
        f"ImageJ=1.54k\\nimages={T_IMGJ * C_IMGJ * D_IMGJ}\\n"
        f"nchannels={C_IMGJ}\\nslices={D_IMGJ}\\nframes={T_IMGJ}\\n"
        f"hyperstack=true\\naxes=TCZYX\\n"
    )
    tifffile.imwrite(filepath, data_tczyx, imagej=True, metadata={"description": description})
    return filepath


@pytest.fixture
def ome_style_tczyx_tiff(tmp_path: Path) -> Path:
    filepath = tmp_path / "ome_tczyx.tif"
    data_tczyx = np.arange(T_OME * C_OME * D_OME * H_OME * W_OME, dtype=np.uint16).reshape(
        T_OME, C_OME, D_OME, H_OME, W_OME
    )
    tifffile.imwrite(filepath, data_tczyx, ome=True, metadata={"axes": "TCZYX"})
    return filepath


@pytest.fixture
def simple_hdf5_file(tmp_path: Path) -> Path:
    filepath = tmp_path / "test_data.h5"
    data_3d = np.arange(D_ZYX * H_ZYX * W_ZYX, dtype=np.float32).reshape(D_ZYX, H_ZYX, W_ZYX)
    data_4d = np.arange(C_IMGJ * D_IMGJ * H_IMGJ * W_IMGJ, dtype=np.int16).reshape(
        C_IMGJ, D_IMGJ, H_IMGJ, W_IMGJ
    )
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data_3d_dhw", data=data_3d)
        f.create_dataset("data_4d_cdhw", data=data_4d)
    return filepath


class TestPathConfigLoading:
    def test_load_simple_zyx_tiff_slicing(self, simple_zyx_tiff: Path):
        source_cfg = FileSource(
            file=simple_zyx_tiff, begin_slice=2, end_slice=5, expected_format_hint="ZYX"
        )
        path_cfg = PathConfig(data=source_cfg)
        loaded_data = path_cfg._load_from_source(path_cfg.data, method=None)
        assert loaded_data is not None
        assert loaded_data.shape == (3, H_ZYX, W_ZYX)
        raw_file_data = tifffile.imread(simple_zyx_tiff)
        assert np.array_equal(loaded_data, raw_file_data[2:5])

    def test_load_imagej_tczyx_tiff_slicing_t(self, imagej_style_tczyx_tiff: Path):
        source_cfg = FileSource(
            file=imagej_style_tczyx_tiff,
            begin_slice=1,
            end_slice=2,
            expected_format_hint="TCZYX",
            expected_shape_hint=(T_IMGJ, C_IMGJ, D_IMGJ, H_IMGJ, W_IMGJ),
        )
        path_cfg = PathConfig(data=source_cfg)
        loaded_data = path_cfg._load_from_source(path_cfg.data, method=None)
        assert loaded_data is not None
        assert loaded_data.shape == (1, C_IMGJ, D_IMGJ, H_IMGJ, W_IMGJ)

    def test_load_ome_tczyx_tiff_slicing_t(self, ome_style_tczyx_tiff: Path):
        source_cfg = FileSource(
            file=ome_style_tczyx_tiff,
            begin_slice=0,
            end_slice=1,
            expected_format_hint="TCZYX",
            expected_shape_hint=(T_OME, C_OME, D_OME, H_OME, W_OME),
        )
        path_cfg = PathConfig(data=source_cfg)
        loaded_data = path_cfg._load_from_source(path_cfg.data, method=None)
        assert loaded_data is not None
        assert loaded_data.shape == (1, C_OME, D_OME, H_OME, W_OME)

    def test_load_hdf5_named_dataset_slicing(self, simple_hdf5_file: Path):
        source_cfg = FileSource(
            file=simple_hdf5_file, begin_slice=1, end_slice=2, expected_format_hint="data_4d_cdhw"
        )
        path_cfg = PathConfig(data=source_cfg)
        loaded_data = path_cfg._load_from_source(path_cfg.data, method=None)
        assert loaded_data is not None
        assert loaded_data.shape == (1, D_IMGJ, H_IMGJ, W_IMGJ)
        with h5py.File(simple_hdf5_file, "r") as f:
            expected_raw_data = f["data_4d_cdhw"][:]
        assert np.array_equal(loaded_data, expected_raw_data[1:2])

    def test_path_resolution_absolute(self, simple_zyx_tiff: Path):
        source_cfg = FileSource(file=simple_zyx_tiff)
        path_cfg = PathConfig(data=source_cfg)
        assert path_cfg.data.resolved_path == simple_zyx_tiff.resolve()
        assert path_cfg.data.is_available

    def test_path_resolution_relative_to_base_dir(self, tmp_path: Path):
        base_dir = tmp_path / "my_base"
        base_dir.mkdir()
        data_sub_dir = base_dir / "data_files"
        data_sub_dir.mkdir()
        test_file = data_sub_dir / "data.tif"
        write_dummy_tiff(test_file)
        source_cfg = FileSource(file="data_files/data.tif")
        path_cfg = PathConfig(base_dir=base_dir, data=source_cfg)
        assert path_cfg.data.resolved_path == test_file.resolve()

    def test_path_resolution_relative_to_source_dir(self, tmp_path: Path):
        specific_data_dir = tmp_path / "specific_data"
        specific_data_dir.mkdir()
        test_file = specific_data_dir / "data.tif"
        write_dummy_tiff(test_file)
        source_cfg = FileSource(dir=specific_data_dir, file="data.tif")
        path_cfg = PathConfig(data=source_cfg)
        assert path_cfg.data.resolved_path == test_file.resolve()

    def test_path_resolution_integer_index(self, tmp_path: Path):
        data_dir = tmp_path / "indexed_files"
        data_dir.mkdir()
        file0 = data_dir / "aaa_file0.tif"
        write_dummy_tiff(file0)
        file1 = data_dir / "bbb_file1.dat"
        file1.touch()
        file2 = data_dir / "ccc_file2.tif"
        write_dummy_tiff(file2)
        source_cfg = FileSource(dir=data_dir, file=1)
        path_cfg = PathConfig(data=source_cfg)
        assert path_cfg.data.resolved_path.name == "bbb_file1.dat"
        source_cfg_out_of_bounds = FileSource(dir=data_dir, file=10)
        with pytest.raises(FileIndexOutOfRangeError):
            PathConfig(data=source_cfg_out_of_bounds)

    def test_file_not_found(self, tmp_path: Path):
        # First case: absolute path
        non_existent_abs = tmp_path / "nonexistent.tif"
        with pytest.raises(FileNotFoundError):
            PathConfig(data=FileSource(file=non_existent_abs))
        # Second case: relative path
        non_existent_rel_file = Path("nonexistent_relative.tif")
        if non_existent_rel_file.exists():
            non_existent_rel_file.unlink()
        with pytest.raises(FileNotFoundError):
            PathConfig(data=FileSource(file="nonexistent_relative.tif"))

    def test_directory_not_found(self):
        # Only the PathConfig call should be inside the raises block
        with pytest.raises(DirectoryNotFoundError):
            PathConfig(data=FileSource(dir="non_existent_dir/", file=0))

    def test_load_data_only(self, simple_zyx_tiff: Path):
        path_cfg = PathConfig(data=FileSource(file=simple_zyx_tiff, expected_format_hint="ZYX"))
        ssu_data = path_cfg.load_data_only()
        assert isinstance(ssu_data, SSUnetData)
        assert ssu_data.primary_data is not None
        assert ssu_data.secondary_data is None
        assert ssu_data.primary_data.shape == (D_ZYX, H_ZYX, W_ZYX)

    def test_load_data_only_no_file_error(self):
        # Only the PathConfig call should be inside the raises block
        with pytest.raises(ConfigError, match="PathConfig: 'data.file' is required."):
            PathConfig(data=FileSource(file=None))

    def test_load_reference_and_ground_truth_all_present(
        self, ome_style_tczyx_tiff: Path, tmp_path: Path
    ):
        dummy_data_file = tmp_path / "dummy_data_for_pathconfig.tif"
        write_dummy_tiff(dummy_data_file)

        ground_truth_file_source = ome_style_tczyx_tiff

        path_cfg = PathConfig(
            data=FileSource(file=dummy_data_file),
            reference=FileSource(
                file=ome_style_tczyx_tiff,
                begin_slice=0,
                end_slice=1,
                expected_format_hint="TCZYX",
                expected_shape_hint=(T_OME, C_OME, D_OME, H_OME, W_OME),
            ),
            ground_truth=FileSource(
                file=ground_truth_file_source,
                begin_slice=1,
                end_slice=2,
                expected_format_hint="TCZYX",
                expected_shape_hint=(T_OME, C_OME, D_OME, H_OME, W_OME),
            ),
        )
        ssu_data = path_cfg.load_reference_and_ground_truth()
        assert ssu_data.primary_data is not None
        assert ssu_data.secondary_data is not None

        expected_final_shape = (1, C_OME, D_OME, H_OME, W_OME)
        assert ssu_data.primary_data.shape == expected_final_shape
        assert ssu_data.secondary_data.shape == expected_final_shape

    def test_load_reference_and_ground_truth_gt_missing(
        self, simple_zyx_tiff: Path, tmp_path: Path
    ):
        dummy_data_file = tmp_path / "dummy_data_for_pathconfig2.tif"
        write_dummy_tiff(dummy_data_file)
        ref_slice_len = 5
        path_cfg = PathConfig(
            data=FileSource(file=dummy_data_file),
            reference=FileSource(
                file=simple_zyx_tiff,
                begin_slice=0,
                end_slice=ref_slice_len,
                expected_format_hint="ZYX",
            ),
        )
        ssu_data = path_cfg.load_reference_and_ground_truth()
        assert ssu_data.primary_data is not None
        assert ssu_data.secondary_data is not None
        assert ssu_data.primary_data.shape == (ref_slice_len, H_ZYX, W_ZYX)
        assert ssu_data.secondary_data.shape == (ref_slice_len, H_ZYX, W_ZYX)
        raw_ref_data = tifffile.imread(simple_zyx_tiff)[0:ref_slice_len]
        norm_ref_data = path_cfg._normalize_ground_truth(raw_ref_data)
        assert np.allclose(ssu_data.secondary_data, norm_ref_data, atol=1e-6)

    def test_load_reference_and_ground_truth_ref_missing_fallback_to_data(self, tmp_path: Path):
        data_as_ref_gt_file = tmp_path / "data_as_ref_gt.tif"
        data_shape = (D_ZYX, H_ZYX, W_ZYX)
        write_dummy_tiff(data_as_ref_gt_file, shape=data_shape)
        path_cfg = PathConfig(data=FileSource(file=data_as_ref_gt_file, expected_format_hint="ZYX"))
        ssu_data = path_cfg.load_reference_and_ground_truth()
        assert ssu_data.primary_data is not None
        assert ssu_data.secondary_data is not None
        assert ssu_data.primary_data.shape == data_shape
        assert ssu_data.secondary_data.shape == data_shape
        raw_data = tifffile.imread(data_as_ref_gt_file)
        norm_data = path_cfg._normalize_ground_truth(raw_data)
        assert np.allclose(ssu_data.secondary_data, norm_data, atol=1e-6)

    def test_invalid_slice_range_in_config(self, simple_zyx_tiff: Path):
        # Only the PathConfig call should be inside the raises block
        with pytest.raises(InvalidSliceRangeError):
            PathConfig(data=FileSource(file=simple_zyx_tiff, begin_slice=5, end_slice=2))

    def test_slicing_on_specific_dim_name_tiff(self, ome_style_tczyx_tiff: Path):
        source_cfg = FileSource(
            file=ome_style_tczyx_tiff,
            begin_slice=1,
            end_slice=2,
            expected_format_hint="TCZYX",
            expected_shape_hint=(T_OME, C_OME, D_OME, H_OME, W_OME),
            slice_dimension_name="Z",
        )
        path_cfg = PathConfig(data=source_cfg)
        loaded_data = path_cfg._load_from_source(path_cfg.data, method=None)
        assert loaded_data is not None
        assert loaded_data.shape == (T_OME, C_OME, 1, H_OME, W_OME)

    def test_slicing_on_specific_dim_index_hdf5(self, simple_hdf5_file: Path):
        source_cfg = FileSource(
            file=simple_hdf5_file,
            begin_slice=2,
            end_slice=5,
            expected_format_hint="data_4d_cdhw",
            slice_dimension_name="1",
        )
        path_cfg = PathConfig(data=source_cfg)
        loaded_data = path_cfg._load_from_source(path_cfg.data, method=None)
        assert loaded_data is not None
        assert loaded_data.shape == (C_IMGJ, 3, H_IMGJ, W_IMGJ)
