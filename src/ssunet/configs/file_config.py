"""Configuration for the file import."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import h5py
import numpy as np
from tifffile import TiffFile  # Added TiffPage, though not explicitly used in final load

from ..constants import EPSILON, LOGGER
from ..exceptions import (
    ConfigError,  # Assuming ConfigError is moved to ssunet.exceptions
    DirectoryNotFoundError,
    FileIndexOutOfRangeError,
    FileNotFoundError,
    InvalidHDF5DatasetError,
    InvalidSliceRangeError,
    NoDataFileAvailableError,
    UnknownFileTypeError,
)
from .data_config import SSUnetData

PathLike = str | Path
FileInput = int | str | Path


# Removed local ConfigError, assuming it's in ssunet.exceptions


class FileType(Enum):
    """Enum for file types."""

    DATA = auto()
    REFERENCE = auto()
    GROUND_TRUTH = auto()


@dataclass
class FileSourceConfig:
    """Configuration for a single data source (e.g., data, reference, ground_truth)."""

    dir: PathLike | None = None
    file: FileInput | None = None
    begin_slice: int = 0
    end_slice: int = -1
    # Optional user hint for ambiguous TIFF dimension interpretation
    # Examples: "ZYX", "CZYX", "TCZYX", "ASSUME_DHW_STACK"
    # If provided, this can guide permutation if tifffile.series.axes is missing/ambiguous.
    expected_format_hint: str | None = None
    expected_shape_hint: tuple[int, ...] | None = None
    # Optional: name of the dimension to slice along if not the first after permutation.
    # E.g., "Z", "T". If None, slices along the first dimension of the (permuted) array.
    slice_dimension_name: str | None = None

    resolved_path: Path | None = field(init=False, default=None)
    is_available: bool = field(init=False, default=False)


@dataclass
class PathConfig:
    """Configuration for paths."""

    base_dir: PathLike | None = None
    data: FileSourceConfig = field(default_factory=FileSourceConfig)
    reference: FileSourceConfig | None = None
    ground_truth: FileSourceConfig | None = None

    def __post_init__(self) -> None:
        if self.base_dir:
            self.base_dir = Path(self.base_dir)
            if not self.base_dir.exists():
                LOGGER.warning(
                    f"PATH.base_dir {self.base_dir} does not exist. Relative paths may fail."
                )

        self._resolve_file_source(self.data, FileType.DATA, is_required=True)
        if self.reference:  # Ensure self.reference is not None before calling
            self._resolve_file_source(self.reference, FileType.REFERENCE)
        if self.ground_truth:  # Ensure self.ground_truth is not None
            self._resolve_file_source(self.ground_truth, FileType.GROUND_TRUTH)

        self._validate_all_slices()

    def _resolve_file_source(
        self, source_cfg: FileSourceConfig, file_type: FileType, is_required: bool = False
    ) -> None:
        if source_cfg.file is None:
            if is_required:
                raise ConfigError(  # Use imported ConfigError
                    f"PathConfig: '{file_type.name.lower()}.file' is required but not provided."
                )
            source_cfg.is_available = False
            return

        file_input = source_cfg.file
        source_dir_path: Path | None = None
        if source_cfg.dir:
            source_dir_path = Path(source_cfg.dir)
            if not source_dir_path.is_absolute() and self.base_dir and self.base_dir.exists():
                source_dir_path = self.base_dir / source_dir_path

            if not source_dir_path.exists():
                raise DirectoryNotFoundError(source_dir_path)

        resolved_file: Path | None = None

        if isinstance(file_input, int):
            if (
                not source_dir_path
            ):  # This check implies source_dir_path MUST exist for int indexing
                raise ConfigError(
                    f"'{file_type.name.lower()}.dir' must be provided and exist when "
                    f"'{file_type.name.lower()}.file' is an integer index."
                )
            try:
                files_in_dir = sorted([f for f in source_dir_path.iterdir() if f.is_file()])
                if not (0 <= file_input < len(files_in_dir)):
                    raise FileIndexOutOfRangeError(file_type, file_input)
                resolved_file = files_in_dir[file_input]
            except Exception as err:  # Catch generic StopIteration or other fs errors
                raise FileIndexOutOfRangeError(file_type, file_input) from err

        elif isinstance(file_input, str | Path):
            file_path = Path(file_input)
            if not file_path.is_absolute():
                if self.base_dir and self.base_dir.exists():
                    resolved_file = self.base_dir / file_path
                elif (
                    source_dir_path and source_dir_path.exists()
                ):  # Fallback if base_dir not set/exist
                    resolved_file = source_dir_path / file_path
                else:  # Relative to CWD
                    resolved_file = file_path.resolve()
            else:  # Absolute path
                resolved_file = file_path

            if not resolved_file.exists():
                raise FileNotFoundError(file_type, resolved_file)
        else:
            raise ConfigError(
                f"Invalid type for '{file_type.name.lower()}.file': {type(file_input)}. "
                "Expected str, Path, or int."
            )

        source_cfg.resolved_path = resolved_file
        source_cfg.is_available = True

    def _validate_all_slices(self) -> None:
        sources_to_validate = {}  # Use a dictionary
        if self.data and self.data.is_available:  # data is always present due to default_factory
            sources_to_validate["data"] = self.data
        if self.reference and self.reference.is_available:
            sources_to_validate["reference"] = self.reference
        if self.ground_truth and self.ground_truth.is_available:
            sources_to_validate["ground_truth"] = self.ground_truth

        for name, source_cfg_val in sources_to_validate.items():
            # source_cfg_val will not be None here due to checks above
            begin, end = source_cfg_val.begin_slice, source_cfg_val.end_slice
            if begin < 0 or (end != -1 and end <= begin):
                raise InvalidSliceRangeError(name, begin, end)

    def _load_from_source(
        self, source_cfg: FileSourceConfig | None, method: Callable | None
    ) -> np.ndarray | None:
        """Loads data from a file source, handling TIFF, HDF5, or custom method.

        :param source_cfg: The file source configuration.
        :param method: Optional custom method for loading non-TIFF/HDF5 files.
        :return: Loaded data as np.ndarray or None.
        """
        if not source_cfg or not source_cfg.is_available or not source_cfg.resolved_path:
            return None

        path_to_load = source_cfg.resolved_path
        begin_slice_user, end_slice_user = source_cfg.begin_slice, source_cfg.end_slice
        data: np.ndarray | None = None

        if path_to_load.suffix.lower() in [".tif", ".tiff"]:
            with TiffFile(str(path_to_load)) as tif:
                if not tif.series or not tif.series[0] or not tif.series[0].shape:
                    raise UnknownFileTypeError(
                        f"Cannot determine series shape for TIFF: {path_to_load}"
                    )

                series = tif.series[0]
                raw_data_from_tiff = series.asarray()
                axes_hint = source_cfg.expected_format_hint
                shape_hint = source_cfg.expected_shape_hint
                series_axes_str = series.axes.upper() if series.axes else ""
                axes_from_tiff = (axes_hint.upper() if axes_hint else series_axes_str) or ""

                current_data_for_processing = raw_data_from_tiff
                current_axes = axes_from_tiff

                # --- Refined shape deduction for OME and ImageJ ---
                intended_nd_shape = None
                if axes_hint and raw_data_from_tiff.ndim < len(axes_hint):
                    # OME-TIFF: parse shape from ome_metadata
                    if (
                        hasattr(tif, "is_ome") and tif.is_ome
                    ):  # Check if tifffile identifies it as OME
                        ome_md = tif.ome_metadata  # This should be a dict
                        if isinstance(ome_md, dict):  # Ensure it's a dict
                            image_metadata = ome_md.get("Image")
                            if isinstance(image_metadata, list):  # OME can have multiple Images
                                image_metadata = image_metadata[0]  # Assume first image

                            if isinstance(image_metadata, dict) and "Pixels" in image_metadata:
                                pixels = image_metadata["Pixels"]
                                if isinstance(pixels, dict):
                                    try:
                                        upper_axes_hint = axes_hint.upper()
                                        intended_nd_shape = tuple(
                                            pixels[f"Size{ax}"] for ax in upper_axes_hint
                                        )
                                        LOGGER.debug(
                                            f"OME metadata parsed shape: {intended_nd_shape} "
                                            f"for axes {axes_hint}"
                                        )
                                    except KeyError as e:
                                        LOGGER.warning(
                                            f"OME metadata parsing failed for Size{e} "
                                            f"using axes_hint '{axes_hint}'. Pixels: {pixels}"
                                        )
                                        intended_nd_shape = None
                                    except OSError as e:
                                        LOGGER.warning(f"Error parsing OME metadata for shape: {e}")
                                        intended_nd_shape = None
                        else:
                            LOGGER.warning(
                                f"tif.ome_metadata was not a dictionary for {path_to_load}. "
                                f"Type: {type(ome_md)}"
                            )
                    # ImageJ: parse shape from imagej_metadata
                    elif hasattr(tif, "imagej_metadata") and tif.imagej_metadata:
                        md = tif.imagej_metadata
                        try:
                            intended_nd_shape = (
                                md.get("frames", 1),
                                md.get("channels", 1),
                                md.get("slices", 1),
                                raw_data_from_tiff.shape[-2],
                                raw_data_from_tiff.shape[-1],
                            )
                        except OSError:
                            intended_nd_shape = None
                    # Fallback to shape_hint if provided
                    elif shape_hint and len(shape_hint) == len(axes_hint):
                        intended_nd_shape = shape_hint

                # Try to reshape if possible
                if intended_nd_shape and np.prod(raw_data_from_tiff.shape) == np.prod(
                    intended_nd_shape
                ):
                    try:
                        current_data_for_processing = raw_data_from_tiff.reshape(intended_nd_shape)
                        current_axes = axes_hint
                        LOGGER.info(
                            f"Reshaped TIFF data to {intended_nd_shape} using metadata and/or"
                            f"   expected_format_hint '{axes_hint}'"
                        )
                    except ValueError as e:
                        LOGGER.warning(
                            f"Failed to reshape TIFF data to {intended_nd_shape} for axes "
                            f"'{axes_hint}': {e}"
                        )
                        current_data_for_processing = raw_data_from_tiff
                        current_axes = series_axes_str
                elif (
                    shape_hint
                    and raw_data_from_tiff.ndim == 3
                    and len(shape_hint) == len(axes_hint)
                ):
                    if np.prod(raw_data_from_tiff.shape) == np.prod(shape_hint):
                        try:
                            current_data_for_processing = raw_data_from_tiff.reshape(shape_hint)
                            current_axes = axes_hint
                            LOGGER.info(
                                f"Reshaped TIFF data to {shape_hint} using expected_shape_hint and"
                                f"   expected_format_hint '{axes_hint}'"
                            )
                        except OSError as e:
                            LOGGER.warning(
                                f"Failed to reshape TIFF data to {shape_hint} "
                                f"for axes '{axes_hint}': {e}"
                            )

                permuted_data = current_data_for_processing
                final_axes_order_str = current_axes

                # Standard internal target orders
                # 5D: TCDHW
                # 4D: CDHW
                # 3D: DHW

                # --- Begin new permutation logic for TZYX and 5D cases ---
                final_ndim = raw_data_from_tiff.ndim
                if axes_from_tiff and final_ndim > 0:
                    if final_ndim == 4 and axes_from_tiff == "TZYX":
                        # Input is (T,Z,Y,X). Expand to (T,1,Z,Y,X) for BasePatchDataset 5D path.
                        # This makes C_eff=1, D_eff=Z. total_time_frames will be T.
                        permuted_data = raw_data_from_tiff[:, np.newaxis, :, :, :]
                        final_axes_order_str = "TCDHW"  # Semantic meaning after expand_dims
                        LOGGER.info(
                            f"Interpreted TZYX as T(1)ZYX (TCDHW semantically). "
                            f"New shape: {permuted_data.shape}"
                        )
                    elif final_ndim == 5:
                        if axes_from_tiff == "TCZYX":
                            permuted_data = raw_data_from_tiff
                            final_axes_order_str = "TCDHW"
                        elif axes_from_tiff == "TZCYX":
                            permuted_data = np.transpose(raw_data_from_tiff, (0, 2, 1, 3, 4))
                            final_axes_order_str = "TCDHW"
                        else:
                            LOGGER.warning(f"Unknown 5D axes '{axes_from_tiff}'. Using raw order.")
                            final_axes_order_str = axes_from_tiff
                    else:
                        # Fallback to original permutation map for other cases
                        permutation_map = {
                            # Target: DHW (3D)
                            "ZYX": ((0, 1, 2), "DHW"),  # No change
                            "YXZ": ((1, 0, 2), "DHW"),
                            "XYZ": ((2, 0, 1), "DHW"),
                            "QYX": ((0, 1, 2), "DHW"),
                            "IYX": ((0, 1, 2), "DHW"),
                            "CZYX": ((0, 1, 2, 3), "CDHW"),
                            "ZCYX": ((1, 0, 2, 3), "CDHW"),
                            "CZXY": ((0, 1, 3, 2), "CDHW"),
                            "TZYX": ((0, 1, 2, 3), "CDHW"),  # If not handled above
                            # Target: TCDHW (5D)
                            "TCZYX": ((0, 1, 2, 3, 4), "TCDHW"),
                            "TZCYX": ((0, 2, 1, 3, 4), "TCDHW"),
                        }
                        if axes_from_tiff in permutation_map:
                            perm_indices, target_axes = permutation_map[axes_from_tiff]
                            if len(perm_indices) == raw_data_from_tiff.ndim:
                                try:
                                    permuted_data = np.transpose(raw_data_from_tiff, perm_indices)
                                    final_axes_order_str = target_axes
                                    LOGGER.info(
                                        f"Permuted '{axes_from_tiff}' to '{final_axes_order_str}', "
                                        f"new_shape={permuted_data.shape}"
                                    )
                                except ValueError as e:
                                    LOGGER.error(
                                        f"Permutation from '{axes_from_tiff}' failed: {e}. "
                                        f"Using raw order."
                                    )
                                    final_axes_order_str = axes_from_tiff  # Revert
                            else:
                                LOGGER.warning(
                                    f"Permutation map for '{axes_from_tiff}' has incorrect ndim. "
                                    f"Using raw order."
                                )
                                final_axes_order_str = axes_from_tiff
                        else:
                            LOGGER.warning(
                                f"TIFF {path_to_load} has axes '{axes_from_tiff}' "
                                f"not in explicit permutation map. Using raw data order."
                            )
                            # permuted_data is already raw_data_from_tiff
                elif not axes_from_tiff and raw_data_from_tiff.ndim > 0:  # No axes info at all
                    LOGGER.warning(
                        f"TIFF {path_to_load} has no axes metadata. "
                        f"Interpreting based on ndim {raw_data_from_tiff.shape} as default."
                    )

                    if raw_data_from_tiff.ndim == 5:
                        final_axes_order_str = "TCDHW"
                    elif raw_data_from_tiff.ndim == 4:
                        final_axes_order_str = "CDHW"
                    elif raw_data_from_tiff.ndim == 3:
                        final_axes_order_str = "DHW"
                    else:
                        final_axes_order_str = "?" * raw_data_from_tiff.ndim
                    # permuted_data is already raw_data_from_tiff
                    LOGGER.info(
                        f"Guessed axes '{final_axes_order_str}' for shape {permuted_data.shape}"
                    )

                # --- Slicing ---
                data_to_slice = permuted_data
                slicing_dim_idx = 0  # Default: slice along the first dimension

                if (
                    source_cfg.slice_dimension_name
                    and final_axes_order_str
                    and "?" not in final_axes_order_str
                ):
                    user_slice_dim_name = source_cfg.slice_dimension_name.upper()

                    if user_slice_dim_name in axes_from_tiff:
                        try:
                            original_semantic_idx = axes_from_tiff.index(user_slice_dim_name)

                            if original_semantic_idx < len(final_axes_order_str):
                                target_slice_char_in_permuted_axes = final_axes_order_str[
                                    original_semantic_idx
                                ]
                                slicing_dim_idx = final_axes_order_str.index(
                                    target_slice_char_in_permuted_axes
                                )
                                LOGGER.info(
                                    f"Slicing on user dimension '{user_slice_dim_name}'"
                                    f"(original index {original_semantic_idx}), "
                                    f"which corresponds to '{target_slice_char_in_permuted_axes}'"
                                    f"(index {slicing_dim_idx}) in permuted axes "
                                    f"'{final_axes_order_str}'."
                                )
                            else:
                                LOGGER.warning(
                                    f"Original index for slice dimension '{user_slice_dim_name}' "
                                    f"is out of bounds for permuted axes '{final_axes_order_str}'. "
                                    f"Slicing on dim 0."
                                )
                        except ValueError:
                            LOGGER.warning(
                                f"Could not determine slice index for dimension "
                                f"'{user_slice_dim_name}' using axes '{axes_from_tiff}' "
                                f"and permuted axes '{final_axes_order_str}'. "
                                f"Slicing on dim 0."
                            )
                    else:
                        LOGGER.warning(
                            f"User-specified slice_dimension_name '{user_slice_dim_name}' "
                            f"not found in derived initial axes '{axes_from_tiff}'. "
                            f"Slicing on dim 0."
                        )

                if data_to_slice.ndim == 0:  # Should not happen for image data
                    raise UnknownFileTypeError(f"TIFF data resulted in a scalar for {path_to_load}")
                if slicing_dim_idx >= data_to_slice.ndim:
                    raise ConfigError(
                        f"Slicing dimension index {slicing_dim_idx} "
                        f"out of bounds for data ndim {data_to_slice.ndim}"
                    )

                num_elements_in_slice_dim = data_to_slice.shape[slicing_dim_idx]

                actual_begin_slice = begin_slice_user
                actual_end_slice = (
                    num_elements_in_slice_dim if end_slice_user == -1 else end_slice_user
                )

                if not (
                    0 <= actual_begin_slice < num_elements_in_slice_dim
                    and actual_begin_slice < actual_end_slice <= num_elements_in_slice_dim
                ):
                    raise InvalidSliceRangeError(
                        f"{path_to_load} (axes: {final_axes_order_str}, "
                        f"shape: {data_to_slice.shape}, "
                        f"slicing_dim_idx: {slicing_dim_idx})",
                        actual_begin_slice,
                        actual_end_slice,
                    )

                # Create slice object for the specified dimension
                slice_obj = [slice(None)] * data_to_slice.ndim
                slice_obj[slicing_dim_idx] = slice(actual_begin_slice, actual_end_slice)
                data = data_to_slice[tuple(slice_obj)]

                LOGGER.debug(
                    f"Sliced on dim_idx {slicing_dim_idx} "
                    f"('{final_axes_order_str[slicing_dim_idx]}' if axes known) "
                    f"from {actual_begin_slice} to {actual_end_slice}. "
                    f"Final data shape: {data.shape}"
                )

        elif path_to_load.suffix.lower() in [".h5", ".hdf5"]:
            with h5py.File(str(path_to_load), "r") as f:
                # Allow specifying dataset name, default to first key
                dataset_name_to_load = source_cfg.expected_format_hint or next(iter(f.keys()))
                if not dataset_name_to_load or dataset_name_to_load not in f:
                    raise InvalidHDF5DatasetError(
                        f"HDF5 file {path_to_load} has no dataset named "
                        f"'{dataset_name_to_load}' or no datasets at all."
                    )

                dataset = f.get(dataset_name_to_load)
                if not isinstance(dataset, h5py.Dataset):
                    raise InvalidHDF5DatasetError(
                        f"HDF5 key '{dataset_name_to_load}' in {path_to_load} is not a Dataset."
                    )

                # HDF5 slicing: Assume slice_dimension_name refers to index if provided, else 0
                slicing_dim_idx = 0
                if source_cfg.slice_dimension_name:
                    try:
                        slicing_dim_idx = int(
                            source_cfg.slice_dimension_name
                        )  # Allow int string for dim index
                        if not (0 <= slicing_dim_idx < dataset.ndim):
                            raise ValueError("Index out of bounds")
                    except ValueError:
                        LOGGER.warning(
                            f"HDF5 slice_dimension_name '{source_cfg.slice_dimension_name}' "
                            f"is not a valid integer index. Slicing on dim 0."
                        )
                        slicing_dim_idx = 0

                num_elements_in_slice_dim = dataset.shape[slicing_dim_idx]
                actual_begin_slice = begin_slice_user
                actual_end_slice = (
                    num_elements_in_slice_dim if end_slice_user == -1 else end_slice_user
                )

                if not (
                    0 <= actual_begin_slice < num_elements_in_slice_dim
                    and actual_begin_slice < actual_end_slice <= num_elements_in_slice_dim
                ):
                    raise InvalidSliceRangeError(
                        f"{path_to_load} (HDF5 dataset '{dataset_name_to_load}' "
                        f"shape: {dataset.shape}, slicing_dim_idx: {slicing_dim_idx})",
                        actual_begin_slice,
                        actual_end_slice,
                    )

                slice_obj = [slice(None)] * dataset.ndim
                slice_obj[slicing_dim_idx] = slice(actual_begin_slice, actual_end_slice)
                data = np.array(dataset[tuple(slice_obj)])
                LOGGER.debug(
                    f"HDF5: Sliced dataset '{dataset_name_to_load}' on dim {slicing_dim_idx} "
                    f"from {actual_begin_slice} to {actual_end_slice}. "
                    f"Final shape: {data.shape}"
                )

        elif method:
            data = method(path_to_load, begin=begin_slice_user, end=end_slice_user)
        else:
            raise UnknownFileTypeError(path_to_load)

        if data is None:  # Should be caught by earlier specific errors
            raise RuntimeError(f"Data loading resulted in None for {path_to_load}")

        return data.astype(np.float32)

    def load_data_only(self, method: Callable | None = None) -> SSUnetData:
        data_arr = self._load_from_source(self.data, method)
        if data_arr is None:  # Should only happen if data source itself is None/unavailable
            raise NoDataFileAvailableError(
                "Primary data file could not be loaded (source config issue or loading failed)."
            )
        return SSUnetData(primary_data=data_arr)

    def load_reference_only(self, method: Callable | None = None) -> SSUnetData:
        # Check if reference is configured and available
        if not self.reference or not self.reference.is_available:
            LOGGER.info(
                "Reference not configured/available, attempting to load primary data "
                "for load_reference_only."
            )
            return self.load_data_only(method)  # Fallback to primary data

        ref_arr = self._load_from_source(self.reference, method)
        if ref_arr is None:  # Loading explicitly configured reference failed
            LOGGER.warning(
                "Configured reference file could not be loaded. Falling back to primary data."
            )
            return self.load_data_only(method)
        return SSUnetData(primary_data=ref_arr)

    def load_data_and_ground_truth(self, method: Callable | None = None) -> SSUnetData:
        data_arr = self._load_from_source(self.data, method)
        if data_arr is None:
            raise NoDataFileAvailableError("Primary data file could not be loaded.")

        # Try to load ground_truth first
        gt_arr = self._load_from_source(self.ground_truth, method)

        if gt_arr is None and self.reference and self.reference.is_available:
            LOGGER.info("Ground truth not available/loaded, attempting to load reference instead.")
            gt_arr = self._load_from_source(self.reference, method)

        if gt_arr is None:  # If both GT and Ref failed or were not available
            LOGGER.warning(
                "Neither ground truth nor reference data available/loaded for "
                "load_data_and_ground_truth. Secondary data will be None."
            )
            # could raise error or return SSUnetData with secondary_data=None

        normalized_gt_arr = self._normalize_ground_truth(gt_arr) if gt_arr is not None else None
        return SSUnetData(primary_data=data_arr, secondary_data=normalized_gt_arr)

    def load_reference_and_ground_truth(self, method: Callable | None = None) -> SSUnetData:
        # Load reference; if not available or fails, try primary data as reference
        ref_arr = self._load_from_source(self.reference, method)
        if ref_arr is None:
            LOGGER.info(
                "Reference not available/loaded, attempting to load primary data as reference."
            )
            ref_arr = self._load_from_source(self.data, method)
            if ref_arr is None:  # If primary data also fails
                raise NoDataFileAvailableError(
                    "Neither reference nor primary data could be loaded."
                )

        # Load ground_truth; if not available or fails, use the loaded ref_arr (primary data)
        gt_arr = self._load_from_source(self.ground_truth, method)
        if gt_arr is None:
            LOGGER.info(
                "Ground truth not available/loaded, using the loaded reference/primary data "
                "as ground truth."
            )
            gt_arr = ref_arr  # ref_arr is guaranteed to be non-None here

        normalized_gt_arr = self._normalize_ground_truth(gt_arr)  # gt_arr is non-None
        return SSUnetData(primary_data=ref_arr, secondary_data=normalized_gt_arr)

    def _normalize_ground_truth(self, ground_truth: np.ndarray | None) -> np.ndarray | None:
        if ground_truth is None:
            return None

        # Ensure it's float for calculations
        gt_float = ground_truth.astype(np.float32)
        gt_min, gt_max = np.min(gt_float), np.max(gt_float)

        if gt_max == gt_min:  # Constant image
            # If constant is 0, return 0. Otherwise, maybe 0.5 or 1.
            # Or just return as is if normalization is not well-defined.
            LOGGER.debug("Ground truth is a constant image. Returning as is.")
            return gt_float

        # Heuristic for 8-bit like data (0-255 range)
        if 1.0 < gt_max <= 255.0 and gt_min >= 0:
            LOGGER.debug("Normalizing ground truth from approx 8-bit range to [0,1].")
            return (gt_float - gt_min) / (gt_max - gt_min + EPSILON)  # Normalize to 0-1 robustly

        # Heuristic for 16-bit like data or other large ranges
        elif gt_max > 255.0 and gt_min >= 0:
            LOGGER.debug("Normalizing ground truth from >8-bit range to [0,1] by its own min/max.")
            return (gt_float - gt_min) / (gt_max - gt_min + EPSILON)  # Normalize to 0-1 robustly

        # If already in ~[0,1] range or negative values present, return as is
        # (or apply a different normalization if needed for negative values)
        LOGGER.debug(
            "Ground truth appears to be already in [0,1] range or contains negative values. "
            "Returning as is."
        )
        return gt_float


# inspect_tiff_dimensions can remain as a utility if needed for manual inspection
# but PathConfig now handles common cases internally.


@dataclass
class SplitParams:
    method: str = "signal"
    min_p: float = EPSILON
    max_p: float = 1.0 - EPSILON  # Ensure this is less than 1.0
    p_list: list[float] | None = field(default_factory=list)
    normalize_target: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        valid_methods = ["signal", "fixed", "list"]
        if self.method not in valid_methods:
            raise ConfigError(  # Use imported ConfigError
                f"Invalid SPLIT.method '{self.method}'. Must be one of {valid_methods}."
            )
        if self.method == "signal":
            if not (
                0.0 <= self.min_p < self.max_p <= 1.0
            ):  # p values for signal should be in [0,1]
                # Allow min_p == max_p for signal if user wants fixed p via signal method
                if self.min_p == self.max_p and 0.0 <= self.min_p <= 1.0:
                    pass
                else:
                    raise ConfigError(
                        f"For 'signal' method, "
                        f"0.0 <= min_p ({self.min_p}) < max_p ({self.max_p}) <= 1.0 "
                        "must hold."
                    )
        elif self.method == "list" and self.p_list:
            for p_val in self.p_list:
                if not (0.0 < p_val < 1.0):  # p values in list should be (0,1) for binomial prob
                    raise ConfigError(
                        f"p_list values must be between 0 and 1 (exclusive), got {p_val}"
                    )
        # seed is handled by the class using it (e.g. BinomDataset)
