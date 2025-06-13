"""Configuration for the file import."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import h5py
import numpy as np
from tifffile import TiffFile

from ..constants import EPSILON, LOGGER
from ..datasets.ssunet_data import SSUnetData
from ..exceptions import (
    ConfigError,
    DirectoryNotFoundError,
    FileIndexOutOfRangeError,
    FileNotFoundError,
    InvalidHDF5DatasetError,
    InvalidSliceRangeError,
    NoDataFileAvailableError,
    UnknownFileTypeError,
)

PathLike = str | Path
FileInput = int | str | Path


class FileType(Enum):
    """Enum for file types."""

    DATA = auto()
    REFERENCE = auto()
    GROUND_TRUTH = auto()


@dataclass
class FileSource:
    """Configuration for a single data source (e.g., data, reference, ground_truth)."""

    dir: PathLike | None = None
    file: FileInput | None = None
    begin_slice: int = 0
    end_slice: int = -1
    expected_format_hint: str | None = None
    expected_shape_hint: tuple[int, ...] | None = None
    slice_dimension_name: str | None = None

    resolved_path: Path | None = field(init=False, default=None)
    is_available: bool = field(init=False, default=False)


@dataclass
class PathConfig:
    """Configuration for paths."""

    base_dir: PathLike | None = None
    data: FileSource = field(default_factory=FileSource)
    reference: FileSource | None = None
    ground_truth: FileSource | None = None

    _raw_data_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.base_dir:
            self.base_dir = Path(self.base_dir)
            if not self.base_dir.exists():
                LOGGER.warning(
                    f"PATH.base_dir {self.base_dir} does not exist. Relative paths may fail."
                )
        self._resolve_file_source(self.data, FileType.DATA, is_required=True)
        if self.reference:
            self._resolve_file_source(self.reference, FileType.REFERENCE)
        if self.ground_truth:
            self._resolve_file_source(self.ground_truth, FileType.GROUND_TRUTH)
        self._validate_all_slices()

    def _resolve_file_source(
        self, source_cfg: FileSource, file_type: FileType, is_required: bool = False
    ) -> None:
        # ... (resolution logic remains the same)
        if source_cfg.file is None:
            if is_required:
                raise ConfigError(
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
            if not source_dir_path:
                raise ConfigError(
                    f"'{file_type.name.lower()}.dir' must be provided and exist when "
                    f"'{file_type.name.lower()}.file' is an integer index."
                )
            try:
                files_in_dir = sorted([f for f in source_dir_path.iterdir() if f.is_file()])
                if not (0 <= file_input < len(files_in_dir)):
                    raise FileIndexOutOfRangeError(file_type, file_input)
                resolved_file = files_in_dir[file_input]
            except Exception as err:
                raise FileIndexOutOfRangeError(file_type, file_input) from err

        elif isinstance(file_input, str | Path):
            file_path = Path(file_input)
            if not file_path.is_absolute():
                if self.base_dir and self.base_dir.exists():
                    resolved_file = self.base_dir / file_path
                elif source_dir_path and source_dir_path.exists():
                    resolved_file = source_dir_path / file_path
                else:
                    resolved_file = file_path.resolve()
            else:
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
        # ... (validation logic remains the same)
        sources_to_validate = {}
        if self.data and self.data.is_available:
            sources_to_validate["data"] = self.data
        if self.reference and self.reference.is_available:
            sources_to_validate["reference"] = self.reference
        if self.ground_truth and self.ground_truth.is_available:
            sources_to_validate["ground_truth"] = self.ground_truth

        for name, source_cfg_val in sources_to_validate.items():
            begin, end = source_cfg_val.begin_slice, source_cfg_val.end_slice
            if begin < 0 or (end != -1 and end <= begin):
                raise InvalidSliceRangeError(name, begin, end)

    def _get_raw_data_cache_key(
        self, source_cfg: FileSource, file_suffix_lower: str, method: Callable | None
    ) -> tuple:
        """Creates a key for caching the raw, fully loaded and processed (but un-sliced) data."""
        if file_suffix_lower in [".tif", ".tiff"]:
            return (
                source_cfg.resolved_path,
                "tiff",
                source_cfg.expected_format_hint,  # Affects reshape/permutation
                source_cfg.expected_shape_hint,  # Affects reshape
            )
        elif file_suffix_lower in [".h5", ".hdf5"]:
            return (
                source_cfg.resolved_path,
                "hdf5",
                source_cfg.expected_format_hint,  # HDF5 dataset name
            )
        elif method:
            return (
                source_cfg.resolved_path,
                id(method),  # Custom loader identity
            )
        else:
            # This should ideally be caught before calling this function
            raise UnknownFileTypeError(f"Cannot determine cache key for {source_cfg.resolved_path}")

    def _load_from_source(
        self, source_cfg: FileSource | None, method: Callable | None
    ) -> np.ndarray | None:
        if not source_cfg or not source_cfg.is_available or not source_cfg.resolved_path:
            return None

        path_to_load = source_cfg.resolved_path
        file_suffix_lower = path_to_load.suffix.lower()

        raw_cache_key = self._get_raw_data_cache_key(source_cfg, file_suffix_lower, method)

        data_to_slice_from: np.ndarray
        final_axes_order_str_for_slicing: str | None = None

        if raw_cache_key in self._raw_data_cache:
            LOGGER.debug(f"Raw data cache HIT for key: {raw_cache_key}")
            cached_item = self._raw_data_cache[raw_cache_key]
            if file_suffix_lower in [".tif", ".tiff"]:
                data_to_slice_from, final_axes_order_str_for_slicing = cached_item
            else:
                data_to_slice_from = cached_item
        else:
            LOGGER.debug(f"Raw data cache MISS for key: {raw_cache_key}. Loading from disk.")

            # Temporary variables for data loaded from disk before final processing for cache
            intermediate_loaded_data: np.ndarray | None = None

            # --- Load full data from disk ---
            if file_suffix_lower in [".tif", ".tiff"]:
                with TiffFile(str(path_to_load)) as tif:
                    if not tif.series or not tif.series[0] or not tif.series[0].shape:
                        raise UnknownFileTypeError(
                            f"Cannot determine series shape for TIFF: {path_to_load}"
                        )
                    series = tif.series[0]
                    raw_data_from_tiff = series.asarray()  # Load ENTIRE series

                    axes_hint = source_cfg.expected_format_hint
                    shape_hint = source_cfg.expected_shape_hint
                    series_axes_str = series.axes.upper() if series.axes else ""
                    # This is the initial interpretation of axes, before permutation
                    current_axes = (axes_hint.upper() if axes_hint else series_axes_str) or ""

                    current_data_for_processing = raw_data_from_tiff

                    # --- Refined shape deduction (same as original) ---
                    intended_nd_shape = None
                    if axes_hint and raw_data_from_tiff.ndim < len(axes_hint):
                        # OME/ImageJ/shape_hint logic... (copied from original)
                        if hasattr(tif, "is_ome") and tif.is_ome:
                            ome_md = tif.ome_metadata
                            if isinstance(ome_md, dict):
                                image_metadata = ome_md.get("Image")
                                if isinstance(image_metadata, list):
                                    image_metadata = image_metadata[0]
                                if isinstance(image_metadata, dict) and "Pixels" in image_metadata:
                                    pixels = image_metadata["Pixels"]
                                    if isinstance(pixels, dict):
                                        try:
                                            intended_nd_shape = tuple(
                                                pixels[f"Size{ax}"] for ax in axes_hint.upper()
                                            )
                                        except (KeyError, OSError) as e:
                                            LOGGER.warning(
                                                f"OME metadata shape parsing failed: {e}"
                                            )
                            else:
                                LOGGER.warning(f"tif.ome_metadata not a dict for {path_to_load}")
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
                                pass  # intended_nd_shape remains None
                        elif shape_hint and len(shape_hint) == len(axes_hint):
                            intended_nd_shape = shape_hint

                    if intended_nd_shape and np.prod(raw_data_from_tiff.shape) == np.prod(
                        intended_nd_shape
                    ):
                        try:
                            current_data_for_processing = raw_data_from_tiff.reshape(
                                intended_nd_shape
                            )
                            current_axes = axes_hint  # Update current_axes if reshape based on hint
                            LOGGER.info(
                                f"Reshaped TIFF data to {intended_nd_shape} using hint "
                                f"'{axes_hint}'"
                            )
                        except (ValueError, OSError) as e:
                            LOGGER.warning(
                                f"Failed to reshape TIFF to {intended_nd_shape} for axes "
                                f"'{axes_hint}': {e}"
                            )
                    elif (
                        shape_hint
                        and raw_data_from_tiff.ndim == 3
                        and len(shape_hint) == len(axes_hint or "")
                        and np.prod(raw_data_from_tiff.shape) == np.prod(shape_hint)
                    ):
                        try:
                            current_data_for_processing = raw_data_from_tiff.reshape(shape_hint)
                            current_axes = axes_hint
                            LOGGER.info(
                                f"Reshaped 3D TIFF data to {shape_hint} using shape_hint and "
                                f"format_hint '{axes_hint}'"
                            )
                        except OSError as e:
                            LOGGER.warning(f"Failed to reshape 3D TIFF data: {e}")

                    # --- Permutation logic (copied & adapted from original) ---
                    permuted_data = current_data_for_processing
                    # final_axes_order_str_for_slicing will be the axes string *after* permutation
                    final_axes_order_str_for_slicing = current_axes

                    final_ndim = permuted_data.ndim
                    if current_axes and final_ndim > 0:
                        if final_ndim == 4 and current_axes == "TZYX":
                            permuted_data = permuted_data[:, np.newaxis, :, :, :]
                            final_axes_order_str_for_slicing = "TCDHW"
                        elif final_ndim == 5:
                            if current_axes == "TCZYX":
                                final_axes_order_str_for_slicing = "TCDHW"
                            elif current_axes == "TZCYX":
                                permuted_data = np.transpose(permuted_data, (0, 2, 1, 3, 4))
                                final_axes_order_str_for_slicing = "TCDHW"
                        else:
                            permutation_map = {
                                "ZYX": ((0, 1, 2), "DHW"),
                                "YXZ": ((1, 0, 2), "DHW"),
                                "XYZ": ((2, 0, 1), "DHW"),
                                "QYX": ((0, 1, 2), "DHW"),
                                "IYX": ((0, 1, 2), "DHW"),
                                "CZYX": ((0, 1, 2, 3), "CDHW"),
                                "ZCYX": ((1, 0, 2, 3), "CDHW"),
                                "CZXY": ((0, 1, 3, 2), "CDHW"),
                                "TZYX": (
                                    (0, 1, 2, 3),
                                    "CDHW",
                                ),  # Fallback if not 4D TZYX handled above
                                "TCZYX": ((0, 1, 2, 3, 4), "TCDHW"),
                                "TZCYX": ((0, 2, 1, 3, 4), "TCDHW"),
                            }
                            if current_axes in permutation_map:
                                perm_indices, target_axes = permutation_map[current_axes]
                                if len(perm_indices) == permuted_data.ndim:
                                    try:
                                        permuted_data = np.transpose(permuted_data, perm_indices)
                                        final_axes_order_str_for_slicing = target_axes
                                    except ValueError as e:
                                        LOGGER.error(
                                            f"Permutation from '{current_axes}' failed: {e}"
                                        )
                                # else: final_axes_order_str_for_slicing remains current_axes
                    elif not current_axes and permuted_data.ndim > 0:  # No axes info
                        if permuted_data.ndim == 5:
                            final_axes_order_str_for_slicing = "TCDHW"
                        elif permuted_data.ndim == 4:
                            final_axes_order_str_for_slicing = "CDHW"
                        elif permuted_data.ndim == 3:
                            final_axes_order_str_for_slicing = "DHW"
                        else:
                            final_axes_order_str_for_slicing = "?" * permuted_data.ndim

                    intermediate_loaded_data = permuted_data
                    # `final_axes_order_str_for_slicing` is now set

            elif file_suffix_lower in [".h5", ".hdf5"]:
                with h5py.File(str(path_to_load), "r") as f:
                    dataset_name = source_cfg.expected_format_hint or next(iter(f.keys()), None)
                    if not dataset_name or dataset_name not in f:
                        raise InvalidHDF5DatasetError(
                            f"HDF5: Dataset '{dataset_name}' not found in {path_to_load}"
                        )
                    dataset = f.get(dataset_name)
                    if not isinstance(dataset, h5py.Dataset):
                        raise InvalidHDF5DatasetError(
                            f"HDF5: Key '{dataset_name}' is not a Dataset in {path_to_load}"
                        )
                    intermediate_loaded_data = dataset[:]  # Load ENTIRE dataset
                # final_axes_order_str_for_slicing remains None for HDF5

            elif method:
                intermediate_loaded_data = method(path_to_load)

            else:
                raise UnknownFileTypeError(path_to_load)

            if intermediate_loaded_data is None:
                raise RuntimeError(f"Full data loading resulted in None for {path_to_load}")

            # Process for caching: convert to float32 and ensure C-contiguity
            data_to_slice_from = np.ascontiguousarray(intermediate_loaded_data.astype(np.float32))

            # Store in cache
            if file_suffix_lower in [".tif", ".tiff"]:
                self._raw_data_cache[raw_cache_key] = (
                    data_to_slice_from,
                    final_axes_order_str_for_slicing,
                )
            else:  # HDF5, custom
                self._raw_data_cache[raw_cache_key] = data_to_slice_from

        # --- Slicing Logic (applied to `data_to_slice_from`) ---
        begin_slice_user, end_slice_user = source_cfg.begin_slice, source_cfg.end_slice
        slicing_dim_idx = 0  # Default: slice along the first dimension

        if data_to_slice_from.ndim == 0:  # Should be caught earlier but good check
            raise UnknownFileTypeError(f"Data to slice is a scalar for {path_to_load}")

        # Determine slicing_dim_idx based on slice_dimension_name
        if source_cfg.slice_dimension_name:
            if file_suffix_lower in [".tif", ".tiff"] and source_cfg.expected_format_hint:
                original_hint_axes = source_cfg.expected_format_hint.upper()
                user_slice_dim_char = source_cfg.slice_dimension_name.upper()
                try:
                    slicing_dim_idx = original_hint_axes.index(user_slice_dim_char)
                    LOGGER.info(
                        f"TIFF: Slicing on user dimension '{user_slice_dim_char}' "
                        f"(index {slicing_dim_idx} from hint '{original_hint_axes}') "
                        f"for data with final axes '{final_axes_order_str_for_slicing}'."
                    )
                except ValueError:
                    LOGGER.warning(
                        f"TIFF: slice_dimension_name '{user_slice_dim_char}' not found in "
                        f"expected_format_hint '{original_hint_axes}'. "
                        "Attempting search in final axes."
                    )
                    if (
                        final_axes_order_str_for_slicing
                        and "?" not in final_axes_order_str_for_slicing
                    ):
                        try:
                            slicing_dim_idx = final_axes_order_str_for_slicing.index(
                                user_slice_dim_char
                            )
                            LOGGER.info(
                                f"TIFF: Slicing on user dimension '{user_slice_dim_char}' "
                                f"(index {slicing_dim_idx}) found in final axes "
                                f"'{final_axes_order_str_for_slicing}'."
                            )
                        except ValueError:
                            LOGGER.warning(
                                f"TIFF: slice_dimension_name '{user_slice_dim_char}' "
                                f"not found in final axes '{final_axes_order_str_for_slicing}'. "
                                "Slicing on dim 0."
                            )
                            slicing_dim_idx = 0
                    else:
                        LOGGER.warning(
                            f"TIFF: Cannot use slice_dimension_name '{user_slice_dim_char}' "
                            f"as final axes unknown or unusable. Slicing on dim 0."
                        )
                        slicing_dim_idx = 0
            elif file_suffix_lower in [".h5", ".hdf5"]:  # HDF5 or custom (if name is int)
                try:
                    slicing_dim_idx = int(source_cfg.slice_dimension_name)
                    if not (0 <= slicing_dim_idx < data_to_slice_from.ndim):
                        LOGGER.warning(
                            f"HDF5/Custom: slice_dimension_name index "
                            f"{slicing_dim_idx} out of bounds "
                            f"for ndim {data_to_slice_from.ndim}. Slicing on dim 0."
                        )
                        slicing_dim_idx = 0
                except ValueError:
                    LOGGER.warning(
                        f"HDF5/Custom: slice_dimension_name '{source_cfg.slice_dimension_name}' "
                        f"is not a valid integer index. Slicing on dim 0."
                    )
                    slicing_dim_idx = 0
            # else custom method without integer slice_dimension_name also defaults to 0

        if not (0 <= slicing_dim_idx < data_to_slice_from.ndim):  # Final safety check
            raise ConfigError(
                f"Slicing dimension index {slicing_dim_idx} "
                f"out of bounds for data ndim {data_to_slice_from.ndim}"
            )

        num_elements_in_slice_dim = data_to_slice_from.shape[slicing_dim_idx]
        actual_begin_slice = begin_slice_user
        actual_end_slice = num_elements_in_slice_dim if end_slice_user == -1 else end_slice_user

        if not (
            0 <= actual_begin_slice < num_elements_in_slice_dim
            and actual_begin_slice < actual_end_slice <= num_elements_in_slice_dim
        ):
            current_axes_for_error = (
                final_axes_order_str_for_slicing if final_axes_order_str_for_slicing else "N/A"
            )
            raise InvalidSliceRangeError(
                f"{path_to_load} (shape: {data_to_slice_from.shape}, "
                f"slicing_dim_idx: {slicing_dim_idx}, axes: '{current_axes_for_error}')",
                actual_begin_slice,
                actual_end_slice,
            )

        slice_obj_list = [slice(None)] * data_to_slice_from.ndim
        slice_obj_list[slicing_dim_idx] = slice(actual_begin_slice, actual_end_slice)

        sliced_data = data_to_slice_from[tuple(slice_obj_list)]

        # Ensure the final returned array is C-contiguous
        final_data = np.ascontiguousarray(sliced_data)  # data_to_slice_from was already float32

        s_dim_char = "?"
        if final_axes_order_str_for_slicing and slicing_dim_idx < len(
            final_axes_order_str_for_slicing
        ):
            s_dim_char = final_axes_order_str_for_slicing[slicing_dim_idx]

        LOGGER.debug(
            f"Sliced on dim_idx {slicing_dim_idx} ('{s_dim_char}') "
            f"from {actual_begin_slice} to {actual_end_slice}. "
            f"Original full shape: {data_to_slice_from.shape}, "
            f"Final sliced shape: {final_data.shape}, Contiguous: {final_data.flags.c_contiguous}"
        )
        return final_data

    def load_data_only(self, method: Callable | None = None) -> SSUnetData:
        data_arr = self._load_from_source(self.data, method)
        if data_arr is None:
            raise NoDataFileAvailableError("Primary data file could not be loaded.")
        return SSUnetData(primary_data=data_arr)

    def load_reference_only(self, method: Callable | None = None) -> SSUnetData:
        if not self.reference or not self.reference.is_available:
            LOGGER.info("Reference not configured/available, loading primary data instead.")
            return self.load_data_only(method)
        ref_arr = self._load_from_source(self.reference, method)
        if ref_arr is None:
            LOGGER.warning("Configured reference file load failed. Falling back to primary data.")
            return self.load_data_only(method)
        return SSUnetData(primary_data=ref_arr)

    def load_data_and_ground_truth(self, method: Callable | None = None) -> SSUnetData:
        data_arr = self._load_from_source(self.data, method)
        if data_arr is None:
            raise NoDataFileAvailableError("Primary data file could not be loaded.")

        gt_arr = self._load_from_source(self.ground_truth, method)
        if gt_arr is None and self.reference and self.reference.is_available:
            LOGGER.info("Ground truth not available/loaded, trying reference.")
            gt_arr = self._load_from_source(self.reference, method)

        if gt_arr is None:
            LOGGER.warning(
                "Neither ground truth nor reference available/loaded. Secondary data is None."
            )

        normalized_gt_arr = self._normalize_ground_truth(gt_arr) if gt_arr is not None else None
        return SSUnetData(primary_data=data_arr, secondary_data=normalized_gt_arr)

    def load_reference_and_ground_truth(self, method: Callable | None = None) -> SSUnetData:
        ref_arr = self._load_from_source(self.reference, method)
        if ref_arr is None:
            LOGGER.info("Reference not available/loaded, trying primary data as reference.")
            ref_arr = self._load_from_source(self.data, method)
            if ref_arr is None:
                raise NoDataFileAvailableError(
                    "Neither reference nor primary data could be loaded."
                )

        gt_arr = self._load_from_source(self.ground_truth, method)
        if gt_arr is None:
            LOGGER.info(
                "Ground truth not available/loaded, using loaded reference/primary as ground truth."
            )
            gt_arr = ref_arr  # ref_arr is guaranteed non-None here

        normalized_gt_arr = self._normalize_ground_truth(gt_arr)  # gt_arr non-None
        return SSUnetData(primary_data=ref_arr, secondary_data=normalized_gt_arr)

    def _normalize_ground_truth(self, ground_truth: np.ndarray | None) -> np.ndarray | None:
        if ground_truth is None:
            return None
        gt_float = ground_truth.astype(np.float32)  # Already float32 from _load_from_source
        gt_min, gt_max = np.min(gt_float), np.max(gt_float)

        if abs(gt_max - gt_min) < EPSILON:
            LOGGER.debug("Ground truth is constant. Returning as is.")
            return gt_float
        if 1.0 < gt_max <= 255.0 and gt_min >= 0:
            LOGGER.debug("Normalizing ground truth (approx 8-bit) to [0,1].")
            return (gt_float - gt_min) / (gt_max - gt_min + EPSILON)
        elif gt_max > 255.0 and gt_min >= 0:
            LOGGER.debug("Normalizing ground truth (>8-bit) to [0,1].")
            return (gt_float - gt_min) / (gt_max - gt_min + EPSILON)
        LOGGER.debug("Ground truth in [0,1] or has negatives. Returning as is.")
        return gt_float
