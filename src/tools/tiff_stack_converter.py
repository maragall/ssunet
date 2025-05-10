import argparse
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog

import numpy as np
import tifffile

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def convert_3d_to_4d_tiff_stack(input_filepath_str: str | Path, t_factor: int):
    """
    Converts a 3D TIFF stack into a 4D TIFF stack by reshaping the first dimension,
    and also saves a mean projection across the new 'T' dimension.

    Args:
        input_filepath_str: Path to the input 3D TIFF file.
        t_factor: The number of time points (T) to create.
                  The original depth will be divided by this factor.
    """
    try:
        input_filepath = Path(input_filepath_str)
        if not input_filepath.is_file():
            LOGGER.error(f"Input file not found: {input_filepath}")
            return

        LOGGER.info(f"Reading input TIFF: {input_filepath}")
        original_stack = tifffile.imread(str(input_filepath))

        if original_stack.ndim != 3:
            LOGGER.error(
                f"Input TIFF is not 3-dimensional. Expected (D, H, W), got {original_stack.shape}."
            )
            return

        original_depth, height, width = original_stack.shape
        LOGGER.info(
            f"Original stack dimensions: Depth={original_depth}, Height={height}, Width={width}"
        )

        if t_factor <= 0:
            LOGGER.error("t_factor must be a positive integer.")
            return
        if t_factor > original_depth:
            LOGGER.error(
                f"t_factor ({t_factor}) cannot be greater than original depth ({original_depth})."
            )
            return

        # Calculate new Z dimension and effective depth for reshaping
        new_z_dim = original_depth // t_factor
        effective_depth_for_reshape = new_z_dim * t_factor

        if effective_depth_for_reshape == 0:  # handles t_factor > original_depth
            LOGGER.error(
                f"Resulting new Z dimension is 0. t_factor ({t_factor}) is too large "
                f"for original depth ({original_depth})."
            )
            return

        if effective_depth_for_reshape < original_depth:
            LOGGER.warning(
                f"Original depth ({original_depth}) is not perfectly divisible "
                f"by t_factor ({t_factor}). "
                f"Trailing {original_depth - effective_depth_for_reshape} slices "
                f"will be truncated. Using first {effective_depth_for_reshape} slices."
            )
            stack_to_reshape = original_stack[:effective_depth_for_reshape, :, :]
        else:
            stack_to_reshape = original_stack

        # Reshape to (T, Z_new, H, W)
        # Original (D,H,W) needs to become (T,Z,H,W)
        # This requires D = T*Z.
        # We have stack_to_reshape which is (T*Z_new, H, W)
        # We want to view it as (T, Z_new, H, W)
        try:
            reshaped_4d_stack = stack_to_reshape.reshape(t_factor, new_z_dim, height, width)
            LOGGER.info(
                f"Reshaped to 4D stack: T={t_factor}, Z_new={new_z_dim}, H={height}, W={width}"
            )
        except ValueError as e:
            LOGGER.error(
                f"Could not reshape stack. Original shape: {stack_to_reshape.shape}, "
                f"Target T={t_factor}, Z_new={new_z_dim}. Error: {e}"
            )
            return

        # --- Save the 4D stack ---
        output_filename_4d = (
            f"{input_filepath.stem}_T{t_factor}_Z{new_z_dim}{input_filepath.suffix}"
        )
        output_filepath_4d = input_filepath.parent / output_filename_4d

        LOGGER.info(f"Saving 4D stack to: {output_filepath_4d}")
        try:
            # tifffile.imwrite can save N-D arrays.
            # For ImageJ compatibility for 5D-like (T,Z,C,Y,X), metadata is useful.
            # Here we have (T,Z,Y,X) effectively, if we consider Z as the new "depth" and T as time.
            # Providing axes metadata helps other software interpret it.
            tifffile.imwrite(
                output_filepath_4d,
                reshaped_4d_stack,
                imagej=True,  # Helps ImageJ recognize it as a hyperstack
                metadata={"axes": "TZYX"},  # T=time, Z=slices_per_timepoint, Y=height, X=width
            )
            LOGGER.info("4D stack saved successfully.")
        except Exception as e:
            LOGGER.error(f"Error saving 4D stack: {e}", exc_info=True)
            return

        # --- Calculate and save mean projection ---
        LOGGER.info("Calculating mean projection across 'T' dimension...")
        mean_projected_stack = np.mean(
            reshaped_4d_stack.astype(np.float32), axis=0
        )  # Mean along T (axis 0)
        mean_projected_stack = mean_projected_stack / np.max(mean_projected_stack) * 255
        mean_projected_stack = mean_projected_stack.astype(np.uint8)
        LOGGER.info(
            f"Mean projected stack shape: {mean_projected_stack.shape}"
        )  # Should be (Z_new, H, W)

        output_filename_mean = f"{input_filepath.stem}_mean_T{t_factor}{input_filepath.suffix}"
        output_filepath_mean = input_filepath.parent / output_filename_mean

        LOGGER.info(f"Saving mean projected stack to: {output_filepath_mean}")
        try:
            # This is a 3D stack (Z_new, H, W)
            tifffile.imwrite(
                output_filepath_mean,
                mean_projected_stack.astype(original_stack.dtype),  # Save with original dtype
                imagej=True,  # Optional, for consistency
                metadata={"axes": "ZYX"},
            )
            LOGGER.info("Mean projected stack saved successfully.")
        except Exception as e:
            LOGGER.error(f"Error saving mean projected stack: {e}", exc_info=True)

    except FileNotFoundError:
        LOGGER.error(f"Input file not found: {input_filepath_str}")
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 3D TIFF stack to a 4D (T,Z,H,W) "
        "stack and create a T-mean projection."
    )
    parser.add_argument(
        "--input_path", type=str, help="Path to the input 3D TIFF file.", default=None
    )
    parser.add_argument(
        "--t_factor",
        type=int,
        required=False,  # Make it not strictly required if GUI will ask
        help="Number of time points (T) to create. Original depth will be divided by this.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    t_factor = args.t_factor

    if not input_path:
        LOGGER.info("No input file path provided via command line. Opening GUI file dialog...")
        root_gui = tk.Tk()
        root_gui.withdraw()  # Hide the main tkinter window
        input_path = filedialog.askopenfilename(
            title="Select 3D TIFF stack input file",
            filetypes=(("TIFF files", "*.tif *.tiff"), ("All files", "*.*")),
        )
        if not input_path:
            LOGGER.info("No input file selected via GUI. Exiting.")
            return
        LOGGER.info(f"Input file selected via GUI: {input_path}")

    if t_factor is None:  # If not provided via CLI and GUI was used for path
        root_gui_t = tk.Tk()
        root_gui_t.withdraw()
        t_factor_str = simpledialog.askstring(
            "Input T Factor",
            "Enter the number of time points (T factor):",
            parent=root_gui_t,  # Make dialog appear on top
        )
        root_gui_t.destroy()  # Destroy the hidden root for t_factor input

        if t_factor_str:
            try:
                t_factor = int(t_factor_str)
            except ValueError:
                LOGGER.error(f"Invalid T factor entered: '{t_factor_str}'. Must be an integer.")
                return
        else:
            LOGGER.info("No T factor provided. Exiting.")
            return

    if input_path and t_factor is not None:  # Ensure t_factor was successfully obtained
        convert_3d_to_4d_tiff_stack(input_path, t_factor)
    else:
        LOGGER.error("Missing input path or T factor. Cannot proceed.")


if __name__ == "__main__":
    main()
