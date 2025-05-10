import argparse  # For command-line arguments
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np  # For np.prod
import tifffile

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def get_tiff_dimensions_and_metadata(filepath: str | Path) -> dict | None:
    """
    Inspects a TIFF file and returns its key dimensional information and metadata.
    (Implementation from previous response - no changes needed here)
    """
    try:
        filepath = Path(filepath)
        if not filepath.is_file():
            LOGGER.error(f"File not found: {filepath}")
            return None

        with tifffile.TiffFile(str(filepath)) as tif:
            info = {"filepath": str(filepath)}

            info["num_pages"] = len(tif.pages)

            try:
                img_array = tifffile.imread(str(filepath))  # Basic read
                info["raw_imread_shape"] = img_array.shape
                del img_array
            except OSError as e:
                info["raw_imread_shape"] = f"Error reading with imread: {e}"

            if tif.series and len(tif.series) > 0:
                info["series_count"] = len(tif.series)
                series0 = tif.series[0]
                info["series_axes"] = series0.axes.upper() if series0.axes else "N/A"
                info["series_shape"] = series0.shape
                info["series_dtype"] = str(series0.dtype)
                info["series_ndim"] = series0.ndim

                if info.get("raw_imread_shape") != "N/A" and np.prod(series0.shape) != np.prod(
                    info.get("raw_imread_shape", 0)
                ):
                    LOGGER.warning(
                        f"Product of series shape {series0.shape} does not match "
                        f"product of imread shape {info.get('raw_imread_shape')}"
                    )
            else:
                info["series_count"] = 0
                info["series_axes"] = "N/A"
                info["series_shape"] = "N/A"
                info["series_dtype"] = "N/A"
                info["series_ndim"] = "N/A"

            if tif.pages:
                first_page = tif.pages[0]
                info["first_page_shape"] = first_page.shape
                info["first_page_axes"] = first_page.axes.upper() if first_page.axes else "N/A"
                info["first_page_dtype"] = str(first_page.dtype)
                info["first_page_samplesperpixel"] = first_page.samplesperpixel
                info["first_page_photometric"] = str(first_page.photometric)

            info["imagej_metadata"] = tif.imagej_metadata if tif.is_imagej else None
            info["ome_metadata_present"] = tif.is_ome

            return info

    except Exception as e:
        LOGGER.error(f"Error inspecting TIFF file {filepath}: {e}", exc_info=True)
        return None


def display_tiff_info_gui(info: dict | None):
    """
    Displays TIFF info in a simple Tkinter window.
    (Implementation from previous response - no changes needed here)
    """
    if info is None:
        # This case should ideally be handled before calling display_tiff_info_gui
        # or by passing a specific error message.
        info_str = "Could not retrieve TIFF information or no file processed."
    elif "error" in info and info.get("filepath"):  # Special case for file not found via GUI
        info_str = f"File: {info.get('filepath', 'N/A')}\n\nError: {info['error']}"
    elif "error" in info:
        info_str = f"Error: {info['error']}"
    else:  # Normal info display
        info_str = f"File: {info.get('filepath', 'N/A')}\n\n"
        info_str += "--- General ---\n"
        info_str += f"  Total Pages: {info.get('num_pages', 'N/A')}\n"
        info_str += f"  imread() Shape: {info.get('raw_imread_shape', 'N/A')}\n\n"

        info_str += "--- Tifffile Series (Primary Interpretation) ---\n"
        info_str += f"  Series Count: {info.get('series_count', 'N/A')}\n"
        if info.get("series_count", 0) > 0:
            info_str += f"  Series[0] Axes: {info.get('series_axes', 'N/A')}\n"
            info_str += f"  Series[0] Shape: {info.get('series_shape', 'N/A')}\n"
            info_str += f"  Series[0] Ndim: {info.get('series_ndim', 'N/A')}\n"
            info_str += f"  Series[0] Dtype: {info.get('series_dtype', 'N/A')}\n\n"
        else:
            info_str += "  No series found by tifffile.\n\n"

        info_str += "--- First Page Details ---\n"
        info_str += f"  Shape: {info.get('first_page_shape', 'N/A')}\n"
        info_str += f"  Axes: {info.get('first_page_axes', 'N/A')}\n"
        info_str += f"  Dtype: {info.get('first_page_dtype', 'N/A')}\n"
        info_str += f"  SamplesPerPixel: {info.get('first_page_samplesperpixel', 'N/A')}\n"
        info_str += f"  Photometric: {info.get('first_page_photometric', 'N/A')}\n\n"

        info_str += "--- Specific Metadata ---\n"
        info_str += (
            f"  ImageJ Metadata: {'Present' if info.get('imagej_metadata') else 'Not Present'}\n"
        )
        if info.get("imagej_metadata"):
            imgj_m = info["imagej_metadata"]
            imgj_info = []
            if "axes" in imgj_m:
                imgj_info.append(f"axes: {imgj_m['axes']}")
            if "channels" in imgj_m:
                imgj_info.append(f"channels: {imgj_m['channels']}")
            if "slices" in imgj_m:
                imgj_info.append(f"slices: {imgj_m['slices']}")
            if "frames" in imgj_m:
                imgj_info.append(f"frames: {imgj_m['frames']}")
            if imgj_info:
                info_str += f"    Details: {', '.join(imgj_info)}\n"
            else:
                info_str += f"    Raw ImageJ Meta (first 200 chars): {str(imgj_m)[:200]}...\n"

        info_str += (
            f"  OME Metadata: {'Present' if info.get('ome_metadata_present') else 'Not Present'}\n"
        )

    popup = tk.Tk()
    popup.title("TIFF Dimensions Inspector")

    text_area = tk.Text(popup, wrap=tk.WORD, width=80, height=25, font=("Courier New", 9))
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    text_area.insert(tk.END, info_str)
    text_area.config(state=tk.DISABLED)

    scrollbar = ttk.Scrollbar(text_area, orient=tk.VERTICAL, command=text_area.yview)
    text_area["yscrollcommand"] = scrollbar.set
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    close_button = ttk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack(pady=10)

    popup.mainloop()


def process_file(filepath_str: str | Path):
    """Processes a single file path and displays info."""
    filepath = Path(filepath_str)
    LOGGER.info(f"Inspecting TIFF file: {filepath.name}")

    dim_info = get_tiff_dimensions_and_metadata(filepath)

    if dim_info:
        LOGGER.info("\n--- TIFF Inspection Report (Console) ---")
        for key, value in dim_info.items():
            if key == "imagej_metadata" and value:
                LOGGER.info("  ImageJ Metadata: Present")
                simple_imgj = {
                    k: v for k, v in value.items() if isinstance(v, str | int | float | bool)
                }
                LOGGER.info(f"    (Simplified): {simple_imgj}")
            else:
                LOGGER.info(f"  {key.replace('_', ' ').capitalize()}: {value}")
        LOGGER.info("--------------------------------------\n")
        display_tiff_info_gui(dim_info)
    else:
        msg = f"Could not inspect TIFF file: {filepath}"
        LOGGER.info(msg)
        display_tiff_info_gui({"filepath": str(filepath), "error": msg})


def main():
    """
    Main execution function. Parses command-line arguments or uses GUI for file selection.
    """
    parser = argparse.ArgumentParser(description="Inspect TIFF file dimensions and metadata.")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the TIFF file to inspect.",
        default=None,  # Default to None, so we can check if it was provided
    )
    args = parser.parse_args()

    if args.path:
        process_file(args.path)
    else:
        LOGGER.info("No file path provided via command line. Opening GUI file dialog...")
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        gui_filepath = filedialog.askopenfilename(
            title="Select TIFF file",
            filetypes=(("TIFF files", "*.tif *.tiff"), ("All files", "*.*")),
        )
        if gui_filepath:  # If a file was selected
            LOGGER.info(f"File selected via GUI: {gui_filepath}")
            process_file(gui_filepath)
        else:
            LOGGER.info("No file selected via GUI. Exiting.")
            # Optionally show a message in a simple GUI window
            display_tiff_info_gui({"error": "No file was selected."})


if __name__ == "__main__":
    main()
