import xarray as xr
import glob
import numpy as np
import re


def extract_year(filename):
    """
    Extract year from filename like: yield_1981.nc4
    """
    match = re.search(r"\d{4}", filename)
    if not match:
        raise ValueError(f"Year not found in filename: {filename}")
    return int(match.group())


def load_crop_folder(
    folder_path,
    var_name="var",
    new_name="yield"
):
    """
    Load yearly NetCDF files into a single xarray Dataset
    with a proper time dimension.
    """

    files = sorted(glob.glob(f"{folder_path}/*.nc4"))

    if not files:
        raise ValueError(f"No .nc4 files found in {folder_path}")

    datasets = []

    for f in files:
        year = extract_year(f)

        ds = xr.open_dataset(f)

        da = ds[var_name]

        # Replace fill values with NaN
        fill = da.attrs.get("_FillValue")
        if fill is not None:
            da = da.where(da != fill)

        # Add time coordinate
        da = da.expand_dims(time=[year])

        datasets.append(da)

    combined = xr.concat(datasets, dim="time")

    # Rename variable for clarity
    combined = combined.rename(new_name)

    # Ensure consistent dtype
    combined = combined.astype("float32")

    return combined
