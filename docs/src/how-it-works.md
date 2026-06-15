# How It Works

## Query flow

A single `get_density` call follows this path:

1. Validate the requested time, latitude, longitude, altitude, and interpolation mode.
2. Select the WAM-IPE version folder for the requested time.
3. Build the NOAA S3 keys for the two nearest 10-minute files.
4. Download missing NetCDF files into the local cache.
5. Open those files through the dataset pool.
6. Load coordinate grids and the requested variable.
7. Interpolate each file in latitude, longitude, altitude, and file-local time.
8. Blend the two file values linearly to the requested `DateTime`.

## Product selection

`WAMInterpolator(product="wfs")` uses forecast data. `WAMInterpolator(product="wrs")` uses real-time nowcast data. Internally, the package knows the archive/cycle layout for each product and can resolve the exact file names needed for 10-minute WAM-IPE output cadence.

The package also chooses the correct WAM-IPE root version for historical dates:

- `v1.1` from 2023-03-20 21:10:00 through 2023-06-30 21:00:00.
- `v1.2` from 2023-06-30 21:10:00 onward.

## Cache layers

There are two cache layers:

- the on-disk file cache stores downloaded NOAA NetCDF files and a small `metadata.bin` index;
- the in-process dataset pool keeps recently used NetCDF handles open and evicts least-recently-used unpinned files.

This is why repeated trajectory queries are much faster after the first call: file downloads, grid decoding, and dataset opens are reused where possible.

## Interpolation modes

The public constructor accepts:

- `:nearest`
- `:linear`
- `:logz_linear`
- `:logz_quadratic`
- `:sciml`

`:sciml` is normalized onto the package's interpolation implementation so callers can use the same setting across related Space Falcon Lab packages.

## Trajectory optimization

`get_density_trajectory_optimised` groups trajectory points by the pair of WAM-IPE files required for each point. Each file pair is loaded once, and every point in that group is evaluated before moving to the next pair.

Use this path for orbit propagation or any dense time series. Use `get_density` for one-off debugging or interactive inspection.
