# WamIPEDensity.jl

`WamIPEDensity.jl` is the density-focused WAM-IPE access layer used by Space Falcon Lab tooling. It downloads NOAA WAM-IPE NetCDF products, keeps them in a local cache, interpolates them in space and time, and exposes a small Julia API for point, batch, and trajectory density queries.

The package is designed to be useful in two ways:

- as a standalone data access package for WAM-IPE neutral density;
- as a SpaceAGORA atmosphere backend through `SpaceAGORA.WAMIPEDensityAtmosphereModel`.

The examples below use UTC `DateTime` values, latitude and longitude in degrees, and altitude in kilometres unless noted otherwise.

```julia
using Dates
using WamIPEDensity

itp = WAMInterpolator(product="wfs", root_prefix="v1.2", varname="den", interpolation=:sciml)
dt = DateTime(2025, 9, 30, 18, 12, 22)

rho = get_density(itp, dt, -33.4, -153.24, 400.0)
```

## What the package provides

- `WAMInterpolator`: configuration for the NOAA WAM-IPE forecast or nowcast product.
- `get_density`: one density value at one time and location.
- `get_density_batch`: repeated point queries.
- `get_density_trajectory_optimised`: trajectory queries grouped by the required WAM-IPE files.
- `GEOSFPInterpolator`, `NRLMSISEInterpolator`, and `HybridDensityInterpolator`: optional model paths for lower atmosphere, empirical atmosphere, and altitude-aware hybrid use cases.

## Pages

- [Quick Start](quickstart.md): installation and standalone examples.
- [WFS Workflows](wfs-workflows.md): the independent `WFS.get_value`, `dump_sample`, and `dump_all` programs.
- [How It Works](how-it-works.md): cache, file selection, interpolation, and trajectory flow.
- [SpaceAGORA Integration](spaceagora.md): direct use inside `SpaceAGORA.jl`.
- [API Reference](api.md): generated reference for public functions and types.
