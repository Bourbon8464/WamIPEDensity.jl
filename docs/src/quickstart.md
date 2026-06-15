# Quick Start

## Installation

From a local checkout:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

For development from another Julia project:

```julia
using Pkg
Pkg.develop(path="/path/to/WamIPEDensity.jl")
```

## Point density query

Use `WAMInterpolator` to choose the WAM-IPE product, variable, and interpolation method. `wfs` is the forecast stream and `wrs` is the real-time nowcast stream.

```julia
using Dates
using WamIPEDensity

itp = WAMInterpolator(
    product="wfs",
    root_prefix="v1.2",
    varname="den",
    interpolation=:sciml,
)

dt = DateTime(2025, 9, 30, 18, 12, 22)
lat_deg = -33.4
lon_deg = -153.24
alt_km = 400.0

rho = get_density(itp, dt, lat_deg, lon_deg, alt_km)
```

## Nowcast product

```julia
using Dates
using WamIPEDensity

itp = WAMInterpolator(product="wrs", root_prefix="v1.2", varname="den", interpolation=:sciml)
dt = DateTime(2025, 9, 30, 18, 12, 22)

rho = get_density(itp, dt, -33.4, -153.24, 400.0)
```

## Batch query

```julia
using Dates
using WamIPEDensity

itp = WAMInterpolator(product="wfs", varname="den", interpolation=:sciml)

dt0 = DateTime(2025, 9, 30, 18, 0, 0)
dts = [dt0 + Minute(10) * i for i in 0:5]
lats = fill(-33.4, length(dts))
lons = fill(-153.24, length(dts))
alts_km = fill(400.0, length(dts))

rhos = get_density_batch(itp, dts, lats, lons, alts_km)
```

## Trajectory query

`get_density_trajectory_optimised` accepts altitude in metres. By default latitude and longitude are radians, which matches many orbit libraries. Set `angles_in_deg=true` if your trajectory arrays are already degrees.

```julia
using Dates
using WamIPEDensity

itp = WAMInterpolator(product="wfs", varname="den", interpolation=:sciml)

dts = [DateTime(2025, 9, 30, 18, 0, 0) + Minute(10) * i for i in 0:11]
lats_deg = range(-34.0, -32.0; length=length(dts))
lons_deg = range(-154.0, -152.0; length=length(dts))
alts_m = fill(400_000.0, length(dts))

rhos = get_density_trajectory_optimised(
    itp,
    dts,
    collect(lats_deg),
    collect(lons_deg),
    alts_m;
    angles_in_deg=true,
)
```

## Cache behavior

The package downloads public NOAA WAM-IPE NetCDF files into a local cache and reuses open datasets through an internal file handle pool. The default cache path is `./cache`.

```julia
print_cache_stats()
set_max_open_datasets!(25)
clear_grid_cache!()
```

Use `prewarm_cache!` when a simulation will need a known time range and you want to download required files before the main run starts.

```julia
prewarm_cache!(itp, dts)
```
