# SpaceAGORA Integration

`SpaceAGORA.jl` already contains a direct adapter for this package:

```julia
SpaceAGORA.WAMIPEDensityAtmosphereModel
```

The adapter constructs a `WamIPEDensity.WAMInterpolator` and calls:

- `get_density` for scalar density requests;
- `get_density_batch` for repeated point requests;
- `get_density_trajectory_optimised` for trajectory batches.

## Add the package to SpaceAGORA

From the `SpaceAGORA.jl` environment:

```julia
using Pkg
Pkg.activate("/path/to/SpaceAGORA.jl")
Pkg.develop(path="/path/to/WamIPEDensity.jl")
Pkg.instantiate()
```

## Construct the SpaceAGORA density model

```julia
using SpaceAGORA
using WamIPEDensity

density_model = SpaceAGORA.WAMIPEDensityAtmosphereModel(
    product="wfs",
    root_prefix="v1.2",
    varname="den",
    interpolation=:sciml,
    temperature_k=1000.0,
)
```

`temperature_k` is a SpaceAGORA placeholder because WAM-IPE density output supplies neutral density, not a full thermodynamic atmosphere state. The adapter returns zero wind.

## Evaluate through SpaceAGORA's density hook

SpaceAGORA's atmosphere hook uses altitude in metres, latitude and longitude in radians, and elapsed simulation time in seconds.

```julia
using Dates
using SpaceAGORA
using WamIPEDensity

model = SpaceAGORA.WAMIPEDensityAtmosphereModel(
    product="wfs",
    varname="den",
    interpolation=:sciml,
)

h_m = 400_000.0
lat_rad = deg2rad(-33.4)
lon_rad = deg2rad(-153.24)
elapsed_seconds = 0.0

rho, temperature_k, wind_vec = SpaceAGORA.getDensity(
    model,
    h_m,
    lat_rad,
    lon_rad,
    elapsed_seconds,
    false,
)
```

When SpaceAGORA passes the simulation parameter bundle, the adapter uses `p.args.initial_time` plus elapsed seconds to recover the physical UTC `DateTime`.

## Use in a SpaceAGORA environment

Use the WAM-IPE model anywhere SpaceAGORA expects an atmosphere or density model. The exact field name depends on the environment constructor being used, but the pattern is:

```julia
using Dates
using SpaceAGORA
using WamIPEDensity

initial_time = DateTime(2025, 9, 30, 18, 12, 22)

atmosphere = SpaceAGORA.WAMIPEDensityAtmosphereModel(
    product="wfs",
    varname="den",
    interpolation=:sciml,
)

# Example shape. Use the same environment constructor your mission script
# already uses, and pass `atmosphere` as its density/atmosphere model.
# env = make_no_gram_environment(earth; atmosphere_model=atmosphere, initial_time=initial_time)
```

## Higher-level WFS variables beside a SpaceAGORA run

SpaceAGORA's current direct adapter is density-focused. If you also need WFS ion or electron temperature values, query them as side-car data at the same timestamps used by the simulation:

```julia
using Dates
using WFS

wfs_itp = WFS.WFSInterpolator(
    product="wfs",
    stream="ipe10",
    varname="ion_temperature",
    interpolation=:sciml,
    cache_dir="cache",
)

dt = DateTime(2025, 9, 30, 18, 12, 22)
lon_deg = -153.24
lat_deg = -33.4
alt_km = 400.0

ion_temperature = WFS.get_value(wfs_itp, dt, lon_deg, lat_deg, alt_km)
```

For density-driven force models, prefer `WAMIPEDensityAtmosphereModel`. For diagnostic ionosphere variables, keep the `WFS` query explicit so it is clear which variable is being sampled.
