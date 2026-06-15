# WFS Workflows

The following examples are the independent `WFS` workflows you can run directly. They are useful when you want ionosphere variables, sample dumps, or all-product reports without going through a full SpaceAGORA simulation.

These examples assume the higher-level `WFS` package is available in the active Julia environment. They use a `cache` folder beside the working directory.

## Ion temperature from forecast data

```julia
print("\033c")
using Revise, Dates, WFS

itp = WFS.WFSInterpolator(
    product="wfs",
    stream="ipe10",
    varname="ion_temperature",
    interpolation=:sciml,
    cache_dir="cache",
)

dt = DateTime(2025, 9, 30, 18, 12, 22)
value = WFS.get_value(itp, dt, -153.24, -33.4, 400.0)
```

## Electron temperature from nowcast data

```julia
print("\033c")
using Revise, Dates, WFS

itp = WFS.WFSInterpolator(
    product="wrs",
    stream="ipe10",
    varname="electron_temperature",
    interpolation=:sciml,
    cache_dir="cache",
)

dt = DateTime(2025, 9, 30, 18, 12, 22)
value = WFS.get_value(itp, dt, -153.24, -33.4, 400.0)
```

## Sample dump from `ipe05`

```julia
print("\033c")
using Revise, WFS, Dates

itp = WFS.WFSInterpolator(product="wfs", stream="ipe05", cache_dir="cache")
dt = DateTime(2025, 9, 28, 18, 12, 22)

rep = WFS.dump_sample(itp, dt, 16, 16, 700.0)
```

## Sample dump from `ipe10`

```julia
print("\033c")
using Revise, WFS, Dates

itp = WFS.WFSInterpolator(product="wfs", stream="ipe10", cache_dir="cache")
dt = DateTime(2025, 9, 29, 18, 12, 22)

rep = WFS.dump_sample(itp, dt, 16, 16, 700.0)
```

```julia
print("\033c")
using Revise, WFS, Dates

itp = WFS.WFSInterpolator(product="wfs", stream="ipe10", cache_dir="cache")
dt = DateTime(2025, 9, 28, 18, 10, 00)

rep = WFS.dump_sample(itp, dt, 16, 16, 700.0)
```

## Dump every configured report

```julia
print("\033c")
using WFS, Dates

dt = DateTime(2025, 9, 28, 18, 20, 00)
lon = 16
lat = 16
alt = 700.0

all_reports = WFS.dump_all(dt, lon, lat, alt)
```

## Coordinate order

The `WFS.get_value` examples above pass longitude first and latitude second:

```julia
WFS.get_value(itp, dt, lon_deg, lat_deg, alt_km)
```

`WamIPEDensity.get_density` passes latitude first and longitude second:

```julia
get_density(itp, dt, lat_deg, lon_deg, alt_km)
```

Keep that distinction visible when moving snippets between standalone WFS scripts and the density-only SpaceAGORA adapter.
