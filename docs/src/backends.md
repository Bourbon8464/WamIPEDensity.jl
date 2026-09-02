# Backends

`WamIPEDensity.jl` exposes four density backends, each reached through a
configuration struct. This page documents what each backend does, the
altitude window it serves, its data source, and any caveats that affect
accuracy or availability.

| Backend | Struct | Altitude range (defaults) | Data source | Cache directory |
|---|---|---|---|---|
| WAM-IPE | `WAMInterpolator` | unbounded; practically LEO altitudes | NOAA WAM-IPE on Amazon S3 (`noaa-nws-wam-ipe-pds`, us-east-1) | `./cache` |
| GEOS-FP | `GEOSFPInterpolator` | `0.0` - `70.0` km (configurable via `min_alt_km` / `max_alt_km`) | NASA NCCS GEOS-FP data-share portal over HTTP | `./cache_geosfp` |
| NRLMSISE-00 | `NRLMSISEInterpolator` | `0.0` - `100.0` km (configurable) | Empirical model + space-weather indices from `SpaceIndices.jl` | (none; `≈/.julia/...` for indices) |
| Hybrid | `HybridDensityInterpolator` | 0 km - LEO (routes by altitude) | Combination of the above | Combination of the above |

---

## WAM-IPE

The Whole Atmosphere Model + Ionosphere-Plasmosphere Electrodynamics is the
NOAA Space Weather Prediction Center's operational model. WAM-IPE output is
published openly on the Amazon S3 bucket `noaa-nws-wam-ipe-pds` in the
`us-east-1` region. WAM-IPE output is published in two product streams:

- **`wrs`** - **WAM-IPE Real-time Stream** (nowcast), initialised from
  near-real-time data assimilation. **This is the primary product** and the
  one you should use for operational work.
- **`wfs`** - **WAM-IPE Forecast Stream**, initialised from NCEP/GFS.
  A fallback when WRS data is unavailable.

Each stream is published at a 10-minute output cadence. The package finds the
two bracketing 10-minute files for a requested `DateTime`, downloads them
into the local cache (atomic `.part` -> `mv`), opens them through the
per-thread dataset pool, decodes the coordinate grids, and performs spatial
interpolation on each file before a linear blend in time between the two.

### WRS cycle preference

WRS data is published in 6-hourly cycles (00, 06, 12, 18 UTC). Each cycle's
files are available in the folder with that hour. However, there is a 3-hour
offset: cycle 00z contains data from 03:10 UTC onwards, cycle 06z from
09:10 UTC onwards, cycle 12z from 15:10 UTC onwards, and cycle 18z from
21:10 UTC onwards.

The `_wrs_cycles(dt)` function returns the cycle hours in preference order,
trying the most recent cycle first and falling back to earlier cycles:

| Query time (UTC) | Cycle preference order |
|---|---|
| 00:00 - 02:59 | 18z (prev day) -> 12z -> 06z -> 00z |
| 03:00 - 08:59 | 00z (today) -> 18z (prev) -> 12z -> 06z |
| 09:00 - 14:59 | 06z (today) -> 00z -> 18z (prev) -> 12z |
| 15:00 - 20:59 | 12z (today) -> 06z -> 00z -> 18z (prev) |
| 21:00 - 23:59 | 18z (today) -> 12z -> 06z -> 00z |

This means a query at 12:00 UTC uses the 06z cycle (whose files cover 09:10-14:50),
not the 12z cycle (which only starts at 15:10). The query always gets the
most-recent available nowcast data with no gaps.

### Versions and `_VERSION_WINDOWS`

The S3 layout has a `v1.1` and a `v1.2` prefix. The package auto-selects the
right prefix from the requested `DateTime`:

| Version | Window |
|---|---|
| `v1.1` | `2023-03-20 21:10:00` through `2023-06-30 21:09:59` (inclusive) |
| `v1.2` | `2023-06-30 21:10:00` onward (open-ended) |

warning **The boundary between `v1.1` and `v1.2` was mechanically patched.** An
earlier version of the constant left an uncovered 10-minute gap between
`2023-06-30 21:00:00` and `2023-06-30 21:10:00` in which every query
threw `"No WAM-IPE version mapping covers $dt"`. The gap has been closed by
extending `v1.1` to `21:09:59`. This boundary was **not** verified against
the actual NOAA WAM-IPE v1.1->v1.2 cutover. Verify against NOAA's records
before relying on data near that timestamp.

You can override the auto-selected version by passing `root_prefix` to
`WAMInterpolator`:

```julia
itp = WAMInterpolator(product="wfs", root_prefix="v1.1",
                      varname="den", interpolation=:sciml)
```

### Interpolation modes

The `interpolation` field of `WAMInterpolator` accepts:

- `:nearest`
- `:linear`
- `:logz_linear`
- `:logz_quadratic`
- `:sciml` - an alias for `:logz_quadratic`, kept so callers in related Space
  Falcon Lab packages can use the same setting across packages.

### What a user can query

WAM-IPE has no intrinsic lower-altitude bound enforced at the validator
level - the only enforced constraints are `isfinite(lat)` and `lat in [-90, 90]`
and `isfinite(lon)` and `alt > 0`. The actual usable altitude band is
governed by the WAM-IPE model's vertical grid (LEO altitudes are well
covered; lower altitudes may be outside the grid and the bracket search will
clamp to the edge with an `@debug` log message).

---

## GEOS-FP

`GEOS-FP` is NASA GMAO's modern-era retrospective reanalysis. The package
uses it for lower-atmosphere density estimates (0-70 km by default). Files
are downloaded as raw `.nc4` from
`https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/...` via plain HTTP
`GET` with `readtimeout=120`, and cached locally.

### URL pattern

With the defaults `root_url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"`
and `collection = "inst3_3d_asm_Np"`, an example URL for `2024-05-15 12:00 UTC`
is:

```
https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2024/M05/D15/GEOS.fp.asm.inst3_3d_asm_Np.20240515_1200.V01.nc4
```

Fields configurable on the `GEOSFPInterpolator` struct:

| Field | Default | Purpose |
|---|---|---|
| `root_url` | `"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"` | Base of the NASA GMAO URL |
| `collection` | `"inst3_3d_asm_Np"` | Sub-collection name in the path |
| `cache_dir` | `DEFAULT_GEOS_CACHE_DIR` (`./cache_geosfp`) | On-disc cache root |
| `qv_varname` | `"QV"` | Specific humidity variable name |
| `t_varname` | `"T"` | Temperature variable name |
| `z_varname` | `"H"` | Geopotential height variable name |
| `lev_varname` | `"lev"` | Vertical level coordinate name |
| `min_alt_km` | `0.0` | Lower validator bound |
| `max_alt_km` | `70.0` | Upper validator bound |

The `density_varname` field is reserved but unused (it defaults to `""` and
is not read by the GEOS-FP loader).

The `GEOSFPInterpolator` is an immutable `Base.@kwdef` struct, so to change
any field you construct a new one:

```julia
geos = GEOSFPInterpolator(cache_dir = "/path/to/geos_cache",
                          max_alt_km = 80.0,
                          interpolation = :sciml)
```

### Density construction

Density is reconstructed internally from temperature, specific humidity, and
geopotential height, rather than read as a stored variable:

```julia
Tv = T .* (1 .+ 0.61 .* QV)            # virtual temperature
V  = P ./ (R_D_GEOS .* Tv)             # ideal gas law with dry-air gas constant
```

where `R_D_GEOS = 287.05` J/(kg*K) is the gas constant for dry air,
`G0_GEOS = 9.80665` m/s^2 is standard gravity, and `P` is the 1-D pressure
profile broadcast against the 4-D `(lon, lat, z, time)` field. (The
broadcasting replaces an earlier `repeat()` call that materialised a full
`lon×lat×nz×nt` array on every load - see
[How It Works](how-it-works.md#Density-reconstruction-in-_geos_load_grids).)

### Hydostatic fill

When the GEOS-FP file's `H` (geopotential height) field contains missing
values along the vertical profile, the package fills them using a
hydrostatic integration based on the bracketing pressure levels and the
mean temperatures between them. The relevant code lives in
`_hydrostatic_fill!`.

### Dynamic altitude bounds

A subtle but important point: the **`max_alt_km = 70.0` validator bound and
the dynamic ceiling from the `H` field are different things**. The validator
bound is what the public `get_density` checks against before doing any I/O.
The dynamic ceiling is what the Hybrid backend uses to route between GEOS-FP
and NRLMSISE-00 (see [Hybrid](#Hybrid) below). The dynamic ceiling is
computed by `_geos_altitude_bounds`, which loads the two bracketing files,
reads the `H` field from each, applies hydrostatic fill where needed, and
returns the overlapping valid range
`(max(zmin_lo, zmin_hi), min(zmax_lo, zmax_hi))`.

### Inspecting GEOS-FP files

`inspect_geos_file` prints variable names, dimension names, sizes,
and selected attributes for a local `.nc4` file.
`inspect_geos_remote_file` downloads one GEOS file for a given
`DateTime` (re-using the cache) and runs the same inspection on it.

---

## NRLMSISE-00

The Naval Research Laboratory Mass Spectrometer, Incoherent Scatter Radar,
Extended model is the standard empirical atmospheric model. The package
calls it through `SatelliteToolboxAtmosphericModels.nrlmsise00`.

### Altitude bounds

`NRLMSISEInterpolator` defaults to `min_alt_km = 0.0`, `max_alt_km = 100.0`.
The validator `_validate_query_args_msis` throws an `ArgumentError` if a
query's altitude falls outside `[min_alt_km, max_alt_km]`. **To query above
100 km with this backend you must widen the bounds explicitly:**

```julia
msis_wide = NRLMSISEInterpolator(min_alt_km = 0.0, max_alt_km = 500.0)
rho       = get_density(msis_wide, dt, 45.0, -75.0, 400.0)
```

This default was a deliberate choice - above ≈100 km the empirical model is
less reliable than WAM-IPE - but it occasionally surprises users who assume
`get_density(msis, ..., 400)` will succeed. See
the Known issues section of the
[README on GitHub](https://github.com/Bourbon8464/WamIPEDensity.jl#known-issues-and-caveats).

### Module-level `SpaceIndices.init()` guard

The first call to any `NRLMSISEInterpolator` method triggers
`SpaceIndices.init()`, which downloads solar/geomagnetic index files from
`celestrak.org`, `sol.spacenvironment.net`, and `kp.gfz.de` (handled
transparently by `SpaceIndices.jl`). The init is **guarded by a
module-level lock and a module-level flag** (`_MSIS_INIT_LOCK` and
`_MSIS_INITIALIZED`), so:

- The init runs at most once per Julia session, regardless of how many
  `NRLMSISEInterpolator` instances are constructed.
- Concurrent first calls from multiple threads will not race to call
  `SpaceIndices.init()` redundantly.
- The interpolator argument to `_init_msis_indices!` is kept for API
  stability but unused - there is no per-instance `space_indices_initialized`
  field.

The `try`/`catch` around `SpaceIndices.init()` emits an `@warn` if the
fetch fails and continues. The empirical model can still produce a density
value if the indices were previously downloaded and cached locally by
`SpaceIndices.jl`.

---

## Hybrid

The `HybridDensityInterpolator` routes each request to GEOS-FP, NRLMSISE-00,
or WAM-IPE based on altitude. The routing logic in `_select_backend` is:

1. If `alt_km > itp.msis_max_alt_km` -> **`:wam`** (WAM-IPE).
2. Otherwise, look up the GEOS-FP dynamic altitude bounds for the 3-hour
   bucket containing `dt` (cached in `itp.geos_bounds_cache`, keyed by
   `_datetime_floor_3hr(dt)`). If not yet cached, compute them via
   `_geos_altitude_bounds(itp.geos, dt)` and store them.
3. If `geos_zmin <= alt_km <= geos_zmax` -> **`:geos`** (GEOS-FP).
4. Otherwise -> **`:msis`** (NRLMSISE-00).

### Default crossover altitudes

| Band | Backend | Default crossover |
|---|---|---|
| 0 km -> ≈GEOS dynamic ceiling (often ≈70 km) | GEOS-FP | `(geos_zmin, geos_zmax)` for the 3-hour bucket |
| ≈GEOS dynamic ceiling -> `msis_max_alt_km` (default 100 km) | NRLMSISE-00 | `msis_max_alt_km` |
| `> msis_max_alt_km` | WAM-IPE | `msis_max_alt_km` |

### warning The GEOS->NRLMSISE boundary is data-dependent

The crossover between GEOS-FP and NRLMSISE-00 is **not** a fixed number. It
is the dynamic ceiling of the `H` field in the bracketing GEOS-FP file for
that 3-hour bucket, cached in `geos_bounds_cache`. In rare cases the dynamic
ceiling can exceed the `max_alt_km` validator bound on the
`GEOSFPInterpolator` (which is `70.0` by default and is *not* consulted by
`_select_backend`). A subsequent `get_density(geos, ..., 85.0)` call could
then throw `ArgumentError: GEOS-FP backend only supports 0.0-70.0 km`
even though `_select_backend` happily routed the request to GEOS at 85 km.
In practice the GEOS `H` field rarely exceeds ≈80 km, but if you rely on the
hybrid backend near the boundary, you should be aware of this.

### Preset hybrid configurations

Two preset configurations are provided:

```julia
leo        = leo_config()         # msis 0-130 km, GEOS 0-70 km, msis_max_alt_km=130.0
lower_atmo = lower_atmo_config()  # msis 0-100 km, GEOS 0-70 km, msis_max_alt_km=70.0
```

`leo_config` is intended for LEO-drag calculations: NRLMSISE-00 covers
70-130 km and WAM-IPE covers everything above. `lower_atmo_config` is
optimised for lower-atmosphere work: GEOS-FP covers 0-70 km and NRLMSISE-00
covers 70-100 km.

### Constructing a custom hybrid configuration

```julia
hyb = HybridDensityInterpolator(
    geos           = GEOSFPInterpolator(max_alt_km = 80.0, interpolation = :sciml),
    msis           = NRLMSISEInterpolator(min_alt_km = 0.0, max_alt_km = 150.0),
    wam            = WAMInterpolator(product = "wfs", interpolation = :sciml),
    msis_max_alt_km = 150.0,
)
```

The `geos_bounds_cache` field can be cleared manually (it is a plain
`Dict{DateTime, Tuple{Float64, Float64}}`) to force a recompute of the GEOS
dynamic bounds for the next 3-hour bucket.
