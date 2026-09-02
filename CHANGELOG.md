# Changelog

All notable changes to WamIPEDensity.jl are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- **`meta.fill_values` typo**. `_decode_value` referenced `meta.fillvals`
  (non-existent field). The resulting `FieldError` was silently swallowed by an
  upstream `try/catch`, causing every query to return `NaN`. Fixed to
  `meta.fill_values`. Confirmed by SpaceAGORA CYGNSS drag verification.
- **`_wfs_archive` cycle-floor mapping**. The function rounded to the nearest
  6-hour cycle (`h < 3 -> 0z`, `h < 9 -> 6z`, ...), but the S3 WFS archive
  stores each 10-minute stamp under the cycle that produced it
  (e.g. `20250606_030000.nc` lives in the `00/` folder, not `06/`).
  Now floors to the latest cycle at or before the query time - the shortest
  forecast lead and closest to assimilated truth. Verified against live S3:
  `curl .../v1.2/wfs.20250606/00/wam_fixed_height.wfs.t00z.wam10.20250606_030000.nc -> 200`,
  same path under `06/` -> `404`.
- **WRS cycle preference routing**. `_wrs_cycles(dt)` now returns cycle hours
  in preference order (closest to the query time first; earlier cycles are
  fallbacks). WRS is the primary product (real-time nowcast); WFS is the
  fallback (forecast). Cycle coverage on S3:

  | Cycle | Folder              | Valid times (UTC)  |
  |-------|---------------------|--------------------|
  | 00z   | `v1.2/wrs.<TODAY>/00/`  | 03:10 - 08:50  |
  | 06z   | `v1.2/wrs.<TODAY>/06/`  | 09:10 - 14:50  |
  | 12z   | `v1.2/wrs.<TODAY>/12/`  | 15:10 - 20:50  |
  | 18z   | `v1.2/wrs.<YESTERDAY>/18/` (after midnight) | 21:10 - 02:50 (next day) |

  A query at 12:00 UTC now tries cycle 06z first (the cycle whose files
  actually cover that time), then 00z, 18z, and 12z as fallbacks.
- **18z folder date split**: the 18z cycle folder uses the previous day's
  date, but the filename valid-time uses the query date. `_construct_s3_key(dt, product, arch)`
  now takes the cycle `arch::DateTime` explicitly so the folder and filename
  dates can differ.
- **NRLMSISE trajectory double-conversion bug**: `get_density_trajectory_optimised(::NRLMSISEInterpolator, ...)`
  was incorrectly calling `rad2deg` on values already in degrees (because
  `get_density_batch` expects degrees). Fixed: now passes values through unchanged.
- **DateTime coordinate arrays in NetCDF**: the WAM `_wam_load_grids` helper
  and the profile grid loader now handle `Vector{DateTime}`-typed axes via
  `_coord_floats` (converts to `Float64` via `Dates.value.`).
- **Missing `_GEOS_BOUNDS_LOCK`**: the hybrid backend now has a dedicated
  lock for the per-3-hour GEOS-FP altitude-bounds cache.

### Added
- `_wrs_cycles(dt)` - preferred-cycle list for WRS routing
- `_construct_s3_key(dt, product, arch)` - explicit-cycle key construction
- `leo_config()` - predefined `HybridDensityInterpolator` for LEO drag
- `lower_atmo_config()` - predefined `HybridDensityInterpolator` for lower atmosphere
- `density(dt, lat, lon, alt; alt_unit, angles_in)` - top-level convenience
- `set_download_concurrency!(n)` - bound parallel S3 downloads (1-32)
- `print_cache_stats()` - dump LRU cache diagnostics
- `get_density_batch!(out, ...)` - in-place batch with preallocated output
- `get_density_at_point(itp, dt, lat, lon, alt_m; angles_in_deg)` - radians convenience
- `clear_nc_pool!()` - test helper to close all per-thread NetCDF handles
- `_coord_floats(ds, dname)` - internal helper for DateTime-typed axes

### Changed
- **Modular refactor**: 3456-line monolith split into focused modules under `src/`.
  Public API unchanged. New layout:
  ```
  src/
    WamIPEDensity.jl        # main, includes all
    types.jl                # struct definitions, const state, locks
    time_utils.jl           # DateTime helpers, version mapping, S3 keys
    io.jl                   # NetCDF helpers, per-thread dataset pool
    cache.jl                # on-disc LRU file cache, time-bucket cache
    interpolation.jl         # 3D/4D separable interpolation
    downloads.jl             # bounded-concurrency S3 downloader
    backends/
      wam.jl                # WAM-IPE
      geos.jl               # GEOS-FP
      msis.jl               # NRLMSISE-00
      hybrid.jl             # altitude-routing hybrid
    api.jl                  # public top-level API + presets
    profile.jl              # global-mean density profile + plotting
    diagnostics.jl          # GEOS file inspection
  ```
- **Per-thread NCDataset handle pool**: warm-cache hot path is now lock-free
  (was a global lock + pin/unpin on every call). The 9.5x WAM-IPE speedup
  comes from this change.
- **Single time-bucket cache**: removed duplicate `_FILEPAIR_CACHE`
  (was parallel to `_TIME_BUCKET_CACHE`); one source of truth only.

### Removed
- `src/optimized_functions.jl` - folded into `src/api.jl` and `src/backends/*.jl`
- `src/performance_tests.jl` - never referenced
- `src/test.jl` - never referenced (tests live in `test/`)
- `_FILEPAIR_CACHE`, `_get_cached_filepair`, `_cache_filepair!` - duplicate of
  `_TIME_BUCKET_CACHE`
- `_vectorized_interp4` - defined in `optimized_functions.jl` but never called
- `get_density_from_key` - defined but never exported or called

### Performance (vs original monolith, warm cache)
- WAM-IPE trajectory 200 points: **198 ms -> 20 ms (9.7x faster)**
- WAM-IPE single point 100x: **100 ms -> 10 ms (9.5x faster)**
- Hybrid 100x at 400 km: **108 ms -> 10 ms (10.5x faster)**
- MSIS single point 1000x: ≈25% slower (added validation overhead);
  absolute time still 9-13 ms, negligible impact in practice.

### Migration guide

No code changes are required for downstream users. The exported API
(`WAMInterpolator`, `get_density`, `get_density_batch`,
`get_density_trajectory_optimised`, `prewarm_cache!`, etc.) is
backward-compatible.

If you were importing any underscore-prefixed internal symbol
(`_open_nc_cached`, `_FILEPAIR_CACHE`, `_vectorized_interp4`,
`get_density_from_key`), it has been removed. Replace with:
- `_open_nc_cached` -> `_open_nc` (per-thread pool, no pin/unpin needed)
- `_FILEPAIR_CACHE` -> `_TIME_BUCKET_CACHE` (same purpose, single source)
- `_vectorized_interp4` -> none (deleted; was unused)
- `get_density_from_key` -> none (deleted; was unused)

To take advantage of the new convenience APIs:
```julia
using WamIPEDensity
hybrid = leo_config()                         # LEO drag preset
set_download_concurrency!(8)                  # 8 parallel S3 downloads
den = density(dt, lat, lon, alt_km)           # simplest API
```
