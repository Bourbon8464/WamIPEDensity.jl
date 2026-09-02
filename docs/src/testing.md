# Testing

`WamIPEDensity.jl` has three test entry points. This page explains what each
covers, the exact commands to run them, what "SKIPPED (no S3 access)" means,
and how CI uses them.

## Test files

| File | Runs offline? | What it tests |
|---|---|---|
| `test/test_offline.jl` | **Yes - fully offline** | A curated subset of the no-network checks plus regression guards for recent fixes. This is the **fast pre-PR check** that CI enforces strictly. |
| `test/runtests.jl` | **Partial - most WAM-IPE / GEOS-FP testsets self-skip without network access** | The full suite, organised into 9 `@testset`s by feature. CI runs this file but treats the self-skipping testsets as soft. |
| `test_smoke.jl` (repo root) | Mostly yes | A tiny end-to-end sanity script: module loads, NRLMSISE returns finite density at one point, argument validation rejects invalid inputs. The WAM-IPE block tries S3 and reports `SKIPPED` if unreachable. |

The `test/Project.toml` environment pins the test-only dependencies
(`BenchmarkTools`, `Test`, `Printf`, `Random`, `Statistics`, `Dates`,
`WamIPEDensity`). The `Project.toml` at the repo root has **no**
`[extras]`/`[targets]` section, so the tests are not run via `Pkg.test()` -
they are invoked directly with `julia --project=test`.

## Commands

From the repository root:

```bash
# (1) Offline-only checks - the fast, strict CI gate (≈30 seconds)
julia --project=test -e 'using Pkg; Pkg.instantiate(); include("test/test_offline.jl")'

# (2) The full test suite - several testsets self-skip without S3/HTTP
julia --project=test -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'

# (3) The end-to-end smoke script
julia --project=. test_smoke.jl
```

Tip: substitute `Pkg.develop(PackageSpec(path=pwd()))` for
`Pkg.instantiate()` if you want the test environment to use uncommitted
edits you've made to `src/`:

```bash
julia --project=test -e '
  using Pkg
  Pkg.develop(PackageSpec(path=pwd()))
  Pkg.instantiate()
  include("test/test_offline.jl")
'
```

To run a single `@testset` from the full suite, use one of two approaches:

- Edit `test/runtests.jl` to comment out the `@testset` blocks you don't
  want to run.
- Or wrap the desired block in a `@testset "tmp" begin ... end` and `include`
  the file from the REPL.

## The full test suite (`test/runtests.jl`)

The file has one outer `@testset "WamIPEDensity.jl Tests"` wrapping nine
nested testsets:

| # | `@testset` | What it tests | Network needed? |
|---|---|---|---|
| 1 | `Module Loading & Exports` | `isdefined(WamIPEDensity, ...)` for 11 exported names | No |
| 2 | `Argument Validation` | Invalid interpolation mode; out-of-bounds latitude; non-finite inputs; negative altitude | No |
| 3 | `WAM-IPE Single Point Retrieval` | `get_density` on a documented point | **Yes (S3)** |
| 4 | `WAM-IPE Small Batch (10 points)` | `get_density_batch` for 10 timestamps | **Yes (S3)** |
| 5 | `2-Week LEO Trajectory Test (WAM-IPE)` | 2 016-point ISS-like orbit; standard vs optimised trajectory agreement; `@benchmark`; `print_cache_stats()` | **Yes (S3)** |
| 6 | `2-Week LEO Trajectory Test (Hybrid Model)` | 144-point one-day hybrid trajectory | **Yes (S3 + HTTP for GEOS-FP)** |
| 7 | `Cache Behavior` | Same point twice in a row; warm vs cold speedup | **Yes (S3)** |
| 8 | `Interpolation Modes` | `get_density` for each of `:nearest`, `:linear`, `:logz_linear`, `:logz_quadratic`, `:sciml` | **Yes (S3)** |
| 9 | `NRLMSISE Standalone` | `get_density(msis, ...)` at 100 km; 24-hour batch | Mostly no (only the initial `SpaceIndices.init()` fetch) |

Each network testset uses a `try`/`catch` pattern that detects the message
`"Could not fetch files"` and prints `SKIPPED (no S3 access)` instead of
failing the suite. This means a developer without AWS credentials still gets
a passing run for the offline testsets, and CI is not broken by the
expected network absence.

### Helper functions in `runtests.jl`

The file also defines three small helpers:

- `generate_leo_trajectory(start_dt, duration; n_points=1000)` - builds an
  ISS-like circular orbit (inclination 51.6°, altitude 400-450 km, ≈90
  minute period). Returns `(dts, lats, lons, altitudes)` with `lats`/`lons`
  in **radians** and `altitudes` in **metres**, matching the
  `get_density_trajectory*` conventions.
- `print_timing_report(label, elapsed_sec, n_points)` - formatted
  `ms/point` and `points/second` printout.
- `validate_densities(densities, label)` - asserts `min > 1e-15`,
  `max < 1e-8`, prints a density range summary. NaN entries are tolerated
  (the testset reports the fraction finite before asserting positivity on
  the finite subset).

## The offline file (`test/test_offline.jl`)

This is a curated subset of the no-network checks, plus regression guards
for the recent Tier 1-3 fixes:

- Module loads; the 11 public exports are defined.
- All four interpolators can be constructed without throwing.
- The `NRLMSISEInterpolator` has only the three documented fields
  (`interpolation`, `min_alt_km`, `max_alt_km`) - a regression guard for
  the move from a per-instance `space_indices_initialized` field to the
  module-level `_MSIS_INITIALIZED`/`_MSIS_INIT_LOCK` guard.
- `get_density(msis, ...)` at 90 km (inside the default window) returns a
  finite, positive density.
- `get_density(msis, ...)` at 400 km on a widened-bounds
  `NRLMSISEInterpolator(min_alt_km=0.0, max_alt_km=500.0)` returns a
  finite, positive density - the Task 2.2 runtime smoke guard.
- Argument validation: invalid interpolation, lat > 90, NaN/Inf inputs,
  negative altitude all throw `ArgumentError`.
- `clean_cache!(; cache_max_bytes=2_000_000_000)` is callable and does not
  throw on a non-existent directory.
- The `density(...)` convenience wrapper routes to a default hybrid backend
  and returns a finite density.

The offline file is the **strict CI gate** - it must pass cleanly on every
push and PR.

## CI

The `.github/workflows/test.yml` workflow runs both test files on every push
to `main` and on every pull request:

```yaml
- name: Run offline test suite
  run: julia --project=test -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("test/test_offline.jl")'

- name: Run full test suite (network-dependent testsets self-skip)
  run: julia --project=test -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("test/runtests.jl")'
  continue-on-error: true
```

The offline file is the strict gate (it must pass cleanly without network).
The full suite is `continue-on-error: true` because most of its testsets
self-skip on a CI runner that is allowed to reach S3 but might be rate
limited or denied - the file is run for visibility on GitHub Actions logs,
not as a hard failure.

To rebuild the docs site, see the separate `.github/workflows/docs.yml`
workflow and the [Contributing](contributing.md) page.

## A note on test performance

`test/runtests.jl` testset 5 runs `@benchmark` against the standard and
optimised 2-week trajectory paths. With `samples=3` and `evals=1` the
benchmark adds a few seconds of fixed cost; it's there to surface
performance regressions rather than as a tight microbenchmark. A cold run
(uncached files) takes several minutes if the S3 downloads are not yet in
`./cache`; a warm run is dominated by file I/O on the cached NetCDF files
and the GPU/CPU interpolation cost.
