# Contributing

This is an open-source project and fixes on top of the existing code are
very welcome. This page explains the workflow, what to run before opening a
pull request, and the items that need maintainer sign-off before they
should be attempted.

## Workflow

1. **Open an issue first** for any non-trivial change. A short design
   sketch saves time on both sides and keeps the public API stable.
2. **Fork and branch** from `main`. Keep the branch focused on a single
   concern - if you've fixed a typo and refactored the cache layer in the
   same branch, please split the two into separate PRs.
3. **Run the offline tests** before pushing (see
   [Testing](testing.md#Commands) for the exact commands). The offline
   file is the strict CI gate; if it fails, CI will fail.
4. **Run the full test suite** if you have network access. The WAM-IPE and
   GEOS-FP testsets self-skip without AWS / NASA portal access, so a
   passing run on your machine with network access is stronger evidence
   than a blank-passing run on CI.
5. **Update the documentation** alongside your code change. The docs live
   in `docs/src/*.md` (markdown) and in the `"""..."""` docstrings above
   each public function in `src/*.jl`. Documenter renders the docstrings
   via the `@docs` blocks in `docs/src/api.md`.
6. **Open the PR** with a short description and a reference to the issue
   you opened in step 1.

## Project layout

```
src/
  WamIPEDensity.jl          # Main module. Includes all submodules, exports.
  types.jl                  # Struct definitions: WAMInterpolator, GEOSFPInterpolator,
                            # NRLMSISEInterpolator, HybridDensityInterpolator,
                            # GridMetadata, _FileCache, and all global constants.
  time_utils.jl             # DateTime helpers, version mapping, S3 key construction.
  io.jl                     # NetCDF helpers, CF decode, per-thread dataset pool,
                            # grid metadata loader.
  cache.jl                  # On-disc LRU file cache with serialised metadata,
                            # time-bucket pair cache, clean_cache!, print_cache_stats.
  interpolation.jl          # 3-D / 4-D separable interpolation (linear,
                            # log-z, SciML quadratic) and WAM-IPE core evaluator.
  downloads.jl              # AWS configuration, bounded-concurrency downloader.
  backends/
    wam.jl                  # WAM-IPE backend: file resolution, density, batch, trajectory.
    geos.jl                 # GEOS-FP backend: density reconstruction via
                            # hydrostatic integration, altitude bounds cache.
    msis.jl                 # NRLMSISE-00 backend via SatelliteToolbox.
    hybrid.jl               # Altitude-routing hybrid: GEOS -> MSIS -> WAM.
  api.jl                    # Public top-level API: density(), leo_config(),
                            # lower_atmo_config(), batch wrappers.
  profile.jl                # Global-mean density profile and Plots.jl helpers.
  diagnostics.jl            # GEOS-FP file inspection helpers.
test/
  runtests.jl                # Full test suite (some testsets need S3 access)
  test_offline.jl            # Curated offline-only checks (the CI gate)
  Project.toml               # Test-only deps
docs/
  make.jl                    # Documenter build + deploy script
  src/*.md                    # Markdown source for the docs site
.github/workflows/
  docs.yml                    # Builds and deploys the docs site (gh-pages)
  test.yml                    # Runs the offline + full test suites
  TagBot.yml                  # Julia registry tag automation
```

## Running the tests

See the [Testing](testing.md) page for the exact commands. The short
version:

```bash
# Strict CI gate - should pass in ≈45 s without any network access
julia --project=. -e 'include("test/test_offline.jl")'

# Full suite - most WAM-IPE / GEOS-FP testsets self-skip without network
julia --project=. -e 'include("test/runtests.jl")'
```

## Building the docs site locally

```bash
julia --project=docs -e '
  using Pkg
  Pkg.develop(PackageSpec(path="."))
  Pkg.instantiate()
  include("docs/make.jl")
'
```

Then open `docs/build/index.html` in a browser. The docs environment is
tracked in `docs/Project.toml`.

## Style conventions

- **Julia 1.11 or later.** The `Project.toml` declares `julia = "1.11"`.
- **`Base.@kwdef` structs** for all interpolation configuration types -
  this generates keyword constructors and lets the docs show field
  defaults.
- **British English in prose, mixed spelling in identifiers.** The
  trajectory family is `get_density_trajectory_optimised` (British); the
  batch family is `get_density_batch_optimized` (American). Both are
  deliberate - do not rename one to match the other. The same rule applies
  inside docstrings: mirror the function name's spelling and write
  surrounding prose in British English.
- **No new dependencies without discussion.** Adding a package to
  `Project.toml` widens the install footprint; please raise it in the
  issue thread first.
- **Do not run tests via `Pkg.test()`.** The root `Project.toml` has no
  `[extras]`/`[targets]` section by design - the test environment is kept
  in `test/Project.toml` instead. Use `julia --project=test ...` as shown
  above.
- **Do not silence `@warn` with `@debug`.** The Tier 1-3 fixes made a
  deliberate choice to surface interpolation errors at the default log
  level - please don't undo that without a discussion.

## Public API stability

The package is not yet at a 1.0 release, so breaking changes are
technically allowed. As a courtesy:

- Don't rename or remove an exported symbol without deprecation.
- Don't add a required positional argument to a public function - use a
  keyword with a default instead.
- Don't change the semantic meaning of an existing keyword (e.g. flipping
  `angles_in_deg=false` from "radians expected" to "degrees expected").

The internal `_`-prefixed exports (`_vectorized_interp3_linear`,
`_vectorized_interp4`, `_batch_process_time_buckets`) are **not** part of
the stable API. They are exported only so internal callers across files
can use them; signature or behaviour may change between releases.

## Do NOT attempt without maintainer sign-off

The following items require an architectural decision, not a mechanical
patch. **Please list them in your issue thread and ask for maintainer
sign-off before implementing them.** They are listed here so contributors
are aware they exist.

### 1. NetCDF / HDF5 thread-safety across `Threads.@threads` batch calls

The underlying C libraries may not support concurrent reads on shared
dataset handles. Two safe approaches exist:

- a **global read lock**, which loses parallelism but is straightforward;
- **per-thread dataset handles**, which preserve parallelism but require a
  larger refactor around the in-process dataset pool.

The maintainer needs to choose which approach to take before any
implementation work is done. Do not just add `lock(...) do ... end`
 wrappers around the dataset reads - that will deadlock in patterns used
by `get_density_batch_parallel`.

### 2. Consolidating `get_density_batch` / `get_density_trajectory` onto the `_optimised` / `_optimized` grouped-by-file-pair logic

The package currently maintains two parallel implementations: the
documented public `get_density_batch` and `get_density_trajectory`
functions, and the time-bucket-grouped `get_density_batch_optimized`,
`get_density_batch_parallel`, and `get_density_trajectory_optimised`
variants. Folding the former into the latter would reduce duplication but
changes the performance characteristics of documented public functions
(some callers rely on the per-call cost profile of the unoptimised path
for benchmarking). This should be reviewed before the consolidation is
made.

### 3. A lightweight GEOS metadata-only loader for `_geos_altitude_bounds`

The current `_geos_altitude_bounds` loads the full `T`/`QV`/`H` arrays for
each bracketing GEOS-FP file just to read the `z` bounds. A new
metadata-only loader mirroring `_load_grid_metadata` - reading only the
coordinate axes and the `H` field - would save significant I/O. The
difficulty is that the GEOS `H` field can contain missing values along
the vertical profile, and the current fallback is the hydrostatic fill in
`_hydrostatic_fill!`. The lightweight loader needs to handle that fallback
path safely before it can replace the full-array load.

## License

The package is released under the MIT license - see `LICENSE` at the
repository root. By contributing, you agree that your contributions will
be licensed under the same terms.
