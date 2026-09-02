using Test
using Dates
using WamIPEDensity

# Offline-only test file. This is the strict CI gate - it must pass cleanly
# without any network access (no S3, no NASA portal) beyond the initial
# SpaceIndices download, which is handled transparently and degrades
# gracefully if a fresh fetch fails.
#
# What this file covers:
#   1. Module loads and exports the documented public symbols.
#   2. All four interpolators can be constructed with their default kwargs.
#   3. NRLMSISEInterpolator has ONLY the three documented fields, with no
#      leftover space_indices_initialized - a regression guard for the move
#      to the module-level _MSIS_INITIALIZED / _MSIS_INIT_LOCK guard.
#   4. get_density on the empirical MSIS backend returns finite, positive
#      density at 90 km (within the default 0-100 km window).
#   5. get_density on a widened-bounds MSIS interpolator returns finite,
#      positive density at 400 km - the Task 2.2 runtime smoke guard.
#   6. Argument validation rejects bad inputs (invalid interpolation mode,
#      out-of-bounds latitude, non-finite inputs, negative altitude).
#   7. clean_cache! is callable with the new cache_max_bytes kwarg and does
#      not throw on a non-existent directory.
#   8. The density() convenience wrapper routes to a default hybrid backend
#      and returns a finite density.

@testset "WamIPEDensity.jl offline checks" begin

# ----------------------------------------------------------------------
@testset "Module exports" begin
# ----------------------------------------------------------------------
    # The 11 names tested in the existing runtests.jl "Module Loading & Exports".
    for sym in (:WAMInterpolator, :GEOSFPInterpolator, :NRLMSISEInterpolator,
                :HybridDensityInterpolator, :get_density, :get_density_batch,
                :get_density_trajectory, :get_density_trajectory_optimised,
                :prewarm_cache!, :clear_grid_cache!, :print_cache_stats)
        @test isdefined(WamIPEDensity, sym)
    end

    # The added public API documented in docs/src/api.md but not in the
    # original test file - exercise them too.
    for sym in (:density, :leo_config, :lower_atmo_config,
                :clean_cache!, :get_density_batch!,
                :get_density_batch_optimized, :get_density_batch_parallel,
                :get_density_at_point)
        @test isdefined(WamIPEDensity, sym)
    end
end

# ----------------------------------------------------------------------
@testset "Constructor smoke" begin
# ----------------------------------------------------------------------
    # None of these should throw on construction. (Construction does not
    # touch the network.)
    @test WAMInterpolator() isa WAMInterpolator
    @test GEOSFPInterpolator() isa GEOSFPInterpolator
    @test NRLMSISEInterpolator() isa NRLMSISEInterpolator
    @test HybridDensityInterpolator() isa HybridDensityInterpolator

    # Preset hybrid configs return HybridDensityInterpolator.
    @test leo_config() isa HybridDensityInterpolator
    @test lower_atmo_config() isa HybridDensityInterpolator
end

# ----------------------------------------------------------------------
@testset "NRLMSISEInterpolator field regression guard" begin
# ----------------------------------------------------------------------
    # After the Tier 2.2 fix, NRLMSISEInterpolator must have exactly three
    # fields: interpolation, min_alt_km, max_alt_km. The old per-instance
    # space_indices_initialized field was removed in favour of the
    # module-level _MSIS_INITIALIZED / _MSIS_INIT_LOCK guard.
    fields = fieldnames(NRLMSISEInterpolator)
    @test collect(fields) == [:interpolation, :min_alt_km, :max_alt_km]

    # The two module-level state objects must exist (the init guard).
    @test isdefined(WamIPEDensity, :_MSIS_INIT_LOCK)
    @test isdefined(WamIPEDensity, :_MSIS_INITIALIZED)

    # The old field name must not exist as a field anywhere in the module.
    @test !(:space_indices_initialized in fields)
end

# ----------------------------------------------------------------------
@testset "NRLMSISE single point (default window)" begin
# ----------------------------------------------------------------------
    # 90 km is inside the default 0-100 km MSIS window and needs no network
    # beyond SpaceIndices.init() on the very first call.
    itp = NRLMSISEInterpolator()
    dt  = DateTime(2024, 5, 15, 12, 0, 0)

    den = get_density(itp, dt, 0.0, 0.0, 90.0)
    @test isfinite(den)
    @test den > 0
end

# ----------------------------------------------------------------------
@testset "NRLMSISE single point (widened bounds, 400 km)" begin
# ----------------------------------------------------------------------
    # The Task 2.2 runtime smoke guard: widen the validator window so the
    # 400 km query is accepted, then confirm the module-level init guard
    # worked and a finite density came back.
    itp = NRLMSISEInterpolator(min_alt_km=0.0, max_alt_km=500.0)
    dt  = DateTime(2024, 5, 15, 12, 0, 0)

    den = get_density(itp, dt, 45.0, -75.0, 400.0)
    @test isfinite(den)
    @test den > 0
end

# ----------------------------------------------------------------------
@testset "Argument validation" begin
# ----------------------------------------------------------------------
    dt = DateTime(2024, 5, 15, 12, 0, 0)

    # WAMInterpolator accepts the :sciml alias (it is normalised to
    # :logz_quadratic internally).
    @test WAMInterpolator(interpolation=:sciml).interpolation == :sciml
    @test WAMInterpolator(interpolation=:nearest).interpolation == :nearest

    # Invalid interpolation mode: the constructor accepts it (Symbol is a
    # Symbol); the validator runs at query time.
    itp_invalid = WAMInterpolator(interpolation=:invalid)
    @test_throws ArgumentError get_density(itp_invalid, dt, 0.0, 0.0, 400.0)

    # Out-of-bounds latitude (WAM backend validates lat in [-90, 90]).
    itp = WAMInterpolator()
    @test_throws ArgumentError get_density(itp, dt,  91.0, 0.0, 400.0)
    @test_throws ArgumentError get_density(itp, dt, -91.0, 0.0, 400.0)

    # Non-finite inputs.
    @test_throws ArgumentError get_density(itp, dt, NaN, 0.0, 400.0)
    @test_throws ArgumentError get_density(itp, dt, 0.0, Inf, 400.0)

    # Negative altitude.
    @test_throws ArgumentError get_density(itp, dt, 0.0, 0.0, -100.0)

    # NRLMSISE out-of-window altitude (default 0-100 km): 400 km should be
    # rejected by _validate_query_args_msis.
    msis = NRLMSISEInterpolator()
    @test_throws ArgumentError get_density(msis, dt, 0.0, 0.0, 400.0)

    # GEOS-FP out-of-window altitude (default 0-70 km): 400 km should be
    # rejected by _validate_query_args_geos before any HTTP fetch is done.
    geos = GEOSFPInterpolator()
    @test_throws ArgumentError get_density(geos, dt, 0.0, 0.0, 400.0)
end

# ----------------------------------------------------------------------
@testset "clean_cache! new signature" begin
# ----------------------------------------------------------------------
    # clean_cache! must accept the cache_max_bytes kwarg (the Tier 3.3 fix)
    # and must not throw on a non-existent cache directory. It returns an
    # Int (the number of files deleted) - zero for an empty or missing dir.
    n = clean_cache!(;
        cache_dir      = joinpath(mktempdir(), "does_not_exist"),
        max_age        = Day(30),
        cache_max_bytes = 2_000_000_000,
    )
    @test n == 0

    # Also confirm the call without the new kwarg still works (the default
    # applies) and that the signed Int return is preserved.
    @test clean_cache!(cache_dir=mktempdir()) isa Integer
end

# ----------------------------------------------------------------------
@testset "density convenience wrapper" begin
# ----------------------------------------------------------------------
    # density(...) routes to a default HybridDensityInterpolator and uses
    # degrees + kilometres by default. Requesting 90 km lets the hybrid
    # backend route to NRLMSISE (no S3), so this is offline-safe.
    dt = DateTime(2024, 5, 15, 12, 0, 0)

    den = density(dt, 45.0, -75.0, 90.0; alt_unit=:km, angles_in=:deg)
    @test isfinite(den)
    @test den > 0

    # The metres + radians alias should agree numerically with the
    # kilometres + degrees form above at the same physical point.
    den_alt = density(dt, deg2rad(45.0), deg2rad(-75.0), 90_000.0;
                      alt_unit=:m, angles_in=:rad)
    @test isfinite(den_alt)
    @test den_alt > 0
    @test den ≈ den_alt rtol=1e-9
end

# ----------------------------------------------------------------------
@testset "WRS cycle preference routing" begin
    # _wrs_cycles must return cycles in preference order; first = closest to query.
    # Coverage windows (v1.2/wrs.YYYYMMDD/HH/):
    #   18z prev  -> 21:10 to next 02:50
    #   00z today -> 03:10 to 08:50
    #   06z today -> 09:10 to 14:50
    #   12z today -> 15:10 to 20:50
    #   18z today -> 21:10 to next 02:50
    cycles = WamIPEDensity._wrs_cycles
    @test cycles(DateTime(2025,6,6, 1,30)) == [18,12,6,0]
    @test cycles(DateTime(2025,6,6, 7,30)) == [0,18,12,6]
    @test cycles(DateTime(2025,6,6,11,30)) == [6,0,18,12]
    @test cycles(DateTime(2025,6,6,17,30)) == [12,6,0,18]
    @test cycles(DateTime(2025,6,6,22,30)) == [18,12,6,0]
end

# ----------------------------------------------------------------------
@testset "WFS cycle-floor routing (offline)" begin
    # Each WFS cycle folder contains all files for that cycle's run.
    # Verified against S3: folder 00Z has files valid at 00:00-14:50+,
    # folder 06Z from 06:00, folder 12Z from 12:00, folder 18Z from 18:00.
    # Routing must floor to the latest cycle at or before the query time
    # (shortest forecast lead = closest to assimilated truth).
    archive = WamIPEDensity._wfs_archive
    @test hour(archive(DateTime(2025,6,6, 0,0)))  == 0
    @test hour(archive(DateTime(2025,6,6, 3,0)))  == 0   # was 06Z before fix -> NaN on S3
    @test hour(archive(DateTime(2025,6,6, 5,59))) == 0
    @test hour(archive(DateTime(2025,6,6, 6,0)))  == 6
    @test hour(archive(DateTime(2025,6,6,11,59))) == 6
    @test hour(archive(DateTime(2025,6,6,12,0)))  == 12
    @test hour(archive(DateTime(2025,6,6,18,0)))  == 18
    @test hour(archive(DateTime(2025,6,6,23,59))) == 18
end

# ----------------------------------------------------------------------
@testset "Cache utility functions (offline)" begin
# ----------------------------------------------------------------------
    # These don't touch the network - they just inspect / manage in-memory
    # state. They must not throw.
    @test set_max_open_datasets!(16) isa Integer
    @test set_max_open_datasets!(8)  isa Integer   # restore default

    clear_grid_cache!()   # returns nothing; just must not throw
    print_cache_stats()   # returns nothing; just must not throw (writes to stdout)
end

end  # @testset "WamIPEDensity.jl offline checks"

println("\n" * "=" ^ 60)
println("  Offline test suite passed.")
println("=" ^ 60)
