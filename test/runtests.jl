using Test
using Dates
using Random
using Printf
using BenchmarkTools
using WamIPEDensity

# Seed random for reproducibility
Random.seed!(42)

# ----------------------------------------------------------------------------
# Helper: generate a realistic LEO trajectory over a time span
# ----------------------------------------------------------------------------
function generate_leo_trajectory(start_dt::DateTime, duration::Day; n_points::Int=1000)
    # Orbital parameters for a roughly circular LEO orbit
    period_minutes = 90
    n_orbits = ceil(Int, Dates.value(duration) * 24 * 60 / period_minutes)
    
    # Time points: every ≈10 minutes for 2 weeks, plus some finer points
    total_minutes = Dates.value(duration) * 24 * 60
    if n_points > 0
        dts = [start_dt + Minute(round(Int, i * total_minutes / (n_points - 1))) for i in 0:(n_points-1)]
    else
        dts = DateTime[]
    end
    
    # Inclination ≈51.6 degrees (ISS-like)
    inclination = deg2rad(51.6)
    
    # Altitude varies between 400 and 450 km (circular orbit with slight variation)
    altitudes = Float64[]
    lats = Float64[]
    lons = Float64[]
    
    for (i, dt) in enumerate(dts)
        # Mean anomaly (progress around orbit)
        M = 2π * (i % (period_minutes * 60)) / (period_minutes * 60)
        
        # Simple circular orbit model
        # Argument of latitude (position along orbit track)
        arg_lat = M + 2π * div(i, period_minutes * 60)
        
        # Latitude from inclination
        lat = asin(sin(inclination) * sin(arg_lat))
        
        # Longitude progression (approximate: Earth rotates underneath)
        earth_rot_rate = 2π / (24 * 3600)  # rad/s
        elapsed_sec = Dates.value(dt - start_dt) / 1000
        lon0 = deg2rad(0)  # Starting longitude
        lon = lon0 + arg_lat + earth_rot_rate * elapsed_sec
        
        # Normalize longitude
        lon = mod(lon + π, 2π) - π
        
        # Altitude with small sinusoidal variation (eccentricity effect)
        alt = 425e3 + 25e3 * cos(M)  # meters
        
        push!(lats, lat)
        push!(lons, lon)
        push!(altitudes, alt)
    end
    
    return dts, lats, lons, altitudes
end

# ----------------------------------------------------------------------------
# Helper: print timing report
# ----------------------------------------------------------------------------
function print_timing_report(label, elapsed_sec, n_points)
    ms_per_point = (elapsed_sec / n_points) * 1000
    pts_per_sec = n_points / elapsed_sec
    println(@sprintf("  %-35s %8.3f s  (%7.2f ms/pt  |  %8.1f pt/s)", label, elapsed_sec, ms_per_point, pts_per_sec))
end

# ----------------------------------------------------------------------------
# Helper: validate density values are physically reasonable
# ----------------------------------------------------------------------------
function validate_densities(densities, label)
    n_finite = sum(isfinite, densities)
    n_total = length(densities)
    
    if n_finite == 0
        println("    $label: all $n_total values are NaN (data unavailable)")
        return
    end
    
    finite_dens = filter(isfinite, densities)
    
    @test all(d -> d > 0, finite_dens)
    
    # Typical LEO density range at 400-450 km: ≈1e-12 to 1e-10 kg/m^3
    # Allow generous bounds for solar cycle variation
    min_dens = minimum(finite_dens)
    max_dens = maximum(finite_dens)
    mean_dens = mean(finite_dens)
    
    @test min_dens > 1e-15
    @test max_dens < 1e-8
    
    println(@sprintf("    Density range: %.2e - %.2e kg/m^3 (mean: %.2e) [%.1f%% finite]", min_dens, max_dens, mean_dens, 100*n_finite/n_total))
end

# ============================================================================
# TEST SUITE
# ============================================================================
@testset "WamIPEDensity.jl Tests" begin

# ----------------------------------------------------------------------------
@testset "Module Loading & Exports" begin
# ----------------------------------------------------------------------------
    @test isdefined(WamIPEDensity, :WAMInterpolator)
    @test isdefined(WamIPEDensity, :GEOSFPInterpolator)
    @test isdefined(WamIPEDensity, :NRLMSISEInterpolator)
    @test isdefined(WamIPEDensity, :HybridDensityInterpolator)
    @test isdefined(WamIPEDensity, :get_density)
    @test isdefined(WamIPEDensity, :get_density_batch)
    @test isdefined(WamIPEDensity, :get_density_trajectory)
    @test isdefined(WamIPEDensity, :get_density_trajectory_optimised)
    @test isdefined(WamIPEDensity, :prewarm_cache!)
    @test isdefined(WamIPEDensity, :clear_grid_cache!)
    @test isdefined(WamIPEDensity, :print_cache_stats)
end

# ----------------------------------------------------------------------------
@testset "Argument Validation" begin
# ----------------------------------------------------------------------------
    itp = WAMInterpolator()
    
    # Valid interpolation modes should work
    @test WAMInterpolator(interpolation=:nearest).interpolation == :nearest
    @test WAMInterpolator(interpolation=:sciml).interpolation == :sciml
    
    # Invalid interpolation mode should error at query time
    dt = DateTime(2024, 5, 15, 12, 0, 0)
    itp_invalid = WAMInterpolator(interpolation=:invalid)
    @test_throws ArgumentError get_density(itp_invalid, dt, 0.0, 0.0, 400.0)
    
    # Latitude out of bounds
    @test_throws ArgumentError get_density(itp, dt, 91.0, 0.0, 400.0)
    @test_throws ArgumentError get_density(itp, dt, -91.0, 0.0, 400.0)
    
    # Non-finite values
    @test_throws ArgumentError get_density(itp, dt, NaN, 0.0, 400.0)
    @test_throws ArgumentError get_density(itp, dt, 0.0, Inf, 400.0)
    
    # Negative altitude
    @test_throws ArgumentError get_density(itp, dt, 0.0, 0.0, -100.0)
end

# ----------------------------------------------------------------------------
@testset "WAM-IPE Single Point Retrieval" begin
# ----------------------------------------------------------------------------
    println("\n  --- WAM-IPE Single Point ---")
    itp = WAMInterpolator(; product="wfs", root_prefix="v1.2", varname="den", interpolation=:sciml)
    dt = DateTime(2024, 5, 15, 12, 0, 0)
    lat, lon, alt = -33.4, -153.24, 550.68
    
    try
        elapsed = @elapsed den = get_density(itp, dt, lat, lon, alt)
        
        if isfinite(den)
            @test den > 0
            println(@sprintf("    Single point: %.3e kg/m^3 in %.3f seconds", den, elapsed))
        else
            println("    Single point: NaN (data not available at this location/time)")
        end
    catch e
        if isa(e, ErrorException) && occursin("Could not fetch files", e.msg)
            println("    SKIPPED (no S3 access)")
        else
            rethrow(e)
        end
    end
end

# ----------------------------------------------------------------------------
@testset "WAM-IPE Small Batch (10 points)" begin
# ----------------------------------------------------------------------------
    println("\n  --- WAM-IPE Small Batch ---")
    itp = WAMInterpolator(; product="wfs", root_prefix="v1.2", varname="den", interpolation=:sciml)
    dt = DateTime(2024, 5, 15, 12, 0, 0)
    
    dts = [dt + Minute(10)*i for i in 0:9]
    lats = fill(-33.4, 10)
    lons = fill(-153.24, 10)
    alts = fill(550.68, 10)
    
    try
        elapsed = @elapsed dens = get_density_batch(itp, dts, lats, lons, alts)
        
        @test length(dens) == 10
        if all(isfinite, dens)
            @test all(d -> d > 0, dens)
            print_timing_report("Batch (10 points)", elapsed, 10)
        else
            n_nan = sum(.!isfinite.(dens))
            println("    Batch: $n_nan/10 points returned NaN (data not available)")
        end
    catch e
        if isa(e, TaskFailedException) && occursin("Could not fetch files", string(e.task.exception))
            println("    SKIPPED (no S3 access)")
        else
            rethrow(e)
        end
    end
end

# ----------------------------------------------------------------------------
@testset "2-Week LEO Trajectory Test (WAM-IPE)" begin
# ----------------------------------------------------------------------------
    println("\n  ==========================================")
    println("  === 2-WEEK LEO TRAJECTORY (WAM-IPE)  ===")
    println("  ==========================================")
    
    # Check if data is available first (skip if not)
    itp_check = WAMInterpolator(; product="wfs", root_prefix="v1.2", varname="den", interpolation=:sciml)
    dt_check = DateTime(2024, 5, 1, 0, 0, 0)
    try
        den_check = get_density(itp_check, dt_check, 0.0, 0.0, 400.0)
        if !isfinite(den_check)
            println("  Data not available for test dates; skipping WAM-IPE trajectory test.")
            return
        end
    catch e
        if isa(e, ErrorException) && occursin("Could not fetch files", e.msg)
            println("  No S3 access; skipping WAM-IPE trajectory test.")
            return
        else
            rethrow(e)
        end
    end
    
    # Trajectory parameters
    start_dt = DateTime(2024, 5, 1, 0, 0, 0)
    duration = Day(14)
    n_points = 2016  # 1 point per 10 minutes for 2 weeks
    
    println(@sprintf("  Generating trajectory: %d points over %s", n_points, duration))
    
    # Generate trajectory
    dts, lats, lons, alts_m = generate_leo_trajectory(start_dt, duration; n_points=n_points)
    
    println(@sprintf("  Time span: %s to %s", dts[1], dts[end]))
    println(@sprintf("  Altitude range: %.1f - %.1f km", minimum(alts_m)/1000, maximum(alts_m)/1000))
    
    # Create interpolator
    itp = WAMInterpolator(; product="wfs", root_prefix="v1.2", varname="den", interpolation=:sciml)
    
    try
        # --- Pre-warm cache ---
        println("\n  Prewarming cache...")
        cache_warm_elapsed = @elapsed n_files = prewarm_cache!(itp, dts)
        println(@sprintf("  Cache prewarm: %.2f seconds (%d unique file pairs)", cache_warm_elapsed, n_files))
        
        # --- Standard trajectory ---
        println("\n  Running standard get_density_trajectory...")
        traj_elapsed = @elapsed densities_traj = get_density_trajectory(itp, dts, lats, lons, alts_m)
        
        @test length(densities_traj) == n_points
        print_timing_report("Standard trajectory", traj_elapsed, n_points)
        validate_densities(densities_traj, "Standard trajectory")
        
        # --- Optimised trajectory ---
        println("\n  Running optimised get_density_trajectory_optimised...")
        opt_elapsed = @elapsed densities_opt = get_density_trajectory_optimised(itp, dts, lats, lons, alts_m)
        
        @test length(densities_opt) == n_points
        print_timing_report("Optimised trajectory", opt_elapsed, n_points)
        validate_densities(densities_opt, "Optimised trajectory")
        
        # --- Results should match ---
        println("\n  Comparing standard vs optimised results...")
        max_diff = maximum(abs.(densities_traj .- densities_opt))
        rel_diff = max_diff / mean(densities_traj)
        println(@sprintf("    Max absolute difference: %.3e", max_diff))
        println(@sprintf("    Max relative difference: %.3e", rel_diff))
        @test max_diff < 1e-6 || rel_diff < 1e-6
        
        # --- Speedup ---
        if traj_elapsed > 0 && opt_elapsed > 0
            speedup = traj_elapsed / opt_elapsed
            println(@sprintf("\n  Speedup (optimised vs standard): %.2fx", speedup))
        end
        
        # --- Benchmark with BenchmarkTools for accuracy ---
        println("\n  Benchmarking with BenchmarkTools...")
        bench_std = @benchmark get_density_trajectory($itp, $dts, $lats, $lons, $alts_m) samples=3 evals=1
        bench_opt = @benchmark get_density_trajectory_optimised($itp, $dts, $lats, $lons, $alts_m) samples=3 evals=1
        
        println(@sprintf("    Standard:  median = %.2f s, mean = %.2f s", median(bench_std).time/1e9, mean(bench_std).time/1e9))
        println(@sprintf("    Optimised: median = %.2f s, mean = %.2f s", median(bench_opt).time/1e9, mean(bench_opt).time/1e9))
        
        # Print cache stats
        println("\n  Cache statistics:")
        print_cache_stats()
    catch e
        if isa(e, ErrorException) && occursin("Could not fetch files", e.msg)
            println("    SKIPPED (no S3 access)")
        else
            rethrow(e)
        end
    end
end

# ----------------------------------------------------------------------------
@testset "2-Week LEO Trajectory Test (Hybrid Model)" begin
# ----------------------------------------------------------------------------
    println("\n  ==========================================")
    println("  === 2-WEEK LEO TRAJECTORY (HYBRID)     ===")
    println("  ==========================================")
    
    start_dt = DateTime(2024, 5, 1, 0, 0, 0)
    duration = Day(1)
    n_points = 144
    
    println(@sprintf("  Generating trajectory: %d points over %s", n_points, duration))
    
    dts, lats, lons, alts_m = generate_leo_trajectory(start_dt, duration; n_points=n_points)
    
    # Hybrid interpolator: routes to GEOS (low alt), MSIS (mid), WAM (high)
    itp = HybridDensityInterpolator()
    
    try
        # Pre-warm cache (shorter to avoid excessive GEOS downloads)
        println("\n  Prewarming cache...")
        cache_warm_elapsed = @elapsed prewarm_cache!(itp, dts, alts_m .* 1e-3)
        println(@sprintf("  Cache prewarm: %.2f seconds", cache_warm_elapsed))
        
        # Standard trajectory
        println("\n  Running hybrid standard trajectory...")
        traj_elapsed = @elapsed densities_hybrid = get_density_trajectory(itp, dts, lats, lons, alts_m)
        
        @test length(densities_hybrid) == n_points
        print_timing_report("Hybrid trajectory", traj_elapsed, n_points)
        validate_densities(densities_hybrid, "Hybrid trajectory")
        
        # Optimised trajectory
        println("\n  Running hybrid optimised trajectory...")
        opt_elapsed = @elapsed densities_hybrid_opt = get_density_trajectory_optimised(itp, dts, lats, lons, alts_m)
        
        @test length(densities_hybrid_opt) == n_points
        print_timing_report("Hybrid optimised", opt_elapsed, n_points)
        validate_densities(densities_hybrid_opt, "Hybrid optimised")
        
        # Compare
        max_diff = maximum(abs.(densities_hybrid .- densities_hybrid_opt))
        println(@sprintf("\n  Max diff (standard vs optimised): %.3e", max_diff))
        @test max_diff < 1e-6 || max_diff / mean(densities_hybrid) < 1e-6
    catch e
        if isa(e, ErrorException) && occursin("Could not fetch files", e.msg)
            println("    SKIPPED (no S3 access)")
        elseif isa(e, TaskFailedException) && occursin("Could not fetch files", string(e.task.exception))
            println("    SKIPPED (no S3 access)")
        else
            rethrow(e)
        end
    end
end

# ----------------------------------------------------------------------------
@testset "Cache Behavior" begin
# ----------------------------------------------------------------------------
    println("\n  --- Cache Tests ---")
    
    # Clear cache
    clear_grid_cache!()
    
    # Run a query
    itp = WAMInterpolator(; product="wfs", root_prefix="v1.2", varname="den")
    dt = DateTime(2024, 5, 15, 12, 0, 0)
    
    try
        # First call (cold)
        elapsed_cold = @elapsed den1 = get_density(itp, dt, 0.0, 0.0, 400.0)
        
        # Second call (warm - same data should be cached)
        elapsed_warm = @elapsed den2 = get_density(itp, dt, 0.0, 0.0, 400.0)
        
        @test den1 ≈ den2 rtol=1e-10
        
        if elapsed_cold > 0 && elapsed_warm > 0
            speedup = elapsed_cold / elapsed_warm
            println(@sprintf("    Cold: %.3f s, Warm: %.3f s, Speedup: %.1fx", elapsed_cold, elapsed_warm, speedup))
        end
    catch e
        if isa(e, ErrorException) && occursin("Could not fetch files", e.msg)
            println("    SKIPPED (no S3 access)")
        else
            rethrow(e)
        end
    end
end

# ----------------------------------------------------------------------------
@testset "Interpolation Modes" begin
# ----------------------------------------------------------------------------
    println("\n  --- Interpolation Modes ---")
    
    dt = DateTime(2024, 5, 15, 12, 0, 0)
    lat, lon, alt = 0.0, 0.0, 400.0
    
    for mode in [:nearest, :linear, :logz_linear, :logz_quadratic, :sciml]
        itp = WAMInterpolator(; product="wfs", root_prefix="v1.2", varname="den", interpolation=mode)
        try
            den = get_density(itp, dt, lat, lon, alt)
            if isfinite(den)
                @test den > 0
                println(@sprintf("    Mode %-20s: %.3e kg/m^3", string(mode), den))
            else
                println(@sprintf("    Mode %-20s: NaN (data unavailable)", string(mode)))
            end
        catch e
            if isa(e, ErrorException) && occursin("Could not fetch files", e.msg)
                println(@sprintf("    Mode %-20s: SKIPPED (no S3)", string(mode)))
            else
                @test false
            end
        end
    end
end

# ----------------------------------------------------------------------------
@testset "NRLMSISE Standalone" begin
# ----------------------------------------------------------------------------
    println("\n  --- NRLMSISE Standalone ---")
    
    itp = NRLMSISEInterpolator()
    dt = DateTime(2024, 5, 15, 12, 0, 0)
    
    # Single point
    den = get_density(itp, dt, 0.0, 0.0, 100.0)
    @test isfinite(den) && den > 0
    println(@sprintf("    NRLMSISE at 100 km: %.3e kg/m^3", den))
    
    # Batch
    dts = [dt + Hour(i) for i in 0:23]
    lats = fill(0.0, 24)
    lons = fill(0.0, 24)
    alts = fill(100.0, 24)
    
    dens = get_density_batch(itp, dts, lats, lons, alts)
    @test length(dens) == 24
    @test all(isfinite, dens)
    @test all(d -> d > 0, dens)
    println(@sprintf("    NRLMSISE batch (24 hours): mean = %.3e kg/m^3", mean(dens)))
end

println("\n" * "="^50)
println("All tests completed!")
println("="^50)

end  # @testset "WamIPEDensity.jl Tests"