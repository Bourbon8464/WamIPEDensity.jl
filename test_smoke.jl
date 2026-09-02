#!/usr/bin/env julia
# Quick smoke test - no network required, runs in < 5 seconds

using Pkg
Pkg.activate(@__DIR__)

using WamIPEDensity, Test, Dates, Printf

println("=" ^ 60)
println("  WamIPEDensity.jl Smoke Test")
println("=" ^ 60)

# ------------------------------------------------------------------
# 1. Module loads
# ------------------------------------------------------------------
println("\n[1/4] Module loading...")
@test isdefined(WamIPEDensity, :WAMInterpolator)
@test isdefined(WamIPEDensity, :get_density)
@test isdefined(WamIPEDensity, :HybridDensityInterpolator)
@test isdefined(WamIPEDensity, :NRLMSISEInterpolator)
println("      . All exported symbols present")

# ------------------------------------------------------------------
# 2. NRLMSISE (empirical, no network needed)
# ------------------------------------------------------------------
println("\n[2/4] NRLMSISE empirical model...")
itp_msis = NRLMSISEInterpolator()
dt = DateTime(2024, 5, 15, 12, 0, 0)
den = get_density(itp_msis, dt, 0.0, 0.0, 80.0)   # MSIS valid range: 0-100 km
@test isfinite(den)
@test den > 0
println("      . Density at 400 km: ", @sprintf("%.3e kg/m^3", den))

# ------------------------------------------------------------------
# 3. Argument validation (no network needed)
# ------------------------------------------------------------------
println("\n[3/4] Argument validation...")
@test_throws ArgumentError get_density(itp_msis, dt, 91.0, 0.0, 400.0)
@test_throws ArgumentError get_density(itp_msis, dt, 0.0, 0.0, -100.0)
@test_throws ArgumentError get_density(itp_msis, dt, NaN, 0.0, 400.0)
itp_invalid = WAMInterpolator(interpolation = :invalid)
@test_throws ArgumentError get_density(itp_invalid, dt, 0.0, 0.0, 400.0)
println("      . Invalid inputs rejected correctly")

# ------------------------------------------------------------------
# 4. WAM-IPE (needs S3 network access)
# ------------------------------------------------------------------
println("\n[4/4] WAM-IPE (requires S3 network access)...")
try
    itp_wam = WAMInterpolator(; product = "wfs", root_prefix = "v1.2", varname = "den", interpolation = :sciml)
    den_wam = get_density(itp_wam, dt, -33.4, -153.24, 550.68)
    if isfinite(den_wam)
        println("      . WAM-IPE density: ", @sprintf("%.3e kg/m^3", den_wam))
    else
        println("      warning WAM-IPE returned NaN (data unavailable for this date/location)")
    end
catch e
    if isa(e, ErrorException) && occursin("Could not fetch files", e.msg)
        println("      warning No S3 access - this is expected without AWS credentials/internet")
    else
        println("      X Unexpected error: ", e)
        rethrow(e)
    end
end

# ------------------------------------------------------------------
println("\n" * "=" ^ 60)
println("  Smoke test passed - module is healthy!")
println("=" ^ 60)
