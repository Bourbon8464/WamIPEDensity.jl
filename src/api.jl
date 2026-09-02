# api.jl - top-level public API and convenience helpers.

# --------------------------------------------------------------------------
# Default hybrid interpolator (lazy)
# --------------------------------------------------------------------------

function _get_default_itp()
    if _DEFAULT_ITP[] === nothing
        _DEFAULT_ITP[] = HybridDensityInterpolator()
    end
    return _DEFAULT_ITP[]
end

"""
    density(dt, lat, lon, alt; alt_unit=:km, angles_in=:deg) -> Float64

Return neutral atmospheric density (kg/m^3) using the default hybrid backend.
"""
function density(dt::DateTime, lat::Real, lon::Real, alt::Real;
                 alt_unit::Symbol=:km, angles_in::Symbol=:deg)
    angles_in in (:deg, :rad) || throw(ArgumentError("angles_in must be :deg or :rad; got $angles_in"))
    alt_unit  in (:km,  :m)   || throw(ArgumentError("alt_unit must be :km or :m; got $alt_unit"))
    lat_d = angles_in === :rad ? rad2deg(Float64(lat)) : Float64(lat)
    lon_d = angles_in === :rad ? rad2deg(Float64(lon)) : Float64(lon)
    alt_k = alt_unit === :m ? Float64(alt) / 1000.0 : Float64(alt)
    return get_density(_get_default_itp(), dt, lat_d, lon_d, alt_k)
end

density(dt::AbstractString, lat::Real, lon::Real, alt::Real; kwargs...) =
    density(DateTime(dt), lat, lon, alt; kwargs...)

# --------------------------------------------------------------------------
# Predefined hybrid configurations
# --------------------------------------------------------------------------

"""
    leo_config() -> HybridDensityInterpolator

Configuration for low-Earth-orbit drag calculations.
"""
function leo_config()
    return HybridDensityInterpolator(
        wam = WAMInterpolator(interpolation=:sciml),
        msis = NRLMSISEInterpolator(min_alt_km=0.0, max_alt_km=130.0),
        geos = GEOSFPInterpolator(interpolation=:sciml, max_alt_km=70.0),
        msis_max_alt_km = 130.0,
    )
end

"""
    lower_atmo_config() -> HybridDensityInterpolator

Configuration for lower-atmosphere work.
"""
function lower_atmo_config()
    return HybridDensityInterpolator(
        geos = GEOSFPInterpolator(interpolation=:sciml, max_alt_km=70.0),
        msis = NRLMSISEInterpolator(min_alt_km=0.0, max_alt_km=100.0),
        wam = WAMInterpolator(interpolation=:sciml),
        msis_max_alt_km = 70.0,
    )
end

# Optimised and parallel batch dispatch.
# These were defined in the original optimized_functions.jl; kept here as simple
# dispatch wrappers for API compatibility.

"""
    get_density_batch_optimized(args...; kwargs...) -> Vector{Float64}

**American spelling** alias of `get_density_batch`. Identical behaviour; exists
for callers that prefer American spelling. See `get_density_trajectory_optimised`
for the British spelling variant.
"""
get_density_batch_optimized(args...; kwargs...) = get_density_batch(args...; kwargs...)

"""
    get_density_batch_parallel(args...; kwargs...) -> Vector{Float64}

**American spelling** alias of `get_density_batch`. Currently a thin wrapper
(no internal parallelism); the name is preserved for API compatibility with
earlier versions and may gain real parallelism in a future release.
"""
get_density_batch_parallel(args...; kwargs...)    = get_density_batch(args...; kwargs...)
