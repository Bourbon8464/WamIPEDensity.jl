# backends/msis.jl - NRLMSISE-00 backend.
# Empirical density model (Naval Research Laboratory Mass Spectrometer
# and Incoherent Scatter Radar Exosphere, 2000 version).

# --------------------------------------------------------------------------
# One-time index initialisation
# --------------------------------------------------------------------------

function _init_msis_indices!(itp::NRLMSISEInterpolator)
    lock(_MSIS_INIT_LOCK) do
        _MSIS_INITIALIZED[] && return nothing
        try
            SpaceIndices.init()
        catch err
            @warn "SpaceIndices.init() failed; NRLMSISE-00 may still work with explicit indices." exception=(err, catch_backtrace())
        end
        _MSIS_INITIALIZED[] = true
        return nothing
    end
end

# --------------------------------------------------------------------------
# Single-point density
# --------------------------------------------------------------------------

"""
    get_density(itp::NRLMSISEInterpolator, dt, lat, lon, alt_km) -> Float64

Return density from the NRLMSISE-00 empirical model. Default window
0-100 km; widen with `NRLMSISEInterpolator(min_alt_km=0.0, max_alt_km=...)`.
Degrees, kilometres. Triggers a one-time space-weather index download
under a module-level lock on first call.
"""
function get_density(itp::NRLMSISEInterpolator, dt::DateTime,
                    latq::Real, lonq::Real, alt_km::Real)
    _validate_query_args_msis(itp, dt, latq, lonq, alt_km)
    _init_msis_indices!(itp)
    out = SatelliteToolboxAtmosphericModels.AtmosphericModels.nrlmsise00(
        dt, alt_km * 1000.0, deg2rad(float(latq)), deg2rad(float(lonq))
    )
    return float(out.total_density)
end

# --------------------------------------------------------------------------
# Batch and trajectory
# --------------------------------------------------------------------------

"""
    get_density_batch(itp::NRLMSISEInterpolator, dts, lats, lons, alts_km)
        -> Vector{Float64}

Vector-form NRLMSISE-00 density query.
"""
function get_density_batch(itp::NRLMSISEInterpolator,
                           dts::AbstractVector{<:DateTime},
                           lats::AbstractVector,
                           lons::AbstractVector,
                           alts_km::AbstractVector)
    n = length(dts)
    @assert length(lats) == n == length(lons) == length(alts_km)
    results = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        results[i] = get_density(itp, dts[i], lats[i], lons[i], alts_km[i])
    end
    return results
end

get_density_trajectory(itp::NRLMSISEInterpolator, dts, lats, lons, alts_m; angles_in_deg=false) =
    get_density_batch(itp, dts,
        Float64.(lats), Float64.(lons), Float64.(alts_m) .* 1e-3)

get_density_trajectory_optimised(itp::NRLMSISEInterpolator, dts, lats, lons, alts_m; angles_in_deg=false) =
    get_density_trajectory(itp, dts, lats, lons, alts_m; angles_in_deg=angles_in_deg)

# --------------------------------------------------------------------------
# Point helper
# --------------------------------------------------------------------------

"""
    get_density_at_point(itp::NRLMSISEInterpolator, dt, lat, lon, alt_m;
                         angles_in_deg=false) -> Float64

Single-point NRLMSISE-00 query in orbit-propagator units (metres, radians).
"""
function get_density_at_point(itp::NRLMSISEInterpolator, dt::DateTime,
                             lat::Real, lon::Real, alt_m::Real;
                             angles_in_deg::Bool=false)
    lat_d = angles_in_deg ? float(lat) : rad2deg(float(lat))
    lon_d = angles_in_deg ? float(lon) : rad2deg(float(lon))
    alt_k = float(alt_m) * 1e-3
    return get_density(itp, dt, lat_d, lon_d, alt_k)
end

# --------------------------------------------------------------------------
# Cache prewarm
# --------------------------------------------------------------------------

"""
    prewarm_cache!(itp::NRLMSISEInterpolator, dts) -> Int

NRLMSISE-00 is an empirical model with no on-disc file cache - this method
exists for API completeness and returns `0` unconditionally.
"""
function prewarm_cache!(itp::NRLMSISEInterpolator, dts::AbstractVector{<:DateTime})
    _init_msis_indices!(itp)
    return 0
end
