# backends/hybrid.jl - Hybrid density interpolator.
# Routes queries by altitude: GEOS-FP (low) -> NRLMSISE (mid) -> WAM-IPE (high).

# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

function _select_backend(itp::HybridDensityInterpolator, dt::DateTime, alt_km::Real)
    if alt_km > itp.msis_max_alt_km
        return :wam
    end
    cache_key = _datetime_floor_3hr(dt)
    bounds = lock(_GEOS_BOUNDS_LOCK) do
        get(itp.geos_bounds_cache, cache_key, nothing)
    end
    if bounds === nothing
        bounds = try
            _geos_altitude_bounds(itp.geos, dt)
        catch
            # GEOS download failed (offline or network error).
            # Use conservative bounds: 0-70 km is the nominal GEOS-FP range.
            (0.0, 70.0)
        end
        lock(_GEOS_BOUNDS_LOCK) do
            itp.geos_bounds_cache[cache_key] = bounds
        end
    end
    geos_zmin, geos_zmax = bounds
    return (geos_zmin <= alt_km <= geos_zmax) ? :geos : :msis
end

# --------------------------------------------------------------------------
# Single-point density
# --------------------------------------------------------------------------

"""
    get_density(itp::HybridDensityInterpolator, dt, lat, lon, alt_km) -> Float64

Return density from the **altitude-routing hybrid backend**. Internally
selects GEOS-FP, NRLMSISE-00, or WAM-IPE based on the requested altitude
and the dynamic GEOS ceiling. Same units as the per-backend `get_density`:
degrees, kilometres.
"""
function get_density(itp::HybridDensityInterpolator, dt::DateTime,
                    latq::Real, lonq::Real, alt_km::Real)
    backend = _select_backend(itp, dt, alt_km)
    if backend === :geos
        return get_density(itp.geos, dt, latq, lonq, alt_km)
    elseif backend === :msis
        return get_density(itp.msis, dt, latq, lonq, alt_km)
    else
        return get_density(itp.wam, dt, latq, lonq, alt_km)
    end
end

# --------------------------------------------------------------------------
# Batch and trajectory
# --------------------------------------------------------------------------

"""
    get_density_batch(itp::HybridDensityInterpolator, dts, lats, lons, alts_km)
        -> Vector{Float64}

Vector-form density query through the hybrid backend. Same units as
`get_density`; routing decision is made per element.
"""
function get_density_batch(itp::HybridDensityInterpolator,
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

get_density_trajectory(itp::HybridDensityInterpolator, dts, lats, lons, alts_m; angles_in_deg=false) =
    get_density_batch(itp, dts, lats, lons, Float64.(alts_m) .* 1e-3)

get_density_trajectory_optimised(itp::HybridDensityInterpolator, dts, lats, lons, alts_m; angles_in_deg=false) =
    get_density_trajectory(itp, dts, lats, lons, alts_m; angles_in_deg=angles_in_deg)

# --------------------------------------------------------------------------
# Point helper
# --------------------------------------------------------------------------

"""
    get_density_at_point(itp::HybridDensityInterpolator, dt, lat, lon, alt_m;
                         angles_in_deg=false) -> Float64

Single-point hybrid query in orbit-propagator units (metres, radians).
"""
function get_density_at_point(itp::HybridDensityInterpolator, dt::DateTime,
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
    prewarm_cache!(itp::HybridDensityInterpolator, dts) -> Int

Pre-download the WAM-IPE file pairs needed by the hybrid backend for the
given timestamps. Returns the number of unique file pairs touched.
"""
function prewarm_cache!(itp::HybridDensityInterpolator,
                        dts::AbstractVector{<:DateTime},
                        alts_km::AbstractVector)
    @assert length(dts) == length(alts_km)
    geos_dts = DateTime[]; msis_dts = DateTime[]; wam_dts = DateTime[]
    for i in eachindex(dts, alts_km)
        backend = _select_backend(itp, dts[i], alts_km[i])
        if backend === :geos; push!(geos_dts, dts[i])
        elseif backend === :msis; push!(msis_dts, dts[i])
        else; push!(wam_dts, dts[i])
        end
    end
    geos_n = isempty(geos_dts) ? 0 : prewarm_cache!(itp.geos, geos_dts)
    msis_n = isempty(msis_dts) ? 0 : prewarm_cache!(itp.msis, msis_dts)
    wam_n  = isempty(wam_dts)  ? 0 : prewarm_cache!(itp.wam,  wam_dts)
    return (geos=geos_n, msis=msis_n, wam=wam_n)
end
