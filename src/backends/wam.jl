# backends/wam.jl - WAM-IPE backend.
# WAMInterpolator, file resolution, single-point and trajectory density.

# --------------------------------------------------------------------------
# File resolution
# --------------------------------------------------------------------------

function _aws_cfg_wam(region="us-east-1")
    AWS.AWSConfig(; region=region, creds=nothing)
end

function _get_two_files_exact(itp::WAMInterpolator, dt::DateTime)
    bucket = _datetime_floor_10min(dt)
    if (cached_pair = _get_cached_file_pair(bucket)) !== nothing
        p_lo, p_hi = cached_pair
        isfile(p_lo) && isfile(p_hi) || @goto refresh
        prod_lo = occursin("wfs", p_lo) ? "wfs" : "wrs"
        prod_hi = occursin("wfs", p_hi) ? "wfs" : "wrs"
        return (p_lo, p_hi, prod_lo, prod_hi)
    end
    @label refresh

    dt_lo, dt_hi = _surrounding_10min(dt)
    pref, alt    = _product_fallback_order(itp.product)
    aws          = _aws_cfg_wam(itp.region)

    local _ensure(key) = let lp = normpath(joinpath(DEFAULT_CACHE_DIR, key))
        isfile(lp) ? lp : _download_to_cache(aws, itp.bucket, key; verbose=false)
    end

    function _try_product(dt_file::DateTime, product::String)
        key = _construct_s3_key(dt_file, product)
        try
            return _ensure(key), product
        catch err
            @debug "Could not fetch WAM-IPE product" key=key product=product exception=(err, catch_backtrace())
            return nothing, product
        end
    end

    if itp.product == "wrs"
        aws = _aws_cfg_wam(itp.region)
        function _try_fetch_wrs(dt_file::DateTime)
            for cycle_hh in _wrs_cycles(dt_file)
                # 18z after midnight belongs to previous day's folder
                cycle_date = if cycle_hh == 18 && hour(dt_file) < 3
                    Date(dt_file) - Day(1)
                else
                    Date(dt_file)
                end
                arch = DateTime(cycle_date, Time(cycle_hh))
                key = _construct_s3_key(dt_file, "wrs", arch)
                lp = normpath(joinpath(DEFAULT_CACHE_DIR, key))
                if isfile(lp)
                    return lp, "wrs"
                end
                try
                    p = _download_to_cache(aws, itp.bucket, key; verbose=false)
                    return p, "wrs"
                catch err
                    @debug "WRS cycle $cycle_hh failed for $dt_file" key=key exception=(err, catch_backtrace())
                end
            end
            return nothing, "wrs"
        end
        p_lo, prod_lo = _try_fetch_wrs(dt_lo)
        p_hi, prod_hi = _try_fetch_wrs(dt_hi)
        (p_lo === nothing || p_hi === nothing) &&
            error("Could not fetch WRS files for low=$dt_lo, high=$dt_hi.")
    else
        p_lo, prod_lo = _try_product(dt_lo, pref)
        if p_lo === nothing && pref != alt
            p_lo, prod_lo = _try_product(dt_lo, alt)
        end
        p_hi, prod_hi = _try_product(dt_hi, pref)
        if p_hi === nothing && pref != alt
            p_hi, prod_hi = _try_product(dt_hi, alt)
        end
        (p_lo === nothing || p_hi === nothing) &&
            error("Could not fetch files for low=$dt_lo, high=$dt_hi (tried $pref, $alt).")
    end

    pair = (p_lo, p_hi)
    _cache_file_pair(dt, pair)
    return (p_lo, p_hi, prod_lo, prod_hi)
end

# --------------------------------------------------------------------------
# Single-point density
# --------------------------------------------------------------------------

"""
    get_density(itp::WAMInterpolator, dt, lat, lon, alt_km) -> Float64

Return neutral atmospheric density (kg/m^3) from WAM-IPE at one (time, location,
altitude) point. Latitude and longitude are in **degrees**; altitude in
**kilometres** above mean sea level. The two bracketing 10-minute WAM-IPE
files are downloaded (if not cached) and linearly interpolated in time.
"""
function get_density(itp::WAMInterpolator, dt::DateTime,
                     latq::Real, lonq::Real, alt_km::Real)
    _validate_query_args(itp.interpolation, dt, latq, lonq, alt_km)
    p_lo, p_hi, _, _ = _get_two_files_exact(itp, dt)

    ds_lo = _open_nc(p_lo)
    ds_hi = _open_nc(p_hi)

    t_lo = _parse_valid_time_from_key(p_lo)
    t_hi = _parse_valid_time_from_key(p_hi)
    t_lo === nothing && (t_lo = t_hi)
    t_hi === nothing && (t_hi = t_lo)

    meta_lo = _get_cached_metadata(p_lo, ds_lo, itp.varname)
    meta_hi = _get_cached_metadata(p_hi, ds_hi, itp.varname)

    v_lo = _get_density_wam_core(meta_lo, Float64(lonq), Float64(latq), Float64(alt_km))
    v_hi = _get_density_wam_core(meta_hi, Float64(lonq), Float64(latq), Float64(alt_km))

    if t_lo == t_hi
        return float(v_lo)
    else
        t_lo_val = Float64(Dates.value(t_lo))
        t_hi_val = Float64(Dates.value(t_hi))
        theta = (Float64(Dates.value(dt)) - t_lo_val) / (t_hi_val - t_lo_val)
        return (1.0 - theta) * float(v_lo) + theta * float(v_hi)
    end
end

# --------------------------------------------------------------------------
# Batch density
# --------------------------------------------------------------------------

"""
    get_density_batch(itp, dts, lats, lons, alts_km) -> Vector{Float64}

Vector-form density query. Inputs are parallel arrays; lats/lons in degrees,
alts in kilometres. Returns one density per timestamp.
"""
function get_density_batch(itp::WAMInterpolator,
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

"""
    get_density_batch!(itp, dts, lats, lons, alts_km, out) -> Vector{Float64}

In-place batch query. Fills the pre-allocated `out` vector and returns it;
avoids a per-call allocation.
"""
function get_density_batch!(itp, dts::AbstractVector{<:DateTime},
                            lats::AbstractVector, lons::AbstractVector,
                            alts_km::AbstractVector, out::AbstractVector{Float64})
    n = length(dts)
    @assert length(out) >= n && length(lats) == n == length(lons) == length(alts_km)
    Threads.@threads for i in 1:n
        @inbounds out[i] = get_density(itp, dts[i], lats[i], lons[i], alts_km[i])
    end
    return out
end

# --------------------------------------------------------------------------
# Trajectory (optimised: group by file pair)
# --------------------------------------------------------------------------

"""
    get_density_trajectory_optimised(itp, dts, lats, lons, alts_m;
                                     angles_in_deg=false) -> Vector{Float64}

Optimised trajectory query. Groups trajectory points by the WAM-IPE file
pair each point needs, loads each pair only once, and evaluates all points
in that group before moving on. Designed for orbit-propagator output:
`alts_m` in metres, `lats`/`lons` in **radians** by default - pass
`angles_in_deg=true` to switch to degrees.

**British spelling** (`optimised`); the American-spelled alias
`get_density_trajectory_optimized` also exists.
"""
function get_density_trajectory_optimised(itp::WAMInterpolator,
                                         dts::AbstractVector{<:DateTime},
                                         lats::AbstractVector,
                                         lons::AbstractVector,
                                         alts_m::AbstractVector;
                                         angles_in_deg::Bool=false)
    n = length(dts)
    @assert length(lats) == n == length(lons) == length(alts_m)

    latv  = Vector{Float64}(undef, n)
    lonv  = Vector{Float64}(undef, n)
    altkm = Vector{Float64}(undef, n)

    if angles_in_deg
        @inbounds @simd for i in 1:n
            latv[i]  = Float64(lats[i])
            lonv[i]  = Float64(lons[i])
            altkm[i] = Float64(alts_m[i]) * 1e-3
        end
    else
        @inbounds @simd for i in 1:n
            latv[i]  = rad2deg(Float64(lats[i]))
            lonv[i]  = rad2deg(Float64(lons[i]))
            altkm[i] = Float64(alts_m[i]) * 1e-3
        end
    end

    @inbounds for i in 1:n
        _validate_query_args(itp.interpolation, dts[i], latv[i], lonv[i], altkm[i])
    end

    file_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for i in 1:n
        p_lo, p_hi, _, _ = _get_two_files_exact(itp, dts[i])
        push!(get!(file_groups, (p_lo, p_hi), Int[]), i)
    end

    results = Vector{Float64}(undef, n)

    for ((p_lo, p_hi), indices) in file_groups
        ds_lo = _open_nc(p_lo)
        ds_hi = _open_nc(p_hi)

        t_lo = _parse_valid_time_from_key(p_lo)
        t_hi = _parse_valid_time_from_key(p_hi)
        t_lo === nothing && (t_lo = t_hi)
        t_hi === nothing && (t_hi = t_lo)

        meta_lo = _get_cached_metadata(p_lo, ds_lo, itp.varname)
        meta_hi = _get_cached_metadata(p_hi, ds_hi, itp.varname)

        same_time   = (t_lo == t_hi)
        t_lo_val    = same_time ? 0.0 : Float64(Dates.value(t_lo))
        t_hi_val    = same_time ? 0.0 : Float64(Dates.value(t_hi))
        t_delta_inv = same_time ? 0.0 : 1.0 / (t_hi_val - t_lo_val)

        for idx in indices
            v_lo = _get_density_wam_core(meta_lo, lonv[idx], latv[idx], altkm[idx])
            v_hi = _get_density_wam_core(meta_hi, lonv[idx], latv[idx], altkm[idx])
            if same_time
                results[idx] = float(v_lo)
            else
                theta = (Float64(Dates.value(dts[idx])) - t_lo_val) * t_delta_inv
                results[idx] = (1.0 - theta) * float(v_lo) + theta * float(v_hi)
            end
        end
    end

    return results
end

"""
    get_density_trajectory(itp, dts, lats, lons, alts_m;
                           angles_in_deg=false) -> Vector{Float64}

Plain trajectory query (thin wrapper around `get_density_batch`). Use
`get_density_trajectory_optimised` instead for dense trajectories, where
file-pair grouping eliminates redundant file opens.
"""
function get_density_trajectory(itp::WAMInterpolator,
                               dts::AbstractVector{<:DateTime},
                               lats::AbstractVector,
                               lons::AbstractVector,
                               alts_m::AbstractVector;
                               angles_in_deg::Bool=false)
    return get_density_trajectory_optimised(itp, dts, lats, lons, alts_m; angles_in_deg=angles_in_deg)
end

# --------------------------------------------------------------------------
# Point helper
# --------------------------------------------------------------------------

"""
    get_density_at_point(itp, dt, lat, lon, alt_m; angles_in_deg=false) -> Float64

Single-point query in orbit-propagator units: `alt_m` in metres, `lats`/`lons`
in **radians** by default. Pass `angles_in_deg=true` to switch to degrees.
"""
function get_density_at_point(itp::WAMInterpolator, dt::DateTime,
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
    prewarm_cache!(itp::WAMInterpolator, dts) -> Int

Pre-download the WAM-IPE file pairs that the trajectory `dts` will need and
insert them into the on-disc cache. Returns the number of unique file pairs
touched. Call this before the first trajectory batch to avoid blocking on
S3 downloads during evaluation.
"""
function prewarm_cache!(itp::WAMInterpolator, dts::AbstractVector{<:DateTime})
    unique_files = Set{Tuple{String,String}}()
    for dt in dts
        p_lo, p_hi, _, _ = _get_two_files_exact(itp, dt)
        push!(unique_files, (p_lo, p_hi))
    end
    @info "Pre-downloaded WAM-IPE file pairs" n=length(unique_files)
    return length(unique_files)
end
