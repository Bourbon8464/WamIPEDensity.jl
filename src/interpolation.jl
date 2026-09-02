# interpolation.jl - interpolation primitives.
# Separable 3D/4D interpolation in lon, lat, z (and time).
# All code in British English.

# --------------------------------------------------------------------------
# Grid convention helpers
# --------------------------------------------------------------------------

@inline function _grid_uses_360(lon::AbstractVector)
    return maximum(lon) > 180
end

@inline function _wrap_lon_for_grid(lon::AbstractVector, lonq::Real)
    if _grid_uses_360(lon)
        return lonq < 0 ? lonq + 360 : lonq
    else
        return lonq > 180 ? lonq - 360 : lonq
    end
end

@inline _nearest_index(vec::AbstractVector, x::Real) = findmin(abs.(vec .- x))[2]

# --------------------------------------------------------------------------
# 3-D separable linear interpolation (single time slice)
# --------------------------------------------------------------------------

@inline function _interp3_linear(lat::AbstractVector, lon::AbstractVector,
                                 z::AbstractVector,
                                 Vt::AbstractArray{<:Real,3},
                                 latq::Real, lonq::Real, zq::Real)
    lonq2 = _wrap_lon_for_grid(lon, lonq)

    if length(z) == 1
        Vz = Vt[:, :, 1]
    else
        iz = clamp(searchsortedlast(z, zq), 1, length(z)-1)
        z1, z2 = z[iz], z[iz+1]
        theta_z = (zq - z1) / (z2 - z1)
        Vz1 = Vt[:, :, iz]
        Vz2 = Vt[:, :, iz+1]
        @views Vz = (1-theta_z) .* Vz1 .+ theta_z .* Vz2
    end

    if length(lat) == 1
        Vphi = Vz[:, 1]
    else
        ilat = clamp(searchsortedlast(lat, latq), 1, length(lat)-1)
        phi1, phi2 = lat[ilat], lat[ilat+1]
        theta_lat = (latq - phi1) / (phi2 - phi1)
        @views Vphi = (1-theta_lat) .* Vz[:, ilat] .+ theta_lat .* Vz[:, ilat+1]
    end

    if length(lon) == 1
        return Vphi[1]
    else
        ilon = clamp(searchsortedlast(lon, lonq2), 1, length(lon)-1)
        lon1, lon2 = lon[ilon], lon[ilon+1]
        theta_lon = (lonq2 - lon1) / (lon2 - lon1)
        return (1-theta_lon) * Vphi[ilon] + theta_lon * Vphi[ilon+1]
    end
end

# --------------------------------------------------------------------------
# 3-D log-z linear interpolation
# --------------------------------------------------------------------------

@inline function _interp3_logz_linear(lat::AbstractVector, lon::AbstractVector,
                                       z::AbstractVector,
                                       Vt::AbstractArray{<:Real,3},
                                       latq::Real, lonq::Real, zq::Real)
    lonq2 = _wrap_lon_for_grid(lon, lonq)

    iz = clamp(searchsortedlast(z, zq), 1, length(z)-1)
    z1, z2 = z[iz], z[iz+1]
    theta_z = (zq - z1) / (z2 - z1)

    Vz1_raw = Vt[:, :, iz]
    Vz2_raw = Vt[:, :, iz+1]

    if any(!isfinite, (z1, z2)) || z1 <= 0 || z2 <= 0 ||
       any(x -> x <= 0 || !isfinite(x), Vz1_raw) ||
       any(x -> x <= 0 || !isfinite(x), Vz2_raw)
        Vz = (1-theta_z) .* Vz1_raw .+ theta_z .* Vz2_raw
    else
        logz1 = log(z1); logz2 = log(z2); logzq = log(zq)
        theta_z_log = (logzq - logz1) / (logz2 - logz1)
        Vz1 = log.(Vz1_raw)
        Vz2 = log.(Vz2_raw)
        @views Vz_log = (1-theta_z_log) .* Vz1 .+ theta_z_log .* Vz2
        Vz = exp.(Vz_log)
    end

    ilat = clamp(searchsortedlast(lat, latq), 1, length(lat)-1)
    phi1, phi2 = lat[ilat], lat[ilat+1]
    theta_lat = (latq - phi1) / (phi2 - phi1)
    @views Vphi = (1-theta_lat) .* Vz[:, ilat] .+ theta_lat .* Vz[:, ilat+1]

    ilon = clamp(searchsortedlast(lon, lonq2), 1, length(lon)-1)
    lon1, lon2 = lon[ilon], lon[ilon+1]
    theta_lon = (lonq2 - lon1) / (lon2 - lon1)
    return (1-theta_lon) * Vphi[ilon] + theta_lon * Vphi[ilon+1]
end

# --------------------------------------------------------------------------
# Bilinear in lon/lat at a single z level
# --------------------------------------------------------------------------

@inline function _bilinear_lonlat(lat::AbstractVector, lon::AbstractVector,
                                   grid::AbstractArray{<:Real,2},
                                   latq::Real, lonq::Real)
    lonq2 = _wrap_lon_for_grid(lon, lonq)
    ilat = clamp(searchsortedlast(lat, latq), 1, length(lat)-1)
    phi1, phi2 = lat[ilat], lat[ilat+1]
    theta_lat = (latq - phi1) / (phi2 - phi1)
    ilon = clamp(searchsortedlast(lon, lonq2), 1, length(lon)-1)
    lon1, lon2 = lon[ilon], lon[ilon+1]
    theta_lon = (lonq2 - lon1) / (lon2 - lon1)
    v11 = grid[ilon,   ilat  ]; v21 = grid[ilon+1, ilat  ]
    v12 = grid[ilon,   ilat+1]; v22 = grid[ilon+1, ilat+1]
    return (1-theta_lon)*(1-theta_lat)*v11 + theta_lon*(1-theta_lat)*v21 +
           (1-theta_lon)*theta_lat*v12     + theta_lon*theta_lat*v22
end

# --------------------------------------------------------------------------
# SciML vertical: quadratic in log(z)-log(v)
# --------------------------------------------------------------------------

function _sciml_quad_logz(z::AbstractVector, v::AbstractVector, zq::Real)
    mask = (z .> 0) .& isfinite.(z) .& (v .> 0) .& isfinite.(v)
    z_ok = Float64.(z[mask])
    v_ok = Float64.(v[mask])

    if length(z_ok) == 0
        return NaN
    elseif length(z_ok) == 1
        return v_ok[1]
    end

    p = sortperm(z_ok)
    z_ok = z_ok[p]
    v_ok = v_ok[p]
    zq_clamped = clamp(float(zq), first(z_ok), last(z_ok))
    if zq != zq_clamped
        @debug "Clamped WAM altitude" requested_km=zq used_km=zq_clamped zmin=first(z_ok) zmax=last(z_ok)
    end

    if length(z_ok) == 2
        itp = DataInterpolations.LinearInterpolation(log.(v_ok), log.(z_ok))
        return exp(itp(log(zq_clamped)))
    else
        itp = DataInterpolations.QuadraticSpline(log.(v_ok), log.(z_ok))
        return exp(itp(log(zq_clamped)))
    end
end

@inline function _interp3_bilin_then_quadlogz(lat::AbstractVector, lon::AbstractVector,
                                              z::AbstractVector,
                                              Vt::AbstractArray{<:Real,3},
                                              latq::Real, lonq::Real, zq::Real)
    v_at_levels = Vector{Float64}(undef, length(z))
    @inbounds for k in eachindex(z)
        v_at_levels[k] = _bilinear_lonlat(lat, lon, Vt[:, :, k], latq, lonq)
    end
    return _sciml_quad_logz(z, v_at_levels, zq)
end

# --------------------------------------------------------------------------
# 4-D interpolation (lon, lat, z, time)
# --------------------------------------------------------------------------

@inline function _interp4(lat, lon, z, t, V, latq, lonq, zq, tq; mode::Symbol=:nearest)
    lonq2 = _wrap_lon_for_grid(lon, lonq)

    if length(t) == 1
        if mode == :linear
            return _interp3_linear(lat, lon, z, V[:, :, :, 1], latq, lonq2, zq)
        elseif mode == :logz_linear
            return _interp3_logz_linear(lat, lon, z, V[:, :, :, 1], latq, lonq2, zq)
        elseif mode == :logz_quadratic
            return _interp3_bilin_then_quadlogz(lat, lon, z, V[:, :, :, 1], latq, lonq2, zq)
        else
            ilat = _nearest_index(lat, latq)
            ilon = _nearest_index(lon, lonq2)
            iz   = _nearest_index(z,   zq)
            return V[ilon, ilat, iz, 1]
        end
    end

    if mode == :nearest
        ilat = _nearest_index(lat, latq)
        ilon = _nearest_index(lon, lonq2)
        iz   = _nearest_index(z,   zq)
        it   = _nearest_index(t,   tq)
        return V[ilon, ilat, iz, it]

    elseif mode == :linear
        it = clamp(searchsortedlast(t, tq), 1, length(t)-1)
        theta_t = (tq - t[it]) / (t[it+1] - t[it])
        V1 = V[:, :, :, it]; V2 = V[:, :, :, it+1]
        v1 = _interp3_linear(lat, lon, z, V1, latq, lonq2, zq)
        v2 = _interp3_linear(lat, lon, z, V2, latq, lonq2, zq)
        return (1-theta_t)*v1 + theta_t*v2

    elseif mode == :logz_linear
        it = clamp(searchsortedlast(t, tq), 1, length(t)-1)
        theta_t = (tq - t[it]) / (t[it+1] - t[it])
        V1 = V[:, :, :, it]; V2 = V[:, :, :, it+1]
        v1 = _interp3_logz_linear(lat, lon, z, V1, latq, lonq2, zq)
        v2 = _interp3_logz_linear(lat, lon, z, V2, latq, lonq2, zq)
        return (1-theta_t)*v1 + theta_t*v2

    elseif mode == :logz_quadratic
        it = clamp(searchsortedlast(t, tq), 1, length(t)-1)
        theta_t = (tq - t[it]) / (t[it+1] - t[it])
        V1 = V[:, :, :, it]; V2 = V[:, :, :, it+1]
        v1 = _interp3_bilin_then_quadlogz(lat, lon, z, V1, latq, lonq2, zq)
        v2 = _interp3_bilin_then_quadlogz(lat, lon, z, V2, latq, lonq2, zq)
        return (1-theta_t)*v1 + theta_t*v2

    else
        error("Unsupported interpolation mode: $mode (use :nearest, :linear, :logz_linear, or :logz_quadratic)")
    end
end

# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------

_normalise_interp(s::Symbol) = (s === :sciml ? :logz_quadratic : s)

function _validate_query_args(interp::Symbol, dt::DateTime,
                              latq::Real, lonq::Real, alt_km::Real)::Symbol
    mode = _normalise_interp(interp)
    mode in _ALLOWED_INTERP_NORM ||
        throw(ArgumentError("interpolation must be one of $(collect(_ALLOWED_INTERP_NORM)) or :sciml; got $interp"))
    isfinite(latq) && -90.0 <= latq <= 90.0 ||
        throw(ArgumentError("lat must be finite and in [-90, 90]; got $latq"))
    isfinite(lonq) || throw(ArgumentError("lon must be finite; got $lonq"))
    isfinite(alt_km) || throw(ArgumentError("alt_km must be finite; got $alt_km"))
    alt_km > 0 || throw(ArgumentError("alt_km must be > 0 km; got $alt_km"))
    return mode
end

function _validate_query_args_geos(itp::GEOSFPInterpolator, dt::DateTime,
                                   latq::Real, lonq::Real, alt_km::Real)::Symbol
    mode = _validate_query_args(itp.interpolation, dt, latq, lonq, alt_km)
    itp.min_alt_km <= alt_km <= itp.max_alt_km ||
        throw(ArgumentError("GEOS-FP backend only supports $(itp.min_alt_km)-$(itp.max_alt_km) km; got $alt_km km"))
    return mode
end

function _validate_query_args_msis(itp::NRLMSISEInterpolator, dt::DateTime,
                                   latq::Real, lonq::Real, alt_km::Real)
    isfinite(latq) && -90.0 <= latq <= 90.0 ||
        throw(ArgumentError("lat must be finite and in [-90, 90]; got $latq"))
    isfinite(lonq) || throw(ArgumentError("lon must be finite; got $lonq"))
    isfinite(alt_km) || throw(ArgumentError("alt_km must be finite; got $alt_km"))
    itp.min_alt_km <= alt_km <= itp.max_alt_km ||
        throw(ArgumentError("NRLMSISE backend only supports $(itp.min_alt_km)-$(itp.max_alt_km) km; got $alt_km km"))
    return :nearest
end

# --------------------------------------------------------------------------
# Per-point WAM-IPE density evaluation
# --------------------------------------------------------------------------

@inline function _decode_value(val::Float64, meta::GridMetadata)
    if isnan(val) || ismissing(val) || val in meta.fill_values
        return NaN
    end
    return val * meta.scale_factor + meta.add_offset
end

@inline function _get_bracket(arr, val)
    n = length(arr)
    if val <= arr[1]
        return (1, 2)
    end
    if val >= arr[n]
        return (n-1, n)
    end
    i = searchsortedfirst(arr, val)
    return (i-1, i)
end

@inline function _get_density_wam_core(meta::GridMetadata,
                                       lon_q::Float64,
                                       lat_q::Float64,
                                       z_q::Float64)
    if meta.lon_is_360 && lon_q < 0.0
        lon_q = lon_q + 360.0
    end

    il, ih = _get_bracket(meta.lon, lon_q)
    jl, jh = _get_bracket(meta.lat, lat_q)
    kl, kh = _get_bracket(meta.z,   z_q)

    idx = Vector{Any}(undef, meta.ndims)
    for d in 1:meta.ndims
        if      get(meta.dim_map, :lon,  0) == d; idx[d] = il:ih
        elseif  get(meta.dim_map, :lat,  0) == d; idx[d] = jl:jh
        elseif  get(meta.dim_map, :z,    0) == d; idx[d] = kl:kh
        else;                                        idx[d] = 1:1
        end
    end

    try
        raw = meta.ds[meta.varname][idx...]
        v = Array{Float64}(undef, 2, 2, 2)
        lon_dim = meta.dim_map[:lon]; lat_dim = meta.dim_map[:lat]; z_dim = meta.dim_map[:z]

        for li in 1:2, lj in 1:2, lk in 1:2
            raw_idx = ones(Int, meta.ndims)
            raw_idx[lon_dim] = li; raw_idx[lat_dim] = lj; raw_idx[z_dim] = lk
            val = raw[raw_idx...]
            v[li, lj, lk] = _decode_value(ismissing(val) ? NaN : Float64(val), meta)
        end

        x   = (lon_q - meta.lon[il]) / (meta.lon[ih] - meta.lon[il])
        y   = (lat_q - meta.lat[jl]) / (meta.lat[jh] - meta.lat[jl])
        z_w = (z_q   - meta.z[kl])   / (meta.z[kh]   - meta.z[kl])

        c00 = v[1,1,1] * (1-z_w) + v[1,1,2] * z_w
        c01 = v[1,2,1] * (1-z_w) + v[1,2,2] * z_w
        c10 = v[2,1,1] * (1-z_w) + v[2,1,2] * z_w
        c11 = v[2,2,1] * (1-z_w) + v[2,2,2] * z_w
        c0  = c00 * (1-y) + c01 * y
        c1  = c10 * (1-y) + c11 * y
        return c0 * (1-x) + c1 * x

    catch err
        @warn "Error in WAM-IPE micro-slice interpolation; returning NaN" exception=(err, catch_backtrace())
        return NaN
    end
end
