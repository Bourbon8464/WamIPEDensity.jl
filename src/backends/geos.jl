# backends/geos.jl - GEOS-FP backend.
# Density reconstruction from T, QV, pressure levels via hydrostatic integration.

# --------------------------------------------------------------------------
# GEOS-FP file paths
# --------------------------------------------------------------------------

function _geos_local_path(itp::GEOSFPInterpolator, dt::DateTime)
    yyyy = Dates.format(Date(dt), dateformat"yyyy")
    mm   = Dates.format(Date(dt), dateformat"mm")
    dd   = Dates.format(Date(dt), dateformat"dd")
    hh   = Dates.format(Time(dt), dateformat"HH")
    mkpath(itp.cache_dir)
    return joinpath(itp.cache_dir, itp.collection, yyyy, mm,
                    "GEOSFP_$(itp.collection)_$(yyyy)$(mm)$(dd)_$(hh)00.nc4")
end

function _geos_build_url(itp::GEOSFPInterpolator, dt::DateTime)
    yyyy = Dates.format(Date(dt), dateformat"yyyy")
    mm   = Dates.format(Date(dt), dateformat"mm")
    dd   = Dates.format(Date(dt), dateformat"dd")
    hh   = Dates.format(Time(dt), dateformat"HH")
    return string(itp.root_url,
        "/Y", yyyy, "/M", mm, "/D", dd,
        "/GEOS.fp.asm.", itp.collection, ".",
        yyyy, mm, dd, "_", hh, "00.V01.nc4")
end

# --------------------------------------------------------------------------
# GEOS-FP download
# --------------------------------------------------------------------------

function _geos_download_to_cache(itp::GEOSFPInterpolator, dt::DateTime; verbose::Bool=true)
    local_path = _geos_local_path(itp, dt)
    isfile(local_path) && return local_path

    i_am_downloader, cond = lock(_GEOS_DOWNLOAD_LOCK) do
        isfile(local_path) && return false, nothing
        if local_path in _GEOS_DOWNLOADING
            return false, get!(_GEOS_DOWNLOAD_CONDS, local_path, Condition())
        else
            push!(_GEOS_DOWNLOADING, local_path)
            return true, get!(_GEOS_DOWNLOAD_CONDS, local_path, Condition())
        end
    end

    if !i_am_downloader
        if cond !== nothing
            @debug "Waiting for concurrent GEOS-FP download" path=local_path
            wait(cond)
        end
        isfile(local_path) && return local_path
        error("Concurrent GEOS-FP download did not produce $local_path")
    end

    url = _geos_build_url(itp, dt)
    tmp_path = local_path * ".part"
    mkpath(dirname(local_path))
    verbose && @info "Downloading GEOS-FP file" url=url dest=local_path

    ok = false
    try
        HTTP.open(:GET, url; readtimeout=120) do io
            open(tmp_path, "w") do f
                while !eof(io)
                    write(f, read(io, 1_048_576))
                end
            end
        end
        ok = true
    catch err
        @warn "GEOS download failed for $url" exception=(err, catch_backtrace())
    end

    ok && isfile(tmp_path) && mv(tmp_path, local_path; force=true)
    isfile(tmp_path) && rm(tmp_path; force=true)

    lock(_GEOS_DOWNLOAD_LOCK) do
        delete!(_GEOS_DOWNLOADING, local_path)
        if haskey(_GEOS_DOWNLOAD_CONDS, local_path)
            notify(_GEOS_DOWNLOAD_CONDS[local_path]; all=true)
            delete!(_GEOS_DOWNLOAD_CONDS, local_path)
        end
    end

    ok && isfile(local_path) || error("Failed to download GEOS-FP file for $dt")
    return local_path
end

# --------------------------------------------------------------------------
# File resolution helpers
# --------------------------------------------------------------------------

function _geos_get_two_files_exact(itp::GEOSFPInterpolator, dt::DateTime)
    dt_lo, dt_hi = _geos_surrounding_times(dt)
    p_lo = _geos_download_to_cache(itp, dt_lo; verbose=false)
    p_hi = _geos_download_to_cache(itp, dt_hi; verbose=false)
    return p_lo, p_hi, dt_lo, dt_hi
end

# --------------------------------------------------------------------------
# Dimension classification
# --------------------------------------------------------------------------

function _geos_classify_dim(dname::String, ds::NCDataset)
    lname   = lowercase(dname)
    var     = haskey(ds, dname) ? ds[dname] : nothing
    attrs   = var === nothing ? Dict{String,Any}() : Dict(var.attrib)
    stdname = lowercase(string(get(attrs, "standard_name", "")))
    axis    = uppercase(string(get(attrs, "axis", "")))
    units   = lowercase(string(get(attrs, "units", "")))
    longn   = lowercase(string(get(attrs, "long_name", "")))

    if occursin("time", lname) || axis == "T" || stdname == "time"; return :time; end
    if occursin("lat", lname)  || stdname == "latitude"  || axis == "Y" || occursin("degrees_north", units); return :lat; end
    if occursin("lon", lname)  || stdname == "longitude" || axis == "X" || occursin("degrees_east",  units); return :lon; end
    if occursin("lev", lname)  || occursin("eta", lname)  || occursin("layer", lname) ||
       occursin("height", lname) || occursin("alt", lname)   || lname == "z" || axis == "Z" ||
       occursin("level", longn); return :z; end
    return :unknown
end

function _geos_dim_indices(ds::NCDataset, varname::String)
    haskey(ds, varname) || error("Variable '$varname' not found in GEOS file.")
    v = ds[varname]
    dnames = String.(NCDatasets.dimnames(v))
    lname_to_idx = Dict(lowercase(name) => i for (i, name) in pairs(dnames))

    idx_lon  = get(lname_to_idx, "lon",  nothing)
    idx_lat  = get(lname_to_idx, "lat",  nothing)
    idx_z    = get(lname_to_idx, "lev",  nothing)
    idx_time = get(lname_to_idx, "time", nothing)

    if any(x -> x === nothing, (idx_lon, idx_lat, idx_z, idx_time))
        roles = map(d -> _geos_classify_dim(d, ds), dnames)
        if idx_lon  === nothing; idx_lon  = findfirst(==(:lon),  roles); end
        if idx_lat  === nothing; idx_lat  = findfirst(==(:lat),  roles); end
        if idx_z    === nothing; idx_z    = findfirst(==(:z),    roles); end
        if idx_time === nothing; idx_time = findfirst(==(:time), roles); end
    end

    idx_lon === nothing && error("Could not find longitude dimension for '$varname'. dims=$(dnames)")
    idx_lat === nothing && error("Could not find latitude dimension for '$varname'. dims=$(dnames)")
    idx_z   === nothing && error("Could not find vertical dimension for '$varname'. dims=$(dnames)")
    idx_time === nothing && error("Could not find time dimension for '$varname'. dims=$(dnames)")
    idxs = [idx_lon, idx_lat, idx_z, idx_time]
    length(unique(idxs)) == 4 ||
        error("Duplicate dimension assignments for '$varname': lon=$idx_lon lat=$idx_lat z=$idx_z time=$idx_time. dims=$(dnames)")

    return idx_lon, idx_lat, idx_z, idx_time, dnames
end

# --------------------------------------------------------------------------
# Permutation and coordinate helpers
# --------------------------------------------------------------------------

function _geos_permute4(A::AbstractArray, idx_lon::Int, idx_lat::Int, idx_z::Int, idx_time::Int)
    perm = (idx_lon, idx_lat, idx_z, idx_time)
    return perm == (1,2,3,4) ? Array(A) : Array(PermutedDimsArray(A, perm))
end

_geos_coord_vector(ds::NCDataset, dname::String, fallback_len::Int) =
    haskey(ds, dname) ? collect(ds[dname][:]) : collect(1:fallback_len)

function _geos_level_pressure_pa(ds::NCDataset, itp::GEOSFPInterpolator)
    haskey(ds, itp.lev_varname) || error("GEOS file missing level coordinate '$(itp.lev_varname)'")
    lev_raw = ds[itp.lev_varname][:]
    lev = map(x -> ismissing(x) ? NaN : Float64(x), lev_raw)
    attrs = Dict(ds[itp.lev_varname].attrib)
    units = lowercase(string(get(attrs, "units", "")))
    if occursin("hpa", units) || occursin("millibar", units) || occursin("mb", units)
        return lev .* 100.0
    elseif occursin(r"\bpa\b", units)
        return lev
    else
        error("Unsupported GEOS pressure-level units '$units' on $(itp.lev_varname)")
    end
end

# --------------------------------------------------------------------------
# Hydrostatic fill
# --------------------------------------------------------------------------

function _hydrostatic_fill!(z_km::Vector{Float64}, T_col::Vector{Float64}, p_pa::Vector{Float64})
    nz = length(z_km)
    @assert length(T_col) == nz
    @assert length(p_pa)  == nz
    sfc = argmax(p_pa)

    if !isfinite(z_km[sfc])
        z_km[sfc] = 0.0
    end

    for k in (sfc - 1):-1:1
        T1 = isfinite(T_col[k])     ? T_col[k]     : 250.0
        T2 = isfinite(T_col[k + 1]) ? T_col[k + 1] : T1
        Tmid = 0.5 * (T1 + T2)
        dz_m = R_D_GEOS * Tmid / G0_GEOS * log(p_pa[k + 1] / p_pa[k])
        if !isfinite(z_km[k + 1])
            error("Hydrostatic fill failed: z_km[$(k+1)] is not finite while filling upward.")
        end
        z_km[k] = z_km[k + 1] + dz_m / 1000.0
    end

    for k in (sfc + 1):nz
        T1 = isfinite(T_col[k - 1]) ? T_col[k - 1] : 250.0
        T2 = isfinite(T_col[k])     ? T_col[k]     : T1
        Tmid = 0.5 * (T1 + T2)
        dz_m = R_D_GEOS * Tmid / G0_GEOS * log(p_pa[k - 1] / p_pa[k])
        if !isfinite(z_km[k - 1])
            error("Hydrostatic fill failed: z_km[$(k-1)] is not finite while filling downward.")
        end
        z_km[k] = z_km[k - 1] - dz_m / 1000.0
    end

    any(!isfinite, z_km) && error("Hydrostatic fill produced non-finite z_km values.")
    return z_km
end

# --------------------------------------------------------------------------
# Variable reading helpers
# --------------------------------------------------------------------------

function _geos_read_var(ds::NCDataset, varname::String)
    v = ds[varname]
    raw = Array(v)
    return map(x -> ismissing(x) ? NaN : Float64(x), raw)
end

function _nanmean_profile_lonlat(A::AbstractArray{<:Real,4})
    nz = size(A, 3)
    out = Vector{Float64}(undef, nz)
    @inbounds for k in 1:nz
        acc = 0.0; cnt = 0
        @views for val in A[:, :, k, 1]
            if isfinite(val); acc += val; cnt += 1; end
        end
        out[k] = cnt == 0 ? NaN : acc / cnt
    end
    return out
end

# --------------------------------------------------------------------------
# Grid loading
# --------------------------------------------------------------------------

function _geos_load_grids(ds::NCDataset, itp::GEOSFPInterpolator;
                         file_time::Union{DateTime,Nothing}=nothing)
    idx_lon, idx_lat, idx_z, idx_time, dnames = _geos_dim_indices(ds, itp.t_varname)

    T  = _geos_permute4(_geos_read_var(ds, itp.t_varname),  idx_lon, idx_lat, idx_z, idx_time)
    QV = _geos_permute4(_geos_read_var(ds, itp.qv_varname), idx_lon, idx_lat, idx_z, idx_time)
    H  = _geos_permute4(_geos_read_var(ds, itp.z_varname),  idx_lon, idx_lat, idx_z, idx_time)

    nz = size(T, 3)
    size(T) == size(QV) || error("T shape $(size(T)) != QV shape $(size(QV))")
    size(H, 3) == nz    || error("H vertical size $(size(H,3)) != T vertical size $nz")

    p_lev = _geos_level_pressure_pa(ds, itp)
    length(p_lev) == nz || error("Pressure-level length $(length(p_lev)) != T vertical size $nz")

    h_attrs  = Dict(ds[itp.z_varname].attrib)
    h_units  = lowercase(string(get(h_attrs, "units", "m")))
    h_scale = occursin("km", h_units) ? 1.0 : 1.0 / 1000.0

    H_col = _nanmean_profile_lonlat(H)
    z_km  = H_col .* h_scale

    if any(isnan, z_km)
        @warn "H contains NaN at $(count(isnan, z_km)) levels - using hydrostatic fill"
        T_col = _nanmean_profile_lonlat(T)
        _hydrostatic_fill!(z_km, T_col, p_lev)
    end
    any(!isfinite, z_km) && error("GEOS z_km still contains non-finite values after fill.")

    P  = reshape(p_lev, 1, 1, nz, 1)
    Tv = T .* (1 .+ 0.61 .* QV)
    V  = P ./ (R_D_GEOS .* Tv)

    lonname = dnames[idx_lon]; latname = dnames[idx_lat]
    zname   = dnames[idx_z];   tname   = dnames[idx_time]

    lon = _geos_coord_vector(ds, lonname, size(T, 1))
    lat = _geos_coord_vector(ds, latname, size(T, 2))
    t   = _geos_coord_vector(ds, tname,   size(T, 4))

    if !issorted(z_km)
        perm_z = sortperm(z_km)
        z_km = z_km[perm_z]
        V    = V[:, :, perm_z, :]
    end

    return lat, lon, z_km, t, V, (latname, lonname, zname, tname)
end

# --------------------------------------------------------------------------
# GEOS grid cache
# --------------------------------------------------------------------------

function _get_cached_geos_grids(file_path::String, ds::NCDataset,
                                itp::GEOSFPInterpolator, file_time::Union{DateTime,Nothing})
    lock(_GEOS_GRID_CACHE_LOCK) do
        if haskey(_GEOS_GRID_CACHE, file_path)
            return _GEOS_GRID_CACHE[file_path]
        end
        result = _geos_load_grids(ds, itp; file_time=file_time)
        _GEOS_GRID_CACHE[file_path] = result
        if length(_GEOS_GRID_CACHE) > _MAX_GEOS_GRID_CACHE
            delete!(_GEOS_GRID_CACHE, first(keys(_GEOS_GRID_CACHE)))
        end
        return result
    end
end

# --------------------------------------------------------------------------
# Density interpolation from open dataset
# --------------------------------------------------------------------------

function _geos_interp_density_from_loaded(file_path::String, ds::NCDataset,
                                          itp::GEOSFPInterpolator,
                                          dt::DateTime,
                                          latq::Real, lonq::Real,
                                          alt_km::Real, mode::Symbol)
    lat, lon, z, t, V, (latname, lonname, zname, tname) =
        _get_cached_geos_grids(file_path, ds, itp, dt)

    tdts, epoch, scale = _decode_time_units(ds, tname, t)
    tq = epoch === nothing ? dt : _encode_query_time(dt, epoch, scale)

    zmin = minimum(z); zmax = maximum(z)
    zq = clamp(float(alt_km), zmin, zmax)
    zq != float(alt_km) && @info "Clamped GEOS altitude request" requested_km=alt_km used_km=zq zmin=zmin zmax=zmax

    return _interp4(lat, lon, z, tdts, V, latq, lonq, zq, tq; mode=mode)
end

# --------------------------------------------------------------------------
# Density altitude bounds
# --------------------------------------------------------------------------

function _geos_altitude_bounds(itp::GEOSFPInterpolator, dt::DateTime)
    p_lo, p_hi, t_lo, t_hi = _geos_get_two_files_exact(itp, dt)
    ds_lo = _open_nc(p_lo); ds_hi = _open_nc(p_hi)
    try
        _, _, z_lo, _, _, _ = _geos_load_grids(ds_lo, itp; file_time=t_lo)
        _, _, z_hi, _, _, _ = _geos_load_grids(ds_hi, itp; file_time=t_hi)
        zmin = max(minimum(z_lo), minimum(z_hi))
        zmax = min(maximum(z_lo), maximum(z_hi))
        return (zmin, zmax)
    finally
        # no-op: per-thread pool handles lifetime
    end
end

# --------------------------------------------------------------------------
# Single-point density
# --------------------------------------------------------------------------

"""
    get_density(itp::GEOSFPInterpolator, dt, lat, lon, alt_km) -> Float64

Return density from NASA GEOS-FP. Default altitude window 0-70 km.
Bracketing 3-hour files are downloaded (if not cached) and linearly
interpolated in time. Degrees, kilometres.
"""
function get_density(itp::GEOSFPInterpolator, dt::DateTime,
                    latq::Real, lonq::Real, alt_km::Real)
    mode = _validate_query_args_geos(itp, dt, latq, lonq, alt_km)
    p_lo, p_hi, t_lo, t_hi = _geos_get_two_files_exact(itp, dt)
    ds_lo = _open_nc(p_lo); ds_hi = _open_nc(p_hi)

    v_lo = _geos_interp_density_from_loaded(p_lo, ds_lo, itp, t_lo, latq, lonq, alt_km, mode)
    v_hi = _geos_interp_density_from_loaded(p_hi, ds_hi, itp, t_hi, latq, lonq, alt_km, mode)

    if t_lo == t_hi
        return float(v_lo)
    else
        t0 = Float64(Dates.value(t_lo)); t1 = Float64(Dates.value(t_hi))
        tq = Float64(Dates.value(dt))
        θ  = (tq - t0) / (t1 - t0)
        return (1.0 - θ) * float(v_lo) + θ * float(v_hi)
    end
end

# --------------------------------------------------------------------------
# Batch and trajectory
# --------------------------------------------------------------------------

"""
    get_density_batch(itp::GEOSFPInterpolator, dts, lats, lons, alts_km)
        -> Vector{Float64}

Vector-form density query through GEOS-FP. Same units as `get_density`.
"""
function get_density_batch(itp::GEOSFPInterpolator,
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
    get_density_trajectory_optimised(itp::GEOSFPInterpolator, dts, lats, lons,
                                     alts_m; angles_in_deg=false)
        -> Vector{Float64}

Optimised trajectory query through GEOS-FP. Same grouping strategy as the
WAM-IPE version: trajectory points are grouped by file pair so each
GEOS-FP file is loaded only once. **British spelling** (`optimised`).
"""
function get_density_trajectory_optimised(itp::GEOSFPInterpolator,
                                        dts::AbstractVector{<:DateTime},
                                        lats::AbstractVector,
                                        lons::AbstractVector,
                                        alts_m::AbstractVector;
                                        angles_in_deg::Bool=false)
    n = length(dts)
    @assert length(lats) == n == length(lons) == length(alts_m)

    latv = Vector{Float64}(undef, n); lonv = Vector{Float64}(undef, n)
    altkm = Vector{Float64}(undef, n)
    if angles_in_deg
        @inbounds @simd for i in 1:n
            latv[i] = Float64(lats[i]); lonv[i] = Float64(lons[i])
            altkm[i] = Float64(alts_m[i]) * 1e-3
        end
    else
        @inbounds @simd for i in 1:n
            latv[i] = rad2deg(Float64(lats[i])); lonv[i] = rad2deg(Float64(lons[i]))
            altkm[i] = Float64(alts_m[i]) * 1e-3
        end
    end

    @inbounds for i in 1:n
        _validate_query_args_geos(itp, dts[i], latv[i], lonv[i], altkm[i])
    end

    file_groups = Dict{Tuple{String,String}, Vector{Int}}()
    time_pairs  = Dict{Tuple{String,String}, Tuple{DateTime,DateTime}}()
    for i in 1:n
        p_lo, p_hi, t_lo, t_hi = _geos_get_two_files_exact(itp, dts[i])
        push!(get!(file_groups, (p_lo, p_hi), Int[]), i)
        time_pairs[(p_lo, p_hi)] = (t_lo, t_hi)
    end

    results = Vector{Float64}(undef, n)
    mode = _normalise_interp(itp.interpolation)

    for ((p_lo, p_hi), indices) in file_groups
        ds_lo = _open_nc(p_lo); ds_hi = _open_nc(p_hi)
        t_lo, t_hi = time_pairs[(p_lo, p_hi)]

        lat_lo, lon_lo, z_lo, tarr_lo, V_lo, names_lo = _geos_load_grids(ds_lo, itp; file_time=t_lo)
        lat_hi, lon_hi, z_hi, tarr_hi, V_hi, names_hi = _geos_load_grids(ds_hi, itp; file_time=t_hi)

        tdts_lo, epoch_lo, scale_lo = _decode_time_units(ds_lo, names_lo[4], tarr_lo)
        tdts_hi, epoch_hi, scale_hi = _decode_time_units(ds_hi, names_hi[4], tarr_hi)
        tq_lo = epoch_lo === nothing ? t_lo : _encode_query_time(t_lo, epoch_lo, scale_lo)
        tq_hi = epoch_hi === nothing ? t_hi : _encode_query_time(t_hi, epoch_hi, scale_hi)

        same_time   = (t_lo == t_hi)
        t_lo_val    = same_time ? 0.0 : Float64(Dates.value(t_lo))
        t_hi_val    = same_time ? 0.0 : Float64(Dates.value(t_hi))
        t_delta_inv = same_time ? 0.0 : 1.0 / (t_hi_val - t_lo_val)

        zlo_min, zlo_max = minimum(z_lo), maximum(z_lo)
        zhi_min, zhi_max = minimum(z_hi), maximum(z_hi)

        for idx in indices
            zq_lo = clamp(altkm[idx], zlo_min, zlo_max)
            zq_hi = clamp(altkm[idx], zhi_min, zhi_max)
            v_lo = _interp4(lat_lo, lon_lo, z_lo, tdts_lo, V_lo, latv[idx], lonv[idx], zq_lo, tq_lo; mode=mode)
            v_hi = _interp4(lat_hi, lon_hi, z_hi, tdts_hi, V_hi, latv[idx], lonv[idx], zq_hi, tq_hi; mode=mode)
            if same_time
                results[idx] = float(v_lo)
            else
                θ = (Float64(Dates.value(dts[idx])) - t_lo_val) * t_delta_inv
                results[idx] = (1.0 - θ) * float(v_lo) + θ * float(v_hi)
            end
        end
    end
    return results
end

get_density_trajectory(itp::GEOSFPInterpolator, dts, lats, lons, alts_m; angles_in_deg=false) =
    get_density_trajectory_optimised(itp, dts, lats, lons, alts_m; angles_in_deg=angles_in_deg)

# --------------------------------------------------------------------------
# Point helper
# --------------------------------------------------------------------------

"""
    get_density_at_point(itp::GEOSFPInterpolator, dt, lat, lon, alt_m;
                         angles_in_deg=false) -> Float64

Single-point GEOS-FP query in orbit-propagator units (metres, radians).
"""
function get_density_at_point(itp::GEOSFPInterpolator, dt::DateTime,
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
    prewarm_cache!(itp::GEOSFPInterpolator, dts) -> Int

Pre-download the GEOS-FP 3-hour files needed for the given timestamps.
Returns the number of unique files touched.
"""
function prewarm_cache!(itp::GEOSFPInterpolator, dts::AbstractVector{<:DateTime})
    unique_files = Set{Tuple{String,String}}()
    for dt in dts
        p_lo, p_hi, _, _ = _geos_get_two_files_exact(itp, dt)
        push!(unique_files, (p_lo, p_hi))
    end
    @info "Pre-downloaded GEOS-FP file pairs" n=length(unique_files)
    return length(unique_files)
end
