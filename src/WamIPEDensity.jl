module WamIPEDensity

using Dates
using Printf
using Statistics
using AWS
using AWSS3
using NCDatasets
using Interpolations
using HTTP
using EzXML
using URIs
using DataInterpolations
using Serialization
using CommonDataModel
using Plots
using CairoMakie 
using CSV, DataFrames 
using FilePathsBase: joinpath
using Base: mkpath


const _WIPED_RUN_START_WALL = Ref{DateTime}(DateTime(0))
const _WIPED_RUN_START_NS   = Ref{Int}(0)
const _WIPED_TIMER_READY    = Ref(false)

function reset_run_timer!()
    _WIPED_RUN_START_WALL[] = now()
    _WIPED_RUN_START_NS[]   = time_ns()
    _WIPED_TIMER_READY[]    = true
    return nothing
end

function _install_run_timer!()
    # mark start
    _WIPED_RUN_START_WALL[] = now()
    _WIPED_RUN_START_NS[]   = time_ns()
    _WIPED_TIMER_READY[]    = true

    atexit() do
        # only print if we actually started (and not during precompile)
        if _WIPED_TIMER_READY[] && ccall(:jl_generating_output, Cint, ()) == 0
            stop_wall     = now()
            cpu_elapsed_s = (time_ns() - _WIPED_RUN_START_NS[]) / 1e9
            wall_elapsed_s = Millisecond(stop_wall - _WIPED_RUN_START_WALL[]).value / 1000

            println("=== WamIPEDensity run timing ===")
            println("Start:  ", Dates.format(_WIPED_RUN_START_WALL[], dateformat"yyyy-mm-dd HH:MM:SS.s"))
            println("End:    ", Dates.format(stop_wall,                dateformat"yyyy-mm-dd HH:MM:SS.s"))
            @printf("CPU elapsed : %.3f s\n", cpu_elapsed_s)
            @printf("Wall elapsed: %.3f s\n", wall_elapsed_s)
        end
    end

    return nothing
end


const DEFAULT_CACHE_DIR = normpath("./cache") # DEFAULT_CACHE_DIR = abspath(joinpath(@__DIR__, "..", "cache")) //Tarun
const _FILEPAIR_CACHE = Dict{Tuple{String,DateTime}, Tuple{String,String,String,String}}()
const _FILEPAIR_LOCK  = ReentrantLock()

export WAMInterpolator, get_density, get_density_batch, get_density_at_point, get_density_trajectory, mean_density_profile, plot_global_mean_profile, plot_global_mean_profile_makie


"""
    WAMInterpolator(; bucket="noaa-nws-wam-ipe-pds", product="wfs", varname="den",
                      region="us-east-1", interpolation=:sciml)

Configuration object for accessing and interpolating WAM-IPE data on S3.

- `bucket`: S3 bucket name (public). Default: `"noaa-nws-wam-ipe-pds"`.
- `product::String` — Product subfolder prefix, typically `"wfs"` (forecast) or `"wrs"` (Real-time Nowcast).
- `varname`: NetCDF variable name for neutral density (set to your target; defaults `"den"`)
- `region`: AWS region (WAM-IPE public data is in `us-east-1`)
- `interpolation`: `:nearest`, `:linear`, `:logz_linear`, `:logz_quadratic` or `:sciml`
"""
Base.@kwdef struct WAMInterpolator
    bucket::String = "noaa-nws-wam-ipe-pds"
    root_prefix::String = "v1.2" # S3 root prefix for WAM-IPE data
    product::String = "wfs"
    varname::String = "den"
    region::String = "us-east-1"
    interpolation::Symbol = :sciml
end

# Helpers 
function _cache_filepair!(product::String, dt::DateTime,
                          p_lo::String, p_hi::String, prod_lo::String, prod_hi::String)
    lock(_FILEPAIR_LOCK) do
        _FILEPAIR_CACHE[(product, _datetime_floor_10min(dt))] = (p_lo, p_hi, prod_lo, prod_hi)
    end
end

function _get_cached_filepair(product::String, dt::DateTime)
    lock(_FILEPAIR_LOCK) do
        get(_FILEPAIR_CACHE, (product, _datetime_floor_10min(dt)), nothing)
    end
end

function _cf_decode!(A::AbstractArray, var)
    # get a string-keyed attribute dict regardless of concrete var type
    attrs_any = try
        # works for NCDatasets.Variable
        Dict(var.attrib)
    catch
        # works for CFVariable and friends
        Dict(CommonDataModel.attributes(var))
    end

    sf = haskey(attrs_any, "scale_factor") ? float(attrs_any["scale_factor"]) : 1.0
    ao = haskey(attrs_any, "add_offset")   ? float(attrs_any["add_offset"])   : 0.0

    fillvals = Set{Float64}()
    for k in ("_FillValue","missing_value")
        if haskey(attrs_any, k)
            v = attrs_any[k]
            if v isa AbstractArray
                for x in v; push!(fillvals, float(x)); end
            else
                push!(fillvals, float(v))
            end
        end
    end

    # decode into Float64 buffer
    B = Float64.(A)

    # mask fills → NaN
    if !isempty(fillvals)
        @inbounds for i in eachindex(B)
            @fastmath if B[i] in fillvals; B[i] = NaN; end
        end
    end

    # apply affine decoding if present
    if sf != 1.0 || ao != 0.0
        @inbounds @fastmath B .= B .* sf .+ ao
    end

    return B
end


# --- helpers ---
# Convert the dataset's vertical axis to kilometers (vector form)
function _z_to_km(z::AbstractVector, ds::NCDataset, zname::String)
    units = get(ds[zname].attrib, "units", "")
    kind  = _classify_vertical_units(units)
    if kind === :km
        return Float64.(z)
    elseif kind === :m
        return Float64.(z) ./ 1000
    elseif kind === :pressure
        error("Vertical axis '$zname' uses pressure units ('$units'); cannot convert to altitude (km).")
    elseif kind === :index || kind === :missing || kind === :unknown
        error("Unsupported or missing vertical units '$units' on '$zname'. Expected kilometers ('km') or meters ('m').")
    end
end

# Mean over lon & lat for every z level (ignores NaN/Fill)
function _mean_lonlat_over_z(V3::AbstractArray{<:Real,3})
    @assert ndims(V3) == 3  # lon×lat×z
    nl, nt, nz = size(V3)
    out = Vector{Float64}(undef, nz)
    @inbounds for k in 1:nz
        acc = 0.0; cnt = 0
        @views for val in V3[:, :, k]
            if isfinite(val)
                acc += val; cnt += 1
            end
        end
        out[k] = cnt == 0 ? NaN : acc / cnt
    end
    return out
end

"""
    mean_density_profile(itp::WAMInterpolator, dt::DateTime)
        -> (alt_km::Vector{Float64}, dens_mean::Vector{Float64})

Returns the global-mean neutral density profile at time `dt`, produced by
averaging across all longitudes and latitudes at each altitude level, with
linear time interpolation between the two bracketing files.
"""
function mean_density_profile(itp::WAMInterpolator, dt::DateTime)
    # Resolve the two files bracketing dt
    p_lo, p_hi, _, _ = _get_two_files_exact(itp, dt)

    ds_lo = _open_nc_cached(p_lo)
    ds_hi = _open_nc_cached(p_hi)

    try
        # Parse valid times from filenames
        t_lo = _parse_valid_time_from_key(p_lo)
        t_hi = _parse_valid_time_from_key(p_hi)
        t_lo === nothing && (t_lo = t_hi)
        t_hi === nothing && (t_hi = t_lo)

        # Load grids and values (lon×lat×z×time)
        latL, lonL, zL, tL, VL, namesL = _load_grids(ds_lo, itp.varname; file_time=t_lo)
        latH, lonH, zH, tH, VH, namesH = _load_grids(ds_hi, itp.varname; file_time=t_hi)

        # Convert z to km for output/plotting
        alt_km_L = _z_to_km(zL, ds_lo, namesL[3])
        alt_km_H = _z_to_km(zH, ds_hi, namesH[3])
        if !isequal(alt_km_L, alt_km_H)
            # Simple safeguard: WAM/IPE fixed-height products should match;
            # if not, we interpolate the high profile onto the low z grid.
            @warn "Vertical grids differ slightly; interpolating high onto low grid."
        end

        # For single-time files, VL[:,:,:,1] / VH[:,:,:,1]
        prof_lo = _mean_lonlat_over_z(@view VL[:, :, :, 1])
        prof_hi = _mean_lonlat_over_z(@view VH[:, :, :, 1])

        # Temporal blend at query time
        if t_lo == t_hi
            return (alt_km_L, prof_lo)
        else
            # Linear interpolation in time for each altitude level
            t0 = Dates.value(t_lo)
            t1 = Dates.value(t_hi)
            tq = Dates.value(dt)
            θ = clamp((tq - t0) / (t1 - t0), 0.0, 1.0)

            # Ensure both profiles align on the same z (assume same grid)
            if length(prof_lo) != length(prof_hi) || length(alt_km_L) != length(alt_km_H)
                # If grids mismatch, interpolate prof_hi onto alt_km_L
                itp_hi = DataInterpolations.LinearInterpolation(prof_hi, alt_km_H)
                prof_hi = itp_hi.(alt_km_L)
            end

            prof = @. (1-θ)*prof_lo + θ*prof_hi
            return (alt_km_L, prof)
        end
    finally
        _unpin_nc_cached(p_lo)
        _unpin_nc_cached(p_hi)
    end
end

"""
    plot_global_mean_profile(itp::WAMInterpolator, dt::DateTime;
                             alt_max_km::Real=500, savepath::Union{Nothing,String}=nothing)

Plots the global-mean density profile (density vs altitude, log x-axis).
Returns the Plots.jl plot object. If `savepath` is given, saves the figure.
"""
function plot_global_mean_profile(itp::WAMInterpolator, dt::DateTime;
                                  alt_max_km::Real=500, savepath::Union{Nothing,String}=nothing)
    alt_km, dens = mean_density_profile(itp, dt)

    # Clamp/clean for plotting
    mask = .!(isnan.(dens) .| isinf.(dens))
    altp = alt_km[mask]
    denp = dens[mask]

    p = Plots.plot(
        denp, altp;
        xscale = :log10,
        xlabel = "Density, kg/m^3",
        ylabel = "Altitude, km",
        legend = false,
        framestyle = :box,
        grid = true, 
        title = "Global Mean Density — " * Dates.format(dt, dateformat"yyyy-mm-dd HH:MM 'UTC'")
    )
    Plots.ylims!(p, (0, min(alt_max_km, maximum(altp))))

    if savepath !== nothing
        Plots.savefig(p, String(savepath))
    end
    return p
end


function _extend_profile_to_zero(alt_km::AbstractVector{<:Real},
                                 dens::AbstractVector{<:Real})
    if any(abs.(alt_km) .<= 1e-8)
        return collect(alt_km), collect(dens)
    end
    mask = .!(isnan.(dens) .| isinf.(dens) .| (dens .<= 0))
    a = collect(alt_km[mask]); d = collect(dens[mask])
    if length(a) < 2
        return vcat(0.0, collect(alt_km)), vcat(first(dens), collect(dens))
    end
    p = sortperm(a)
    a1, a2 = a[p[1]], a[p[2]]
    d1, d2 = d[p[1]], d[p[2]]
    d0 = (a2 == a1) ? d1 : begin
        m = (log(d2) - log(d1)) / (a2 - a1)
        b = log(d1) - m*a1
        val = exp(b)
        (isfinite(val) && val > 0) ? val : d1
    end
    return vcat(0.0, collect(alt_km)), vcat(d0, collect(dens))
end

"""
    plot_global_mean_profile_makie(itp, dt;
        alt_max_km = nothing,
        extend_to0 = false,
        savepath   = nothing,     # if provided, overrides auto path for PNG only
        export_csv = false,
        base_dir   = "plots")     # auto base directory for assets

Creates (if needed) `plots/<product>/<YYYYMMDDTHHMMSS>/` and saves
`global_mean_profile.png` (and `.csv` if requested) there.
Returns `(fig, ax, png_path, csv_path_or_nothing)`.
"""
function plot_global_mean_profile_makie(itp::WAMInterpolator, dt::DateTime;
    alt_max_km::Union{Nothing,Real}=nothing,
    extend_to0::Bool=false,
    savepath::Union{Nothing,String}=nothing,
    export_csv::Bool=false,
    base_dir::AbstractString="plots",
)
    alt_km, dens = mean_density_profile(itp, dt)

    # clean/log-safe; then optional 0-km extension
    mask = .!(isnan.(dens) .| isinf.(dens) .| (dens .<= 0))
    altp = alt_km[mask]; denp = dens[mask]
    if extend_to0
        altp, denp = _extend_profile_to_zero(altp, denp)
    end

    stamp   = Dates.format(dt, dateformat"yyyymmddTHHMMSS")
    outdir  = joinpath(base_dir, itp.product, stamp)
    mkpath(outdir)

    default_png = joinpath(outdir, "global_mean_profile.png")
    png_path    = savepath === nothing ? default_png : String(savepath)
    csv_path    = export_csv ? joinpath(outdir, "global_mean_profile.csv") : nothing

    CairoMakie.activate!()
    fig = CairoMakie.Figure(resolution = (800, 600))
    ax  = CairoMakie.Axis(fig[1,1];
        xlabel = "Density (kg·m⁻³)",
        ylabel = "Altitude (km)",
        xscale = CairoMakie.log10,
        title  = "Global Mean Density — " * Dates.format(dt, dateformat"yyyy-mm-dd HH:MM 'UTC'")
    )
    CairoMakie.lines!(ax, denp, altp)
    if alt_max_km !== nothing
        CairoMakie.ylims!(ax, nothing, float(alt_max_km))
    end
    CairoMakie.autolimits!(ax)

    CairoMakie.save(png_path, fig)

    if export_csv
        open(csv_path, "w") do io
            write(io, "altitude_km,density_kg_m3\n")
            @inbounds for i in eachindex(altp)
                write(io, string(altp[i], ",", denp[i], "\n"))
            end
        end
    end

    return fig, ax, png_path, csv_path
end


"""
    _aws_cfg(region) returns AWS.AWSConfig

Create an unsigned AWS config for `region`.
This avoids credential requirements for public WAM-IPE objects.
"""
function _aws_cfg(region::String)
    AWS.AWSConfig(; region=region, creds=nothing)
end

mutable struct _DSPool
    map::Dict{String,NCDataset}      # path -> open dataset
    pins::Dict{String,Int}           # path -> active users
    last::Dict{String,Int64}         # path -> last use (time_ns)
    max_open::Int                    # cap on simultaneously open datasets
    lock::ReentrantLock
end

const _DSPOOL = _DSPool(Dict{String,NCDataset}(),
                        Dict{String,Int}(),
                        Dict{String,Int64}(),
                        16,                           # default: keep up to 16 files open
                        ReentrantLock())

# touch for LRU
@inline function _ds_touch!(pool::_DSPool, path::String)
    pool.last[path] = time_ns()
end

function _ds_evict_unpinned!(pool::_DSPool)
    while length(pool.map) > pool.max_open
        unpinned = [p for (p,c) in pool.pins if c == 0]
        isempty(unpinned) && return 
        victim = argmin(p -> get(pool.last, p, 0), unpinned)
        try
            close(pool.map[victim])
        catch
            # ignore close errors
        end
        delete!(pool.map, victim)
        delete!(pool.pins, victim)
        delete!(pool.last, victim)
    end
end

function _open_nc_cached(path::String)
    lock(_DSPOOL.lock) do
        if haskey(_DSPOOL.map, path)
            _DSPOOL.pins[path] = get(_DSPOOL.pins, path, 0) + 1
            _ds_touch!(_DSPOOL, path)
            return _DSPOOL.map[path]
        else
            ds = NCDataset(path, "r")
            _DSPOOL.map[path] = ds
            _DSPOOL.pins[path] = 1
            _ds_touch!(_DSPOOL, path)
            _ds_evict_unpinned!(_DSPOOL)   # keep pool bounded
            return ds
        end
    end
end

# Unpin after use (keeps file open for reuse unless evicted later)
function _unpin_nc_cached(path::String)
    lock(_DSPOOL.lock) do
        if haskey(_DSPOOL.pins, path)
            _DSPOOL.pins[path] = max(0, _DSPOOL.pins[path] - 1)
            _ds_touch!(_DSPOOL, path)
            _ds_evict_unpinned!(_DSPOOL)
        end
    end
    return nothing
end

# Optional: allow users to change cap at runtime
function set_max_open_datasets!(n::Integer)
    lock(_DSPOOL.lock) do
        _DSPOOL.max_open = max(1, Int(n))
        _ds_evict_unpinned!(_DSPOOL)
    end
    return _DSPOOL.max_open
end


"""
    _cache_path(cache_dir, key) returns String

Joins `cache_dir` and S3 `key` using a normalised path while keeping the
remote directory structure (e.g. `v1.2/wfs...`).
"""

_cache_path(cache_dir::AbstractString, key::AbstractString) =   
    normpath(joinpath(cache_dir, key))  # preserves v1.2/…/… structure

"""
    _download_to_cache(aws, bucket, key; cache_dir=DEFAULT_CACHE_DIR,
                       cache_max_bytes=2_000_000_000, verbose=true) returns String

Download `key` into an on-disk, thread-safe LRU cache rooted at `cache_dir`.
Returns the local path, atomic and safe for concurrent use.
"""

function _download_to_cache(aws::AWS.AWSConfig, bucket::String, key::String;
                            cache_dir::AbstractString=DEFAULT_CACHE_DIR,
                            cache_max_bytes::Int=2_000_000_000,
                            verbose::Bool=false) 
    cache = _get_cache(cache_dir, cache_max_bytes)
    return _cache_get_file!(cache, aws, bucket, key; verbose=verbose)
end


"""
    _open_nc_from_s3(aws, bucket, key; cache_dir=DEFAULT_CACHE_DIR)

Opens the NetCDF from local cache if present, otherwise downloads from S3 into cache.
Returns `(ds, path)`; Caller must `close(ds)` when done. The cache file is kept for reuse.
"""

function _open_nc_from_s3(aws::AWS.AWSConfig, bucket::String, key::String;
                          cache_dir::AbstractString=DEFAULT_CACHE_DIR,
                          cache_max_bytes::Int=2_000_000_000)
    local_path = _download_to_cache(aws, bucket, key;
                                    cache_dir=cache_dir,
                                    cache_max_bytes=cache_max_bytes,
                                    verbose=true)
    return NCDataset(local_path, "r"), local_path
end

"""
    print_cache_stats(; cache_dir=DEFAULT_CACHE_DIR, cache_max_bytes=2_000_000_000)

Prints the current cache directory, capacity, usage, and LRU/MRU keys.
Useful for debugging what is stored on disk.
"""

function print_cache_stats(; cache_dir::AbstractString=DEFAULT_CACHE_DIR, cache_max_bytes::Int=2_000_000_000)
    cache = _get_cache(cache_dir, cache_max_bytes)
    lock(cache.lock) do
        println("Cache dir: ", cache.dir)
        println("Capacity : ", round(cache.max_bytes/1e9, digits=2), " GB")
        println("Used     : ", round(cache.bytes/1e9, digits=3), " GB  (", length(cache.map), " files)")
        if !isempty(cache.order)
            println("LRU head: ", first(cache.order))
            println("MRU tail: ", last(cache.order))
        end
    end
end


#  Version mapping
const _VERSION_WINDOWS = (
    ("v1.1", DateTime(2023,3,20,21,10,0), DateTime(2023,6,30,21,0,0)), # inclusive start/ end
    ("v1.2", DateTime(2023,6,30,21,10,0), nothing), # open-ended
)
"""
    _version_for(dt) returns String

Returns the S3 version root (e.g. `"v1.2"`) that applies to `dt` based on
internal date windows.
"""
function _version_for(dt::DateTime)::String
    for (v, lo, hi) in _VERSION_WINDOWS
        if dt >= lo && (hi === nothing || dt <= hi)
            return v
        end
    end
    error("No WAM-IPE version mapping covers $dt")
end

"""
    _model_for_version(v) returns String

Map a version root (e.g. `"v1.1"`, `"v1.2"`) to its model token used in filenames
(e.g. `"wam10"`, `"gsm10"`).
"""

_model_for_version(v::String) = v == "v1.2" ? "wam10" :
                                v == "v1.1" ? "gsm10" :
                                error("Unknown version $v")



const _CACHE_META_FILE = "metadata.bin"

mutable struct _FileCache
    dir::String
    max_bytes::Int64
    map::Dict{String,String}          # key -> local_path
    sizes::Dict{String,Int64}         # key -> bytes
    order::Vector{String}             # LRU order, oldest at index 1
    bytes::Int64                      # current bytes on disk
    downloading::Set{String}          # keys currently being downloaded
    conds::Dict{String,Condition}     # key -> Condition for waiters
    lock::ReentrantLock               # global cache lock
end

# cache instances keyed by (dir, max_bytes)
const _CACHES = Dict{Tuple{String,Int64}, _FileCache}()

"""
    _cache_meta_path(dir) returns String

Returns the path of the cache metadata file in `dir`.
"""

function _cache_meta_path(dir::AbstractString)
    joinpath(dir, _CACHE_META_FILE)
end

"""
    _load_cache(dir, max_bytes) returns _FileCache

Load/initialises the on-disk cache metadata for `dir`. This is tolerant of
corrupt/old metadata and recreates missing fields.
"""

function _load_cache(dir::AbstractString, max_bytes::Int64)
    mkpath(dir)
    meta = _cache_meta_path(dir)
    if isfile(meta)
        try
            open(meta, "r") do io
                obj = deserialize(io)
                if obj isa _FileCache
                    obj.bytes = sum(values(obj.sizes))
                    obj.order = [k for k in obj.order if haskey(obj.map, k)]
                    obj.lock = ReentrantLock()
                    empty!(obj.downloading); empty!(obj.conds)
                    return obj
                end
            end
        catch
        end
    end
    return _FileCache(
        String(dir),
        Int64(max_bytes),
        Dict{String,String}(),
        Dict{String,Int64}(),
        String[],
        0,
        Set{String}(),
        Dict{String,Condition}(),
        ReentrantLock()
    )
end

"""
    _save_cache(cache) returns Nothing

Persist cache metadata to disk.
"""

function _save_cache(cache::_FileCache)
    mkpath(cache.dir)
    open(_cache_meta_path(cache.dir), "w") do io
        serialize(io, cache)
    end
    return nothing
end

# Mark a key as most recently used
"""
    _lru_touch!(cache, key)

Mark `key` as most-recently-used in `cache`.
"""

function _lru_touch!(cache::_FileCache, key::String)
    # remove if present
    idx = findfirst(==(key), cache.order)
    if idx !== nothing
        deleteat!(cache.order, idx)
    end
    push!(cache.order, key)
end

# Evict least-recently used files until under budget

"""
    _evict_until_under_budget!(cache)

Deletes least-recently-used files from disk until the cache fits within
`cache.max_bytes`.
"""
function _evict_until_under_budget!(cache::_FileCache)
    while cache.bytes > cache.max_bytes && !isempty(cache.order)
        victim = first(cache.order)
        popfirst!(cache.order)
        if haskey(cache.map, victim)
            local_path = cache.map[victim]
            sz = get(cache.sizes, victim, 0)
            try
                isfile(local_path) && rm(local_path; force=true)
            catch
                # ignore I/O errors on delete
            end
            delete!(cache.map, victim)
            delete!(cache.sizes, victim)
            cache.bytes = max(0, cache.bytes - sz)
        end
    end
end

# main: return local path for (bucket,key), downloading if necessary
"""
    _cache_get_file!(cache, aws, bucket, key; verbose=true) returns String

Core cache routine. If present, return the cached path. Otherwise download the
object (AWSS3 stream with HTTP fallback), store it atomically, update LRU and
size accounting, possibly evicting older files. Safe for concurrent threads.
"""
function _cache_get_file!(cache::_FileCache, aws::AWS.AWSConfig, bucket::String, key::String;
                          verbose::Bool=false)
    local_path = normpath(joinpath(cache.dir, key))

    # fast path: already on disk and recorded
    lock(cache.lock) do
        if haskey(cache.map, key) && isfile(cache.map[key])
            _lru_touch!(cache, key)
            _save_cache(cache)
            verbose && println("[cache] hit: ", cache.map[key])
            return cache.map[key]
        end

        if key in cache.downloading
            cond = get!(cache.conds, key) do
                Condition()
            end
            verbose && println("[cache] wait: ", key)
            wait(cond)
            if haskey(cache.map, key) && isfile(cache.map[key])
                _lru_touch!(cache, key)
                _save_cache(cache)
                return cache.map[key]
            else
                error("Download failed for $key (woken without file present)")
            end
        end

        push!(cache.downloading, key)
        cache.conds[key] = get(cache.conds, key, Condition())
    end

    tmp_path = local_path * ".part"
    mkpath(dirname(local_path))
    verbose && println("[cache] get:  s3://$bucket/$key -> ", local_path)

    ok = false
    bytes_written::Int64 = 0

    # First try S3 streaming
    try
        io = AWSS3.s3_get(aws, bucket, key; return_stream=true)
        open(tmp_path, "w") do f
            while !eof(io)
                chunk = read(io, 1_048_576)  # 1 MiB
                write(f, chunk)
                bytes_written += sizeof(chunk)
            end
        end
        ok = true
    catch
        # Fallback HTTP streaming with timeout
        try
            url = "https://$bucket.s3.amazonaws.com/$key"
            HTTP.open(:GET, url; readtimeout=60) do http_io
                open(tmp_path, "w") do f
                    while !eof(http_io)
                        chunk = read(http_io, 1_048_576)
                        write(f, chunk)
                        bytes_written += sizeof(chunk)
                    end
                end
            end
            ok = true
        catch
            ok = false
        end
    end

    # atomically move into place if successful
    if ok
        mv(tmp_path, local_path; force=true)
    else
        # cleanup temp
        isfile(tmp_path) && rm(tmp_path; force=true)
    end

    lock(cache.lock) do
        # notify and clear downloading flag regardless of success
        if haskey(cache.conds, key)
            notify(cache.conds[key]; all=true)
            delete!(cache.conds, key)
        end
        delete!(cache.downloading, key)

        if !ok || !isfile(local_path)
            error("Failed to download s3://$bucket/$key")
        end

        # record size
        sz = try
            filesize(local_path)
        catch
            bytes_written > 0 ? bytes_written : 0
        end

        cache.map[key] = local_path
        cache.sizes[key] = sz
        cache.bytes += sz
        _lru_touch!(cache, key)

        _evict_until_under_budget!(cache)
        _save_cache(cache)

        return local_path
    end
end

"""
    _get_cache(cache_dir, max_bytes) returns _FileCache

Gets/creates the shared cache object for `(cache_dir, max_bytes)`.
"""

function _get_cache(cache_dir::AbstractString, max_bytes::Int64)
    key = (String(cache_dir), Int64(max_bytes))
    if haskey(_CACHES, key)
        return _CACHES[key]
    else
        cache = _load_cache(cache_dir, max_bytes)
        return (_CACHES[key] = cache)
    end
end

# Single file download attempt (no interpolation)
"""
    _try_download(itp, dt, product) returns Union{String,Nothing}

Attempt to download a single NetCDF corresponding to an exact 10-minute
stamp `dt` under `product` (e.g. `"wfs"`). Returns local path on success,
`nothing` on failure.
"""
function _try_download(itp::WAMInterpolator, dt::DateTime, product::String)
    aws = _aws_cfg(itp.region)
    key = _construct_s3_key(dt, product)

    if _have_in_cache(key)
        return normpath(joinpath(DEFAULT_CACHE_DIR, key))
    end

    try
        return _download_to_cache(aws, itp.bucket, key; cache_dir=DEFAULT_CACHE_DIR, verbose=true)
    catch
        return nothing
    end
end


# Returns one of: :km, :m, :pressure, :index, :missing, :unknown
"""
    _classify_vertical_units(units_raw) returns Symbol

Classify vertical coordinate, heuristically, units into one of
`:km`, `:m`, `:pressure`, `:index`, `:missing`, or `:unknown`.
Used to validate/convert altitude queries.
"""

function _classify_vertical_units(units_raw::AbstractString)
    s = lowercase(strip(String(units_raw)))
    isempty(s) && return :missing

    # common kilometer spellings
    if occursin(r"\bkm\b", s) || occursin("kilometer", s) || occursin("kilometre", s)
        return :km
    end

    # plain meters (avoid mm/cm false-positives)
    if (occursin(r"\bm\b", s) || occursin("meter", s) || occursin("metre", s)) &&
       !occursin(r"\bmm\b", s) && !occursin(r"\bcm\b", s) && !occursin("km", s)
        return :m
    end

    # pressure coordinates (not geometric height)
    if occursin(r"\bpa\b", s) || occursin(r"\bhpa\b", s) || occursin(r"\bmb\b", s) ||
       occursin("pascal", s) || occursin("pressure", s)
        return :pressure
    end

    # index/level-ish (not physical distance)
    if occursin("level", s) || occursin("index", s) || occursin("layer", s)
        return :index
    end

    return :unknown
end

"""
    _datetime_floor_10min(dt) returns DateTime

Floor `dt` to the nearest 10-minute boundary.
"""

#  Ten-minute bracketing 
function _datetime_floor_10min(dt::DateTime)
    m  = minute(dt)
    mm = m - (m % 10)
    DateTime(Date(dt), Time(hour(dt), mm))
end

_surrounding_10min(dt::DateTime) = (_datetime_floor_10min(dt),
                                    _datetime_floor_10min(dt) + Minute(10))


#  Cycle hour rules (folder selection) 
# For WRS:
#   if dt < 03:00 → prev day 18:00
#   if dt < 09:00 → 00:00
#   if dt < 15:00 → 06:00
#   if dt < 21:00 → 12:00
#   else          → 18:00

"""
    _wrs_archive(dt) returns DateTime

Select the cycle hour for the WRS product that should contain `dt`.
This controls which S3 folder (…/HH/) to search.
"""
function _wrs_archive(dt::DateTime)::DateTime
    h = hour(dt)
    if h < 3
        return DateTime(Date(dt) - Day(1), Time(18))
    elseif h < 9
        return DateTime(Date(dt), Time(0))
    elseif h < 15
        return DateTime(Date(dt), Time(6))
    elseif h < 21
        return DateTime(Date(dt), Time(12))
    else
        return DateTime(Date(dt), Time(18))
    end
end

# For WFS:
#   if dt < 03:00 → same day 00:00
#   if dt < 09:00 → 06:00
#   if dt < 15:00 → 12:00
#   if dt < 21:00 → 18:00
#   else          → next day 00:00
"""
    _wfs_archive(dt) returns DateTime

Select the cycle hour for the WFS product that should contain `dt`.
This controls which S3 folder (…/HH/) to search.
"""

function _wfs_archive(dt::DateTime)::DateTime
    h = hour(dt)
    if h < 3
        return DateTime(Date(dt), Time(0))
    elseif h < 9
        return DateTime(Date(dt), Time(6))
    elseif h < 15
        return DateTime(Date(dt), Time(12))
    elseif h < 21
        return DateTime(Date(dt), Time(18))
    else
        return DateTime(Date(dt) + Day(1), Time(0))
    end
end

#  Exact S3 key construction 
"""
    _construct_s3_key(dt, product) returns String

Build the exact S3 key for a given 10-minute stamp `dt` and `product`
(`"wfs"` or `"wrs"`). The filename encodes `dt`, while the folder encodes the
chosen cycle hour.
"""

function _construct_s3_key(dt::DateTime, product::String)::String
    v       = _version_for(dt)
    model   = _model_for_version(v)
    # choose archive cycle by product
    arch    = product == "wrs" ? _wrs_archive(dt) :
              product == "wfs" ? _wfs_archive(dt) :
              error("Unknown product $product")
    ymd_dir = Dates.format(Date(arch), dateformat"yyyymmdd")
    HH_dir  = @sprintf("%02d", hour(arch))
    # filename encodes the EXACT target dt (10-minute stamp), not the cycle hour
    ymd     = Dates.format(Date(dt), dateformat"yyyymmdd")
    HMS     = Dates.format(Time(dt), dateformat"HHMMSS")
    HHfile  = @sprintf("%02d", hour(arch))  # tHHz uses cycle hour
    return @sprintf("%s/%s.%s/%s/wam_fixed_height.%s.t%sz.%s.%s_%s.nc",
                    v, product, ymd_dir, HH_dir, product, HHfile, model, ymd, HMS)
end

_product_fallback_order(product::String) = product == "wfs" ? ("wfs","wrs") : ("wrs","wfs")

# Build a WRS key but forcing the archive (cycle) hour
function _construct_wrs_key_with_cycle(dt::DateTime, arch::DateTime)::String
    v     = _version_for(dt)
    model = _model_for_version(v)

    ymd_dir = Dates.format(Date(arch), dateformat"yyyymmdd")
    HH_dir  = @sprintf("%02d", hour(arch))    # folder: .../<HH>/

    ymd     = Dates.format(Date(dt), dateformat"yyyymmdd")
    HMS     = Dates.format(Time(dt), dateformat"HHMMSS")
    HHfile  = @sprintf("%02d", hour(arch))    # tHHz uses cycle hour

    return @sprintf("%s/%s.%s/%s/wam_fixed_height.%s.t%sz.%s.%s_%s.nc",
                    v, "wrs", ymd_dir, HH_dir,
                    "wrs", HHfile, model, ymd, HMS)
end


# Returns two local file paths for the bracketing stamps
"""
    _get_two_files_exact(itp, dt) returns (low_path, high_path, low_product, high_product)

Resolve and fetch the two local files that bracket the 10-minute stamp `dt`.
Prefers the configured product, but will fall back to the alternate product if
necessary. Throws if either side cannot be found.
"""

const _WRS_00Z_FIRST_TIME = Time(3, 10, 0)  # first valid file under 00Z folder is ..._031000.nc
_have_in_cache(key::AbstractString; cache_dir::AbstractString=DEFAULT_CACHE_DIR) =
    isfile(normpath(joinpath(cache_dir, key)))


@inline function _both_exist(p1::AbstractString, p2::AbstractString)
    isfile(p1) && isfile(p2)
end

function _get_two_files_exact(itp::WAMInterpolator, dt::DateTime)
    # RAM cache check (per product, per floored 10-min bucket)
    if (cached = _get_cached_filepair(itp.product, dt)) !== nothing
        p_lo, p_hi, prod_lo_used, prod_hi_used = cached
        if _both_exist(p_lo, p_hi)
            return (p_lo, p_hi, prod_lo_used, prod_hi_used)
        end
        # fall through to refresh if files were evicted on disk
    end

    dt_lo, dt_hi = _surrounding_10min(dt)
    pref, alt    = _product_fallback_order(itp.product)
    aws          = _aws_cfg(itp.region)

    # prefer local file if present; otherwise pull once into cache dir
    local function _local_path_for_key(key::String)
        normpath(joinpath(DEFAULT_CACHE_DIR, key))
    end
    local function _ensure_local(key::String)
        lp = _local_path_for_key(key)
        return isfile(lp) ? lp :
               _download_to_cache(aws, itp.bucket, key; cache_dir=DEFAULT_CACHE_DIR, verbose=false)
    end
    local function _try_product(dt_file::DateTime, product::String)
        key = _construct_s3_key(dt_file, product)
        try
            return _ensure_local(key)
        catch
            return nothing
        end
    end

    # Special WRS cycle fallback: try same-day 00Z, then prev-day 18Z
    if itp.product == "wrs"
        local function _try_wrs_from_cycle(dt_file::DateTime, arch::DateTime)
            key = _construct_wrs_key_with_cycle(dt_file, arch)
            try
                return _ensure_local(key)
            catch
                return nothing
            end
        end
        local function _resolve_wrs_stamp(dt_file::DateTime)
            arch_00 = DateTime(Date(dt_file), Time(0))
            arch_18 = DateTime(Date(dt_file) - Day(1), Time(18))
            (p00 = _try_wrs_from_cycle(dt_file, arch_00)) !== nothing && return (p00, "wrs")
            (p18 = _try_wrs_from_cycle(dt_file, arch_18)) !== nothing && return (p18, "wrs")
            return (nothing, "wrs")
        end

        p_lo_path, prod_lo_used = _resolve_wrs_stamp(dt_lo)
        p_hi_path, prod_hi_used = _resolve_wrs_stamp(dt_hi)

        if p_lo_path === nothing || p_hi_path === nothing
            missing = String[]
            p_lo_path === nothing && push!(missing, "low @ $(dt_lo) (wrs 00Z, then prev 18Z)")
            p_hi_path === nothing && push!(missing, "high @ $(dt_hi) (wrs 00Z, then prev 18Z)")
            error("Could not fetch WRS files for $(join(missing, "; ")).")
        end

        _cache_filepair!(itp.product, dt, p_lo_path, p_hi_path, prod_lo_used, prod_hi_used)
        return (p_lo_path, p_hi_path, prod_lo_used, prod_hi_used)
    end

    # Generic (WFS as pref with WRS fallback, or vice versa)
    p_lo = _try_product(dt_lo, pref)
    prod_lo = p_lo === nothing ? ((p = _try_product(dt_lo, alt)) === nothing ? nothing : (p, alt)) : (p_lo, pref)

    p_hi = _try_product(dt_hi, pref)
    prod_hi = p_hi === nothing ? ((p = _try_product(dt_hi, alt)) === nothing ? nothing : (p, alt)) : (p_hi, pref)

    if prod_lo === nothing || prod_hi === nothing
    missing = String[]
    prod_lo === nothing && push!(missing, "low @ $(dt_lo)")
    prod_hi === nothing && push!(missing, "high @ $(dt_hi)")
    error("Could not fetch files for $(join(missing, ", ")); tried $(pref), $(alt).")
end


    p_lo_path, prod_lo_used = prod_lo
    p_hi_path, prod_hi_used = prod_hi

    if prod_lo_used != prod_hi_used
        @debug "[mix] Using mixed products: low=$(prod_lo_used), high=$(prod_hi_used)"
    end

    _cache_filepair!(itp.product, dt, p_lo_path, p_hi_path, prod_lo_used, prod_hi_used)
    return (p_lo_path, p_hi_path, prod_lo_used, prod_hi_used)
end


#  Time utilities
function _decode_time_units(ds::NCDataset, tname::String, t::AbstractVector)
    units = get(ds[tname].attrib, "units", "")
    cal   = lowercase(string(get(ds[tname].attrib, "calendar", "gregorian")))
    m = match(r"(seconds|minutes|hours|days)\s+since\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?", units)
    if m === nothing
        return t, nothing, nothing  # keep axis as already provided (often DateTime)
    end
    scale = m.captures[1]
    epoch_date = Date(m.captures[2])
    epoch_time = m.captures[3] === nothing ? Time(0) : Time(m.captures[3])
    epoch = DateTime(epoch_date, epoch_time)

    if eltype(t) <: DateTime
        tnum = [_encode_query_time(tt, epoch, scale) for tt in t]
        return tnum, epoch, scale
    else
        return collect(t), epoch, scale
    end
end


# Convert query DateTime into the numeric coordinate used in the file
function _encode_query_time(dtq::DateTime,
                            epoch::Union{DateTime,Nothing},
                            scale::Union{AbstractString,Nothing})
    epoch === nothing && return float(dtq.value)

    delta_ms = Dates.value(dtq - epoch)  # milliseconds

    # If scale missing, default to days
    s = scale === nothing ? "days" : lowercase(String(scale))

    if startswith(s, "sec")       # "seconds since ..."
        return delta_ms / 1_000
    elseif startswith(s, "min")   # "minutes since ..."
        return delta_ms / 60_000
    elseif startswith(s, "hour")  # "hours since ..."
        return delta_ms / 3_600_000
    else                          # treat anything else as "days since ..."
        return delta_ms / 86_400_000
    end
end

# Parse ...YYYYMMDD_HHMMSS.nc at the end of the key
_parse_valid_time_from_key(key::AbstractString) = let m = match(r"(\d{8})_(\d{6})\.nc$", key)
    m === nothing && return nothing
    ymd, hms = m.captures
    DateTime(parse(Int, ymd[1:4]), parse(Int, ymd[5:6]), parse(Int, ymd[7:8]),
             parse(Int, hms[1:2]), parse(Int, hms[3:4]), parse(Int, hms[5:6]))
end

function _pick_file(objs::AbstractVector; target_dt::Union{DateTime,Nothing}=nothing)
    isempty(objs) && return nothing
    if target_dt === nothing
        return sort(objs, by = o -> String(o["Key"]))[end]
    end

    # Build (delta, key, obj) so ties on delta break by lexicographically latest key
    scored = map(objs) do o
        key = String(o["Key"])
        vt  = _parse_valid_time_from_key(key)
        delta   = vt === nothing ? Day(9999) : abs(target_dt - vt)
        (delta, key, o)
    end
    _, idx = findmin(scored)
    return scored[idx][3]   # the `o`
end

#  Grid conventions (longitude wrapping) 
# Decide grid convention quickly: if any lon > 180, treat as [0, 360); else assume [-180, 180]
_grid_uses_360(lon::AbstractVector) = maximum(lon) > 180

# Wrap lonq to match the grid’s convention
function _wrap_lon_for_grid(lon_grid::AbstractVector, lonq::Real)
    if _grid_uses_360(lon_grid)
        return lonq < 0 ? lonq + 360 : lonq
    else
        return lonq > 180 ? lonq - 360 : lonq
    end
end

# Extract grids and variable
# NEW: accepts 3D (lon,lat,z) or 4D (lon,lat,z,time)
# If 3D, we synthesize a 1-point time axis using `file_time` (DateTime).
function _load_grids(ds::NCDataset, varname::String; file_time::Union{DateTime,Nothing}=nothing)
    haskey(ds, varname) || error("Variable '$varname' not found; pass the correct varname.")
    v = ds[varname]

    dnames = String.(NCDatasets.dimnames(v))  # names like "time","height","latitude","longitude", etc.

    function classify_dim(dname::String)
        lname = lowercase(dname)
        var   = haskey(ds, dname) ? ds[dname] : nothing  # coord var 
        attrs = var === nothing ? Dict{String,Any}() : Dict(var.attrib)

        stdname = lowercase(string(get(attrs, "standard_name", "")))
        axis    = uppercase(string(get(attrs, "axis", "")))
        units   = lowercase(string(get(attrs, "units", "")))

        # Detect TIME
        if occursin("time", lname) || axis == "T" || stdname == "time"
            return :time
        end

        # Detect LAT
        if occursin("lat", lname) || stdname == "latitude" || axis == "Y" || occursin("degrees_north", units)
            return :lat
        end

        # Detect LON
        if occursin("lon", lname) || stdname == "longitude" || axis == "X" || occursin("degrees_east", units)
            return :lon
        end

        # Detect VERTICAL
        if occursin("lev", lname) || occursin("height", lname) || occursin("alt", lname) || lname == "z" || axis == "Z"
            return :z
        end

        # Some WAM/IPE files use generic X/Y
        if lname in ("x","grid_xt","i","nx")
            return :lon
        end
        if lname in ("y","grid_yt","j","ny")
            return :lat
        end

        return :unknown
    end

    roles = map(classify_dim, dnames)
    Vraw  = Array(v)
    nd    = ndims(Vraw)

    if nd == 4
        # Current axis indices in Vraw:
        idx_lon  = findfirst(==( :lon  ), roles)
        idx_lat  = findfirst(==( :lat  ), roles)
        idx_z    = findfirst(==( :z    ), roles)
        idx_time = findfirst(==( :time ), roles)

        idx_lon === nothing  && error("Could not find longitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
        idx_lat === nothing  && error("Could not find latitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
        idx_z   === nothing  && error("Could not find vertical dimension for '$varname'. dims=$(dnames) roles=$(roles)")
        idx_time === nothing && error("Could not find time dimension for '$varname'. dims=$(dnames) roles=$(roles)")

        latname = dnames[idx_lat]; lonname = dnames[idx_lon]; zname = dnames[idx_z]; tname = dnames[idx_time]
        lat = haskey(ds, latname) ? collect(ds[latname][:]) : collect(1:size(Vraw, idx_lat))
        lon = haskey(ds, lonname) ? collect(ds[lonname][:]) : collect(1:size(Vraw, idx_lon))
        z   = haskey(ds, zname)   ? collect(ds[zname][:])   : collect(1:size(Vraw, idx_z))
        t   = haskey(ds, tname)   ? collect(ds[tname][:])   : collect(1:size(Vraw, idx_time))

        perm = (idx_lon, idx_lat, idx_z, idx_time)
        V    = perm == (1,2,3,4) ? Vraw : Array(PermutedDimsArray(Vraw, perm))

        V = _cf_decode!(V, v)

        latunits = lowercase(string(get(ds[latname].attrib, "units", "")))
        lonunits = lowercase(string(get(ds[lonname].attrib, "units", "")))
        if !occursin("degrees_north", latunits); @warn "Latitude units are '$latunits' (expected degrees_north)."; end
        if !occursin("degrees_east",  lonunits); @warn "Longitude units are '$lonunits' (expected degrees_east)."; end

        return lat, lon, z, t, V, (latname, lonname, zname, tname)

    elseif nd == 3
        # Expect lon/lat/z only; synthesize time using `file_time`
        idx_lon  = findfirst(==( :lon ), roles)
        idx_lat  = findfirst(==( :lat ), roles)
        idx_z    = findfirst(==( :z   ), roles)

        idx_lon === nothing && error("Could not find longitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
        idx_lat === nothing && error("Could not find latitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
        idx_z   === nothing && error("Could not find vertical dimension for '$varname'. dims=$(dnames) roles=$(roles)")

        file_time === nothing && error("3-D variable requires `file_time` to synthesize a 1-point time axis.")

        latname = dnames[idx_lat]; lonname = dnames[idx_lon]; zname = dnames[idx_z]; tname = "time"
        lat = haskey(ds, latname) ? collect(ds[latname][:]) : collect(1:size(Vraw, idx_lat))
        lon = haskey(ds, lonname) ? collect(ds[lonname][:]) : collect(1:size(Vraw, idx_lon))
        z   = haskey(ds, zname)   ? collect(ds[zname][:])   : collect(1:size(Vraw, idx_z))
        t   = [file_time]  # synthesized one-element DateTime axis

        perm = (idx_lon, idx_lat, idx_z)
        V3   = perm == (1,2,3) ? Vraw : Array(PermutedDimsArray(Vraw, perm))
        # Expand to 4-D by adding a singleton time dimension at the end
        V    = reshape(V3, size(V3,1), size(V3,2), size(V3,3), 1)

        V = _cf_decode!(V, v)

        latunits = lowercase(string(get(ds[latname].attrib, "units", "")))
        lonunits = lowercase(string(get(ds[lonname].attrib, "units", "")))
        if !occursin("degrees_north", latunits)
            @warn "Latitude units are '$latunits' (expected degrees_north). Results may be incorrect."
        end
        if !occursin("degrees_east", lonunits)
            @warn "Longitude units are '$lonunits' (expected degrees_east). Results may be incorrect."
        end

        return lat, lon, z, t, V, (latname, lonname, zname, tname)

    else
        error("Expected 3D or 4D var '$varname', got ndims=$(nd) with dims=$(dnames)")
    end
end



# Convert query altitude (km) to the dataset's vertical axis units.
# Only supports 'km' and 'm' reliably. Anything else is rejected with a clear error.
function _maybe_convert_alt(z::AbstractVector, alt_km::Real, ds::NCDataset, zname::String)
    units = get(ds[zname].attrib, "units", "")
    kind  = _classify_vertical_units(units)

    if kind === :km
        return alt_km
    elseif kind === :m
        return alt_km * 1000
    elseif kind === :pressure
        error("Vertical axis '$zname' uses pressure units ('$units'); cannot convert altitude in km to pressure levels.")
    elseif kind === :index
        error("Vertical axis '$zname' has index/level units ('$units'); cannot convert altitude in km to an index.")
    elseif kind === :missing
        error("Vertical axis '$zname' is missing a 'units' attribute; cannot safely convert altitude.")
    else
        error("Unsupported vertical units '$units' on '$zname'. Expected kilometers ('km') or meters ('m').")
    end
end


# Find nearest indices
_nearest_index(vec::AbstractVector, x::Real) = findmin(abs.(vec .- x))[2]

# 3-D separable linear interpolation over (lon, lat, z) for a single time slice Vt (size: length(lon)×length(lat)×length(z))
function _interp3_linear(lat::AbstractVector, lon::AbstractVector, z::AbstractVector,
                         Vt::AbstractArray{<:Real,3}, latq::Real, lonq::Real, zq::Real)

    lonq2 = _wrap_lon_for_grid(lon, lonq)

    # z axis
    if length(z) == 1
        Vz = Vt[:, :, 1]               # lon×lat
    else
        iz = clamp(searchsortedlast(z, zq), 1, length(z)-1)
        z1, z2 = z[iz], z[iz+1]
        theta_z = (zq - z1) / (z2 - z1)
        Vz1 = Vt[:, :, iz]
        Vz2 = Vt[:, :, iz+1]
        Vz  = (1-theta_z).*Vz1 .+ theta_z.*Vz2   # lon×lat
    end

    # lat axis
    if length(lat) == 1
        Vphi = Vz[:, 1]                  # lon
    else
        ilat = clamp(searchsortedlast(lat, latq), 1, length(lat)-1)
        phi1, phi2 = lat[ilat], lat[ilat+1]
        theta_lat = (latq - phi1) / (phi2 - phi1)
        Vphi = (1-theta_lat).*Vz[:, ilat] .+ theta_lat.*Vz[:, ilat+1]  # lon
    end

    # lon axis
    if length(lon) == 1
        return Vphi[1]
    else
        ilon = clamp(searchsortedlast(lon, lonq2), 1, length(lon)-1)
        lon1, lon2 = lon[ilon], lon[ilon+1]
        theta_lon = (lonq2 - lon1) / (lon2 - lon1)
        return (1-theta_lon)*Vphi[ilon] + theta_lon*Vphi[ilon+1]
    end
end


# 3-D separable linear interpolation where the vertical step is done in log-space
# (interpolate log(V) vs log(z), then exponentiate), while lat/lon remain linear.
function _interp3_logz_linear(lat::AbstractVector, lon::AbstractVector, z::AbstractVector,
                              Vt::AbstractArray{<:Real,3}, latq::Real, lonq::Real, zq::Real)

    lonq2 = _wrap_lon_for_grid(lon, lonq)
    
    #  vertical (z) step in log-space 
    iz = clamp(searchsortedlast(z, zq), 1, length(z)-1)
    z1, z2 = z[iz], z[iz+1]
    theta_z = (zq - z1) / (z2 - z1)

    # Ensure strictly positive values for log; if any nonpositive, fall back to linear z.
    Vz1_raw = Vt[:, :, iz]
    Vz2_raw = Vt[:, :, iz+1]

    if any(!isfinite, (z1, z2)) || z1 <= 0 || z2 <= 0 ||
       any(x -> x <= 0 || !isfinite(x), Vz1_raw) ||
       any(x -> x <= 0 || !isfinite(x), Vz2_raw)
        # fallback: ordinary linear-in-z
        Vz = (1-theta_z).*Vz1_raw .+ theta_z.*Vz2_raw
    else
        # log-space interpolation
        logz1 = log(z1); logz2 = log(z2); logzq = log(zq)
        theta_z_log = (logzq - logz1) / (logz2 - logz1)

        Vz1 = log.(Vz1_raw)
        Vz2 = log.(Vz2_raw)
        Vz_log = (1-theta_z_log).*Vz1 .+ theta_z_log.*Vz2
        Vz = exp.(Vz_log)  # now lon×lat slice at the requested z
    end

    #  lat step (linear) 
    ilat = clamp(searchsortedlast(lat, latq), 1, length(lat)-1)
    phi1, phi2 = lat[ilat], lat[ilat+1]
    theta_lat = (latq - phi1) / (phi2 - phi1)
    Vphi = (1-theta_lat).*Vz[:, ilat] .+ theta_lat.*Vz[:, ilat+1]  # now lon

    #  lon step (linear) 
    ilon = clamp(searchsortedlast(lon, lonq2), 1, length(lon)-1)
    lon1, lon2 = lon[ilon], lon[ilon+1]
    theta_lon = (lonq2 - lon1) / (lon2 - lon1)
    return (1-theta_lon)*Vphi[ilon] + theta_lon*Vphi[ilon+1]
end

#  SciML vertical helper (quadratic in log(z) on log(values)) 
# Uses DataInterpolations.jl; falls back to linear in log-space or constants if needed.
function _sciml_quad_logz(z::AbstractVector, v::AbstractVector, zq::Real)
    # keep only strictly positive, finite pairs (required for log)
    mask = (z .> 0) .& isfinite.(z) .& (v .> 0) .& isfinite.(v)
    z_ok = z[mask]; v_ok = v[mask]

    if length(z_ok) == 0
        return NaN
    elseif length(z_ok) == 1
        return v_ok[1]
    elseif length(z_ok) == 2
        # linear in log-space between two nearest
        itp = DataInterpolations.LinearInterpolation(log.(v_ok), log.(z_ok))
        return exp(itp(log(zq)))
    else
        # quadratic in log-space using all available points
        # DataInterpolations constructors are (data, knots)
        itp = DataInterpolations.QuadraticSpline(log.(v_ok), log.(z_ok))
        return exp(itp(log(zq)))
    end
end

# Interpolate in lon-lat-z-time (nearest / linear / logz_linear / logz_quadratic).
# For single-time: :linear/:logz_linear/:logz_quadratic do 3-D spatial; :nearest picks nearest grid point.
function _interp4(lat, lon, z, t, V, latq, lonq, zq, tq; mode::Symbol=:nearest)
    lonq2 = _wrap_lon_for_grid(lon, lonq)

    #  Single-time case 
    if length(t) == 1
        if mode == :linear
            Vt = V[:, :, :, 1]
            return _interp3_linear(lat, lon, z, Vt, latq, lonq2, zq)
        elseif mode == :logz_linear
            Vt = V[:, :, :, 1]
            return _interp3_logz_linear(lat, lon, z, Vt, latq, lonq2, zq)
        elseif mode == :logz_quadratic
            Vt = V[:, :, :, 1]
            return _interp3_bilin_then_quadlogz(lat, lon, z, Vt, latq, lonq2, zq)
        else
            ilat = _nearest_index(lat, latq)
            ilon = _nearest_index(lon, lonq2)
            iz   = _nearest_index(z, zq)
            return V[ilon, ilat, iz, 1]
        end
    end

    #  Multi-time cases 
    if mode == :nearest
        ilat = _nearest_index(lat, latq)
        ilon = _nearest_index(lon, lonq2)
        iz   = _nearest_index(z, zq)
        it   = _nearest_index(t, tq)
        return V[ilon, ilat, iz, it]

    elseif mode == :linear
        it = clamp(searchsortedlast(t, tq), 1, length(t)-1)
        theta_t = (tq - t[it]) / (t[it+1] - t[it])
        V1 = V[:, :, :, it]
        V2 = V[:, :, :, it+1]
        v1 = _interp3_linear(lat, lon, z, V1, latq, lonq2, zq)
        v2 = _interp3_linear(lat, lon, z, V2, latq, lonq2, zq)
        return (1-theta_t)*v1 + theta_t*v2

    elseif mode == :logz_linear
        it = clamp(searchsortedlast(t, tq), 1, length(t)-1)
        theta_t = (tq - t[it]) / (t[it+1] - t[it])
        V1 = V[:, :, :, it]
        V2 = V[:, :, :, it+1]
        v1 = _interp3_logz_linear(lat, lon, z, V1, latq, lonq2, zq)
        v2 = _interp3_logz_linear(lat, lon, z, V2, latq, lonq2, zq)
        return (1-theta_t)*v1 + theta_t*v2

    elseif mode == :logz_quadratic
        it = clamp(searchsortedlast(t, tq), 1, length(t)-1)
        theta_t = (tq - t[it]) / (t[it+1] - t[it])
        V1 = V[:, :, :, it]
        V2 = V[:, :, :, it+1]
        v1 = _interp3_bilin_then_quadlogz(lat, lon, z, V1, latq, lonq2, zq)
        v2 = _interp3_bilin_then_quadlogz(lat, lon, z, V2, latq, lonq2, zq)
        return (1-theta_t)*v1 + theta_t*v2

    else
    error("Unsupported interpolation mode: $mode (use :nearest, :linear, :logz_linear, or :logz_quadratic)")
    end
end

function _bilinear_lonlat(lat::AbstractVector, lon::AbstractVector,
                          grid::AbstractArray{<:Real,2}, latq::Real, lonq::Real)
    lonq2 = _wrap_lon_for_grid(lon, lonq)

    ilat = clamp(searchsortedlast(lat, latq), 1, length(lat)-1)
    phi1, phi2 = lat[ilat], lat[ilat+1]
    theta_lat = (latq - phi1) / (phi2 - phi1)

    ilon = clamp(searchsortedlast(lon, lonq2), 1, length(lon)-1)
    lon1, lon2 = lon[ilon], lon[ilon+1]
    theta_lon = (lonq2 - lon1) / (lon2 - lon1)

    v11 = grid[ilon,   ilat  ]
    v21 = grid[ilon+1, ilat  ]
    v12 = grid[ilon,   ilat+1]
    v22 = grid[ilon+1, ilat+1]

    return (1-theta_lon)*(1-theta_lat)*v11 + theta_lon*(1-theta_lat)*v21 + (1-theta_lon)*theta_lat*v12 + theta_lon*theta_lat*v22
end

# Local quadratic in log-space using 3 nearest points; fallback to linear if needed
function _quad_interp_logz(z::AbstractVector, v::AbstractVector, zq::Real)
    # keep only strictly positive, finite pairs (required for log)
    mask = map(i -> z[i] > 0 && isfinite(z[i]) && v[i] > 0 && isfinite(v[i]), eachindex(z))
    z_ok = collect(z[mask]); v_ok = collect(v[mask])

    if length(z_ok) == 0
        return NaN
    elseif length(z_ok) == 1
        return v_ok[1]  # only one level → best we can do
    end

    x = log.(z_ok)
    y = log.(v_ok)
    xq = log(zq)

    # choose up to 3 nearest nodes in x
    ord = sortperm(abs.(x .- xq))
    sel = x[ord[1:min(3, length(x))]]
    sely = y[ord[1:min(3, length(y))]]

    if length(sel) == 2
        # linear in log-space between two nearest
        (x1, x2) = (sel[1], sel[2]); (y1, y2) = (sely[1], sely[2])
        t = (xq - x1) / (x2 - x1)
        yq = (1-t)*y1 + t*y2
        return exp(yq)
    else
        # quadratic via Lagrange basis on 3 points (sorted by x for stability)
        p = sortperm(sel)
        x1, x2, x3 = sel[p[1]], sel[p[2]], sel[p[3]]
        y1, y2, y3 = sely[p[1]], sely[p[2]], sely[p[3]]

        denom1 = (x1-x2)*(x1-x3)
        denom2 = (x2-x1)*(x2-x3)
        denom3 = (x3-x1)*(x3-x2)
        # guard degenerate spacing
        if denom1 == 0 || denom2 == 0 || denom3 == 0
            # fall back to linear using the two closest
            (x1L, x2L) = (sel[p[1]], sel[p[2]]); (y1L, y2L) = (sely[p[1]], sely[p[2]])
            t = (xq - x1L) / (x2L - x1L)
            yq = (1-t)*y1L + t*y2L
            return exp(yq)
        end

        L1 = ((xq-x2)*(xq-x3)) / denom1
        L2 = ((xq-x1)*(xq-x3)) / denom2
        L3 = ((xq-x1)*(xq-x2)) / denom3
        yq = L1*y1 + L2*y2 + L3*y3
        return exp(yq)
    end
end

# Bilinear in lon/lat at each z-level, then quadratic in log(z)-log(v) across all levels
function _interp3_bilin_then_quadlogz(lat::AbstractVector, lon::AbstractVector, z::AbstractVector,
                                      Vt::AbstractArray{<:Real,3}, latq::Real, lonq::Real, zq::Real)

    # Build v(z_k) = bilinear lon/lat value at each level
    v_at_levels = Vector{Float64}(undef, length(z))
    for k in eachindex(z)
        # Vt is lon×lat×z
        @views v_at_levels[k] = _bilinear_lonlat(lat, lon, Vt[:, :, k], latq, lonq)
    end
    return _sciml_quad_logz(z, v_at_levels, zq)   # ← use SciML
end



#  Guardrails 
const _ALLOWED_INTERP_NORM = Set([:nearest, :linear, :logz_linear, :logz_quadratic])

function _validate_query_args(interp::Symbol, dt::DateTime, latq::Real, lonq::Real, alt_km::Real)::Symbol
    mode = _normalize_interp(interp)  # map :sciml → :logz_quadratic

    # allow users to pass :sciml, but enforce normalized membership
    mode in _ALLOWED_INTERP_NORM ||
        throw(ArgumentError("interpolation must be one of $(collect(_ALLOWED_INTERP_NORM)) or :sciml; got $interp"))

    isfinite(latq) && -90.0 <= latq <= 90.0 ||
        throw(ArgumentError("lat must be finite and in [-90, 90]; got $latq"))

    isfinite(lonq) || throw(ArgumentError("lon must be finite; got $lonq"))
    isfinite(alt_km) || throw(ArgumentError("alt_km must be finite; got $alt_km"))
    alt_km > 0 || throw(ArgumentError("alt_km must be > 0 km (needed for vertical interpolation); got $alt_km"))

    return mode
end

# Treat :sciml as an alias of :logz_quadratic
_normalize_interp(s::Symbol) = (s === :sciml ? :logz_quadratic : s)


#  Public API 

"""
    get_density(itp::WAMInterpolator, dt::DateTime, lat::Real, lon::Real, alt_km::Real)

Return neutral density at (`dt`, `lat`, `lon`, `alt_km`) using WAM‑IPE outputs.
"""

function get_density(itp::WAMInterpolator, dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    mode = _validate_query_args(itp.interpolation, dt, latq, lonq, alt_km)

    # 1) Find local cached file paths (does S3 download if missing)
    p_lo, p_hi, prod_lo, prod_hi = _get_two_files_exact(itp, dt)
    @debug "[fetch] Using files: low=[$(prod_lo)] $(basename(p_lo)), high=[$(prod_hi)] $(basename(p_hi))"

    # 2) Open via pooled handles (pin); do NOT close—just unpin in finally
    ds_lo = _open_nc_cached(p_lo)
    ds_hi = _open_nc_cached(p_hi)

    # 3) Parse valid times (YYYYMMDD_HHMMSS from filename)
    t_lo = _parse_valid_time_from_key(p_lo)
    t_hi = _parse_valid_time_from_key(p_hi)
    t_lo === nothing && (t_lo = t_hi)
    t_hi === nothing && (t_hi = t_lo)

    try
        lat, lon, z, t, V, (latname, lonname, zname, tname) =
            _load_grids(ds_lo, itp.varname; file_time=t_lo)
        tdts, epoch, scale = _decode_time_units(ds_lo, tname, t)
        tq_lo = (epoch === nothing) ? t_lo : _encode_query_time(t_lo, epoch, scale)
        zq_lo = _maybe_convert_alt(z, alt_km, ds_lo, zname)
        v_lo  = _interp4(lat, lon, z, tdts, V, latq, lonq, zq_lo, tq_lo; mode=mode)

        lat2, lon2, z2, t2, V2, (latname2, lonname2, zname2, tname2) =
            _load_grids(ds_hi, itp.varname; file_time=t_hi)
        tdts2, epoch2, scale2 = _decode_time_units(ds_hi, tname2, t2)
        tq_hi = (epoch2 === nothing) ? t_hi : _encode_query_time(t_hi, epoch2, scale2)
        zq_hi = _maybe_convert_alt(z2, alt_km, ds_hi, zname2)
        v_hi  = _interp4(lat2, lon2, z2, tdts2, V2, latq, lonq, zq_hi, tq_hi; mode=mode)

        # 4) Temporal blend at query dt
        if t_lo == t_hi
            return float(v_lo)
        else
            itp_t = DataInterpolations.LinearInterpolation(
                [float(v_lo), float(v_hi)],
                [Dates.value(t_lo), Dates.value(t_hi)]
            )
            return itp_t(Dates.value(dt))
        end
    finally
        # unpin (keeps files open in pool for reuse)
        _unpin_nc_cached(p_lo)
        _unpin_nc_cached(p_hi)
    end
end



"""
    get_density_batch(itp, dts, lats, lons, alts_km) -> Vector{Float64}

Vectorized call matching Python API.
"""
function get_density_batch(itp::WAMInterpolator, dts::AbstractVector{<:DateTime},
                           lats::AbstractVector, lons::AbstractVector, alts_km::AbstractVector)
    n = length(dts)
    @assert length(lats)==n==length(lons)==length(alts_km)
    
    # Parallel version
    results = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        results[i] = get_density(itp, dts[i], lats[i], lons[i], alts_km[i])
    end
    return results
end

function get_density_from_key(itp::WAMInterpolator, key::AbstractString,
                              dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    mode = _normalize_interp(itp.interpolation)

    # Ensure the file is present in on-disk cache; get local path
    aws = _aws_cfg(itp.region)
    local_path = _download_to_cache(aws, itp.bucket, String(key); cache_dir=DEFAULT_CACHE_DIR, verbose=true)

    # Open via pooled handles and unpin after
    ds = _open_nc_cached(local_path)
    try
        t_file = _parse_valid_time_from_key(String(key))
        lat, lon, z, t, V, (latname, lonname, zname, tname) =
            _load_grids(ds, itp.varname; file_time=t_file)

        tdts, epoch, scale = _decode_time_units(ds, tname, t)
        tq = (epoch === nothing) ? (t_file === nothing ? dt : t_file) : _encode_query_time(dt, epoch, scale)

        zq = _maybe_convert_alt(z, alt_km, ds, zname)
        return _interp4(lat, lon, z, tdts, V, latq, lonq, zq, tq; mode=mode)
    finally
        _unpin_nc_cached(local_path)
    end
end


"""
    get_density_at_point(itp, dt, lat, lon, alt_m;
                         angles_in_deg = false)

Wrapper around `get_density` that works directly with altitude in metres and
angles in either radians or degrees.

Arguments

- `itp::WAMInterpolator` : configuration object.
- `dt::DateTime`         : physical time of the state (UTC).
- `lat::Real`            : latitude (rad by default).
- `lon::Real`            : longitude (rad by default).
- `alt_m::Real`          : geometric altitude in metres.

Keyword arguments
--
- `angles_in_deg::Bool=false` : set to `true` if `lat`/`lon` are already in
    degrees. Otherwise they are assumed to be in radians and converted.
"""
function get_density_at_point(itp::WAMInterpolator,
                              dt::DateTime,
                              lat::Real,
                              lon::Real,
                              alt_m::Real;
                              angles_in_deg::Bool = false)

    # Convert to degrees if coming from typical orbital libraries (radians)
    lat_deg = angles_in_deg ? float(lat) : rad2deg(float(lat))
    lon_deg = angles_in_deg ? float(lon) : rad2deg(float(lon))

    # Altitude metres → kilometres
    alt_km = float(alt_m) * 1e-3

    return get_density(itp, dt, lat_deg, lon_deg, alt_km)
end


"""
    get_density_trajectory(itp, dts, lats, lons, alts_m;
                           angles_in_deg = false)

Vectorised wrapper around `get_density` for a full trajectory.

Arguments

- `dts::AbstractVector{<:DateTime}` : time stamps along the trajectory.
- `lats::AbstractVector`            : latitudes (rad by default).
- `lons::AbstractVector`            : longitudes (rad by default).
- `alts_m::AbstractVector`          : altitudes in metres.

Keyword arguments
--
- `angles_in_deg::Bool=false` : set to `true` if `lats`/`lons` are already in
    degrees; otherwise they are assumed to be in radians.

Returns
-
`Vector{Float64}` of neutral densities, same length as `dts`.
"""
function get_density_trajectory(itp::WAMInterpolator,
                                dts::AbstractVector{<:DateTime},
                                lats::AbstractVector,
                                lons::AbstractVector,
                                alts_m::AbstractVector;
                                angles_in_deg::Bool = false)
    # after you construct `times :: Vector{DateTime}`
    WamIPEDensity.prewarm_cache!(WAM_DEFAULT_INTERP, times)

    n = length(dts)
    @assert length(lats)    == n "lats length must match dts"
    @assert length(lons)    == n "lons length must match dts"
    @assert length(alts_m)  == n "alts_m length must match dts"

    # Copy into plain Float64 vectors
    latv  = Float64.(lats)
    lonv  = Float64.(lons)
    altkm = Float64.(alts_m) .* 1e-3

    if !angles_in_deg
        latv .= rad2deg.(latv)
        lonv .= rad2deg.(lonv)
    end

    return get_density_batch(itp, dts, latv, lonv, altkm)
end

function get_density_trajectory_optimized(itp::WAMInterpolator,
                                         dts::AbstractVector{<:DateTime},
                                         lats::AbstractVector,
                                         lons::AbstractVector,
                                         alts_m::AbstractVector;
                                         angles_in_deg::Bool = false)
    
    WamIPEDensity.prewarm_cache!(WAM_DEFAULT_INTERP, times)

    n = length(dts)
    latv = angles_in_deg ? Float64.(lats) : rad2deg.(Float64.(lats))
    lonv = angles_in_deg ? Float64.(lons) : rad2deg.(Float64.(lons))
    altkm = Float64.(alts_m) .* 1e-3
    
    # Group queries by which file pair they need
    file_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for i in 1:n
        p_lo, p_hi, _, _ = _get_two_files_exact(itp, dts[i])
        key = (p_lo, p_hi)
        push!(get!(file_groups, key, Int[]), i)
    end
    
    results = Vector{Float64}(undef, n)
    
    # Process each file pair only once
    for ((p_lo, p_hi), indices) in file_groups
        ds_lo = _open_nc_cached(p_lo)
        ds_hi = _open_nc_cached(p_hi)
        
        try
            # Load grids once per file pair
            t_lo = _parse_valid_time_from_key(p_lo)
            t_hi = _parse_valid_time_from_key(p_hi)
            
            lat_lo, lon_lo, z_lo, t_lo_arr, V_lo, names_lo = _load_grids(ds_lo, itp.varname; file_time=t_lo)
            lat_hi, lon_hi, z_hi, t_hi_arr, V_hi, names_hi = _load_grids(ds_hi, itp.varname; file_time=t_hi)
            
            tdts_lo, epoch_lo, scale_lo = _decode_time_units(ds_lo, names_lo[4], t_lo_arr)
            tdts_hi, epoch_hi, scale_hi = _decode_time_units(ds_hi, names_hi[4], t_hi_arr)
            
            # Interpolate all points using this file pair
            mode = _normalize_interp(itp.interpolation)
            for idx in indices
                zq_lo = _maybe_convert_alt(z_lo, altkm[idx], ds_lo, names_lo[3])
                zq_hi = _maybe_convert_alt(z_hi, altkm[idx], ds_hi, names_hi[3])
                
                tq_lo = (epoch_lo === nothing) ? t_lo : _encode_query_time(t_lo, epoch_lo, scale_lo)
                tq_hi = (epoch_hi === nothing) ? t_hi : _encode_query_time(t_hi, epoch_hi, scale_hi)
                
                v_lo = _interp4(lat_lo, lon_lo, z_lo, tdts_lo, V_lo, latv[idx], lonv[idx], zq_lo, tq_lo; mode=mode)
                v_hi = _interp4(lat_hi, lon_hi, z_hi, tdts_hi, V_hi, latv[idx], lonv[idx], zq_hi, tq_hi; mode=mode)
                
                # Temporal interpolation
                if t_lo == t_hi
                    results[idx] = float(v_lo)
                else
                    itp_t = DataInterpolations.LinearInterpolation(
                        [float(v_lo), float(v_hi)],
                        [Dates.value(t_lo), Dates.value(t_hi)]
                    )
                    results[idx] = itp_t(Dates.value(dts[idx]))
                end
            end
        finally
            _unpin_nc_cached(p_lo)
            _unpin_nc_cached(p_hi)
        end
    end
    
    return results
end

function prewarm_cache!(itp::WAMInterpolator, dts::AbstractVector{<:DateTime})
    unique_files = Set{Tuple{String,String}}()
    for dt in dts
        p_lo, p_hi, _, _ = _get_two_files_exact(itp, dt)
        push!(unique_files, (p_lo, p_hi))
    end
    
    println("Pre-downloading $(length(unique_files)) unique file pairs...")
    # Files are already downloaded by _get_two_files_exact
    return length(unique_files)
end

function __init__()
    if ccall(:jl_generating_output, Cint, ()) == 0
        _install_run_timer!()
    end
end


WamIPEDensity.reset_run_timer!()   # marks the start time right now


end # module