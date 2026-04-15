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
using CSV, DataFrames 
using FilePathsBase: joinpath
using Base: mkpath
using SatelliteToolbox
using SatelliteToolboxAtmosphericModels

# EXPORTS

export WAMInterpolator, GEOSFPInterpolator, NRLMSISEInterpolator, HybridDensityInterpolator,
       get_density, get_density_batch, get_density_at_point, 
       get_density_trajectory, get_density_trajectory_optimised, mean_density_profile, 
       plot_global_mean_profile, plot_global_mean_profile_plots,
       prewarm_cache!, set_max_open_datasets!, print_cache_stats, clear_grid_cache!,
       inspect_geos_file, inspect_geos_remote_file

# CONSTANTS AND GLOBAL STATE
const _CACHE_META_FILE = "metadata.bin"
const DEFAULT_CACHE_DIR = normpath("./cache")
# GEOS-FP constants
const DEFAULT_GEOS_CACHE_DIR = normpath("./cache_geosfp")
const R_D_GEOS  = 287.05          # J/(kg*K)
const G0_GEOS   = 9.80665         # m/s^2
const P_TOP_GEOS = 0.01           # Pa fallback top pressure for DELP integration

# Run timer state
const _WIPED_RUN_START_WALL = Ref{DateTime}(DateTime(0))
const _WIPED_RUN_START_NS   = Ref{Int}(0)
const _WIPED_TIMER_READY    = Ref(false)

# File pair cache for temporal interpolation
const _FILEPAIR_CACHE = Dict{Tuple{String,DateTime}, Tuple{String,String,String,String}}()
const _FILEPAIR_LOCK  = ReentrantLock()

# Allowed interpolation modes
const _ALLOWED_INTERP_NORM = Set([:nearest, :linear, :logz_linear, :logz_quadratic])

# Version windows for WAM-IPE data
const _VERSION_WINDOWS = (
    ("v1.1", DateTime(2023,3,20,21,10,0), DateTime(2023,6,30,21,0,0)), # inclusive start/end
    ("v1.2", DateTime(2023,6,30,21,10,0), nothing), # open-ended
)

# Special WRS cycle constants
const _WRS_00Z_FIRST_TIME = Time(3, 10, 0)  # first valid file under 00Z folder is ..._031000.nc

# Cache metadata
const _CACHE_META_FILE = "metadata.bin"

# MAIN DATA STRUCTURE

Base.@kwdef struct WAMInterpolator
    bucket::String = "noaa-nws-wam-ipe-pds"
    root_prefix::String = "v1.2" # S3 root prefix for WAM-IPE data
    product::String = "wfs"
    varname::String = "den"
    region::String = "us-east-1"
    interpolation::Symbol = :sciml
end


Base.@kwdef struct GEOSFPInterpolator
    root_url::String = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
    collection::String = "inst3_3d_asm_Np"
    interpolation::Symbol = :sciml
    cache_dir::String = DEFAULT_GEOS_CACHE_DIR

    # variable names confirmed from your file
    qv_varname::String = "QV"
    t_varname::String = "T"
    z_varname::String = "H"
    lev_varname::String = "lev"
    density_varname::String = ""

    min_alt_km::Float64 = 0.0
    max_alt_km::Float64 = 70.0
end
Base.@kwdef struct NRLMSISEInterpolator
    interpolation::Symbol = :nearest   # kept only for API symmetry
    min_alt_km::Float64 = 0.0
    max_alt_km::Float64 = 100.0
    space_indices_initialized::Base.RefValue{Bool} = Ref(false)
end


Base.@kwdef struct HybridDensityInterpolator
    geos::GEOSFPInterpolator = GEOSFPInterpolator()
    msis::NRLMSISEInterpolator = NRLMSISEInterpolator()
    wam::WAMInterpolator = WAMInterpolator()
    msis_max_alt_km::Float64 = 100.0
    geos_bounds_cache::Dict{DateTime, Tuple{Float64, Float64}} = Dict{DateTime, Tuple{Float64, Float64}}()
end

# AWS CONFIGURATION


function _aws_cfg(region::String)
    AWS.AWSConfig(; region=region, creds=nothing)
end

# NETCDF DATASET POOLING (LRU CACHE FOR OPEN FILES)

mutable struct _DSPool
    map::Dict{String,NCDataset}      # path → open dataset
    pins::Dict{String,Int}           # path → active users
    last::Dict{String,Int64}         # path → last use (time_ns)
    max_open::Int                    # cap on simultaneously open datasets
    lock::ReentrantLock
end

const _DSPOOL = _DSPool(
    Dict{String,NCDataset}(),
    Dict{String,Int}(),
    Dict{String,Int64}(),
    50,                           
    ReentrantLock()
)

# Touch for LRU
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

# FILE CACHE SYSTEM (ON-DISC LRU CACHE)

mutable struct _FileCache
    dir::String
    max_bytes::Int64
    map::Dict{String,String}          # key → local_path
    sizes::Dict{String,Int64}         # key → bytes
    order::Vector{String}             # LRU order, oldest at index 1
    bytes::Int64                      # current bytes on disc
    downloading::Set{String}          # keys currently being downloaded
    conds::Dict{String,Condition}     # key → Condition for waiters
    lock::ReentrantLock               # global cache lock
end

# Cache instances keyed by (dir, max_bytes)
const _CACHES = Dict{Tuple{String,Int64}, _FileCache}()


function _cache_meta_path(dir::AbstractString)
    joinpath(dir, _CACHE_META_FILE)
end

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


function _save_cache(cache::_FileCache)
    mkpath(cache.dir)
    open(_cache_meta_path(cache.dir), "w") do io
        serialize(io, cache)
    end
    return nothing
end

function _lru_touch!(cache::_FileCache, key::String)
    # remove if present
    idx = findfirst(==(key), cache.order)
    if idx !== nothing
        deleteat!(cache.order, idx)
    end
    push!(cache.order, key)
end


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

function _cache_get_file!(cache::_FileCache, aws::AWS.AWSConfig, bucket::String, key::String;
                          verbose::Bool=true)
    local_path = normpath(joinpath(cache.dir, key))

    # fast path: already on disc and recorded
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

function _get_cache(cache_dir::AbstractString, max_bytes::Int64)
    key = (String(cache_dir), Int64(max_bytes))
    if haskey(_CACHES, key)
        return _CACHES[key]
    else
        cache = _load_cache(cache_dir, max_bytes)
        return (_CACHES[key] = cache)
    end
end


_cache_path(cache_dir::AbstractString, key::AbstractString) =   
    normpath(joinpath(cache_dir, key))  # preserves v1.2/…/… structure


function _download_to_cache(aws::AWS.AWSConfig, bucket::String, key::String;
                            cache_dir::AbstractString=DEFAULT_CACHE_DIR,
                            cache_max_bytes::Int=2_000_000_000,
                            verbose::Bool=true) 
    cache = _get_cache(cache_dir, cache_max_bytes)
    return _cache_get_file!(cache, aws, bucket, key; verbose=verbose)
end

function _open_nc_from_s3(aws::AWS.AWSConfig, bucket::String, key::String;
                          cache_dir::AbstractString=DEFAULT_CACHE_DIR,
                          cache_max_bytes::Int=2_000_000_000)
    local_path = _download_to_cache(aws, bucket, key;
                                    cache_dir=cache_dir,
                                    cache_max_bytes=cache_max_bytes,
                                    verbose=true)
    return NCDataset(local_path, "r"), local_path
end


function _geos_local_path(itp::GEOSFPInterpolator, dt::DateTime)
    yyyy = Dates.format(Date(dt), dateformat"yyyy")
    mm   = Dates.format(Date(dt), dateformat"mm")
    dd   = Dates.format(Date(dt), dateformat"dd")
    hh   = Dates.format(Time(dt), dateformat"HH")
    mkpath(itp.cache_dir)
    return joinpath(itp.cache_dir,
                    itp.collection,
                    yyyy,
                    mm,
                    "GEOSFP_" * itp.collection * "_" * yyyy * mm * dd * "_" * hh * "00.nc4")
end


function _geos_build_url(itp::GEOSFPInterpolator, dt::DateTime)
    yyyy = Dates.format(Date(dt), dateformat"yyyy")
    mm   = Dates.format(Date(dt), dateformat"mm")
    dd   = Dates.format(Date(dt), dateformat"dd")
    hh   = Dates.format(Time(dt), dateformat"HH")

    return string(
        itp.root_url,
        "/Y", yyyy,
        "/M", mm,
        "/D", dd,
        "/GEOS.fp.asm.",
        itp.collection, ".",
        yyyy, mm, dd, "_", hh, "00.V01.nc4"
    )
end


function _geos_download_to_cache(itp::GEOSFPInterpolator, dt::DateTime; verbose::Bool=true)
    local_path = _geos_local_path(itp, dt)
    isfile(local_path) && return local_path

    url = _geos_build_url(itp, dt)
    tmp_path = local_path * ".part"
    mkpath(dirname(local_path))

    verbose && println("[geos] get: ", url, " -> ", local_path)

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
        ok = false
        @warn "GEOS download failed for $url" exception=(err, catch_backtrace())
    end

    if ok
        mv(tmp_path, local_path; force=true)
        return local_path
    else
        isfile(tmp_path) && rm(tmp_path; force=true)
        error("Failed to download GEOS-FP file for $dt")
    end
end

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

_have_in_cache(key::AbstractString; cache_dir::AbstractString=DEFAULT_CACHE_DIR) =
    isfile(normpath(joinpath(cache_dir, key)))

# FILE PAIR CACHING (FOR TEMPORAL INTERPOLATION)

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

# GRID CACHING (AVOID RE-LOADING NETCDF DATA)
const _GRID_CACHE = Dict{String, Tuple}()
const _GRID_CACHE_LOCK = ReentrantLock()
const _GRID_360_CACHE = Dict{UInt64, Bool}()
const _GRID_360_LOCK = ReentrantLock()
const _MAX_GRID_CACHE_SIZE = 20  # Keep last 20 file grids in RAM

function _get_cached_grids(file_path::String, ds::NCDataset, varname::String, file_time::DateTime)    
    lock(_GRID_CACHE_LOCK) do
        if haskey(_GRID_CACHE, file_path)
            return _GRID_CACHE[file_path]
        end
        
        # Load grids (expensive operation - do once per file)
        grids = _load_grids(ds, varname; file_time=file_time)
        
        # Cache it
        _GRID_CACHE[file_path] = grids
        
        # Limit cache size (LRU eviction)
        if length(_GRID_CACHE) > _MAX_GRID_CACHE_SIZE
            # Remove first (oldest) entry
            delete!(_GRID_CACHE, first(keys(_GRID_CACHE)))
        end
        
        return grids
    end
end

function clear_grid_cache!()
    lock(_GRID_CACHE_LOCK) do
        empty!(_GRID_CACHE)
    end
end

# VERSION AND MODEL MAPPING


function _version_for(dt::DateTime)::String
    for (v, lo, hi) in _VERSION_WINDOWS
        if dt >= lo && (hi === nothing || dt <= hi)
            return v
        end
    end
    error("No WAM-IPE version mapping covers $dt")
end


_model_for_version(v::String) = v == "v1.2" ? "wam10" :
                                v == "v1.1" ? "gsm10" :
                                error("Unknown version $v")

# DATE/TIME UTILITIES


@inline function _datetime_floor_10min(dt::DateTime)
    m  = minute(dt)
    mm = m - (m % 10)
    DateTime(Date(dt), Time(hour(dt), mm))
end

_surrounding_10min(dt::DateTime) = (_datetime_floor_10min(dt),
                                    _datetime_floor_10min(dt) + Minute(10))


@inline function _datetime_floor_3hr(dt::DateTime)
    hh = hour(dt) - (hour(dt) % 3)
    return DateTime(Date(dt), Time(hh))
end


function _geos_surrounding_times(dt::DateTime)
    t_lo = _datetime_floor_3hr(dt)
    t_hi = t_lo == dt ? t_lo : t_lo + Hour(3)
    return t_lo, t_hi
end


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


_parse_valid_time_from_key(key::AbstractString) = let m = match(r"(\d{8})_(\d{6})\.nc$", key)
    m === nothing && return nothing
    ymd, hms = m.captures
    DateTime(parse(Int, ymd[1:4]), parse(Int, ymd[5:6]), parse(Int, ymd[7:8]),
             parse(Int, hms[1:2]), parse(Int, hms[3:4]), parse(Int, hms[5:6]))
end

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

# S3 KEY CONSTRUCTION AND FILE RESOLUTION

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
        # fall through to refresh if files were evicted on disc
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
            arch_primary = _wrs_archive(dt_file)
            (p = _try_wrs_from_cycle(dt_file, arch_primary)) !== nothing && return (p, "wrs")

            # fallback: adjacent cycles in case of gaps
            arch_prev = arch_primary - Hour(6)
            (p = _try_wrs_from_cycle(dt_file, arch_prev)) !== nothing && return (p, "wrs")

            arch_next = arch_primary + Hour(6)
            (p = _try_wrs_from_cycle(dt_file, arch_next)) !== nothing && return (p, "wrs")

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

function _geos_get_two_files_exact(itp::GEOSFPInterpolator, dt::DateTime)
    dt_lo, dt_hi = _geos_surrounding_times(dt)
    p_lo = _geos_download_to_cache(itp, dt_lo; verbose=false)
    p_hi = _geos_download_to_cache(itp, dt_hi; verbose=false)
    return p_lo, p_hi, dt_lo, dt_hi
end

function _geos_altitude_bounds(itp::GEOSFPInterpolator, dt::DateTime)
    p_lo, p_hi, t_lo, t_hi = _geos_get_two_files_exact(itp, dt)
    ds_lo = _open_nc_cached(p_lo)
    ds_hi = _open_nc_cached(p_hi)

    try
        _, _, z_lo, _, _, _ = _geos_load_grids(ds_lo, itp; file_time=t_lo)
        _, _, z_hi, _, _, _ = _geos_load_grids(ds_hi, itp; file_time=t_hi)

        # Use overlapping valid range between the two bracketing files
        zmin = max(minimum(z_lo), minimum(z_hi))
        zmax = min(maximum(z_lo), maximum(z_hi))

        return (zmin, zmax)
    finally
        _unpin_nc_cached(p_lo)
        _unpin_nc_cached(p_hi)
    end
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

# NETCDF DATA LOADING AND CF CONVENTIONS

function _cf_decode!(A::AbstractArray, var)
    attrs_any = try
        Dict(var.attrib)
    catch
        Dict(CommonDataModel.attributes(var))
    end

    sf = haskey(attrs_any, "scale_factor") ? float(attrs_any["scale_factor"]) : 1.0
    ao = haskey(attrs_any, "add_offset")   ? float(attrs_any["add_offset"])   : 0.0

    fillvals = Set{Float64}()
    for k in ("_FillValue", "missing_value")
        if haskey(attrs_any, k)
            v = attrs_any[k]
            if v isa AbstractArray
                for x in v
                    if !ismissing(x)
                        push!(fillvals, float(x))
                    end
                end
            else
                if !ismissing(v)
                    push!(fillvals, float(v))
                end
            end
        end
    end

    B = map(A) do x
        ismissing(x) ? NaN : Float64(x)
    end

    if !isempty(fillvals)
        @inbounds for i in eachindex(B)
            if B[i] in fillvals
                B[i] = NaN
            end
        end
    end

    if sf != 1.0 || ao != 0.0
        @inbounds B .= B .* sf .+ ao
    end

    return B
end

function _classify_vertical_units(units_raw::AbstractString)
    s = lowercase(strip(String(units_raw)))
    isempty(s) && return :missing

    # common kilometre spellings
    if occursin(r"\bkm\b", s) || occursin("kilometer", s) || occursin("kilometre", s)
        return :km
    end

    # plain metres (avoid mm/cm false-positives)
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

# Extract grids and variable
# Accepts 3D (lon,lat,z) or 4D (lon,lat,z,time)
# If 3D, we synthesise a 1-point time axis using `file_time` (DateTime).
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
        # Expect lon/lat/z only; synthesise time using `file_time`
        idx_lon  = findfirst(==( :lon ), roles)
        idx_lat  = findfirst(==( :lat ), roles)
        idx_z    = findfirst(==( :z   ), roles)

        idx_lon === nothing && error("Could not find longitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
        idx_lat === nothing && error("Could not find latitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
        idx_z   === nothing && error("Could not find vertical dimension for '$varname'. dims=$(dnames) roles=$(roles)")

        file_time === nothing && error("3-D variable requires `file_time` to synthesise a 1-point time axis.")

        latname = dnames[idx_lat]; lonname = dnames[idx_lon]; zname = dnames[idx_z]; tname = "time"
        lat = haskey(ds, latname) ? collect(ds[latname][:]) : collect(1:size(Vraw, idx_lat))
        lon = haskey(ds, lonname) ? collect(ds[lonname][:]) : collect(1:size(Vraw, idx_lon))
        z   = haskey(ds, zname)   ? collect(ds[zname][:])   : collect(1:size(Vraw, idx_z))
        t   = [file_time]  # synthesised one-element DateTime axis

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
        error("Unsupported vertical units '$units' on '$zname'. Expected kilometres ('km') or metres ('m').")
    end
end

# Convert the dataset's vertical axis to kilometres (vector form)
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
        error("Unsupported or missing vertical units '$units' on '$zname'. Expected kilometres ('km') or metres ('m').")
    end
end

function _geos_classify_dim(dname::String, ds::NCDataset)
    lname = lowercase(dname)
    var   = haskey(ds, dname) ? ds[dname] : nothing
    attrs = var === nothing ? Dict{String,Any}() : Dict(var.attrib)

    stdname = lowercase(string(get(attrs, "standard_name", "")))
    axis    = uppercase(string(get(attrs, "axis", "")))
    units   = lowercase(string(get(attrs, "units", "")))
    longn   = lowercase(string(get(attrs, "long_name", "")))

    # time
    if occursin("time", lname) || axis == "T" || stdname == "time"
        return :time
    end

    # latitude
    if occursin("lat", lname) || stdname == "latitude" || axis == "Y" ||
       occursin("degrees_north", units)
        return :lat
    end

    # longitude
    if occursin("lon", lname) || stdname == "longitude" || axis == "X" ||
       occursin("degrees_east", units)
        return :lon
    end

    # vertical / model level
    if occursin("lev", lname) || occursin("eta", lname) || occursin("layer", lname) ||
       occursin("height", lname) || occursin("alt", lname) || lname == "z" ||
       axis == "Z" || occursin("level", longn)
        return :z
    end

    return :unknown
end

function _geos_dim_indices(ds::NCDataset, varname::String)
    haskey(ds, varname) || error("Variable '$varname' not found in GEOS file.")
    v = ds[varname]
    dnames = String.(NCDatasets.dimnames(v))

    # Build a case-insensitive name → index map
    lname_to_idx = Dict(lowercase(name) => i for (i, name) in pairs(dnames))

    # Try exact canonical GEOS dim names first (lon, lat, lev, time)
    idx_lon  = get(lname_to_idx, "lon",  nothing)
    idx_lat  = get(lname_to_idx, "lat",  nothing)
    idx_z    = get(lname_to_idx, "lev",  nothing)
    idx_time = get(lname_to_idx, "time", nothing)

    # Fall back to heuristic classification only for dims not yet found
    if any(x -> x === nothing, (idx_lon, idx_lat, idx_z, idx_time))
        roles = map(d -> _geos_classify_dim(d, ds), dnames)

        if idx_lon  === nothing; idx_lon  = findfirst(==(:lon),  roles); end
        if idx_lat  === nothing; idx_lat  = findfirst(==(:lat),  roles); end
        if idx_z    === nothing; idx_z    = findfirst(==(:z),    roles); end
        if idx_time === nothing; idx_time = findfirst(==(:time), roles); end
    end

    # Final validation
    idx_lon  === nothing && error("Could not find longitude dimension for '$varname'. dims=$(dnames)")
    idx_lat  === nothing && error("Could not find latitude dimension for '$varname'. dims=$(dnames)")
    idx_z    === nothing && error("Could not find vertical dimension for '$varname'. dims=$(dnames)")
    idx_time === nothing && error("Could not find time dimension for '$varname'. dims=$(dnames)")

    # Sanity check: all four indices must be distinct
    idxs = [idx_lon, idx_lat, idx_z, idx_time]
    length(unique(idxs)) == 4 || error(
        "Duplicate dimension assignments for '$varname': lon=$idx_lon lat=$idx_lat z=$idx_z time=$idx_time. dims=$(dnames)"
    )

    return idx_lon, idx_lat, idx_z, idx_time, dnames
end


function _geos_permute4(A::AbstractArray, idx_lon::Int, idx_lat::Int, idx_z::Int, idx_time::Int)
    perm = (idx_lon, idx_lat, idx_z, idx_time)
    return perm == (1,2,3,4) ? Array(A) : Array(PermutedDimsArray(A, perm))
end


function _geos_coord_vector(ds::NCDataset, dname::String, fallback_len::Int)
    return haskey(ds, dname) ? collect(ds[dname][:]) : collect(1:fallback_len)
end


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


function _geos_get_height4(ds::NCDataset,
                           itp::GEOSFPInterpolator,
                           idx_lon::Int, idx_lat::Int, idx_z::Int, idx_time::Int,
                           dnames::Vector{String},
                           shape_ref::NTuple{4,Int})

    candidates = String[]
    if !isempty(itp.z_varname)
        push!(candidates, itp.z_varname)
    end
    append!(candidates, ["H", "Z", "HEIGHT", "h", "ght"])

    for zname in candidates
        if haskey(ds, zname)
            raw = _cf_decode!(ds[zname][:], ds[zname])
            ndims(raw) == 4 || continue

            z_idx_lon, z_idx_lat, z_idx_z, z_idx_time, _ = _geos_dim_indices(ds, zname)
            z4 = _geos_permute4(raw, z_idx_lon, z_idx_lat, z_idx_z, z_idx_time)

            if size(z4) == shape_ref
                attrs = Dict(ds[zname].attrib)
                units = lowercase(string(get(attrs, "units", "")))

                if occursin(r"\bkm\b", units) || occursin("kilometer", units) || occursin("kilometre", units)
                    return z4 .* 1000.0
                else
                    return z4
                end
            end
        end
    end

    return nothing
end


function _hydrostatic_fill!(z_km::Vector{Float64}, T_col::Vector{Float64}, p_pa::Vector{Float64})
    nz = length(z_km)
    @assert length(T_col) == nz
    @assert length(p_pa) == nz

    # Surface = largest pressure
    sfc = argmax(p_pa)

    # Seed surface altitude if missing
    if !isfinite(z_km[sfc])
        z_km[sfc] = 0.0
    end

    # Upward integration: from surface toward top
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

    # Downward integration: below surface if present
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

    if any(!isfinite, z_km)
        error("Hydrostatic fill produced non-finite z_km values.")
    end

    return z_km
end

function _geos_read_var(ds::NCDataset, varname::String)
    v = ds[varname]
    raw = Array(v)
    return map(x -> ismissing(x) ? NaN : Float64(x), raw)
end

function _nanmean_profile_lonlat(A::AbstractArray{<:Real,4})
    # A is (lon, lat, z, time); use first time slice
    nz = size(A, 3)
    out = Vector{Float64}(undef, nz)

    @inbounds for k in 1:nz
        acc = 0.0
        cnt = 0
        for x in @view A[:, :, k, 1]
            if isfinite(x)
                acc += x
                cnt += 1
            end
        end
        out[k] = cnt == 0 ? NaN : acc / cnt
    end

    return out
end

function _geos_load_grids(ds::NCDataset, itp::GEOSFPInterpolator; file_time::Union{DateTime,Nothing}=nothing)

    # ---- T ----
    # Use ds[var][:] which gives correct Julia-order array matching dimnames order.
    # NCDatasets applies CF decoding (scale/offset/fill→missing) automatically.
    # We just need to convert missing→NaN ourselves.
    idx_lon, idx_lat, idx_z, idx_time, dnames = _geos_dim_indices(ds, itp.t_varname)

    T = _geos_read_var(ds, itp.t_varname)
    T = _geos_permute4(T, idx_lon, idx_lat, idx_z, idx_time)

    # ---- QV ----
    qv_il, qv_ia, qv_iz, qv_it, _ = _geos_dim_indices(ds, itp.qv_varname)
    QV = _geos_read_var(ds, itp.qv_varname)
    QV = _geos_permute4(QV, qv_il, qv_ia, qv_iz, qv_it)

    # ---- H ----
    h_il, h_ia, h_iz, h_it, _ = _geos_dim_indices(ds, itp.z_varname)
    H = _geos_read_var(ds, itp.z_varname)
    H = _geos_permute4(H, h_il, h_ia, h_iz, h_it)

    # ---- size checks ----
    nz = size(T, 3)
    size(T) == size(QV) || error("T shape $(size(T)) != QV shape $(size(QV))")
    size(H, 3) == nz    || error("H vertical size $(size(H,3)) != T vertical size $nz")

    @info "GEOS raw shapes" T=size(T) QV=size(QV) H=size(H)

    # ---- pressure ----
    p_lev = _geos_level_pressure_pa(ds, itp)
    length(p_lev) == nz || error("Pressure-level length $(length(p_lev)) != T vertical size $nz. " *
        "T dimnames=$(dnames), idx_lon=$idx_lon, idx_lat=$idx_lat, idx_z=$idx_z, idx_time=$idx_time, " *
        "size(ds[T][:])=$(size(ds[itp.t_varname][:]))")

    # ---- altitude axis in km ----
    h_attrs = Dict(ds[itp.z_varname].attrib)
    h_units = lowercase(string(get(h_attrs, "units", "m")))
    h_scale = occursin("km", h_units) ? 1.0 : 1.0/1000.0

    H_col = _nanmean_profile_lonlat(H)
    z_km  = H_col .* h_scale

    if any(isnan, z_km)
        @warn "H contains NaN at $(count(isnan, z_km)) levels — using hydrostatic fill"
        T_col = _nanmean_profile_lonlat(T)
        _hydrostatic_fill!(z_km, T_col, p_lev)
    end

    if any(!isfinite, z_km)
    error("GEOS z_km still contains non-finite values after fill.")
    end
    # ---- density ----
    P  = reshape(p_lev, 1, 1, nz, 1)
    P  = repeat(P, size(T,1), size(T,2), 1, size(T,4))
    Tv = T .* (1 .+ 0.61 .* QV)
    V  = P ./ (R_D_GEOS .* Tv)

    # ---- coordinate vectors ----
    lonname = dnames[idx_lon]
    latname = dnames[idx_lat]
    zname   = dnames[idx_z]
    tname   = dnames[idx_time]

    lon = _geos_coord_vector(ds, lonname, size(T, 1))
    lat = _geos_coord_vector(ds, latname, size(T, 2))
    t   = _geos_coord_vector(ds, tname,   size(T, 4))

    @info "GEOS z range" zmin=minimum(z_km) zmax=maximum(z_km)

    if !issorted(z_km)
        perm_z = sortperm(z_km)
        z_km = z_km[perm_z]
        V    = V[:, :, perm_z, :]
    end

    return lat, lon, z_km, t, V, (latname, lonname, zname, tname)
end

# GRID UTILITIES (LONGITUDE WRAPPING)

# Decide grid convention quickly: if any lon > 180, treat as [0, 360); else assume [-180, 180]

@inline function _grid_uses_360(lon::AbstractVector)
    h = hash(lon)
    lock(_GRID_360_LOCK) do
        if haskey(_GRID_360_CACHE, h)
            return _GRID_360_CACHE[h]
        end
        result = maximum(lon) > 180
        _GRID_360_CACHE[h] = result
        return result
    end
end

# Wrap lonq to match the grid's convention
@inline function _wrap_lon_for_grid(lon_grid::AbstractVector, lonq::Real)
    if _grid_uses_360(lon_grid)
        return lonq < 0 ? lonq + 360 : lonq
    else
        return lonq > 180 ? lonq - 360 : lonq
    end
end

# Find nearest indices
@inline _nearest_index(vec::AbstractVector, x::Real) = findmin(abs.(vec .- x))[2]

# INTERPOLATION FUNCTIONS

# 3-D separable linear interpolation over (lon, lat, z) for a single time slice
@inline function _interp3_linear(lat::AbstractVector, lon::AbstractVector, z::AbstractVector,
                         Vt::AbstractArray{<:Real,3}, latq::Real, lonq::Real, zq::Real)::Float64

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
@inline function _interp3_logz_linear(lat::AbstractVector, lon::AbstractVector, z::AbstractVector,
                              Vt::AbstractArray{<:Real,3}, latq::Real, lonq::Real, zq::Real)

    lonq2 = _wrap_lon_for_grid(lon, lonq)
    
    # vertical (z) step in log-space
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

    # lat step (linear)
    ilat = clamp(searchsortedlast(lat, latq), 1, length(lat)-1)
    phi1, phi2 = lat[ilat], lat[ilat+1]
    theta_lat = (latq - phi1) / (phi2 - phi1)
    Vphi = (1-theta_lat).*Vz[:, ilat] .+ theta_lat.*Vz[:, ilat+1]  # now lon

    # lon step (linear)
    ilon = clamp(searchsortedlast(lon, lonq2), 1, length(lon)-1)
    lon1, lon2 = lon[ilon], lon[ilon+1]
    theta_lon = (lonq2 - lon1) / (lon2 - lon1)
    return (1-theta_lon)*Vphi[ilon] + theta_lon*Vphi[ilon+1]
end

@inline function _bilinear_lonlat(lat::AbstractVector, lon::AbstractVector,
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

    return (1-theta_lon)*(1-theta_lat)*v11 + theta_lon*(1-theta_lat)*v21 + 
           (1-theta_lon)*theta_lat*v12 + theta_lon*theta_lat*v22
end

# SciML vertical helper (quadratic in log(z) on log(values))
# Uses DataInterpolations.jl; falls back to linear in log-space or constants if needed.
# function _sciml_quad_logz(z::AbstractVector, v::AbstractVector, zq::Real)
#     # keep only strictly positive, finite pairs (required for log)
#     mask = (z .> 0) .& isfinite.(z) .& (v .> 0) .& isfinite.(v)
#     z_ok = z[mask]; v_ok = v[mask]

#     if length(z_ok) == 0
#         return NaN
#     elseif length(z_ok) == 1
#         return v_ok[1]
#     elseif length(z_ok) == 2
#         # linear in log-space between two nearest
#         itp = DataInterpolations.LinearInterpolation(log.(v_ok), log.(z_ok))
#         return exp(itp(log(zq)))
#     else
#         # quadratic in log-space using all available points
#         itp = DataInterpolations.QuadraticSpline(log.(v_ok), log.(z_ok))
#         return exp(itp(log(zq)))
#     end
# end

############ for now since we havent set up from 70km to 100km ################

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

    
    # clamp query into valid vertical range
    zq_clamped = clamp(float(zq), first(z_ok), last(z_ok))
    
    if zq != zq_clamped
        @info "Clamped WAM altitude request" requested_km=zq used_km=zq_clamped zmin=first(z_ok) zmax=last(z_ok)
    end

    if length(z_ok) == 2
        itp = DataInterpolations.LinearInterpolation(log.(v_ok), log.(z_ok))
        return exp(itp(log(zq_clamped)))
    else
        itp = DataInterpolations.QuadraticSpline(log.(v_ok), log.(z_ok))
        return exp(itp(log(zq_clamped)))
    end
end

# Bilinear in lon/lat at each z-level, then quadratic in log(z)-log(v) across all levels
@inline function _interp3_bilin_then_quadlogz(lat::AbstractVector, lon::AbstractVector, z::AbstractVector,
                                      Vt::AbstractArray{<:Real,3}, latq::Real, lonq::Real, zq::Real)::Float64

    # Build v(z_k) = bilinear lon/lat value at each level
    v_at_levels = Vector{Float64}(undef, length(z))
    for k in eachindex(z)
        @views v_at_levels[k] = _bilinear_lonlat(lat, lon, Vt[:, :, k], latq, lonq)
    end
    return _sciml_quad_logz(z, v_at_levels, zq)
end

# Interpolate in lon-lat-z-time (nearest / linear / logz_linear / logz_quadratic)
@inline function _interp4(lat, lon, z, t, V, latq, lonq, zq, tq; mode::Symbol=:nearest)
    lonq2 = _wrap_lon_for_grid(lon, lonq)

    # Single-time case
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

    # Multi-time cases
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

# VALIDATION AND NORMALISATION

# Treat :sciml as an alias of :logz_quadratic
@inline _normalise_interp(s::Symbol) = (s === :sciml ? :logz_quadratic : s)

function _validate_query_args(interp::Symbol, dt::DateTime, latq::Real, lonq::Real, alt_km::Real)::Symbol
    mode = _normalise_interp(interp)

    # allow users to pass :sciml, but enforce normalised membership
    mode in _ALLOWED_INTERP_NORM ||
        throw(ArgumentError("interpolation must be one of $(collect(_ALLOWED_INTERP_NORM)) or :sciml; got $interp"))

    isfinite(latq) && -90.0 <= latq <= 90.0 ||
        throw(ArgumentError("lat must be finite and in [-90, 90]; got $latq"))

    isfinite(lonq) || throw(ArgumentError("lon must be finite; got $lonq"))
    isfinite(alt_km) || throw(ArgumentError("alt_km must be finite; got $alt_km"))
    alt_km > 0 || throw(ArgumentError("alt_km must be > 0 km (needed for vertical interpolation); got $alt_km"))

    return mode
end

function _validate_query_args_geos(itp::GEOSFPInterpolator, dt::DateTime,
                                   latq::Real, lonq::Real, alt_km::Real)::Symbol
    mode = _normalise_interp(itp.interpolation)

    mode in _ALLOWED_INTERP_NORM ||
        throw(ArgumentError("interpolation must be one of $(collect(_ALLOWED_INTERP_NORM)) or :sciml; got $(itp.interpolation)"))

    isfinite(latq) && -90.0 <= latq <= 90.0 ||
        throw(ArgumentError("lat must be finite and in [-90, 90]; got $latq"))

    isfinite(lonq) || throw(ArgumentError("lon must be finite; got $lonq"))
    isfinite(alt_km) || throw(ArgumentError("alt_km must be finite; got $alt_km"))
    itp.min_alt_km <= alt_km <= itp.max_alt_km ||
        throw(ArgumentError("GEOS-FP backend only supports $(itp.min_alt_km)–$(itp.max_alt_km) km; got $alt_km km"))

    return mode
end

function _validate_query_args_msis(itp::NRLMSISEInterpolator, dt::DateTime,
                                   latq::Real, lonq::Real, alt_km::Real)::Symbol
    isfinite(latq) && -90.0 <= latq <= 90.0 ||
        throw(ArgumentError("lat must be finite and in [-90, 90]; got $latq"))

    isfinite(lonq) || throw(ArgumentError("lon must be finite; got $lonq"))
    isfinite(alt_km) || throw(ArgumentError("alt_km must be finite; got $alt_km"))
    itp.min_alt_km <= alt_km <= itp.max_alt_km ||
        throw(ArgumentError("NRLMSISE backend only supports $(itp.min_alt_km)–$(itp.max_alt_km) km; got $alt_km km"))

    return :nearest
end

function _init_msis_indices!(itp::NRLMSISEInterpolator)
    if itp.space_indices_initialized[]
        return nothing
    end

    try
        SpaceIndices.init()
    catch err
        @warn "SpaceIndices.init() failed; NRLMSISE-00 may still work if indices are passed explicitly." exception=(err, catch_backtrace())
    end

    itp.space_indices_initialized[] = true
    return nothing
end

@inline function _select_backend(itp::HybridDensityInterpolator, dt::DateTime, alt_km::Real)
    if alt_km > itp.msis_max_alt_km
        return :wam
    end

    geos_zmin, geos_zmax = get!(itp.geos_bounds_cache, dt) do
        _geos_altitude_bounds(itp.geos, dt)
    end

    if geos_zmin <= alt_km <= geos_zmax
        return :geos
    else
        return :msis
    end
end
# PUBLIC API - DENSITY RETRIEVAL

# =========================
# GEOS-FP DENSITY INTERPOLATION HELPERS
# =========================

function _geos_interp_density_from_loaded(ds::NCDataset,
                                          itp::GEOSFPInterpolator,
                                          dt::DateTime,
                                          latq::Real,
                                          lonq::Real,
                                          alt_km::Real,
                                          mode::Symbol)

    lat, lon, z, t, V, (latname, lonname, zname, tname) =
        _geos_load_grids(ds, itp; file_time=dt)

    tdts, epoch, scale = _decode_time_units(ds, tname, t)
    tq = (epoch === nothing) ? dt : _encode_query_time(dt, epoch, scale)

    zmin = minimum(z)
    zmax = maximum(z)

    zq_raw = float(alt_km)
    zq = clamp(zq_raw, zmin, zmax)

    if zq != zq_raw
        @info "Clamped GEOS altitude request" requested_km=zq_raw used_km=zq zmin=zmin zmax=zmax
    end

    return _interp4(lat, lon, z, tdts, V, latq, lonq, zq, tq; mode=mode)
end


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
    _get_cached_grids(p_lo, ds_lo, itp.varname, t_lo)
        tdts, epoch, scale = _decode_time_units(ds_lo, tname, t)
        tq_lo = (epoch === nothing) ? t_lo : _encode_query_time(t_lo, epoch, scale)
        zq_lo = _maybe_convert_alt(z, alt_km, ds_lo, zname)
        v_lo  = _interp4(lat, lon, z, tdts, V, latq, lonq, zq_lo, tq_lo; mode=mode)

        lat2, lon2, z2, t2, V2, (latname2, lonname2, zname2, tname2) =
    _get_cached_grids(p_hi, ds_hi, itp.varname, t_hi)
        tdts2, epoch2, scale2 = _decode_time_units(ds_hi, tname2, t2)
        tq_hi = (epoch2 === nothing) ? t_hi : _encode_query_time(t_hi, epoch2, scale2)
        zq_hi = _maybe_convert_alt(z2, alt_km, ds_hi, zname2)
        v_hi  = _interp4(lat2, lon2, z2, tdts2, V2, latq, lonq, zq_hi, tq_hi; mode=mode)

        # 4) Temporal blend at query dt
        if t_lo == t_hi
            return float(v_lo)
        else
            t_lo_val = Float64(Dates.value(t_lo))
            t_hi_val = Float64(Dates.value(t_hi))
            theta_t = (Float64(Dates.value(dt)) - t_lo_val) / (t_hi_val - t_lo_val)
            return (1.0 - theta_t) * float(v_lo) + theta_t * float(v_hi)
        end

    finally
        # unpin (keeps files open in pool for reuse)
        _unpin_nc_cached(p_lo)
        _unpin_nc_cached(p_hi)
    end
end

function get_density(itp::NRLMSISEInterpolator, dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    _validate_query_args_msis(itp, dt, latq, lonq, alt_km)
    _init_msis_indices!(itp)

    out = SatelliteToolboxAtmosphericModels.AtmosphericModels.nrlmsise00(
        dt,
        alt_km * 1000.0,
        deg2rad(float(latq)),
        deg2rad(float(lonq))
    )

    return float(out.total_density)
end

function get_density(itp::GEOSFPInterpolator, dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    mode = _validate_query_args_geos(itp, dt, latq, lonq, alt_km)

    # 1) Resolve/cached bracketing files
    p_lo, p_hi, t_lo, t_hi = _geos_get_two_files_exact(itp, dt)

    # 2) Open via pooled handles
    ds_lo = _open_nc_cached(p_lo)
    ds_hi = _open_nc_cached(p_hi)

    try
        v_lo = _geos_interp_density_from_loaded(ds_lo, itp, t_lo, latq, lonq, alt_km, mode)
        v_hi = _geos_interp_density_from_loaded(ds_hi, itp, t_hi, latq, lonq, alt_km, mode)

        # 3) Temporal blend
        if t_lo == t_hi
            return float(v_lo)
        else
            t0 = Float64(Dates.value(t_lo))
            t1 = Float64(Dates.value(t_hi))
            tq = Float64(Dates.value(dt))
            θ  = (tq - t0) / (t1 - t0)
            return (1.0 - θ) * float(v_lo) + θ * float(v_hi)
        end

    finally
        _unpin_nc_cached(p_lo)
        _unpin_nc_cached(p_hi)
    end
end


function get_density(itp::HybridDensityInterpolator, dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    backend = _select_backend(itp, dt, alt_km)

    if backend == :geos
        return get_density(itp.geos, dt, latq, lonq, alt_km)
    elseif backend == :msis
        return get_density(itp.msis, dt, latq, lonq, alt_km)
    else
        return get_density(itp.wam, dt, latq, lonq, alt_km)
    end
end


function get_density_batch(itp::WAMInterpolator, dts::AbstractVector{<:DateTime},
                           lats::AbstractVector, lons::AbstractVector, alts_km::AbstractVector)
    n = length(dts)
    # @assert length(lats)==n==length(lons)==length(alts_km)
    @assert length(lats) == n
    @assert length(lons) == n
    @assert length(alts_km) == n

    # Parallel version
    results = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        results[i] = get_density(itp, dts[i], lats[i], lons[i], alts_km[i])
    end
    return results
end

function get_density_batch(itp::GEOSFPInterpolator, dts::AbstractVector{<:DateTime},
                           lats::AbstractVector, lons::AbstractVector, alts_km::AbstractVector)
    n = length(dts)
    @assert length(lats) == n
    @assert length(lons) == n
    @assert length(alts_km) == n

    results = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        results[i] = get_density(itp, dts[i], lats[i], lons[i], alts_km[i])
    end
    return results
end

function get_density_batch(itp::NRLMSISEInterpolator, dts::AbstractVector{<:DateTime},
                           lats::AbstractVector, lons::AbstractVector, alts_km::AbstractVector)
    n = length(dts)
    @assert length(lats) == n
    @assert length(lons) == n
    @assert length(alts_km) == n

    results = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        results[i] = get_density(itp, dts[i], lats[i], lons[i], alts_km[i])
    end
    return results
end

function get_density_batch(itp::HybridDensityInterpolator,
                           dts::AbstractVector{<:DateTime},
                           lats::AbstractVector,
                           lons::AbstractVector,
                           alts_km::AbstractVector)
    n = length(dts)

    @assert length(lats) == n
    @assert length(lons) == n
    @assert length(alts_km) == n

    results = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        results[i] = get_density(itp, dts[i], lats[i], lons[i], alts_km[i])
    end
    return results
end

function get_density_from_key(itp::WAMInterpolator, key::AbstractString,
                              dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    mode = _normalise_interp(itp.interpolation)

    # Ensure the file is present in on-disc cache; get local path
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


function get_density_at_point(itp::GEOSFPInterpolator,
                              dt::DateTime,
                              lat::Real,
                              lon::Real,
                              alt_m::Real;
                              angles_in_deg::Bool = false)

    lat_deg = angles_in_deg ? float(lat) : rad2deg(float(lat))
    lon_deg = angles_in_deg ? float(lon) : rad2deg(float(lon))
    alt_km  = float(alt_m) * 1e-3

    return get_density(itp, dt, lat_deg, lon_deg, alt_km)
end

function get_density_at_point(itp::NRLMSISEInterpolator,
                              dt::DateTime,
                              lat::Real,
                              lon::Real,
                              alt_m::Real;
                              angles_in_deg::Bool = false)

    lat_deg = angles_in_deg ? float(lat) : rad2deg(float(lat))
    lon_deg = angles_in_deg ? float(lon) : rad2deg(float(lon))
    alt_km  = float(alt_m) * 1e-3

    return get_density(itp, dt, lat_deg, lon_deg, alt_km)
end

function get_density_at_point(itp::HybridDensityInterpolator,
                              dt::DateTime,
                              lat::Real,
                              lon::Real,
                              alt_m::Real;
                              angles_in_deg::Bool = false)
    lat_deg = angles_in_deg ? float(lat) : rad2deg(float(lat))
    lon_deg = angles_in_deg ? float(lon) : rad2deg(float(lon))
    alt_km  = float(alt_m) * 1e-3
    return get_density(itp, dt, lat_deg, lon_deg, alt_km)
end

function get_density_trajectory(itp::WAMInterpolator,
                                dts::AbstractVector{<:DateTime},
                                lats::AbstractVector,
                                lons::AbstractVector,
                                alts_m::AbstractVector;
                                angles_in_deg::Bool = false)
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

function get_density_trajectory(itp::GEOSFPInterpolator,
                                dts::AbstractVector{<:DateTime},
                                lats::AbstractVector,
                                lons::AbstractVector,
                                alts_m::AbstractVector;
                                angles_in_deg::Bool = false)
    n = length(dts)
    @assert length(lats)   == n "lats length must match dts"
    @assert length(lons)   == n "lons length must match dts"
    @assert length(alts_m) == n "alts_m length must match dts"

    latv  = Float64.(lats)
    lonv  = Float64.(lons)
    altkm = Float64.(alts_m) .* 1e-3

    if !angles_in_deg
        latv .= rad2deg.(latv)
        lonv .= rad2deg.(lonv)
    end

    return get_density_batch(itp, dts, latv, lonv, altkm)
end

function get_density_trajectory(itp::NRLMSISEInterpolator,
                                dts::AbstractVector{<:DateTime},
                                lats::AbstractVector,
                                lons::AbstractVector,
                                alts_m::AbstractVector;
                                angles_in_deg::Bool = false)
    n = length(dts)
    @assert length(lats)   == n "lats length must match dts"
    @assert length(lons)   == n "lons length must match dts"
    @assert length(alts_m) == n "alts_m length must match dts"

    latv  = Float64.(lats)
    lonv  = Float64.(lons)
    altkm = Float64.(alts_m) .* 1e-3

    if !angles_in_deg
        latv .= rad2deg.(latv)
        lonv .= rad2deg.(lonv)
    end

    return get_density_batch(itp, dts, latv, lonv, altkm)
end

function get_density_trajectory(itp::HybridDensityInterpolator,
                                dts::AbstractVector{<:DateTime},
                                lats::AbstractVector,
                                lons::AbstractVector,
                                alts_m::AbstractVector;
                                angles_in_deg::Bool = false)
    n = length(dts)
    @assert length(lats)   == n "lats length must match dts"
    @assert length(lons)   == n "lons length must match dts"
    @assert length(alts_m) == n "alts_m length must match dts"

    latv  = Float64.(lats)
    lonv  = Float64.(lons)
    altkm = Float64.(alts_m) .* 1e-3

    if !angles_in_deg
        latv .= rad2deg.(latv)
        lonv .= rad2deg.(lonv)
    end

    return get_density_batch(itp, dts, latv, lonv, altkm)
end

function get_density_trajectory_optimised(itp::WAMInterpolator,
                                         dts::AbstractVector{<:DateTime},
                                         lats::AbstractVector,
                                         lons::AbstractVector,
                                         alts_m::AbstractVector;
                                         angles_in_deg::Bool = false)
    
    n = length(dts)

    # Pre-allocate ALL arrays upfront
    latv = Vector{Float64}(undef, n)
    lonv = Vector{Float64}(undef, n)
    altkm = Vector{Float64}(undef, n)

    # Convert in-place (faster than broadcasting)
    if angles_in_deg
        @inbounds @simd for i in 1:n
            latv[i] = Float64(lats[i])
            lonv[i] = Float64(lons[i])
            altkm[i] = Float64(alts_m[i]) * 1e-3
        end
    else
        @inbounds @simd for i in 1:n
            latv[i] = rad2deg(Float64(lats[i]))
            lonv[i] = rad2deg(Float64(lons[i]))
            altkm[i] = Float64(alts_m[i]) * 1e-3
        end
    end
    
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
            
            lat_lo, lon_lo, z_lo, t_lo_arr, V_lo, names_lo = _get_cached_grids(p_lo, ds_lo, itp.varname, t_lo)
            lat_hi, lon_hi, z_hi, t_hi_arr, V_hi, names_hi = _get_cached_grids(p_hi, ds_hi, itp.varname, t_hi)
            
            tdts_lo, epoch_lo, scale_lo = _decode_time_units(ds_lo, names_lo[4], t_lo_arr)
            tdts_hi, epoch_hi, scale_hi = _decode_time_units(ds_hi, names_hi[4], t_hi_arr)
            
            mode = _normalise_interp(itp.interpolation)

            tq_lo = (epoch_lo === nothing) ? t_lo : _encode_query_time(t_lo, epoch_lo, scale_lo)
            tq_hi = (epoch_hi === nothing) ? t_hi : _encode_query_time(t_hi, epoch_hi, scale_hi)

            same_time = (t_lo == t_hi)
            t_lo_val = same_time ? 0.0 : Float64(Dates.value(t_lo))
            t_hi_val = same_time ? 0.0 : Float64(Dates.value(t_hi))
            t_delta_inv = same_time ? 0.0 : 1.0 / (t_hi_val - t_lo_val)

            for idx in indices
                zq_lo = _maybe_convert_alt(z_lo, altkm[idx], ds_lo, names_lo[3])
                zq_hi = _maybe_convert_alt(z_hi, altkm[idx], ds_hi, names_hi[3])
                
                v_lo = _interp4(lat_lo, lon_lo, z_lo, tdts_lo, V_lo, latv[idx], lonv[idx], zq_lo, tq_lo; mode=mode)
                v_hi = _interp4(lat_hi, lon_hi, z_hi, tdts_hi, V_hi, latv[idx], lonv[idx], zq_hi, tq_hi; mode=mode)
                
                if same_time
                    results[idx] = float(v_lo)
                else
                    theta_t = (Float64(Dates.value(dts[idx])) - t_lo_val) * t_delta_inv
                    results[idx] = (1.0 - theta_t) * float(v_lo) + theta_t * float(v_hi)
                end
            end
        finally
            _unpin_nc_cached(p_lo)
            _unpin_nc_cached(p_hi)
        end
    end
    
    return results
end

function get_density_trajectory_optimised(itp::GEOSFPInterpolator,
                                          dts::AbstractVector{<:DateTime},
                                          lats::AbstractVector,
                                          lons::AbstractVector,
                                          alts_m::AbstractVector;
                                          angles_in_deg::Bool = false)

    n = length(dts)

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

    # Group by file pair
    file_groups = Dict{Tuple{String,String}, Vector{Int}}()
    time_pairs  = Dict{Tuple{String,String}, Tuple{DateTime,DateTime}}()

    for i in 1:n
        p_lo, p_hi, t_lo, t_hi = _geos_get_two_files_exact(itp, dts[i])
        key = (p_lo, p_hi)
        push!(get!(file_groups, key, Int[]), i)
        time_pairs[key] = (t_lo, t_hi)
    end

    results = Vector{Float64}(undef, n)
    mode = _normalise_interp(itp.interpolation)

    for ((p_lo, p_hi), indices) in file_groups
        ds_lo = _open_nc_cached(p_lo)
        ds_hi = _open_nc_cached(p_hi)

        t_lo, t_hi = time_pairs[(p_lo, p_hi)]

        try
            # Load once
            lat_lo, lon_lo, z_lo, tarr_lo, V_lo, names_lo =
                _geos_load_grids(ds_lo, itp; file_time=t_lo)
            lat_hi, lon_hi, z_hi, tarr_hi, V_hi, names_hi =
                _geos_load_grids(ds_hi, itp; file_time=t_hi)

            tdts_lo, epoch_lo, scale_lo = _decode_time_units(ds_lo, names_lo[4], tarr_lo)
            tdts_hi, epoch_hi, scale_hi = _decode_time_units(ds_hi, names_hi[4], tarr_hi)

            tq_lo = (epoch_lo === nothing) ? t_lo : _encode_query_time(t_lo, epoch_lo, scale_lo)
            tq_hi = (epoch_hi === nothing) ? t_hi : _encode_query_time(t_hi, epoch_hi, scale_hi)

            same_time   = (t_lo == t_hi)
            t_lo_val    = same_time ? 0.0 : Float64(Dates.value(t_lo))
            t_hi_val    = same_time ? 0.0 : Float64(Dates.value(t_hi))
            t_delta_inv = same_time ? 0.0 : 1.0 / (t_hi_val - t_lo_val)

            zlo_min, zlo_max = minimum(z_lo), maximum(z_lo)
            zhi_min, zhi_max = minimum(z_hi), maximum(z_hi)

            for idx in indices
                zq_lo = clamp(altkm[idx], zlo_min, zlo_max)
                zq_hi = clamp(altkm[idx], zhi_min, zhi_max)

                v_lo = _interp4(lat_lo, lon_lo, z_lo, tdts_lo, V_lo,
                                latv[idx], lonv[idx], zq_lo, tq_lo; mode=mode)
                v_hi = _interp4(lat_hi, lon_hi, z_hi, tdts_hi, V_hi,
                                latv[idx], lonv[idx], zq_hi, tq_hi; mode=mode)

                if same_time
                    results[idx] = float(v_lo)
                else
                    θ = (Float64(Dates.value(dts[idx])) - t_lo_val) * t_delta_inv
                    results[idx] = (1.0 - θ) * float(v_lo) + θ * float(v_hi)
                end
            end

        finally
            _unpin_nc_cached(p_lo)
            _unpin_nc_cached(p_hi)
        end
    end

    return results
end

function get_density_trajectory_optimised(itp::NRLMSISEInterpolator,
                                          dts::AbstractVector{<:DateTime},
                                          lats::AbstractVector,
                                          lons::AbstractVector,
                                          alts_m::AbstractVector;
                                          angles_in_deg::Bool = false)
    return get_density_trajectory(itp, dts, lats, lons, alts_m; angles_in_deg=angles_in_deg)
end


function get_density_trajectory_optimised(itp::HybridDensityInterpolator,
                                          dts::AbstractVector{<:DateTime},
                                          lats::AbstractVector,
                                          lons::AbstractVector,
                                          alts_m::AbstractVector;
                                          angles_in_deg::Bool = false)
    return get_density_trajectory(itp, dts, lats, lons, alts_m; angles_in_deg=angles_in_deg)
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


function prewarm_cache!(itp::GEOSFPInterpolator, dts::AbstractVector{<:DateTime})
    unique_files = Set{Tuple{String,String}}()
    for dt in dts
        p_lo, p_hi, _, _ = _geos_get_two_files_exact(itp, dt)
        push!(unique_files, (p_lo, p_hi))
    end

    println("Pre-downloading $(length(unique_files)) unique GEOS-FP file pairs...")
    return length(unique_files)
end

function prewarm_cache!(itp::NRLMSISEInterpolator, dts::AbstractVector{<:DateTime})
    _init_msis_indices!(itp)
    return 0
end

function prewarm_cache!(itp::HybridDensityInterpolator,
                        dts::AbstractVector{<:DateTime},
                        alts_km::AbstractVector)
    @assert length(dts) == length(alts_km)

    geos_dts = DateTime[]
    msis_dts = DateTime[]
    wam_dts  = DateTime[]

    for i in eachindex(dts, alts_km)
        backend = _select_backend(itp, dts[i], alts_km[i])
        if backend == :geos
            push!(geos_dts, dts[i])
        elseif backend == :msis
            push!(msis_dts, dts[i])
        else
            push!(wam_dts, dts[i])
        end
    end

    geos_count = isempty(geos_dts) ? 0 : prewarm_cache!(itp.geos, geos_dts)
    msis_count = isempty(msis_dts) ? 0 : prewarm_cache!(itp.msis, msis_dts)
    wam_count  = isempty(wam_dts)  ? 0 : prewarm_cache!(itp.wam,  wam_dts)

    return (geos=geos_count, msis=msis_count, wam=wam_count)
end

# PROFILE AND PLOTTING FUNCTIONS

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

function plot_global_mean_profile_plots(itp::WAMInterpolator, dt::DateTime;
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

    # Create plot with Plots.jl
    p = Plots.plot(
        denp, altp;
        xscale = :log10,
        xlabel = "Density (kg·m⁻³)",
        ylabel = "Altitude (km)",
        legend = false,
        framestyle = :box,
        grid = true,
        title = "Global Mean Density — " * Dates.format(dt, dateformat"yyyy-mm-dd HH:MM 'UTC'"),
        linewidth = 2,
        size = (800, 600),
        dpi = 150
    )
    
    if alt_max_km !== nothing
        Plots.ylims!(p, (0, float(alt_max_km)))
    end

    Plots.savefig(p, png_path)

    if export_csv
        open(csv_path, "w") do io
            write(io, "altitude_km,density_kg_m3\n")
            @inbounds for i in eachindex(altp)
                write(io, string(altp[i], ",", denp[i], "\n"))
            end
        end
    end

    return p, png_path, csv_path
end

function inspect_geos_file(path::AbstractString)
    ds = NCDataset(String(path), "r")
    try
        println("FILE: ", path)
        println("VARIABLES:")
        for k in keys(ds)
            print("  ", k)
            try
                print(" => size=", size(ds[k]))
            catch
            end
            println()

            try
                attrs = Dict(ds[k].attrib)
                for aname in ("units", "long_name", "standard_name", "axis")
                    if haskey(attrs, aname)
                        println("      ", aname, " = ", attrs[aname])
                    end
                end
            catch
            end
        end
    finally
        close(ds)
    end
    return nothing
end

function inspect_geos_remote_file(itp::GEOSFPInterpolator, dt::DateTime)
    path = _geos_download_to_cache(itp, dt; verbose=true)
    inspect_geos_file(path)
    return path
end

end # module