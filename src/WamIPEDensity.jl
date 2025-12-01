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

const DEFAULT_CACHE_DIR = normpath("./cache") # DEFAULT_CACHE_DIR = abspath(joinpath(@__DIR__, "..", "cache"))

export WAMInterpolator, get_density, get_density_batch, get_density_at_point, get_density_trajectory

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

"""
    _aws_cfg(region) returns AWS.AWSConfig

Create an unsigned AWS config for `region`.
This avoids credential requirements for public WAM-IPE objects.
"""
function _aws_cfg(region::String)
    AWS.AWSConfig(; region=region, creds=nothing)
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
                            verbose::Bool=true)
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
                          verbose::Bool=true)
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

    # - update cache (under lock) -
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

        # evict if needed
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

# Build a WRS key but forcing the archive (cycle) hour you want
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

# function _get_two_files_exact(itp::WAMInterpolator, dt::DateTime)
#     dt_lo, dt_hi = _surrounding_10min(dt)
#     pref, alt = _product_fallback_order(itp.product)  # e.g., ("wfs","wrs")

#     #  lower bracket: prefer `pref`, fall back to `alt` 
#     p_lo = _try_download(itp, dt_lo, pref)
#     prod_lo = p_lo === nothing ? begin
#         p = _try_download(itp, dt_lo, alt)
#         p === nothing ? nothing : (p, alt)
#     end : (p_lo, pref)

#     # upper bracket: prefer `pref`, fall back to `alt` 
#     p_hi = _try_download(itp, dt_hi, pref)
#     prod_hi = p_hi === nothing ? begin
#         p = _try_download(itp, dt_hi, alt)
#         p === nothing ? nothing : (p, alt)
#     end : (p_hi, pref)

#     # Validate we got both
#     if prod_lo === nothing || prod_hi === nothing
#         missing_sides = String[]
#         prod_lo === nothing && push!(missing_sides, "low @ $(dt_lo)")
#         prod_hi === nothing && push!(missing_sides, "high @ $(dt_hi)")
#         error("Could not fetch files for $(join(missing_sides, ", ")); tried products $(pref), $(alt).")
#     end

#     p_lo_path, prod_lo_used = prod_lo
#     p_hi_path, prod_hi_used = prod_hi

#     if prod_lo_used != prod_hi_used
#         @info "[mix] Using mixed products: low=$(prod_lo_used), high=$(prod_hi_used)"
#     end

#     return (p_lo_path, p_hi_path, prod_lo_used, prod_hi_used)
# end

const _WRS_00Z_FIRST_TIME = Time(3, 10, 0)  # first valid file under 00Z folder is ..._031000.nc

function _get_two_files_exact(itp::WAMInterpolator, dt::DateTime)
    dt_lo, dt_hi = _surrounding_10min(dt)
    pref, alt    = _product_fallback_order(itp.product)  # ("wfs","wrs") or ("wrs","wfs")

    if itp.product == "wrs"
        aws = _aws_cfg(itp.region)

        # Try to download a specific dt_file from a forced WRS archive (cycle) hour.
        # Returns String local-path on success, or nothing.
        function _try_wrs_from_cycle(dt_file::DateTime, arch::DateTime)
            key = _construct_wrs_key_with_cycle(dt_file, arch)
            try
                return _download_to_cache(aws, itp.bucket, key; cache_dir=DEFAULT_CACHE_DIR, verbose=true)
            catch
                return nothing
            end
        end

        # For a given 10-min valid time, prefer SAME-DAY 00Z folder, else PREV-DAY 18Z folder.
        # Returns (local_path::String, "wrs") or (nothing, "wrs")
        function _resolve_wrs_stamp(dt_file::DateTime)
            arch_00 = DateTime(Date(dt_file), Time(0))               # same day, 00Z
            arch_18 = DateTime(Date(dt_file) - Day(1), Time(18))     # previous day, 18Z

            # 1) Prefer 00Z (some days start around 03:10; if absent, this is nothing)
            if (p00 = _try_wrs_from_cycle(dt_file, arch_00)) !== nothing
                return (p00, "wrs")
            end

            # 2) Fall back to previous day's 18Z
            if (p18 = _try_wrs_from_cycle(dt_file, arch_18)) !== nothing
                return (p18, "wrs")
            end

            return (nothing, "wrs")
        end

        # Resolve both bracketing stamps independently with the 00Z→18Z preference
        p_lo_path, prod_lo_used = _resolve_wrs_stamp(dt_lo)
        p_hi_path, prod_hi_used = _resolve_wrs_stamp(dt_hi)

        # Hard error if either side is missing, with clear guidance
       if p_lo_path === nothing || p_hi_path === nothing
            missing = String[]
            if p_lo_path === nothing
                push!(missing, "low @ $(dt_lo) (tried wrs.$(Dates.format(Date(dt_lo), dateformat"yyyymmdd"))/00 then previous day /18)")
            end
            if p_hi_path === nothing
                push!(missing, "high @ $(dt_hi) (tried wrs.$(Dates.format(Date(dt_hi), dateformat"yyyymmdd"))/00 then previous day /18)")
            end
            error("Could not fetch WRS files for $(join(missing, "; ")). " *
                "This usually means the earliest same-day 00Z products begin later (e.g., ~03:10) " *
                "and the previous 18Z cycle also does not contain that valid stamp.")
        end


        return (p_lo_path, p_hi_path, prod_lo_used, prod_hi_used)
    end

    p_lo = _try_download(itp, dt_lo, pref)
    prod_lo = p_lo === nothing ? begin
        p = _try_download(itp, dt_lo, alt)
        p === nothing ? nothing : (p, alt)
    end : (p_lo, pref)

    p_hi = _try_download(itp, dt_hi, pref)
    prod_hi = p_hi === nothing ? begin
        p = _try_download(itp, dt_hi, alt)
        p === nothing ? nothing : (p, alt)
    end : (p_hi, pref)

    if prod_lo === nothing || prod_hi === nothing
        missing_sides = String[]
        prod_lo === nothing && push!(missing_sides, "low @ $(dt_lo)")
        prod_hi === nothing && push!(missing_sides, "high @ $(dt_hi)")
        error("Could not fetch files for $(join(missing_sides, ", ")); tried products $(pref), $(alt).")
    end

    p_lo_path, prod_lo_used = prod_lo
    p_hi_path, prod_hi_used = prod_hi
    return (p_lo_path, p_hi_path, prod_lo_used, prod_hi_used)
end



#  Time utilities 
function _decode_time_units(ds::NCDataset, tname::String, t::AbstractVector)
    units = get(ds[tname].attrib, "units", "")
    m = match(r"(seconds|minutes|hours|days) since (\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?", units)
    if m === nothing
        return t, nothing, nothing  # keep DateTime axis as is
    end
    scale = m.captures[1]
    epoch_date = Date(m.captures[2])
    epoch_time = m.captures[3] === nothing ? Time(0) : Time(m.captures[3])
    epoch = DateTime(epoch_date, epoch_time)

    if eltype(t) <: DateTime
        tnum = [ _encode_query_time(tt, epoch, scale) for tt in t ]
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

    # Argmin by the first field (delta), ties broken by the second (key)
    # findmin returns (min_tuple, idx)
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

    # Helper to classify a dimension by inspecting name + attributes
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


#  Public API -

"""
    get_density(itp::WAMInterpolator, dt::DateTime, lat::Real, lon::Real, alt_km::Real)

Return neutral density at (`dt`, `lat`, `lon`, `alt_km`) using WAM‑IPE outputs.
"""

function get_density(itp::WAMInterpolator, dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    mode = _validate_query_args(itp.interpolation, dt, latq, lonq, alt_km)

    # 1) Get bracketing local files by constructing exact keys (with 6-hour cycle rules)
    p_lo, p_hi, prod_lo, prod_hi = _get_two_files_exact(itp, dt)
    @info "[fetch] Using files: low=[$(prod_lo)] $(basename(p_lo)), high=[$(prod_hi)] $(basename(p_hi))"

    # 2) Open both datasets
    ds_lo = NCDataset(p_lo, "r")
    ds_hi = NCDataset(p_hi, "r")

    
    # 3) Parse each file’s valid time from its filename (YYYYMMDD_HHMMSS)
    t_lo = _parse_valid_time_from_key(p_lo)
    t_hi = _parse_valid_time_from_key(p_hi)
    t_lo === nothing && (t_lo = t_hi)
    t_hi === nothing && (t_hi = t_lo)

    try
        # - low file spatial value at its own valid time -
        lat, lon, z, t, V, (latname, lonname, zname, tname) =
            _load_grids(ds_lo, itp.varname; file_time=t_lo)
        tdts, epoch, scale = _decode_time_units(ds_lo, tname, t)
        tq_lo = (epoch === nothing) ? t_lo : _encode_query_time(t_lo, epoch, scale)
        zq_lo = _maybe_convert_alt(z, alt_km, ds_lo, zname)
        v_lo  = _interp4(lat, lon, z, tdts, V,  latq, lonq, zq_lo, tq_lo; mode=mode)

        # - high file spatial value at its own valid time -
        lat2, lon2, z2, t2, V2, (latname2, lonname2, zname2, tname2) =
            _load_grids(ds_hi, itp.varname; file_time=t_hi)
        tdts2, epoch2, scale2 = _decode_time_units(ds_hi, tname2, t2)
        tq_hi = (epoch2 === nothing) ? t_hi : _encode_query_time(t_hi, epoch2, scale2)
        zq_hi = _maybe_convert_alt(z2, alt_km, ds_hi, zname2)
        v_hi  = _interp4(lat2, lon2, z2, tdts2, V2, latq, lonq, zq_hi, tq_hi; mode=mode)

        # 4) Temporal linear blend at query dt
        if t_lo == t_hi
            return float(v_lo)
        else
            itp_t = DataInterpolations.LinearInterpolation([float(v_lo), float(v_hi)],
                [Dates.value(t_lo), Dates.value(t_hi)])
            return itp_t(Dates.value(dt))
        end
    finally
        close(ds_lo); close(ds_hi)
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
    [get_density(itp, dts[i], lats[i], lons[i], alts_km[i]) for i in 1:n]
end

function get_density_from_key(itp::WAMInterpolator, key::AbstractString,
                              dt::DateTime, latq::Real, lonq::Real, alt_km::Real)
    mode = _normalize_interp(itp.interpolation)   # or reuse _validate_query_args if you want full checks
    aws = _aws_cfg(itp.region)
    ds, tmp = _open_nc_from_s3(aws, itp.bucket, String(key))
    try
        t_file = _parse_valid_time_from_key(String(key))
        lat, lon, z, t, V, (latname, lonname, zname, tname) = _load_grids(ds, itp.varname; file_time=t_file)

        tdts, epoch, scale = _decode_time_units(ds, tname, t)
        tq = (epoch === nothing) ? (t_file === nothing ? dt : t_file) : _encode_query_time(dt, epoch, scale)

        zq = _maybe_convert_alt(z, alt_km, ds, zname)
        return _interp4(lat, lon, z, tdts, V, latq, lonq, zq, tq; mode=mode)
    finally
        close(ds); isfile(tmp) && rm(tmp; force=true)
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

end # module
