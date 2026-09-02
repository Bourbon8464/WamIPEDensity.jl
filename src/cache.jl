# cache.jl - on-disc LRU file cache and time-bucket pair cache.
# Thread-safe, single-flight download guard.

using Serialization

# --------------------------------------------------------------------------
# _FileCache - on-disc LRU cache with serialised metadata
# --------------------------------------------------------------------------

mutable struct _FileCache
    dir::String
    max_bytes::Int64
    map::Dict{String,String}
    sizes::Dict{String,Int64}
    order::Vector{String}
    access_time::Dict{String,Int}
    access_counter::Int
    bytes::Int64
    downloading::Set{String}
    conds::Dict{String,Condition}
    lock::ReentrantLock
end

function _load_cache(dir::AbstractString, max_bytes::Int64)
    mkpath(dir)
    meta_path = joinpath(dir, _CACHE_META_FILE)
    if isfile(meta_path)
        try
            obj = open(meta_path, "r") do io; deserialize(io) end
            if obj isa _FileCache
                obj.bytes = sum(values(obj.sizes))
                obj.order = [k for k in obj.order if haskey(obj.map, k)]
                obj.lock  = ReentrantLock()
                empty!(obj.downloading)
                empty!(obj.conds)
                return obj
            end
        catch err
            @warn "Cache metadata unreadable - starting fresh" path=meta_path exception=err
        end
    end
    return _FileCache(
        String(dir), Int64(max_bytes),
        Dict{String,String}(), Dict{String,Int64}(),
        String[], Dict{String,Int}(), 0, Int64(0),
        Set{String}(), Dict{String,Condition}(), ReentrantLock()
    )
end

function _save_cache(cache::_FileCache)
    mkpath(cache.dir)
    open(joinpath(cache.dir, _CACHE_META_FILE), "w") do io
        serialize(io, cache)
    end
    return nothing
end

@inline function _lru_touch!(cache::_FileCache, key::String)
    cache.access_counter += 1
    cache.access_time[key] = cache.access_counter
end

function _evict_until_under_budget!(cache::_FileCache)
    while cache.bytes > cache.max_bytes
        isempty(cache.map) && break
        victim = argmin(k -> get(cache.access_time, k, 0), keys(cache.map))
        sz = get(cache.sizes, victim, Int64(0))
        local_path = cache.map[victim]
        try; isfile(local_path) && rm(local_path; force=true); catch; end
        delete!(cache.map, victim)
        delete!(cache.sizes, victim)
        delete!(cache.access_time, victim)
        cache.bytes = max(Int64(0), cache.bytes - sz)
    end
end

# --------------------------------------------------------------------------
# Primary cache interface
# --------------------------------------------------------------------------

function _get_cache(cache_dir::AbstractString, max_bytes::Int64)
    key = (String(cache_dir), Int64(max_bytes))
    if haskey(_CACHES, key)
        return _CACHES[key]
    end
    cache = _load_cache(cache_dir, max_bytes)
    return (_CACHES[key] = cache)
end

# --------------------------------------------------------------------------
# Time-bucket pair cache - single source of truth
# --------------------------------------------------------------------------

"""
    _get_cached_file_pair(dt) -> Union{Tuple{String,String}, Nothing}
"""
function _get_cached_file_pair(dt::DateTime)
    lock(_TIME_BUCKET_LOCK) do
        return get(_TIME_BUCKET_CACHE, _datetime_floor_10min(dt), nothing)
    end
end

"""
    _cache_file_pair(dt, pair)
"""
function _cache_file_pair(dt::DateTime, pair::Tuple{String,String})
    lock(_TIME_BUCKET_LOCK) do
        _TIME_BUCKET_CACHE[_datetime_floor_10min(dt)] = pair
    end
end

# --------------------------------------------------------------------------
# Core download routine - atomic, single-flight, serialised to disc
# --------------------------------------------------------------------------

function _cache_get_file!(cache::_FileCache, aws, bucket::String, key::String;
                          verbose::Bool=true)
    local_path = normpath(joinpath(cache.dir, key))

    # --- cache hit ---
    hit = lock(cache.lock) do
        if haskey(cache.map, key) && isfile(cache.map[key])
            _lru_touch!(cache, key)
            @debug "Cache hit" path=cache.map[key]
            return cache.map[key]
        end
        return nothing
    end
    hit !== nothing && return hit

    # --- atomic single-flight: claim or wait ---
    i_am_downloader, cond = lock(cache.lock) do
        if key in cache.downloading
            return false, get!(cache.conds, key, Condition())
        else
            push!(cache.downloading, key)
            return true, get!(cache.conds, key, Condition())
        end
    end

    if !i_am_downloader
        @debug "Waiting for concurrent download" key=key
        wait(cond)
        result = lock(cache.lock) do
            haskey(cache.map, key) && isfile(cache.map[key]) ? cache.map[key] : nothing
        end
        result !== nothing && return result
        error("Concurrent download failed for $key")
    end

    tmp_path = local_path * ".part"
    mkpath(dirname(local_path))
    verbose && @info "Downloading" bucket=bucket key=key

    ok = false
    bytes_written::Int64 = 0

    try
        io = AWSS3.s3_get(aws, bucket, key; return_stream=true)
        open(tmp_path, "w") do f
            while !eof(io)
                chunk = read(io, 1_048_576)
                write(f, chunk)
                bytes_written += sizeof(chunk)
            end
        end
        ok = true
    catch err
        @debug "S3 download failed; trying HTTP" key=key exception=(err, catch_backtrace())
        try
            url = "https://$(bucket).s3.amazonaws.com/$(key)"
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
        catch err2
            @debug "HTTP fallback failed" key=key exception=(err2, catch_backtrace())
        end
    end

    ok && isfile(tmp_path) && mv(tmp_path, local_path; force=true)
    isfile(tmp_path) && rm(tmp_path; force=true)

    lock(cache.lock) do
        if haskey(cache.conds, key)
            notify(cache.conds[key]; all=true)
            delete!(cache.conds, key)
        end
        delete!(cache.downloading, key)

        ok && isfile(local_path) || error("Failed to download s3://$(bucket)/$(key)")

        sz = try filesize(local_path) catch; max(bytes_written, 0) end
        cache.map[key]   = local_path
        cache.sizes[key] = sz
        cache.bytes    += sz
        _lru_touch!(cache, key)
        _evict_until_under_budget!(cache)
        _save_cache(cache)

        return local_path
    end
end

# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------

_download_to_cache(aws, bucket, key; cache_dir=DEFAULT_CACHE_DIR,
                    cache_max_bytes=2_000_000_000, verbose=true) =
    _cache_get_file!(_get_cache(cache_dir, cache_max_bytes), aws, bucket, key; verbose=verbose)

_have_in_cache(key::AbstractString; cache_dir::AbstractString=DEFAULT_CACHE_DIR) =
    isfile(normpath(joinpath(cache_dir, key)))

# --------------------------------------------------------------------------
# clean_cache!
# --------------------------------------------------------------------------

"""
    clean_cache!(; cache_dir, max_age=Day(30), cache_max_bytes=2e9)

Delete cached WAM-IPE / GEOS-FP files older than `max_age` and synchronise
the in-memory `_FileCache` bookkeeping with what is actually on disc. Pass the
same `cache_max_bytes` you use elsewhere for the same `cache_dir` so the
bookkeeping sync finds the right shared `_FileCache` instance.
"""
function clean_cache!(; cache_dir::AbstractString=DEFAULT_CACHE_DIR,
                        max_age::Period=Day(30),
                        cache_max_bytes::Int=2_000_000_000)
    cutoff = now() - max_age
    count = 0
    isdir(cache_dir) || return 0
    cache = _get_cache(cache_dir, Int64(cache_max_bytes))

    lock(cache.lock) do
        for (root, _, files) in walkdir(cache_dir)
            for file in files
                path = joinpath(root, file)
                endswith(path, ".bin") && continue
                if isfile(path) && Dates.unix2datetime(mtime(path)) < cutoff
                    try; rm(path; force=true); count += 1; @debug "Deleted" path=path; catch; end
                    for (k, mapped) in collect(cache.map)
                        if mapped == path
                            sz = get(cache.sizes, k, Int64(0))
                            delete!(cache.map, k)
                            delete!(cache.sizes, k)
                            delete!(cache.access_time, k)
                            cache.bytes = max(Int64(0), cache.bytes - sz)
                        end
                    end
                end
            end
        end
        _save_cache(cache)
    end

    @info "Cache cleanup complete" deleted=count dir=cache_dir
    return count
end

# --------------------------------------------------------------------------
# print_cache_stats
# --------------------------------------------------------------------------

"""
    print_cache_stats(; cache_dir, cache_max_bytes=2e9)

Print a human-readable summary of the on-disc cache: directory path, capacity,
used space, number of files, and the most-recently-used keys. No return value.
"""
function print_cache_stats(; cache_dir::AbstractString=DEFAULT_CACHE_DIR,
                            cache_max_bytes::Int=2_000_000_000)
    cache = _get_cache(cache_dir, cache_max_bytes)
    lock(cache.lock) do
        println("Cache dir  : ", cache.dir)
        println("Capacity   : ", round(cache.max_bytes/1e9, digits=2), " GB")
        println("Used       : ", round(cache.bytes/1e9, digits=3), " GB  (", length(cache.map), " files)")
        if !isempty(cache.order)
            println("LRU head   : ", first(cache.order))
            println("MRU tail   : ", last(cache.order))
        end
    end
end
