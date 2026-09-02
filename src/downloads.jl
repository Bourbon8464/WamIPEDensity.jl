# downloads.jl - AWS configuration, bounded-concurrency download helpers.
# Provides AWS.AWSConfig, multi-key batched download with serial fallback.

# --------------------------------------------------------------------------
# AWS configuration
# --------------------------------------------------------------------------

"""
    _aws_cfg(region) -> AWS.AWSConfig

Build an unsigned AWS config for public-bucket reads against `region`.
"""
_aws_cfg(region::String) = AWS.AWSConfig(; region=region, creds=nothing)

# --------------------------------------------------------------------------
# Concurrency control
# --------------------------------------------------------------------------

"""
    set_download_concurrency!(n::Integer) -> Int

Set the maximum number of in-flight S3 downloads used by the bounded
batch downloader. Bounded between 1 and `_MAX_DOWNLOAD_CONCURRENCY`.
"""
function set_download_concurrency!(n::Integer)
    v = max(1, min(Int(n), _MAX_DOWNLOAD_CONCURRENCY))
    _DOWNLOAD_CONCURRENCY[] = v
    return v
end

_current_download_concurrency() = _DOWNLOAD_CONCURRENCY[]

# --------------------------------------------------------------------------
# Bounded-concurrency batch download
# --------------------------------------------------------------------------

"""
    _batched_download!(paths, bucket, aws; verbose=false) -> Vector{String}

Download each `(bucket, key)` to its local cache path under a Semaphore-bounded
fan-out. Falls back to serial download if threaded dispatch is not available
(single-threaded Julia). Each individual download uses `_cache_get_file!` so
the standard single-flight guarantee still applies.
"""
function _batched_download!(paths::Vector{String}, bucket::String, aws; verbose::Bool=false)
    n = length(paths)
    concurrency = min(_current_download_concurrency(), n)
    if concurrency <= 1 || n <= 1 || Threads.nthreads() <= 1
        results = Vector{String}(undef, n)
        for i in 1:n
            results[i] = _download_to_cache(aws, bucket, paths[i]; verbose=verbose)
        end
        return results
    end

    results = Vector{String}(undef, n)
    next_idx = Threads.Atomic{Int}(0)
    Threads.@threads for _ in 1:concurrency
        while true
            i = Threads.atomic_add!(next_idx, 1) + 1
            i > n && break
            results[i] = _download_to_cache(aws, bucket, paths[i]; verbose=verbose)
        end
    end
    return results
end
