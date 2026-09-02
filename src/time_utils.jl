# time_utils.jl - DateTime helpers, version mapping, S3 key construction.
# Pure functions, no I/O.

# --------------------------------------------------------------------------
# DateTime floor / bracket helpers
# --------------------------------------------------------------------------

@inline function _datetime_floor_10min(dt::DateTime)
    m  = minute(dt)
    mm = m - (m % 10)
    return DateTime(Date(dt), Time(hour(dt), mm))
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

# --------------------------------------------------------------------------
# Cycle hour routing for WAM-IPE products
# --------------------------------------------------------------------------

# WRS (real-time nowcast) cycle-hour preference list.
# Each cycle covers a 6 h window starting at HH+3h10m.
# WRS is the primary product (real-time nowcast); WFS is the forecast fallback.
#
# Coverage windows on S3 (all v1.2/wrs.YYYYMMDD/HH/):
#   18z prev-day  ->  21:10 to next 02:50  (t18z files on prev YYYYMMDD)
#   00z today     ->  03:10 to 08:50        (t00z files on YYYYMMDD)
#   06z today     ->  09:10 to 14:50        (t06z files on YYYYMMDD)
#   12z today     ->  15:10 to 20:50        (t12z files on YYYYMMDD)
#   18z today     ->  21:10 to next 02:50  (t18z files on YYYYMMDD)
#
# Prefer the cycle closest to the query time; fall back to earlier cycles.

"""
    _wrs_cycles(dt::DateTime) -> Vector{Int}

Return WRS cycle hours to try in preference order. Each integer is the
folder-hour (00, 06, 12, 18). The first entry is the best match; later
entries are fallbacks.
"""
function _wrs_cycles(dt::DateTime)::Vector{Int}
    h = hour(dt)
    if h < 3
        return [18, 12, 6, 0]   # 18z from prev day (covers up to 02:50), then 12z, 06z, 00z
    elseif h < 9
        return [0, 18, 12, 6]    # 00z today (covers 03:10+), then prev 18z, then 12z, 06z
    elseif h < 15
        return [6, 0, 18, 12]    # 06z today (covers 09:10+), then 00z, then prev 18z, 12z
    elseif h < 21
        return [12, 6, 0, 18]    # 12z today (covers 15:10+), then 06z, 00z, then prev 18z
    else
        return [18, 12, 6, 0]   # 18z today (covers 21:10+), then 12z, 06z, 00z
    end
end

"""
    _wrs_archive(dt::DateTime) -> DateTime

Return the preferred WRS cycle DateTime for a query at `dt`.
Kept for backward compatibility; prefer `_wrs_cycles` for multi-fallback routing.
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

# WFS (forecast) cycle hour: 6 h cycle, floor to latest cycle at or before query time.
# Each cycle folder contains files for ALL subsequent valid times until the next cycle starts.
# Verified against S3: folder 00Z has files valid at 00:00 through >=14:50; folder 06Z
# starts at 06:00; folder 12Z at 12:00; folder 18Z at 18:00.
function _wfs_archive(dt::DateTime)::DateTime
    h = hour(dt)
    if h < 6
        return DateTime(Date(dt), Time(0))
    elseif h < 12
        return DateTime(Date(dt), Time(6))
    elseif h < 18
        return DateTime(Date(dt), Time(12))
    else
        return DateTime(Date(dt), Time(18))
    end
end

# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------

_parse_valid_time_from_key(key::AbstractString) = let m = match(_VALID_TIME_REGEX, key)
    m === nothing && return nothing
    ymd, hms = m.captures
    DateTime(parse(Int, ymd[1:4]), parse(Int, ymd[5:6]), parse(Int, ymd[7:8]),
             parse(Int, hms[1:2]), parse(Int, hms[3:4]), parse(Int, hms[5:6]))
end

# --------------------------------------------------------------------------
# NetCDF time axis decoding
# --------------------------------------------------------------------------

function _decode_time_units(ds::NCDataset, tname::String, t::AbstractVector)
    units = get(ds[tname].attrib, "units", "")
    cal   = lowercase(string(get(ds[tname].attrib, "calendar", "gregorian")))
    m = match(r"(seconds|minutes|hours|days)\s+since\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?", units)
    if m === nothing
        return t, nothing, nothing
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

function _encode_query_time(dtq::DateTime,
                            epoch::Union{DateTime,Nothing},
                            scale::Union{AbstractString,Nothing})
    epoch === nothing && return float(dtq.value)

    delta_ms = Dates.value(dtq - epoch)

    s = scale === nothing ? "days" : lowercase(String(scale))

    if startswith(s, "sec")
        return delta_ms / 1_000
    elseif startswith(s, "min")
        return delta_ms / 60_000
    elseif startswith(s, "hour")
        return delta_ms / 3_600_000
    else
        return delta_ms / 86_400_000
    end
end

# --------------------------------------------------------------------------
# Version and model mapping
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# S3 key construction
# --------------------------------------------------------------------------

"""
    _construct_s3_key(dt, product, arch) -> String

Build the S3 object key for a WAM-IPE file.

- `dt` - query DateTime (determines the valid-time date/HMS in the filename)
- `product` - "wrs" or "wfs"
- `arch` - cycle DateTime (determines the folder path and tHH label)

For WRS, `arch` should be the cycle start (e.g. `DateTime(2024,5,14,18)` for the
18z cycle). The folder path uses `Date(arch)`; the filename uses `Date(dt)` for the
valid-time date. This correctly handles the 18z case where the folder is prev-day
but the valid times in the filenames span from that prev day through the next morning.

Two-argument form uses the default cycle routing (`_wrs_archive`/`_wfs_archive`).
"""
function _construct_s3_key(dt::DateTime, product::String, arch::DateTime)::String
    v     = _version_for(dt)
    model = _model_for_version(v)
    ymd_dir = Dates.format(Date(arch), dateformat"yyyymmdd")  # folder date
    HH_dir  = @sprintf("%02d", hour(arch))                  # folder hour
    ymd     = Dates.format(Date(dt),  dateformat"yyyymmdd")  # filename date (valid time)
    HMS     = Dates.format(Time(dt),  dateformat"HHMMSS")      # filename HMS (valid time)
    HHfile  = @sprintf("%02d", hour(arch))                  # tHH label
    return @sprintf("%s/%s.%s/%s/wam_fixed_height.%s.t%sz.%s.%s_%s.nc",
                    v, product, ymd_dir, HH_dir, product, HHfile, model, ymd, HMS)
end

_construct_s3_key(dt::DateTime, product::String)::String =
    _construct_s3_key(dt, product,
        product == "wrs" ? _wrs_archive(dt) :
        product == "wfs" ? _wfs_archive(dt) :
        error("Unknown product $product"))

_product_fallback_order(product::String) = product == "wfs" ? ("wfs","wrs") : ("wrs","wfs")

function _construct_wrs_key_with_cycle(dt::DateTime, arch::DateTime)::String
    v     = _version_for(dt)
    model = _model_for_version(v)
    ymd_dir = Dates.format(Date(arch), dateformat"yyyymmdd")
    HH_dir  = @sprintf("%02d", hour(arch))
    ymd     = Dates.format(Date(dt), dateformat"yyyymmdd")
    HMS     = Dates.format(Time(dt), dateformat"HHMMSS")
    HHfile  = @sprintf("%02d", hour(arch))
    return @sprintf("%s/%s.%s/%s/wam_fixed_height.%s.t%sz.%s.%s_%s.nc",
                    v, "wrs", ymd_dir, HH_dir, "wrs", HHfile, model, ymd, HMS)
end
