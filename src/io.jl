# io.jl - NetCDF helpers, CF decode, per-thread dataset pool.
# British English throughout.

# --------------------------------------------------------------------------
# Per-thread NCDataset pool
# --------------------------------------------------------------------------

"""
    _open_nc(path) -> NCDataset

Open a NetCDF file read-only on the current thread.
Per-thread pool avoids lock contention on the hot path.
Files are opened once per thread and reused; closing is not managed here
(leave handles open for the lifetime of the process).
"""
function _open_nc(path::String)::NCDataset
    tid = Threads.threadid()
    key = (tid, path)
    lock(_THREAD_NC_LOCK) do
        if haskey(_THREAD_NC_POOL, key)
            return _THREAD_NC_POOL[key]
        end
        ds = NCDataset(path, "r")
        _THREAD_NC_POOL[key] = ds
        return ds
    end
end

"""
    clear_nc_pool!()

Close and remove all per-thread NetCDF handles.
Useful in tests to force a clean state.
"""
function clear_nc_pool!()
    lock(_THREAD_NC_LOCK) do
        for ds in values(_THREAD_NC_POOL)
            try close(ds) catch end
        end
        empty!(_THREAD_NC_POOL)
    end
    return nothing
end

"""
    set_max_open_datasets!(n::Integer)

Stub for API compatibility. The per-thread pool has no fixed size limit;
Julia's GC handles reclaim. Kept so existing calling code does not break.
"""
set_max_open_datasets!(n::Integer) = n

# --------------------------------------------------------------------------
# CF attribute decoding
# --------------------------------------------------------------------------

"""
    _cf_decode!(A, var) -> Matrix{Float64}

Apply CF convention decoding to a raw NetCDF array:
scale_factor, add_offset -> missing -> NaN.
Modifies a copy; the input array is unaffected.
"""
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

    B = map(x -> ismissing(x) ? NaN : Float64(x), A)

    if !isempty(fillvals)
        @inbounds for i in eachindex(B)
            if B[i] in fillvals
                B[i] = NaN
            end
        end
    end

    if sf != 1.0 || ao != 0.0
        @inbounds @. B = B * sf + ao
    end

    return B
end

"""
    _classify_vertical_units(units_raw) -> Symbol

Classify a vertical coordinate unit string into `:km`, `:m`, `:pressure`,
`:index`, `:missing`, or `:unknown`.
"""
function _classify_vertical_units(units_raw::AbstractString)
    s = lowercase(strip(String(units_raw)))
    isempty(s) && return :missing

    if occursin(r"\bkm\b", s) || occursin("kilometre", s) || occursin("kilometer", s)
        return :km
    end
    if (occursin(r"\bm\b", s) || occursin("metre", s) || occursin("meter", s)) &&
       !occursin(r"\bmm\b", s) && !occursin(r"\bcm\b", s) && !occursin("km", s)
        return :m
    end
    if occursin(r"\bpa\b", s) || occursin(r"\bhpa\b", s) || occursin(r"\bmb\b", s) ||
       occursin("pascal", s) || occursin("pressure", s)
        return :pressure
    end
    if occursin("level", s) || occursin("index", s) || occursin("layer", s)
        return :index
    end
    return :unknown
end

# --------------------------------------------------------------------------
# Grid metadata loading
# --------------------------------------------------------------------------

function _load_grid_metadata(ds::NCDataset, varname::String)
    haskey(ds, varname) || error("Variable '$varname' not found in dataset.")
    v = ds[varname]
    dnames = String.(NCDatasets.dimnames(v))

    function classify_dim(dname::String)
        lname  = lowercase(dname)
        var    = haskey(ds, dname) ? ds[dname] : nothing
        attrs  = var === nothing ? Dict{String,Any}() : Dict(var.attrib)
        stdname = lowercase(string(get(attrs, "standard_name", "")))
        axis    = uppercase(string(get(attrs, "axis", "")))
        units   = lowercase(string(get(attrs, "units", "")))

        if occursin("time", lname) || axis == "T" || stdname == "time"; return :time; end
        if occursin("lat", lname)  || stdname == "latitude"  || axis == "Y" || occursin("degrees_north", units); return :lat; end
        if occursin("lon", lname)  || stdname == "longitude" || axis == "X" || occursin("degrees_east",  units); return :lon; end
        if occursin("lev", lname)  || occursin("height", lname) || occursin("alt", lname) || lname == "z" || axis == "Z"; return :z; end
        if lname in ("x","grid_xt","i","nx"); return :lon; end
        if lname in ("y","grid_yt","j","ny"); return :lat; end
        return :unknown
    end

    roles = map(classify_dim, dnames)
    nd = ndims(v)

    idx_lon  = findfirst(==(:lon),  roles)
    idx_lat  = findfirst(==(:lat),  roles)
    idx_z    = findfirst(==(:z),    roles)
    idx_time = findfirst(==(:time), roles)

    idx_lon === nothing && error("Could not find longitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
    idx_lat === nothing && error("Could not find latitude dimension for '$varname'. dims=$(dnames) roles=$(roles)")
    idx_z   === nothing && error("Could not find vertical dimension for '$varname'. dims=$(dnames) roles=$(roles)")

    get_coord(i) = haskey(ds, dnames[i]) ? Float64.(collect(ds[dnames[i]][:])) :
                                            collect(Float64, 1.0:1.0:float(size(v, i)))

    lon = get_coord(idx_lon)
    lat = get_coord(idx_lat)
    z_raw = get_coord(idx_z)
    zname = dnames[idx_z]

    z_units = haskey(ds, zname) ? get(ds[zname].attrib, "units", "km") : "km"
    z_kind  = _classify_vertical_units(String(z_units))

    z = if z_kind === :km
        Float64.(z_raw)
    elseif z_kind === :m
        Float64.(z_raw) ./ 1000.0
    elseif z_kind === :pressure
        error("Vertical axis '$zname' uses pressure units ('$z_units'); cannot convert to altitude in km.")
    elseif z_kind === :index || z_kind === :unknown || z_kind === :missing
        error("Unsupported or missing vertical units '$z_units' on '$zname'.")
    else
        error("Unsupported vertical units '$z_units' on '$zname'.")
    end

    attrs_any = try Dict(v.attrib) catch; Dict(CommonDataModel.attributes(v)) end
    sf = haskey(attrs_any, "scale_factor") ? float(attrs_any["scale_factor"]) : 1.0
    ao = haskey(attrs_any, "add_offset")   ? float(attrs_any["add_offset"])   : 0.0

    fillvals = Set{Float64}()
    for k in ("_FillValue", "missing_value")
        if haskey(attrs_any, k)
            val = attrs_any[k]
            if val isa AbstractArray
                for x in val; !ismissing(x) && push!(fillvals, float(x)); end
            else
                !ismissing(val) && push!(fillvals, float(val))
            end
        end
    end

    dim_map = Dict{Symbol,Int}()
    if idx_lon  !== nothing; dim_map[:lon]  = idx_lon;  end
    if idx_lat  !== nothing; dim_map[:lat]  = idx_lat;  end
    if idx_z    !== nothing; dim_map[:z]    = idx_z;    end
    if idx_time !== nothing; dim_map[:time] = idx_time; end

    return GridMetadata(lon, lat, z, ds, varname, sf, ao, fillvals, dim_map, nd, maximum(lon) > 180.0)
end

# --------------------------------------------------------------------------
# Grid cache
# --------------------------------------------------------------------------

"""
    _get_cached_metadata(path, ds, varname) -> GridMetadata

Return cached grid metadata for `path`, loading and caching on demand.
"""
function _get_cached_metadata(path::String, ds::NCDataset, varname::String)
    lock(_GRID_CACHE_LOCK) do
        if haskey(_GRID_CACHE, path)
            meta = _GRID_CACHE[path]
            isopen(meta.ds) && return meta
        end
        meta = _load_grid_metadata(ds, varname)
        _GRID_CACHE[path] = meta
        if length(_GRID_CACHE) > _MAX_GRID_CACHE_SIZE
            delete!(_GRID_CACHE, first(keys(_GRID_CACHE)))
        end
        return meta
    end
end

"""
    clear_grid_cache!()

Empty the in-memory grid metadata cache.
"""
function clear_grid_cache!()
    lock(_GRID_CACHE_LOCK) do
        empty!(_GRID_CACHE)
    end
end
