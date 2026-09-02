# profile.jl - global mean density profile and plotting helpers.

# --------------------------------------------------------------------------
# Mean lon/lat profile
# --------------------------------------------------------------------------

function _mean_lonlat_over_z(V3::AbstractArray{<:Real,3})
    nz = size(V3, 3)
    out = Vector{Float64}(undef, nz)
    @inbounds for k in 1:nz
        acc = 0.0; cnt = 0
        @views for val in V3[:, :, k]
            if isfinite(val); acc += val; cnt += 1; end
        end
        out[k] = cnt == 0 ? NaN : acc / cnt
    end
    return out
end

"""
    mean_density_profile(itp::WAMInterpolator, dt::DateTime)
        -> (alt_km::Vector{Float64}, dens_mean::Vector{Float64})

Global-mean neutral density profile at `dt`, averaged over all longitudes and
latitudes at each altitude level, with linear time interpolation between the
two bracketing files.
"""
function mean_density_profile(itp::WAMInterpolator, dt::DateTime)
    p_lo, p_hi, _, _ = _get_two_files_exact(itp, dt)
    ds_lo = _open_nc(p_lo); ds_hi = _open_nc(p_hi)

    t_lo = _parse_valid_time_from_key(p_lo)
    t_hi = _parse_valid_time_from_key(p_hi)
    t_lo === nothing && (t_lo = t_hi)
    t_hi === nothing && (t_hi = t_lo)

    # _load_grid_metadata returns GridMetadata; mean needs raw grids
    # Use the internal WAM grid loader path
    lat_lo, lon_lo, z_lo, t_lo_arr, VL, names_lo =
        _wam_load_grids(ds_lo, itp.varname; file_time=t_lo)
    lat_hi, lon_hi, z_hi, t_hi_arr, VH, names_hi =
        _wam_load_grids(ds_hi, itp.varname; file_time=t_hi)

    prof_lo = _mean_lonlat_over_z(@view VL[:, :, :, 1])
    prof_hi = _mean_lonlat_over_z(@view VH[:, :, :, 1])

    if t_lo == t_hi
        return z_lo, prof_lo
    else
        t0 = Float64(Dates.value(t_lo))
        t1 = Float64(Dates.value(t_hi))
        tq = Float64(Dates.value(dt))
        θ  = clamp((tq - t0) / (t1 - t0), 0.0, 1.0)
        return z_lo, @. (1-θ)*prof_lo + θ*prof_hi
    end
end

# Internal WAM grid loader (mirrors what WAM backend uses internally).
function _wam_load_grids(ds::NCDataset, varname::String;
                         file_time::Union{DateTime,Nothing}=nothing)
    v = ds[varname]
    dnames = String.(NCDatasets.dimnames(v))
    function classify_dim(dn)
        l = lowercase(dn); attrs = try Dict(ds[dn].attrib) catch; Dict{String,Any}() end
        s = lowercase(string(get(attrs, "standard_name", ""))); ax = uppercase(string(get(attrs, "axis", "")))
        u = lowercase(string(get(attrs, "units", "")))
        if occursin("time",l)||ax=="T"||s=="time"; return :time; end
        if occursin("lat",l)||s=="latitude"||ax=="Y"||occursin("degrees_north",u); return :lat; end
        if occursin("lon",l)||s=="longitude"||ax=="X"||occursin("degrees_east",u); return :lon; end
        if occursin("lev",l)||occursin("height",l)||occursin("alt",l)||l=="z"||ax=="Z"; return :z; end
        if l in ("x","grid_xt","i","nx"); return :lon; end
        if l in ("y","grid_yt","j","ny"); return :lat; end
        return :unknown
    end
    roles = map(classify_dim, dnames)
    nd = ndims(v)

    @inline _coord_floats(ds, dname) = haskey(ds, dname) ? begin
        raw = collect(ds[dname][:])
        if eltype(raw) <: DateTime
            Float64.(Dates.value.(raw))
        else
            Float64.(raw)
        end
    end : collect(Float64, 1.0:1.0:size(v, 0))

    if nd == 4
        idx_lon = findfirst(==(:lon), roles)
        idx_lat = findfirst(==(:lat), roles)
        idx_z   = findfirst(==(:z),   roles)
        idx_tim = findfirst(==(:time), roles)
        latname = dnames[idx_lat]; lonname = dnames[idx_lon]
        zname   = dnames[idx_z];   tname   = dnames[idx_tim]
        lat = haskey(ds, latname) ? _coord_floats(ds, latname) :
                                        collect(Float64, 1.0:1.0:float(size(v, idx_lat)))
        lon = haskey(ds, lonname) ? _coord_floats(ds, lonname) :
                                        collect(Float64, 1.0:1.0:float(size(v, idx_lon)))
        z   = haskey(ds, zname)   ? _coord_floats(ds, zname)   :
                                        collect(Float64, 1.0:1.0:float(size(v, idx_z)))
        t   = haskey(ds, tname)   ? _coord_floats(ds, tname)   :
                                        [file_time === nothing ? 0.0 : Float64(Dates.value(file_time))]
        perm = (idx_lon, idx_lat, idx_z, idx_tim)
        Vraw = perm == (1,2,3,4) ? Array(v) : Array(PermutedDimsArray(Array(v), perm))
        V = _cf_decode!(Vraw, v)
        return lat, lon, z, t, V, (latname, lonname, zname, tname)
    else
        idx_lon = findfirst(==(:lon), roles)
        idx_lat = findfirst(==(:lat), roles)
        idx_z   = findfirst(==(:z),   roles)
        latname = dnames[idx_lat]; lonname = dnames[idx_lon]; zname = dnames[idx_z]
        lat = haskey(ds, latname) ? _coord_floats(ds, latname) :
                                        collect(Float64, 1.0:1.0:float(size(v, idx_lat)))
        lon = haskey(ds, lonname) ? _coord_floats(ds, lonname) :
                                        collect(Float64, 1.0:1.0:float(size(v, idx_lon)))
        z   = haskey(ds, zname)   ? _coord_floats(ds, zname)   :
                                        collect(Float64, 1.0:1.0:float(size(v, idx_z)))
        t   = [file_time === nothing ? 0.0 : Float64(Dates.value(file_time))]
        perm = (idx_lon, idx_lat, idx_z)
        Vraw = perm == (1,2,3) ? Array(v) : Array(PermutedDimsArray(Array(v), perm))
        V    = reshape(Vraw, size(Vraw,1), size(Vraw,2), size(Vraw,3), 1)
        V = _cf_decode!(Vraw, v)
        return lat, lon, z, t, V, (latname, lonname, zname, "time")
    end
end

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

function _extend_profile_to_zero(alt_km, dens)
    if any(abs.(alt_km) .<= 1e-8)
        return collect(alt_km), collect(dens)
    end
    mask = .!(isnan.(dens) .| isinf.(dens) .| (dens .<= 0))
    a = collect(alt_km[mask]); d = collect(dens[mask])
    length(a) < 2 && return vcat(0.0, collect(alt_km)), vcat(first(dens), collect(dens))
    p = sortperm(a)
    a1, a2 = a[p[1]], a[p[2]]; d1, d2 = d[p[1]], d[p[2]]
    m = (log(d2) - log(d1)) / (a2 - a1)
    b = log(d1) - m * a1
    d0 = exp(b)
    return vcat(0.0, collect(alt_km)), vcat(isfinite(d0) && d0 > 0 ? d0 : d1, collect(dens))
end

"""
    plot_global_mean_profile(itp, dt; alt_max_km=500, savepath=nothing) -> Plots.Plot

Plot the global-mean density profile (log-scale x-axis).
"""
function plot_global_mean_profile(itp::WAMInterpolator, dt::DateTime;
                                alt_max_km::Real=500,
                                savepath::Union{Nothing,String}=nothing)
    alt_km, dens = mean_density_profile(itp, dt)
    mask = .!(isnan.(dens) .| isinf.(dens))
    altp = alt_km[mask]; denp = dens[mask]
    p = Plots.plot(denp, altp; xscale=:log10, xlabel="Density (kg/m^3)",
                   ylabel="Altitude (km)", legend=false, framestyle=:box, grid=true,
                   title="Global Mean Density - " * Dates.format(dt, dateformat"yyyy-mm-dd HH:MM 'UTC'"))
    Plots.ylims!(p, (0, min(alt_max_km, maximum(altp))))
    savepath !== nothing && Plots.savefig(p, String(savepath))
    return p
end

"""
    plot_global_mean_profile_plots(itp, dt; ...) -> (Plot, png_path, csv_path)

Create a plot of the global-mean density profile, saving to
`plots/<product>/<stamp>/global_mean_profile.png`.
"""
function plot_global_mean_profile_plots(itp::WAMInterpolator, dt::DateTime;
    alt_max_km::Union{Nothing,Real}=nothing,
    extend_to0::Bool=false,
    savepath::Union{Nothing,String}=nothing,
    export_csv::Bool=false,
    base_dir::AbstractString="plots")
    alt_km, dens = mean_density_profile(itp, dt)
    mask = .!(isnan.(dens) .| isinf.(dens) .| (dens .<= 0))
    altp = alt_km[mask]; denp = dens[mask]
    if extend_to0; altp, denp = _extend_profile_to_zero(altp, denp); end

    stamp  = Dates.format(dt, dateformat"yyyymmddTHHMMSS")
    outdir = mkpath(joinpath(base_dir, itp.product, stamp))
    png_path = savepath === nothing ? joinpath(outdir, "global_mean_profile.png") : String(savepath)
    csv_path = export_csv ? joinpath(outdir, "global_mean_profile.csv") : nothing

    p = Plots.plot(denp, altp; xscale=:log10, xlabel="Density (kg*m⁻^3)",
                   ylabel="Altitude (km)", legend=false, framestyle=:box, grid=true,
                   title="Global Mean Density - " * Dates.format(dt, dateformat"yyyy-mm-dd HH:MM 'UTC'"),
                   linewidth=2, size=(800, 600), dpi=150)
    alt_max_km !== nothing && Plots.ylims!(p, (0, float(alt_max_km)))
    Plots.savefig(p, png_path)
    if export_csv
        open(csv_path, "w") do io
            write(io, "altitude_km,density_kg_m3\n")
            for i in eachindex(altp)
                write(io, "$(altp[i]),$(denp[i])\n")
            end
        end
    end
    return p, png_path, csv_path
end
