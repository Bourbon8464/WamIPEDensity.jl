# diagnostics.jl - GEOS-FP file inspection helpers.

"""
    inspect_geos_file(path)

Print variable names, dimension names, sizes, and selected attributes
for a local GEOS file.
"""
function inspect_geos_file(path::AbstractString)
    ds = NCDataset(String(path), "r")
    try
        println("FILE: ", path)
        println("VARIABLES:")
        for k in keys(ds)
            print("  ", k)
            try; print(" => size=", size(ds[k])); catch; end
            println()
            try
                attrs = Dict(ds[k].attrib)
                for aname in ("units", "long_name", "standard_name", "axis")
                    if haskey(attrs, aname)
                        println("      ", aname, " = ", attrs[aname])
                    end
                end
            catch; end
        end
    finally
        try; close(ds); catch; end
    end
    return nothing
end

"""
    inspect_geos_remote_file(itp, dt) -> String

Download/cache one GEOS file for `dt` and inspect it.
"""
function inspect_geos_remote_file(itp::GEOSFPInterpolator, dt::DateTime)
    path = _geos_download_to_cache(itp, dt; verbose=true)
    inspect_geos_file(path)
    return path
end
