# types.jl - struct definitions and shared state.
# British English throughout. No behaviour - only data.

# --------------------------------------------------------------------------
# WAM-IPE configuration
# --------------------------------------------------------------------------

"""
    WAMInterpolator(; bucket, root_prefix, product, varname, region, interpolation)

Configuration object for accessing and interpolating WAM-IPE data on S3.
"""
Base.@kwdef struct WAMInterpolator
    bucket::String       = "noaa-nws-wam-ipe-pds"
    root_prefix::String  = "v1.2"
    product::String      = "wfs"
    varname::String      = "den"
    region::String       = "us-east-1"
    interpolation::Symbol = :sciml
end

# --------------------------------------------------------------------------
# GEOS-FP configuration
# --------------------------------------------------------------------------

"""
    GEOSFPInterpolator(; ...)

Configuration for GEOS-FP density reconstruction. GEOS-FP is used for the
lower atmosphere and can be combined with WAM-IPE and NRLMSISE through
[`HybridDensityInterpolator`](@ref).
"""
Base.@kwdef struct GEOSFPInterpolator
    root_url::String        = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
    collection::String      = "inst3_3d_asm_Np"
    interpolation::Symbol   = :sciml
    cache_dir::String       = DEFAULT_GEOS_CACHE_DIR
    qv_varname::String      = "QV"
    t_varname::String       = "T"
    z_varname::String       = "H"
    lev_varname::String     = "lev"
    density_varname::String = ""
    min_alt_km::Float64     = 0.0
    max_alt_km::Float64     = 70.0
end

# --------------------------------------------------------------------------
# NRLMSISE-00 configuration
# --------------------------------------------------------------------------

"""
    NRLMSISEInterpolator(; interpolation, min_alt_km, max_alt_km)

Configuration for the NRLMSISE empirical atmosphere backend (Naval Research
Laboratory Mass Spectrometer and Incoherent Scatter Empirical model).
"""
Base.@kwdef struct NRLMSISEInterpolator
    interpolation::Symbol = :nearest
    min_alt_km::Float64   = 0.0
    max_alt_km::Float64   = 100.0
end

# --------------------------------------------------------------------------
# Hybrid configuration
# --------------------------------------------------------------------------

"""
    HybridDensityInterpolator(; geos, msis, wam, msis_max_alt_km)

Altitude-aware density interpolator that routes queries across GEOS-FP,
NRLMSISE, and WAM-IPE backends.
"""
Base.@kwdef struct HybridDensityInterpolator
    geos::GEOSFPInterpolator         = GEOSFPInterpolator()
    msis::NRLMSISEInterpolator       = NRLMSISEInterpolator()
    wam::WAMInterpolator             = WAMInterpolator()
    msis_max_alt_km::Float64         = 100.0
    geos_bounds_cache::Dict{DateTime, Tuple{Float64,Float64}} = Dict{DateTime, Tuple{Float64,Float64}}()
end

# --------------------------------------------------------------------------
# Internal state: constants and global state
# --------------------------------------------------------------------------

const _CACHE_META_FILE     = "metadata.bin"
const DEFAULT_CACHE_DIR    = normpath("./cache")
const DEFAULT_GEOS_CACHE_DIR = normpath("./cache_geosfp")

# Gas constants used for density reconstruction in GEOS-FP.
const R_D_GEOS  = 287.05      # J/(kg*K), specific gas constant for dry air
const G0_GEOS   = 9.80665     # m/s^2, standard gravity

# Allowed interpolation modes (normalised; `:sciml` aliases to `:logz_quadratic`).
const _ALLOWED_INTERP_NORM = Set([:nearest, :linear, :logz_linear, :logz_quadratic])
const _VALID_TIME_REGEX    = r"(\d{8})_(\d{6})\.nc$"

# Version windows for WAM-IPE data.
# NOTE: boundary mechanically patched to remove a 10-minute coverage gap
# (2023-06-30 21:00:00-21:10:00). Verify against NOAA cutover before relying
# on data near that timestamp.
const _VERSION_WINDOWS = (
    ("v1.1", DateTime(2023, 3, 20, 21, 10, 0), DateTime(2023, 6, 30, 21, 9, 59)),
    ("v1.2", DateTime(2023, 6, 30, 21, 10, 0), nothing),
)

# Default backend used by the top-level `density(...)` convenience function.
const _DEFAULT_ITP = Ref{Union{Nothing, HybridDensityInterpolator}}(nothing)

# Default AWS region for unsigned public-bucket reads.
const _AWS_REGION_US_EAST_1 = "us-east-1"

# Maximum number of parallel S3 downloads.
const _DEFAULT_DOWNLOAD_CONCURRENCY = 6
const _MAX_DOWNLOAD_CONCURRENCY     = 32

# --------------------------------------------------------------------------
# Internal types
# --------------------------------------------------------------------------

"""
    GridMetadata

Lightweight, decoded grid description for one opened NetCDF variable.
Loaded once per file, cached, and reused across many queries. Avoids
keeping the full 3D/4D variable in memory.
"""
struct GridMetadata
    lon::Vector{Float64}
    lat::Vector{Float64}
    z::Vector{Float64}
    ds::NCDataset
    varname::String
    scale_factor::Float64
    add_offset::Float64
    fill_values::Set{Float64}
    dim_map::Dict{Symbol,Int}
    ndims::Int
    lon_is_360::Bool
end

# --------------------------------------------------------------------------
# Synchronisation primitives
# --------------------------------------------------------------------------

# Cache lock for the on-disc LRU file cache (used across WAM and GEOS).
const _CACHE_LOCK     = ReentrantLock()
const _CACHES         = Dict{Tuple{String,Int64}, Any}()   # populated in cache.jl
const _GRID_CACHE     = Dict{String, GridMetadata}()
const _GRID_CACHE_LOCK = ReentrantLock()
const _MAX_GRID_CACHE_SIZE = 100

# GEOS-FP single-flight download lock.
const _GEOS_DOWNLOAD_LOCK   = ReentrantLock()
const _GEOS_DOWNLOADING     = Set{String}()
const _GEOS_DOWNLOAD_CONDS  = Dict{String, Condition}()

# GEOS grid cache (per file_path).
const _GEOS_GRID_CACHE       = Dict{String, Any}()
const _GEOS_GRID_CACHE_LOCK  = ReentrantLock()
const _MAX_GEOS_GRID_CACHE   = 4

# Time-bucket file-pair cache (single source of truth).
const _TIME_BUCKET_CACHE = Dict{DateTime, Tuple{String,String}}()
const _TIME_BUCKET_LOCK  = ReentrantLock()

# Per-3-hour GEOS-FP altitude-bound cache lock.
const _GEOS_BOUNDS_LOCK  = ReentrantLock()

# Per-thread NCDataset handle pool.
# Populated lazily on first use; never invalidated mid-process.
const _THREAD_NC_POOL    = Dict{Tuple{Int,String}, NCDataset}()
const _THREAD_NC_LOCK    = ReentrantLock()

# NRLMSISE-00 initialisation latch.
const _MSIS_INIT_LOCK     = ReentrantLock()
const _MSIS_INITIALIZED   = Ref(false)

# Download concurrency settings.
const _DOWNLOAD_CONCURRENCY = Ref(_DEFAULT_DOWNLOAD_CONCURRENCY)
