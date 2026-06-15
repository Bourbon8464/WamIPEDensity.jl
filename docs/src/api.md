# API Reference

## Interpolators

```@docs
WAMInterpolator
GEOSFPInterpolator
NRLMSISEInterpolator
HybridDensityInterpolator
```

## Density Queries

```@docs
get_density
get_density_batch
get_density_at_point
get_density_trajectory
get_density_trajectory_optimised
```

## Cache and Diagnostics

```@docs
prewarm_cache!
set_max_open_datasets!
print_cache_stats
clear_grid_cache!
inspect_geos_file
inspect_geos_remote_file
```

## Profiles

```@docs
mean_density_profile
plot_global_mean_profile
plot_global_mean_profile_plots
```
