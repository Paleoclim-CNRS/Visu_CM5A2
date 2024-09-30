"""
PLots bathymetry computed from grid_T file. 
Then plots histogram showing sum of cells surface 
ordonned by depth bins (500m reange each unless specified - 
defined by "bins" variable) for each subbasin.

Fill variables in "PARAM" section:
    - file_msk_in       : [string format], /path/to/file.nc
    - file_grid_T_in    : [string format], /path/to/grid_T_file.nc
    - file_subbasin_in  : [string format], /path/to/subbasin_file.nc
    - bins              : [list format], bins of depth used for histogram plot
    - cmap_color        : [string format], colormap color
    - cmap_range        : [list format], defines bounds and steps of the different 
                          intervals in the custom colormap
                          [[val_min, val_1, step_1], [...], [val_n, val_max, step_n]]
                            Example:
                                cmap_range = [[0, 10, 1],[10,31,5]]
                                cmap_range = [[0, 10, 1]]
    - contour_lines     : [list format], defines contour lines to display for the plot
                          [contour_min, contour_max, step]

    - projection        : [string format],projection to use of the plot
    - lon_lim           : [float format], longitude limits for plot
    - lat_lim           : [float format], latitude limits for plot
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from utils import (disp_grid_coastline,
                   get_coords_h,
                   get_unit,
                   get_proj,
                   get_bottom_ind,
                   get_var_bottom,
                   get_subbasin_masks,
                   compute_cell_surface,
                   compute_land_mask,
                   param4plot,
                   plot_var_h,
                   plot_cbar,
                   plot_cbar_legend_h,
                   hist_bathy_subbasin,
                   plot_subbasins)

# PARAM
#-----------------------------------------------------------------------#
# Files to load
file_msk_in      = '/Users/anthony/Documents/Model/Data_CM5A2/C30MaTMP_mesh_mask.nc'
file_grid_T_in   = '/Users/anthony/Documents/Model/Data_CM5A2/C30MaTotV1-3X_SE_4805_4854_1M_grid_T.nc'
file_subbasin_in = '/Users/anthony/Documents/Model/Data_CM5A2/subbasins_rupelianTot.nc'

# Bins of depth used for histogram plot
bins = list(range(0,6000,500)) # Default: list(range(0,6000,500))

# Color colormap
cmap_color = 'Spectral_r' # default: 'viridis', 'Spectral_r', 'RdBu_r', 'BrBG_r'

# Custom colormap and contour
cmap_range    = [] # default: [] - format: [[v_min, v_1, stp_1], [...], [v_n, v_max, stp_n]]
contour_lines = [] # default: [] - format: [contour_min, contour_max, step]

# Type of projection
projection = 'robinson' # default: 'robinson', 'polar_N', 'polar_S'

# Zoom
lon_lim = [-180,180] # default: [-180,180]
lat_lim = [-90,90]  # default: [-90,90]
#-----------------------------------------------------------------------#

# Compute bathymetry
#-----------------------------------------------------------------------#
coords                = get_coords_h(file_grid_T_in)
bottom_ind            = get_bottom_ind(file_grid_T_in)
bathy                 = get_var_bottom(file_grid_T_in, 'deptht', bottom_ind)
coords_msk, land_mask = compute_land_mask(file_msk_in)
unit                  = get_unit(file_grid_T_in, "deptht")
#-----------------------------------------------------------------------#

# Compute subbasins mask and cell_surface for histogram plot
#-----------------------------------------------------------------------#
subbasins      = get_subbasin_masks(file_subbasin_in, 'all', 'format_2')
subbasins_hist = get_subbasin_masks(file_subbasin_in, 'all', 'format_3')
cell_surface = compute_cell_surface(file_msk_in)
#-----------------------------------------------------------------------#

# PLOT PARAMS
#-----------------------------------------------------------------------#
# Standard/Custom levels
projDataOut = get_proj(projection)
cmap_param  = param4plot(bathy, cmap_color, cmap_range)
projDataIn  = ccrs.PlateCarree() # projection data are originally in
plot_limits = [lon_lim, lat_lim]
#-----------------------------------------------------------------------#

# BATHY HORIZONTAL PLOT
#-----------------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projDataOut})
mapp    = plot_var_h(ax, projDataIn, projDataOut, plot_limits, coords, bathy*land_mask[0, ::], contour_lines, cmap_param)
cbar    = plot_cbar(ax, mapp, cmap_param)
plot_cbar_legend_h("Bathymetry", unit, cbar, '', '')
disp_grid_coastline(ax, coords_msk[0], coords_msk[1], projDataIn, land_mask[0, ::], 1.25)
#-----------------------------------------------------------------------#

# HISTOGRAM PLOT 
#-----------------------------------------------------------------------#
hist_bathy_subbasin(bathy, subbasins_hist, bins, cell_surface)
#-----------------------------------------------------------------------#

# SUBBASINS PLOT
#-----------------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projDataOut})
mapp    = plot_subbasins(ax, projDataIn, projDataOut, plot_limits, coords, subbasins, 'Accent')
#-----------------------------------------------------------------------#