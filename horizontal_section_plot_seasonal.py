"""
Plot 4 horizontal sections (lon/lat plot) of specified variable from specified file
The first 3 plots are averages: annual, winter(jan-mar), summer (jul-sep)
The 4th plot is difference: winter-summer.

Fill variables in "PARAM" section:
    - file_msk_in   : [string format], /path/to/file.nc
    - file_in       : [string format], /path/to/file.nc

    - variable      : [string format], variable to visualize

    - depth_ind     : [list format], depth indice at which variable will be plotted
                        Example:
                            z = [0] plot variable at surface
                            z = [5, 13] plot variable integrated from depth indice 5 to 12
                            (Remember in python when slicing, indice defining the end is excluded)
                            z = [] plot variable integrated over all depth indices

    - cmap_color     : [string format], colormap color
    - cmap_range     : [list format], defines bounds and steps of the different 
                      intervals in the custom colormap
                      [[val_min, val_1, step_1], [...], [val_n, val_max, step_n]]
                        Example:
                            cmap_range = [[0, 10, 1],[10,31,5]]
                            cmap_range = [[0, 10, 1]]
    - contour_lines : [list format], defines contour lines to display for the plot
                      [contour_min, contour_max, step]

    - projection    : [string format],projection to use of the plot
    - lon_lim       : [float format], longitude limits for plot
    - lat_lim       : [float format], latitude limits for plot
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tools.process_grid import disp_grid_coastline
from tools.process_oce import (get_coords_h,
                               get_unit,
                               get_proj,
                               get_var_seasonal,
                               compute_land_mask,
                               param4plot,
                               plot_var_h,
                               subplot_cbar,
                               subplot_cbar_legend,
                               subplot_title)

# PARAM
#-----------------------------------------------------------------------#
# Files to load
file_msk_in = ''
file_in     = ''

# Variable name
varname = ''

# time and depth selection
depth_ind = [0] # ex: [0] => ind 0 / [6,20] => ave ind 6 to 19 / [] => ave over all depth indices

# Color colormap
cmap_color      = 'Spectral_r' # default: 'viridis', 'Spectral_r'
cmap_color_diff = 'BrBG_r' # default: 'RdBu_r', 'BrBG_r'

# Custom colormap and contour
cmap_range_a   = [] # default: [] - format: [[v_min, v_1, stp_1], [...], [v_n, v_max, stp_n]]
cmap_range_w   = [] # default: [] - format: [[v_min, v_1, stp_1], [...], [v_n, v_max, stp_n]]
cmap_range_s   = [] # default: [] - format: [[v_min, v_1, stp_1], [...], [v_n, v_max, stp_n]]
cmap_range_w_s = [] # default: [] - format: [[v_min, v_1, stp_1], [...], [v_n, v_max, stp_n]]
contour_lines  = [] # default: [] - format: [contour_min, contour_max, step]

# Type of projection
projection = 'robinson' # default: 'robinson', 'polar_N', 'polar_S'

# Zoom
lon_lim = [-180,180] # default: [-180,180]
lat_lim = [-90,90]  # default: [-90,90]
#-----------------------------------------------------------------------#

# Get indices for seasons
#-----------------------------------------------------------------------#
time_ind = {
    "winter": [0,3],
    "summer": [6,9],
    "all": []
}
#-----------------------------------------------------------------------#

# Compute variables
#-----------------------------------------------------------------------#
coords    = get_coords_h(file_in)
variables = get_var_seasonal(file_in,
                             file_msk_in,
                             varname,
                             time_ind,
                             depth_ind)

var_winter = variables[0]
var_summer = variables[1]
var_annual = variables[2]

coords_msk, land_mask = compute_land_mask(file_msk_in)
unit = get_unit(file_in, varname)
#-----------------------------------------------------------------------#

# PLOT PARAMS
#-----------------------------------------------------------------------#
projDataOut = get_proj(projection)
cmap_param_w = param4plot(var_winter, cmap_color, cmap_range_w)
cmap_param_s = param4plot(var_summer, cmap_color, cmap_range_s)
cmap_param_a = param4plot(var_annual, cmap_color, cmap_range_a)
cmap_param_w_s = param4plot(var_winter-var_summer, cmap_color_diff, cmap_range_w_s)
projDataIn = ccrs.PlateCarree()
plot_limits = [lon_lim, lat_lim]
#-----------------------------------------------------------------------#

# PLOT
#-----------------------------------------------------------------------#
fig, axs = plt.subplots(figsize=(10, 20), nrows=4, ncols=1, subplot_kw={'projection': projDataOut})

# Main plot
mapa = plot_var_h(axs[0], projDataIn, projDataOut, plot_limits, coords, var_annual, contour_lines, cmap_param_a)
cbara = subplot_cbar(axs[0], mapa, cmap_param_a)
disp_grid_coastline(axs[0], coords_msk[0], coords_msk[1], projDataIn, land_mask[0, ::], 1.25)

mapw = plot_var_h(axs[1], projDataIn, projDataOut, plot_limits, coords, var_winter, contour_lines, cmap_param_w)
cbarw = subplot_cbar(axs[1], mapw, cmap_param_w)
disp_grid_coastline(axs[1], coords_msk[0], coords_msk[1], projDataIn, land_mask[0, ::], 1.25)

maps = plot_var_h(axs[2], projDataIn, projDataOut, plot_limits, coords, var_summer, contour_lines, cmap_param_s)
cbars = subplot_cbar(axs[2], maps, cmap_param_s)
disp_grid_coastline(axs[2], coords_msk[0], coords_msk[1], projDataIn, land_mask[0, ::], 1.25)

mapdiff = plot_var_h(axs[3], projDataIn, projDataOut, plot_limits, coords, var_winter - var_summer, contour_lines, cmap_param_w_s)
cbardiff = subplot_cbar(axs[3], mapdiff, cmap_param_w_s)
disp_grid_coastline(axs[3], coords_msk[0], coords_msk[1], projDataIn, land_mask[0, ::], 1.25)

# Legend
subplot_cbar_legend(cbara, 'annual')
subplot_cbar_legend(cbarw, 'winter')
subplot_cbar_legend(cbars, 'summer')
subplot_cbar_legend(cbardiff, '(winter-summer)')

subplot_title(axs, varname, unit, depth_ind)

#-----------------------------------------------------------------------#
