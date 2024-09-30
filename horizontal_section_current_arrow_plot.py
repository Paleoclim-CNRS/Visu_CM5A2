"""
Plot horizontal section (lon/lat plot) of specified variable from specified file
and hover a quiver plot dsiplaying the ocean currents

Fill variables in "PARAM" section:
    - file_msk_in : [string format], /path/to/file.nc
    - file_in     : [string format], /path/to/file.nc
    - file_grid_U : [string format], /path/to/file.nc
    - file_grid_V : [string format], /path/to/file.nc

    Note: If you want to make a difference between 2 files, just fill in
          variables file_msk_in_2 and file_in_2, otherwise leave them
          empty strings (e.g file_msk_in_2 = '' and file_in_2 = '')

    - varname   : [string format], variable to visualize
    - varname_u : [string format], variable name of zonal component of current
    - varname_v : [string format], variable name of meridional component of current

    - month : [list format], month(s) at which variable will be plotted
                Example:
                    month = [3] plot variable at march
                    month = [7, 9] plot variable averaged from july to september
                    month = [] plot variable averaged over all time steps

    - depth_ind : [list format], depth indice(s) at which variable will be plotted
                    Example:
                        depth_ind = [0] plot variable at surface
                        depth_ind = [5, 13] plot variable integrated from depth indices 5 to 12
                        (Remember in python when slicing, indice defining the end is excluded)
                        depth_ind = [] plot variable integrated over all depth indices

    - cmap_color    : [string format], colormap color
    - arrow_color   : [string format], arrow colors
    - cmap_range    : [list format], defines bounds and steps of the different 
                      intervals in the custom colormap
                      [[val_min, val_1, step_1], [...], [val_n, val_max, step_n]]
                        Example:
                            cmap_range = [[0, 10, 1],[10,31,5]]
                            cmap_range = [[0, 10, 1]]
    - contour_lines : [list format], defines contour lines to display for the plot
                      [contour_min, contour_max, step]

    - projection : [string format],projection to use of the plot
    - lon_lim    : [float format], longitude limits for plot
    - lat_lim    : [float format], latitude limits for plot
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tools.process_grid import disp_grid_coastline
from tools.process_oce import (get_coords_h,
                               get_unit,
                               get_proj,
                               get_var_h,
                               compute_land_mask,
                               param4plot,
                               plot_var_h,
                               plot_quiver,
                               plot_cbar,
                               plot_cbar_legend_h)

# PARAM
#-----------------------------------------------------------------------#
file_msk_in  = ''
file_in      = ''
file_grid_U  = ''
file_grid_V  = ''

# Variable names
varname   = ''
varname_u = ''
varname_v = ''

# Current vector step
v_step = 3

# time and depth selection
month     = [] # ex: [1] => jan / [3,6] => ave mar to Jun / [] => ave over all time steps
depth_ind = [0] # ex: [0] => ind 0 / [6,20] => ave ind 6 to 19 / [] => ave over all depth indices

# Color colormap
cmap_color  = 'viridis' # default: 'viridis', 'Spectral_r', 'RdBu_r', 'BrBG_r'
arrow_color = 'r' # default: 'b': blue, 'g': green, 'r': red, 'c': cyan,
                  # 'm': magenta, 'y': yellow, 'k': black, 'w': white

# Custom colormap and contour
cmap_range    = [] # default: [] - format: [[v_min, v_1, stp_1], [...], [v_n, v_max, stp_n]]
contour_lines = [] # default: [] - format: [contour_min, contour_max, step]

# Type of projection
projection = 'robinson' # default: 'robinson', 'polar_N', 'polar_S'

# Zoom
lon_lim = [-180,180] # default: [-180,180]
lat_lim = [-90,90]  # default: [-90,90]
#-----------------------------------------------------------------------#

# Extract time indice(s) from month variable
#-----------------------------------------------------------------------#
# Switch to indices for time instead of month
time_ind = [i-1 for i in month]

# In python when slicing, last indices is excluded
# So we add this lines to take in account last month specified by user
if len(time_ind) == 2:
    time_ind[-1] = time_ind[-1] + 1
#-----------------------------------------------------------------------#

# Compute variables
#-----------------------------------------------------------------------#
coords = get_coords_h(file_in)

var2plot   = get_var_h(file_in,
                       file_msk_in,
                       varname,
                       time_ind,
                       depth_ind)
var2plot_u = get_var_h(file_grid_U,
                       file_msk_in,
                       varname_u,
                       time_ind,
                       depth_ind)
var2plot_v = get_var_h(file_grid_V,
                       file_msk_in,
                       varname_v,
                       time_ind,
                       depth_ind)

coords_msk, land_mask = compute_land_mask(file_msk_in)
unit                  = get_unit(file_in, varname)
#-----------------------------------------------------------------------#

# PLOT PARAMS
#-----------------------------------------------------------------------#
projDataOut = get_proj(projection)
cmap_param  = param4plot(var2plot, cmap_color, cmap_range)
projDataIn  = ccrs.PlateCarree()  
plot_limits = [lon_lim, lat_lim]
#-----------------------------------------------------------------------#

# PLOT
#-----------------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projDataOut})
mapp    = plot_var_h(ax, projDataIn, projDataOut, plot_limits, coords, var2plot, contour_lines, cmap_param)
mapp2   = plot_quiver(ax, projDataIn, v_step, coords, var2plot_u, var2plot_v, col = arrow_color)
cbar    = plot_cbar(ax, mapp, cmap_param)
plot_cbar_legend_h(varname, unit, cbar, time_ind, depth_ind)
disp_grid_coastline(ax, coords_msk[0], coords_msk[1], projDataIn, land_mask[0, ::], 1.25)
#-----------------------------------------------------------------------#
