"""
Plot vertical section required variable from required file.
If 2 files are provided, difference will be computed

Fill variables in "PARAM" section:
    - file_msk_in   : [string format], /path/to/file.nc
    - file_in       : [string format], /path/to/file.nc
    - file_in_2     : [string format], /path/to/file.nc - can be empty string
    - varname       : [string format], variable to visualize

    - subbasin_name : [string format], mask to apply to the data.
                        if subbasin_name = 'atlmsk', 'pacmsk' or 'indmsk'
                            data will be masked with the chosen subbasin
                        if subbasin_name = 'all'
                            data will be masked with all subbasin masks at same time
                            (equivalent to applying a land mask) 

    - section       : [string format], will define type of section plot
                            if section = 'lat'
                                section along depth/lat will be plotted
                            if section = 'lon'
                                section along depth/lon will be plotted

    - ind           : [list format], indice(s) at which variable will be plotted, depends on section output.
                            if section = 'lat'
                                ind will refer to lon indices
                            elif section = 'lon'
                                ind will refer to lat indices
                            Example:
                                ind = [0] plot variable at longitue or latitude at indice 0
                                ind = [5, 13] plot variable averaged along lon or lat indices 5 to 12
                                (Remember in python when slicing, indice defining the end is excluded)
                                ind = [] plot variable averaged over all lon or lat indices

    - month         : [list format], month(s) at which variable will be plotted
                         Example:
                             month = [3] plot variable at march
                             month = [7, 9] plot variable averaged from july to september
                             month = [] plot variable averaged over all time steps

    - cmap_color    : [string format], colormap color
    - cmap_range    : [list format], defines bounds and steps of the different 
                      intervals in the custom colormap
                      [[val_min, val_1, step_1], [...], [val_n, val_max, step_n]]
                        Example:
                            cmap_range = [[0, 10, 1],[10,31,5]]
                            cmap_range = [[0, 10, 1]]
    - contour_lines : [list format], defines contour lines to display for the plot
                      [contour_min, contour_max, step]

    If you want a custom domain (not mandatory):
    - lonlat_lim    : float format, longitude/latitudes limits for plot
    - depth_lim     : float format, depth limits for plot
"""

import matplotlib.pyplot as plt
from tools.process_oce import (get_unit,
                   get_coords_var_v,
                   compute_land_mask,
                   get_subbasin_masks,
                   param4plot,
                   plot_var_v,
                   plot_cbar,
                   plot_cbar_legend_v)

# PARAMS
#-----------------------------------------------------------------------#
file_msk_in      = ''
file_in          = ''
file_in_2        = ''
file_subbasin_in = ''

# Variable name
varname = ''

# Longitude/Latitude indice(s) selection
subbasin_name = 'atlmsk' # default value: 'atlmsk', 'pacmsk', 'indmsk' or 'all'
# x_ind = [75,125] # ex: [0] => ind 0 / [6,20] => ave ind 6 to 19 / [] => ave over all lon indices
# y_ind = [] # ex: [0] => ind 0 / [6,20] => ave ind 6 to 19 / [] => ave over all lat indices

section = 'lat' # default value: 'lon' or 'lat'
ind     = [] # ex: [0] => ind 0 / [6,20] => ave ind 6 to 19 / [] => average over all indices
month   = [] # ex: [1] => jan / [3,6] => ave mar to Jun / [] => ave over all time steps

# Custom colormap
cmap_color = 'viridis' # default: 'viridis', 'Spectral_r', 'RdBu_r', 'BrBG_r'

# Custom colormap and contour
cmap_range    = [] # default: [] - format: [[v_min, v_1, stp_1], [...], [v_n, v_max, stp_n]]
contour_lines = [] # default: [] - format: [contour_min, contour_max, step]

# Zoom
lonlat_lim = [-80,60] # default: [-180,180] or [-90,90]
depth_lim  = [-5000,0] # default: [-5000,0]
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
subbasin              = get_subbasin_masks(file_subbasin_in, subbasin_name, 'format_1')
coords, var2plot      = get_coords_var_v(file_in, varname, section, time_ind, ind, subbasin)
if file_in_2:
    coords2, var2plot2  = get_coords_var_v(file_in, varname, section, time_ind, ind, subbasin)
    var2plot = var2plot - var2plot2

coords_msk, land_mask = compute_land_mask(file_msk_in)
unit                  = get_unit(file_in, varname)
#-----------------------------------------------------------------------#

# PLOT PARAMS
#-----------------------------------------------------------------------#
cmap_param  = param4plot(var2plot, cmap_color, cmap_range)
plot_limits = [lonlat_lim, depth_lim]
#-----------------------------------------------------------------------#

# PLOT
#-----------------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(10, 10))
mapp    = plot_var_v(ax, plot_limits, coords, var2plot, contour_lines, cmap_param)
cbar    = plot_cbar(ax, mapp, cmap_param)
plot_cbar_legend_v(varname, unit, cbar, time_ind, section, ind)
#-----------------------------------------------------------------------#
