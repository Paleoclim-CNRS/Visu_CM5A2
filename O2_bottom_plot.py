"""
Gather O2 variable from ptrc_T file at deepest cell over the domain and plot it.

Fill variables in "PARAM" section:
    - file_msk_in   : [string format], /path/to/file.nc
    - file_in       : [string format], /path/to/file.nc
    - cmap_color    : [string format], colormap color
    - cmap_range    : [list format], defines bounds and steps of the different 
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
                               get_bottom_ind,
                               get_var_bottom,
                               compute_land_mask,
                               param4plot,
                               plot_var_h,
                               plot_cbar,
                               plot_cbar_legend_h)

# PARAM
#-----------------------------------------------------------------------#
# Files to load
file_msk_in  = ''
file_ptrc_in = ''

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

# Compute name of O2 bottom file
#-----------------------------------------------------------------------#
# file_suffix = file_ptrc_in.split('/')[-1].replace('ptrc_T.nc','')
# file_O2_bottom = os.path.join('DATA', file_suffix + 'O2_bottom.nc')
#-----------------------------------------------------------------------#

# If the script is launched for the first time with data in file_ptrc_in:
#   - Compute O2_bottom and save it in a new nc file so next time script
#     is launched, load this file to avoid to compute O2_bottom again
#-----------------------------------------------------------------------#
# if not os.path.isfile(file_O2_bottom):
#     compute_o2_bottom(file_ptrc_in, file_O2_bottom)

coords                = get_coords_h(file_ptrc_in)
# O2_bottom             = get_o2_bottom(file_O2_bottom)
bottom_ind            = get_bottom_ind(file_ptrc_in)
O2_bottom             = get_var_bottom(file_ptrc_in, 'O2', bottom_ind)
coords_msk, land_mask = compute_land_mask(file_msk_in)
unit                  = get_unit(file_ptrc_in, "O2")

#-----------------------------------------------------------------------#

# PLOT PARAMS
#-----------------------------------------------------------------------#
# Standard/Custom levels
projDataOut = get_proj(projection)
cmap_param  = param4plot(O2_bottom, cmap_color, cmap_range)
projDataIn  = ccrs.PlateCarree() # projection data are originally in
plot_limits = [lon_lim, lat_lim]
#-----------------------------------------------------------------------#

# PLOT
#-----------------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projDataOut})
mapp    = plot_var_h(ax, projDataIn, projDataOut, plot_limits, coords, O2_bottom, contour_lines, cmap_param)
cbar    = plot_cbar(ax, mapp, cmap_param)
plot_cbar_legend_h("O2", unit, cbar, [], 'bottom')
disp_grid_coastline(ax, coords_msk[0], coords_msk[1], projDataIn, land_mask[0, ::], 1.25)
#-----------------------------------------------------------------------#
