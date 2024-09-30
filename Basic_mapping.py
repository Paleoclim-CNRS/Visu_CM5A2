
import os
import sys
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# Appending new path to system to be able to import from parent folder
sys.path.append('../Diag_CM5A2')
from Toolz import dispGridCoastline, z_masked_overlap

# PARAM
#-----------------------------------------------------------------------#
file_mask_in = '/Users/anthony/Documents/Model/Data_CM5A2/C30MaTMP_mesh_mask.nc'
file_ptrc_in = '/Users/anthony/Documents/Model/Data_CM5A2/C30MaTotV1-3X_SE_4805_4854_1M_ptrc_T.nc'

# Custom colormap
cmapColor = 'viridis'
interv_bounds = []  # default value: []
interv_step = []  # default value: []

# Zoom
x_min = None
x_max = None  # default value: None
y_min = None
y_max = None  # default value: None
#-----------------------------------------------------------------------#

# PLOT
#-----------------------------------------------------------------------#
# Standard/Custom levels
if not interv_bounds:
    cmap = mpl.cm.get_cmap(cmapColor)
    bounds = np.linspace(np.nanmin(O2_bottom), np.nanmax(O2_bottom), 50)
else:
    assert len(interv_bounds) == len(interv_step) + 1, ("Please check interv_bounds and interv_step variables."
                                                        " If len(interv_bounds) = n, len(interv_step) should be n-1.")
    lvl = []
    for i in np.arange(0, len(interv_bounds)-1):
        lvl = np.append(lvl, np.arange(
            interv_bounds[i], interv_bounds[i+1], interv_step[i]))
    bounds = np.round(lvl, 3)
    cmap = plt.get_cmap(cmapColor, len(bounds)-1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Plot projected
projDataIn = ccrs.PlateCarree()  # Projection data are originally in
projDataOut = ccrs.Robinson(central_longitude=0)  # Final projection in plots
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={
                       'projection': projDataOut})
ax.gridlines(draw_labels=True)  # transform = ccrs.Geodetic(),
X, Y, maskedZ = z_masked_overlap(
    ax, lon, lat, O2_bottom, source_projection=ccrs.Geodetic())
if x_min is None and y_min is None:
    map1 = ax.pcolormesh(
        lon,
        lat,
        maskedZ,
        transform=projDataIn,
        cmap=cmap,
        norm=norm,
        shading='nearest'
    )
    dispGridCoastline(lon,
                      lat,
                      projDataIn,
                      land_mask,
                      1.25)
else:
    map1 = ax.pcolormesh(
        lon[y_min:y_max, x_min:x_max],
        lat[y_min:y_max, x_min:x_max],
        maskedZ[y_min:y_max, x_min:x_max],
        transform=projDataIn,
        cmap=cmap,
        norm=norm,
        shading='nearest'
    )
    dispGridCoastline(lon[y_min:y_max, x_min:x_max],
                      lat[y_min:y_max, x_min:x_max],
                      projDataIn,
                      land_mask[y_min:y_max, x_min:x_max],
                      1.25)
if interv_bounds == []:
    # control nbb of ticks for colorbar
    ticks = np.linspace(norm.vmin, norm.vmax, 7, endpoint=True)
    cbar = plt.colorbar(map1, ticks=ticks,
                        orientation='horizontal', extend='both')
else:
    cbar = plt.colorbar(map1, orientation='horizontal', extend='both')
cbar.ax.set_title('O2 at bottom (mmol/m3)', size=18)
cbar.ax.tick_params(labelsize=18)
plt.show()
#-----------------------------------------------------------------------#
