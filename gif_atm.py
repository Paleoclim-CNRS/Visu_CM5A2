"""
Generate animated gif files for wind 850hPa, Precip-Evap and T2M 

filename_atm : atmosphere histmth file
filename_T   : ocean grid_T file
filename_msk : mesh_mask file

out_path : output path where gif files will be saved

manual_lim_[] : use custom contour and ranges for the plot [True/False]
contour_[]    : custom contours (to set only if manual_lim_[] = True)
                must be in list format: [val_1, val_2, ..., val_n]
range_[]      : custom ranges must be
                    - In tuple format (don't forget the comma at the end):
                        ([val_min, val_1, step_1], [...], [val_n, val_max, step_n],)
                    - Or in list format
                        [[val_min, val_1, step_1], [...], [val_n, val_max, step_n]]
                Example:
                    contour_z850        = [1300]
                    range_wind850       = ([0, 10, 1],[10,31,5],)
                    contour_precip_evap = [-8, 0, 13]
                    range_precip_evap   = ([-14, 14, 1],)
cmap_color_[] : color of the cmap for each plot in str format
                you can find all le colormap available here 
                https://matplotlib.org/stable/tutorials/colors/colormaps.html

nbContourLines : number of contour lines plotted (for plots where manual_lim_[] = False)

projDataIn  : input projection (It is unlikely this parameter will need to be modified)
projDataOut : output projection for the plotted data in gif

time_interv : time interval in ms between each frame of gif files 
"""

import cartopy.crs as ccrs
import warnings
from tools.process_atm import Atm

# ignore warnings
warnings.filterwarnings('ignore')

# PARAMS
#---------------------------------------------------------------------#
filename_atm = ''
filename_T   = ''
filename_msk = ''

out_path = ''

manual_lim_wind     = True # True/False
contour_z850        = [1250, 1300] # (m)
range_wind850       = ([0, 10, 1],[10,31,5],) # (m.s^-1)
cmap_color_wind     = 'viridis'

manual_lim_p_e      = True
contour_precip_evap = [-8, 0, 13]  # (mm.d-1)
range_precip_evap   = ([-14, 14, 1],) # (mm.d-1)
cmap_color_precip_evap = 'BrBG' # 'BrBG_r'

manual_lim_t2m      = True
contour_t2m         = [-20, 0, 10, 35] # (°C)
range_t2m           = ([-20,-10,5],[-10,10,1],[10,20,5],[20,35,2],) # (°C)
cmap_color_t2m      = 'viridis'

nbContourLines = 6  # For plots where manual_lim_[] = False
projDataIn = ccrs.PlateCarree()  # Projection data are originally in
projDataOut = ccrs.Robinson(central_longitude=0)  # Final projection in plots

time_interv = 250
#---------------------------------------------------------------------#

# Set up plot params
#--------------------------------------#
ranges = {}

if manual_lim_wind:
    ranges['c_z850'] = contour_z850
    ranges['r_w850'] = range_wind850
if manual_lim_p_e:
    ranges['c_p_e'] = contour_precip_evap
    ranges['r_p_e'] = range_precip_evap
if manual_lim_t2m:
    ranges['c_t2m'] = contour_t2m
    ranges['r_t2m'] = range_t2m

palettes = [cmap_color_wind, cmap_color_precip_evap, cmap_color_t2m]
#--------------------------------------#

# EXECTUTION
#---------------------------------------------------------------------#
atm = Atm(out_path, filename_atm, filename_T, filename_msk, ranges, palettes)

atm.gen_plot_param(nbContourLines)

atm.plot_wind(projDataIn, projDataOut)
atm.make_gif(time_interv)

atm.plot_precip_evap(projDataIn, projDataOut)
atm.make_gif(time_interv)

atm.plot_t2m(projDataIn, projDataOut)
atm.make_gif(time_interv)
#---------------------------------------------------------------------#
