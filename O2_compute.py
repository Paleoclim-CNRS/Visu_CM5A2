#!/usr/bin/env python
"""
Compute saturated OZ

Fill variables in "PARAM" section:
    - file_msk_in   : [string format], /path/to/mesh_mask_file.nc
    - file_ptrc_T   : [string format], /path/to/ptrc_T_file.nc
    - file_grid_T   : [string format], /path/to/grid_T_file.nc
    - file_o2_out   : [string format], /path/to/save/computed/O2_file.nc
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# PARAMS
#-----------------------------------------------------------------------#
file_msk=''
file_ptrc_T=''
file_grid_T=''

file_o2_out = ''
#-----------------------------------------------------------------------#


# LOAD DATASETS AND VARIABLES
#-----------------------------------------------------------------------#
ds_msk = xr.open_dataset(file_msk)
ds_ptrc_T = xr.open_dataset(file_ptrc_T)
ds_grid_T = xr.open_dataset(file_grid_T)

thetao = np.mean(ds_grid_T.thetao.values, axis=0)
so = np.mean(ds_grid_T.so.values, axis=0)
#-----------------------------------------------------------------------#


# Compute variables 
#-----------------------------------------------------------------------#
oxygen_4 = np.mean(ds_ptrc_T.O2.values, axis=0)

ox0 = -58.3877
ox1 = 85.8079
ox2 = 23.8439
ox3 = -0.034892
ox4 = 0.015568
ox5 = -0.0019387
oxyco = 1/22.4144
atcox = 0.20946

T_kel4=thetao+273.15
T_tt_4=T_kel4*0.01
T_tt2_4=T_tt_4*T_tt_4
sal=so
T_log4=np.log(T_tt_4)
oxy4 = ox0+ox1/T_tt_4+ox2*T_log4+sal*(ox3+ox4*T_tt_4+ox5*T_tt2_4)
O2_SAT4 = np.exp(oxy4)*oxyco*atcox*1e6
AOU4=O2_SAT4 - oxygen_4
#-----------------------------------------------------------------------#


# Save new dataset in nc file
#-----------------------------------------------------------------------#
# Create a new dataset from ptrc_T file keeping only coordinates
ds_new = ds_ptrc_T.drop(list(ds_ptrc_T.keys()))

# Creating new variables in dataset newly generated
ds_new = ds_new.assign({'oxygen_4': (['olevel', 'y','x'], oxygen_4)})
ds_new = ds_new.assign( O2_SAT4 = (['olevel', 'y','x'], O2_SAT4))
ds_new = ds_new.assign( AOU4 = (['olevel', 'y','x'], AOU4))

ds_new.oxygen_4.attrs['units'] = '$mmol/m^3$'
ds_new.O2_SAT4.attrs['units'] = '$mmol/m^3$'
ds_new.AOU4.attrs['units'] = '$mmol/m^3$'


# Saving to netcdf
ds_new.to_netcdf(file_o2_out)
#-----------------------------------------------------------------------#


ind = 20

ds_grid_T.deptht.values[ind]


plt.figure()
plt.pcolormesh(oxygen_4[ind,::])
plt.title('oxygen_4')
plt.colorbar()

# plt.figure()
# plt.pcolormesh(O2_SAT4[ind,::])
# plt.title('O2_SAT4')
# plt.colorbar()

# plt.figure()
# plt.pcolormesh(AOU4[ind,::])
# plt.title('AOU4')
# plt.colorbar()


# var_mean = np.nanmean(AOU4[15:20,::],axis=0)


# cmap_range    = ([-300,10,10],)
# cmap_param  = param4plot(var_mean, 'BrBG', cmap_range)



# bounds = np.round(np.append([], np.arange(cmap_range[0], cmap_range[1], cmap_range[2])),3)
# cmap   = plt.get_cmap('BrBG', len(bounds)-1)
# norm   = mpl.colors.BoundaryNorm(bounds, cmap.N)


# var_mean_plot = np.where(var_mean<0, np.nan, var_mean)

# plt.figure()
# plt.pcolormesh(-1*var_mean_plot, cmap=cmap_param["cmap"], norm=cmap_param["norm"], vmin=-300, vmax=0)
# plt.title('AOU4')
# plt.colorbar()
