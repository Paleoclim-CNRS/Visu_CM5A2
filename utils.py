#!/usr/bin/env python
import os
import warnings
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from shapely.errors import ShapelyDeprecationWarning

def z_masked_overlap(axe, x, y, z, source_projection=None):
    """
    for data in projection axe.projection
    find and mask the overlaps (more 1/2 the axe.projection range)
​
    X, Y either the coordinates in axe.projection or longitudes latitudes
    Z the data
    operation one of 'pcorlor', 'pcolormesh', 'countour', 'countourf'
​
    if source_projection is a geodetic CRS data is in geodetic coordinates
    and should first be projected in axe.projection
​
    X, Y are 2D same dimension as Z for contour and contourf
    same dimension as Z or with an extra row and column for pcolor
    and pcolormesh
​
    return ptx, pty, Z
    """
    if not hasattr(axe, 'projection'):
        return x, y, z
    if not isinstance(axe.projection, ccrs.Projection):
        return x, y, z

    if len(x.shape) != 2 or len(y.shape) != 2:
        return x, y, z

    if (source_projection is not None and
            isinstance(source_projection, ccrs.Geodetic)):
        transformed_pts = axe.projection.transform_points(
            source_projection, x, y)
        ptx, pty = transformed_pts[..., 0], transformed_pts[..., 1]
    else:
        ptx, pty = x, y


    with np.errstate(invalid='ignore'):
        # diagonals have one less row and one less columns
        diagonal0_lengths = np.hypot(
            ptx[1:, 1:] - ptx[:-1, :-1],
            pty[1:, 1:] - pty[:-1, :-1]
        )
        diagonal1_lengths = np.hypot(
            ptx[1:, :-1] - ptx[:-1, 1:],
            pty[1:, :-1] - pty[:-1, 1:]
        )
        to_mask = (
            (diagonal0_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            np.isnan(diagonal0_lengths) |
            (diagonal1_lengths > (
                abs(axe.projection.x_limits[1]
                    - axe.projection.x_limits[0])) / 2) |
            np.isnan(diagonal1_lengths)
        )

        # Added to prevent plotting bugs happening with some
        # simulation outputs where border values (around -180°/180°)
        # are not properly masked. The masked band in to_mask var
        # is enlarged by on cell on the left and one cell on the right
        #---------------------------------------------------------#
        # to_mask_true = np.where(to_mask == True)
        # for i in np.arange(0, len(to_mask_true[0])):
        #     to_mask[to_mask_true[0][i], to_mask_true[1][i]-1] = True
        #     to_mask[to_mask_true[0][i], to_mask_true[1][i]+1] = True
        #---------------------------------------------------------#

        # TODO check if we need to do something about surrounding vertices

        # add one extra colum and row for contour and contourf
        if (to_mask.shape[0] == z.shape[0] - 1 and
                to_mask.shape[1] == z.shape[1] - 1):
            to_mask_extended = np.zeros(z.shape, dtype=bool)
            to_mask_extended[:-1, :-1] = to_mask
            to_mask_extended[-1, :] = to_mask_extended[-2, :]
            to_mask_extended[:, -1] = to_mask_extended[:, -2]
            to_mask = to_mask_extended
        if np.any(to_mask):

            Z_mask = getattr(z, 'mask', None)
            to_mask = to_mask if Z_mask is None else to_mask | Z_mask

            z = np.ma.masked_where(to_mask, z)

        return ptx, pty, z

def disp_grid_coastline(axis, lon_data, lat_data, proj_in, land_mask, linewidthdata, zorder=40, line_color='k'):
    """
    zorder defined as an optional argument following https://linux.die.net/diveintopython/html/power_of_introspection/optional_arguments.html
    lon_data      = lon coordinate
    lat_data      = lat coordinate
    proj_in       = original grid projection for lon_data and lat_data (ex: ccrs.PlateCarree())
    land_mask     = mask containing masked value over land cells and anything else (integer) on ocean cells
    linewidthdata = width on the line to be plotted
    line_color    = color of the line
    """

    # land outline... the hard part
    # for each point of the cropped area, determine if it is a coastal point and plot (or not) accordingly
    land_outline_linewidth = linewidthdata
    for ilon in np.arange(0,land_mask.shape[1]-1):
        for ilat in np.arange(0,land_mask.shape[0]-1):
            # is there an ocean to the East or to the West?
            if (land_mask.mask[ilat,ilon] != land_mask.mask[ilat,ilon+1]):
                    lat1 = lat_data[ilat,ilon+1]
                    lat2 = lat_data[ilat+1,ilon+1]
                    lon1 = lon_data[ilat,ilon+1]
                    lon2 = lon_data[ilat+1,ilon+1,]
                    latpts = [lat1, lat2]; #print latpts
                    lonpts = [lon1, lon2]; #print lonpts
                    
                    # IF condition to avoid to plot horizontal line all the way from 180° up to -180° 
                    # for pieces of land spreading before and after 180° (=>split in two pieces in the plot,
                    # 1 part at the very east and the other at the very west) 
                    # (Remove the if condition to see the difference if you don't understand)
                    if (np.abs(lon2-lon1) < 50):
                        axis.plot(lonpts,latpts,'-',linewidth=land_outline_linewidth, color=line_color,zorder=zorder,transform=proj_in)
            # is there an ocean to the North or to the South?
            if (land_mask.mask[ilat,ilon] != land_mask.mask[ilat+1,ilon]):
                    lat1 = lat_data[ilat+1,ilon]
                    lat2 = lat_data[ilat+1,ilon+1,]
                    lon1 = lon_data[ilat+1,ilon]
                    lon2 = lon_data[ilat+1,ilon+1,]
                    latpts = [lat1, lat2]; #print latpts
                    lonpts = [lon1, lon2]; #print lonpts
                    
                    # IF condition to avoid to plot horizontal line all the way from 180° up to -180° 
                    # for pieces of land spreading before and after 180° (=>split in two pieces in the plot,
                    # 1 part at the very east and the other at the very west)
                    # (Remove the if condition to see the difference if you don't understand)
                    if (np.abs(lon2-lon1) < 50):
                        axis.plot(lonpts,latpts,'-',linewidth=land_outline_linewidth, color=line_color,zorder=zorder,transform=proj_in)

def get_coords_h(file_in):
    """Get lon/lat coordinates of provided file"""
    ds_in   = xr.open_dataset(file_in)
    # load lon, lat
    if 'x' in list(ds_in.dims.keys()): # OCE file
        lon = ds_in["nav_lon"].values
        lat = ds_in["nav_lat"].values
    elif 'lon' in list(ds_in.dims.keys()): # ATM file
        lon = ds_in["lon"].values
        lat = ds_in["lat"].values
    return [lon, lat]

def get_unit(file_in, varname):
    """Extract unit variable"""
    ds_in   = xr.open_dataset(file_in)
    try:
        unit = ds_in[varname].attrs["units"]
    except KeyError:
        unit = 'N/A'
    return unit

def get_proj(projection):
    """Get final projection in plots"""
    if projection == 'robinson':
        proj_out = ccrs.Robinson(central_longitude=0)
    elif projection == 'polar_N':
        proj_out = ccrs.AzimuthalEquidistant(central_longitude=0,central_latitude=90)
    elif projection == 'polar_S':
        proj_out = ccrs.AzimuthalEquidistant(central_longitude=0,central_latitude=-90)
    return proj_out

def get_subbasin_masks(file_in, subbasin, usage):
    """Extract subbasin mask"""
    ds_sub   = xr.open_dataset(file_in)
    atlmsk = np.where(ds_sub.atlmsk.values==0, np.nan, ds_sub.atlmsk.values)
    pacmsk = np.where(ds_sub.pacmsk.values==0, np.nan, ds_sub.pacmsk.values)
    indmsk = np.where(ds_sub.indmsk.values==0, np.nan, ds_sub.indmsk.values)
    if subbasin == 'all':
        if usage == 'format_1':
            subbasin = np.nansum(np.dstack((atlmsk, pacmsk, indmsk)),2)
            subbasin = np.where(subbasin==0,np.nan,subbasin)
            return subbasin
        elif usage == 'format_2':
            pacmsk = np.where(pacmsk==1, 2, pacmsk)
            indmsk = np.where(indmsk==1, 3, indmsk)
            subbasin = np.nansum(np.dstack((atlmsk, pacmsk, indmsk)),2)
            subbasin = np.where(subbasin==0,np.nan,subbasin)
            return subbasin
        elif usage == 'format_3':
            return [atlmsk, pacmsk, indmsk]
    elif subbasin == 'atlmsk':
        return atlmsk
    elif subbasin == 'pacmsk':
        return pacmsk
    elif subbasin == 'indmsk':
        return indmsk

def get_var_h(file_in, file_msk_in, varname, time_ind, depth_ind):
    """
    Extract horizontal section of variable at specific 
    time/depth indice or averaged along several time/depth 
    indices depending on the function inputs given.
    """
    # load datasets
    ds_mask = xr.open_dataset(file_msk_in)
    ds_in   = xr.open_dataset(file_in)

    # Make the varname search in the dataset case insensitive
    lst_var = list(ds_in.variables.keys())
    lst_var_lower = [s.casefold() for s in lst_var]
    try:
        i = lst_var_lower.index(varname.casefold())
        varname = lst_var[i]
    except ValueError:
        print(f'Variable {varname} not found in file {file_in}')
    
    # Extract slices from time_ind and depth_ind, allowing to compute mean hereafter
    if not time_ind:
        t_ind = slice(0, None)
        ax1 = 0
    elif len(time_ind) == 1:
        t_ind = time_ind[0]
        ax1 = ()
    elif len(time_ind) == 2:
        t_ind = slice(time_ind[0],time_ind[1])
        ax1 = 0

    if not depth_ind:
        z_ind = slice(0, None)
        ax2 = 0
    elif len(depth_ind) == 1:
        z_ind = depth_ind[0]
        ax2 = ()
    elif len(depth_ind) == 2:
        z_ind = slice(depth_ind[0],depth_ind[1])
        ax2 = 0

    # Compute mean depending of varname dimensions
    if len(ds_in[varname].shape) == 4:
        if ds_in[varname].shape[-1] == 182: # OCE variable
            e3t = np.squeeze(ds_mask["e3t_0"].values)
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", category=RuntimeWarning)
            #     warnings.filterwarnings("ignore", category=RuntimeWarning)
            var2plot = np.nanmean(ds_in[varname].values[t_ind, ::], axis = ax1)
            # Weighted mean along water column
            var2plot = np.nansum(var2plot[z_ind, ::]*e3t[z_ind, ::], axis = ax2)/ np.sum(e3t[z_ind, ::], axis = ax2)
            # with nansum operation above, 0 replaced nans so we put them back
            var2plot = np.where(var2plot == 0, np.nan, var2plot) 
        elif ds_in[varname].shape[-1] == 96: # ATM variable
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", category=RuntimeWarning)
            #     warnings.filterwarnings("ignore", category=RuntimeWarning)
            var2plot = np.nanmean(ds_in[varname].values[t_ind, ::], axis = ax1)
            var2plot = np.nanmean(var2plot[z_ind, ::], axis = ax2)

    elif len(ds_in[varname].shape) == 3:
        var2plot = np.nanmean(ds_in[varname].values[t_ind, ::], axis=ax1)
    elif len(ds_in[varname].shape) == 2:
        var2plot = ds_in[varname].values
    return var2plot

def get_var_seasonal(file_in, file_msk_in, varname, time_ind, depth_ind):
    """
    Extract horizontal section of variable with annual, winter and summer means 
    at specific depth indice or averaged along several depth indices depending 
    on the function inputs given.
    """
    # Winter
    var_winter = get_var_h(file_in,
                         file_msk_in,
                         varname,
                         time_ind["winter"],
                         depth_ind)

    # Summer
    var_summer = get_var_h(file_in,
                         file_msk_in,
                         varname,
                         time_ind["summer"],
                         depth_ind)

    # All year
    var_annual = get_var_h(file_in,
                      file_msk_in,
                      varname,
                      time_ind["all"],
                      depth_ind)

    return [var_winter, var_summer, var_annual]

def get_coords_var_v(file_in, varname, section, time_ind, ind, subbasin):
    """
    Get coordinates of provided file and extract vertical section of 
    variable at specific longitude/latitude indice or averaged along several 
    longitude/latitude indices depending on the function inputs given.
    """
    # Load datasets
    ds_in   = xr.open_dataset(file_in)
    # Load variables
    depth    = ds_in["deptht"].values

    # if type(subbasin) is list:
    #     subbasin = np.nansum(np.dstack((subbasin[0], subbasin[1], subbasin[2])),2)
    #     subbasin = np.where(subbasin==0,np.nan,subbasin)

    # Extract slices from time_ind and depth_ind, allowing to compute mean hereafter
    if not time_ind:
        var = np.mean(ds_in[varname].values, axis=0)
    elif len(time_ind) == 1:
        var = ds_in[varname].values[time_ind[0],::]
    elif len(time_ind) == 2:
        var = np.mean(ds_in[varname].values[time_ind[0]:time_ind[1],::], axis=0)

    var = var*subbasin

    if section == 'lat':
        if len(ind) == 1:
            lat      = ds_in["nav_lat"].values[:,ind[0]]
            var2plot = np.squeeze(var[:,:,ind[0]])
        elif len(ind) == 2:
            lat      = ds_in["nav_lat"].values[:,int(np.round((ind[0]+ind[1])/2))]
            var2plot = np.nanmean(var[:, :, ind[0]:ind[1]], axis=2)
        elif ind == []:
            lat      = ds_in["nav_lat"].values[:,0]
            var2plot = np.nanmean(var, axis=2)
        lat_grid, depth_grid = np.meshgrid(lat, depth)
        return [lat_grid, depth_grid], var2plot
    elif section == 'lon':
        if len(ind) == 1:
            lon      = ds_in["nav_lon"].values[ind[0],:]
            var2plot = np.squeeze(var[:,ind[0],:])
        elif len(ind) == 2:
            lon = ds_in["nav_lon"].values[int(np.round((ind[0]+ind[1])/2)), :]
            var2plot = np.nanmean(var[:, ind[0]:ind[1], :], axis=1)
        elif ind == []:
            lon      = ds_in["nav_lon"].values[0,:]
            var2plot = np.nanmean(var, axis=1)
        lon_grid, depth_grid = np.meshgrid(lon, depth)
        return [lon_grid, depth_grid], var2plot

def get_bottom_ind(file_in):
    """
    Get z indice at bottom of the ocean.
    Can take grid_T, ptrc_T or diad_T files as input.
    """
    ds_in = xr.open_dataset(file_in)
    # Process 02 at bottom
    try:
        var = ds_in["thetao"][0,::]
    except KeyError:
        try:
            var = ds_in["O2"][0,::]
        except KeyError:
            var = ds_in["TPP"][0,::]

    bottom_ind = np.zeros((var.shape[1],var.shape[2]))

    for z in range(31):
        ind_not_nan = np.where(~np.isnan(var[z,::]))
        bottom_ind[ind_not_nan] = z

    return bottom_ind.astype(int)

def get_var_bottom(file_var_in, var, bottom_ind):
    """
    Use variable bottom_ind generated with get_bottom_ind() function to extract 
    2d (lon/lat) variable at bottom of the ocean
    """
    # Load data
    ds_in = xr.open_dataset(file_var_in)

    # Process 02 at bottom
    if len(ds_in[var].shape)==4:
        var_new = np.mean(ds_in[var].values, axis=0)
    elif len(ds_in[var].shape)==3:
        var_new = ds_in[var].values
    elif len(ds_in[var].shape)==1: # typically for computing bathy using var deptht which is 1 dim vector
        var_new = np.tile(ds_in[var].values[:, np.newaxis, np.newaxis], (1, ds_in.dims["y"], ds_in.dims["x"]))

    j_ind, i_ind = np.indices(var_new.shape[1:])

    var_bottom = var_new[bottom_ind, j_ind, i_ind]

    return var_bottom

    # o2_bottom = np.zeros((ds_ptrc["O2"].shape[2],ds_ptrc["O2"].shape[3]))

def compute_land_mask(file_msk_in):
    """Compute the land mask to display coastline for the plots"""
    ds_mask = xr.open_dataset(file_msk_in)
    # Extract lon/lat
    lon = ds_mask["nav_lon"]
    lat = ds_mask["nav_lat"]
    # Compute land mask
    tmask = np.squeeze(ds_mask["tmask"])
    # puts masked value instead of 0 in land_mask
    land_mask = np.ma.masked_where(tmask == 0, tmask)
    return [lon, lat], land_mask

def param4plot(var2plot, cmap_color, cmap_range):
    """
    Generate plot parameters
    Returns a dictionnary with 3 keys:
        - `cmap`
        - `norm`
        - `cmap_range`
    """
    
    if not cmap_range:
        bounds = np.linspace(np.nanmin(var2plot), np.nanmax(var2plot), 50)
        cmap = mpl.cm.get_cmap(cmap_color)
    else:
        bounds = []
        for i in cmap_range:
            numstr = str(i[2])
            if '.' in numstr:
                rounder = numstr[::-1].find('.')
            elif 'e-' in numstr:
                rounder = int(numstr[-1])
            else:
                rounder = 0
            bounds = np.round(np.append(bounds, np.arange(i[0], i[1], i[2])), rounder)
            cmap = plt.get_cmap(cmap_color, len(bounds)-1)
    if len(bounds)==1:
        raise ValueError(f'In cmap_range = {cmap_range}, step must be too high for v_min, v_max provided')
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    cmap_param = {
        "cmap": cmap,
        "norm": norm,
        "cmap_range": cmap_range
    }
    return cmap_param

def plot_var_h(axis, proj_in, proj_out, plot_limits, coords, var2plot, contour_lines, cmap_param):
    """Plot variable over horizontal section"""
    # define limits of plot
    axis.set_extent([plot_limits[0][0], plot_limits[0][1], plot_limits[1][0], plot_limits[1][1]], proj_in)

    # Display meridians and parallels
    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", category=ShapelyDeprecationWarning)
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", ShapelyDeprecationWarning)
        # warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        axis.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    
    # Compute var2plot to be "plotable"
    x, y, masked_z = z_masked_overlap(axis, coords[0], coords[1], var2plot, source_projection=ccrs.Geodetic())

    # Pcolor
    mapp = axis.pcolormesh(coords[0], coords[1], masked_z[0:-1,0:-1], 
                        transform=proj_in, cmap=cmap_param["cmap"], norm=cmap_param["norm"])


    # Contour lines
    if contour_lines:
        contourlines = np.arange(contour_lines[0], contour_lines[1], contour_lines[2])
        cont = axis.contour(x, y, masked_z, contourlines,
                                    transform = proj_out,
                                    colors='k', linewidths=0.7)
        # Add labels over contour lines
        axis.clabel(cont,fmt=' {:.0f} '.format,fontsize='large')
    return mapp

def plot_var_v(axis, plot_limits, coords, var2plot, contour_lines, cmap_param):
    """Plot variable over vertical section"""
    mapp = axis.pcolormesh(
        coords[0],
        -coords[1],
        var2plot,
        cmap=cmap_param["cmap"],
        norm=cmap_param["norm"],
        shading='nearest'
    )
    axis.set_xlim(plot_limits[0][0],plot_limits[0][1])
    axis.set_ylim(plot_limits[1][0],plot_limits[1][1])

    # Contour lines
    if contour_lines:
        contourlines = np.arange(contour_lines[0], contour_lines[1], contour_lines[2])
        cont = axis.contour(coords[0], -coords[1], var2plot, contourlines,                         
                                    colors='k', linewidths=0.7)
        # Add labels over contour lines
        axis.clabel(cont,fmt=' {:.0f} '.format,fontsize='large')
    return mapp

def plot_quiver(axis, proj_in, step, coords, var2plot_u, var2plot_v, col):
    """Plot quivers"""
    mapp = axis.quiver(
    coords[0][::step, ::step],
    coords[1][::step, ::step],
    var2plot_u[::step, ::step],
    var2plot_v[::step, ::step],
    color=col,
    transform=proj_in,
    scale=10,
    width=0.002,
    # width=0.004, headwidth=0.02, headlength=0.04, headaxislength=0.005
    )
    return mapp

def plot_cbar(axis, mapp, cmap_param):
    """Plot colorbar"""
    if not cmap_param["cmap_range"]:
        # control nbb of ticks for colorbar
        ticks = np.linspace(cmap_param["norm"].vmin, cmap_param["norm"].vmax, 7, endpoint=True)
        cbar = plt.colorbar(mapp, ax=axis, ticks=ticks,
                            orientation='horizontal', extend='both')
    else:
        cbar = plt.colorbar(mapp, ax=axis, orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=18)

    return cbar

def plot_cbar_legend_h(varname, unit, cbar, time_ind, depth_ind):
    """Plot colorbar legend for horizontal section plot"""
    leg_init = f'{varname} ({unit})'
    wit = False
    if time_ind == []:
        leg_t = 'all time averaged'
    elif len(time_ind) == 1:
        leg_t = f'time indice {time_ind[0]}'
    elif len(time_ind) == 2:
        leg_t = f'time averaged (ind [{time_ind[0]}-{time_ind[1]-1}])'
    elif time_ind == '':
        leg_t = ''
        wit = True # witnesses that 'and' str will not be necessary'
    if depth_ind == []:
        leg_z = 'all depth integrated'
    elif len(depth_ind) == 1:
        leg_z = f'depth indice {depth_ind[0]}'
    elif len(depth_ind) == 2:
        leg_z = f'depth integrated (ind [{depth_ind[0]}-{depth_ind[1]-1}])'
    elif depth_ind == 'bottom':
        leg_z = 'at bottom'
    elif depth_ind == '':
        leg_z = ''
        wit = True # witnesses that 'and' str will not be necessary'
    if wit:
        str_link = ''
    else:
        str_link = 'and'
    cbar.ax.set_title(f'{leg_init}\n{leg_t} {str_link} {leg_z}', size=16)

def plot_cbar_legend_v(varname, unit, cbar, time_ind, section, ind):
    """Plot colorbar legend for vertical section plot"""
    leg_init = f'{varname} ({unit})'
    wit = False
    if time_ind == []:
        leg_t = 'all time averaged'
    elif len(time_ind) == 1:
        leg_t = f'time indice {time_ind[0]}'
    elif len(time_ind) == 2:
        leg_t = f'time averaged (ind [{time_ind[0]}-{time_ind[1]-1}])'
    elif time_ind == '':
        leg_t = ''
        wit = True # witnesses that 'and' str will not be necessary'
    if ind == []:
        leg_xy = f'all {section} averaged'
    elif len(ind) == 1:
        leg_xy = f'{section} indice {ind[0]}'
    elif len(ind) == 2:
        leg_xy = f'{section} averaged (ind [{ind[0]}-{ind[1]-1}])'
    elif ind == 'bottom':
        leg_xy = 'at bottom'
    elif ind == '':
        leg_xy = ''
        wit = True # witnesses that 'and' str will not be necessary'
    if wit:
        str_link = ''
    else:
        str_link = 'and'
    cbar.ax.set_title(f'{leg_init}\n{leg_t} {str_link} {leg_xy}', size=16)

def subplot_cbar(axis, mapp, cmap_param):
    """Plot colorbars for subplots"""
    if not cmap_param["cmap_range"]:
        # control nbb of ticks for colorbar
        ticks = np.linspace(cmap_param["norm"].vmin, cmap_param["norm"].vmax, 7, endpoint=True)
        cbar = plt.colorbar(mapp, ax=axis, ticks=ticks,
                            orientation='vertical', extend='both')
    else:
        cbar = plt.colorbar(mapp, ax=axis, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=10)
    return cbar

def subplot_cbar_legend(cbar, period):
    """Plot colorbar legend for seasonal horizontal section plot"""
    cbar.ax.set_title(f'{period}', size=15)

def subplot_title(axis, varname, unit, depth_ind):
    """Display main title for subplots"""
    leg_init = f'{varname} ({unit})'
    if not depth_ind:
        leg_z = 'all depth integrated'
    elif len(depth_ind) == 1:
        leg_z = f'depth indice {depth_ind[0]}'
    elif len(depth_ind) == 2:
        leg_z = f'depth integrated (ind [{depth_ind[0]}-{depth_ind[1]-1}])'
    axis[0].set_title(f'{leg_init}\n{leg_z}', size=16)

def compute_cell_surface(file_in):
    """Compute cells surface in km^2"""
    ds_in = xr.open_dataset(file_in)
    cell_surface = np.squeeze(ds_in.e1t.values)*np.squeeze(ds_in.e2t.values)*1e-6
    return cell_surface

def hist_bathy_subbasin(bathy, subbasins, bins, weigth):
    """Plot histograms representing sum of cells surface ordonned by depth ranges surface per subbasin """

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,10), sharey=True)
    axs[0].set_title(f'Sum of cells surface ordonned by depth ranges')

    # hist weighted by cells surface
    axs[0].hist((bathy*subbasins[0]).flatten(), bins, weights=weigth.flatten(), histtype='bar', rwidth=0.8, facecolor='b', label='ATL')
    axs[1].hist((bathy*subbasins[1]).flatten(), bins, weights=weigth.flatten(), histtype='bar', rwidth=0.8, facecolor='g', label='PAC')
    axs[2].hist((bathy*subbasins[2]).flatten(), bins, weights=weigth.flatten(), histtype='bar', rwidth=0.8, facecolor='r', label='IND')

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper right")

    axs[1].set_ylabel('Surface ($km^2$)')
    axs[2].set_xlabel('depth (m)')


    fig2, axs2 = plt.subplots(nrows=3, ncols=1, figsize=(8,10), sharey=True)

    axs2[0].set_title(f'Percentage of surface')

    # hist weighted by cells surface
    axs2[0].hist((bathy*subbasins[0]).flatten(), bins, weights=weigth.flatten()/np.sum(weigth.flatten())*1e2, histtype='bar', rwidth=0.8, facecolor='b', label='ATL')
    axs2[1].hist((bathy*subbasins[1]).flatten(), bins, weights=weigth.flatten()/np.sum(weigth.flatten())*1e2, histtype='bar', rwidth=0.8, facecolor='g', label='PAC')
    axs2[2].hist((bathy*subbasins[2]).flatten(), bins, weights=weigth.flatten()/np.sum(weigth.flatten())*1e2, histtype='bar', rwidth=0.8, facecolor='r', label='IND')

    axs2[0].legend(loc="upper right")
    axs2[1].legend(loc="upper right")
    axs2[2].legend(loc="upper right")

    axs2[1].set_ylabel('Percentage (%)')
    axs2[2].set_xlabel('depth (m)')

def plot_subbasins(axis, proj_in, proj_out, plot_limits, coords, var2plot, cmap):
    """Plot variable over horizontal section"""
    # define limits of plot
    axis.set_extent([plot_limits[0][0], plot_limits[0][1], plot_limits[1][0], plot_limits[1][1]], proj_in)

    # Display meridians and parallels
    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", category=ShapelyDeprecationWarning)
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", ShapelyDeprecationWarning)
        # warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        axis.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

    # Compute var2plot to be "plotable"
    x, y, masked_z = z_masked_overlap(axis, coords[0], coords[1], var2plot, source_projection=ccrs.Geodetic())

    # Pcolor
    mapp = axis.pcolormesh(coords[0], coords[1], masked_z[0:-1,0:-1], 
                        transform=proj_in, cmap=cmap)

    cmapColor_subbasin = mpl.cm.get_cmap(cmap)

    color1 = cmapColor_subbasin(0)
    color2 = cmapColor_subbasin(0.5)
    color3 = cmapColor_subbasin(1.0)

    patch1 = mpatches.Patch(color=color1, label='Atlantic sub basin')
    patch2 = mpatches.Patch(color=color2, label='Pacific sub basin')
    patch3 = mpatches.Patch(color=color3, label='Indian sub basin')

    axis.legend(handles=[patch1, patch2, patch3],bbox_to_anchor=(0.65, 0.83), bbox_transform=plt.gcf().transFigure, prop={'size': 14})


    return mapp


# def get_coords_var_v(file_in, varname, x_ind, y_ind):
#     """
#     Get coordinates of provided file and extract vertical section of 
#     variable at specific longitude/latitude indice or averaged along several 
#     longitude/latitude indices depending on the function inputs given.
#     """
#     # Load datasets
#     ds_in   = xr.open_dataset(file_in)
#     # Load variables
#     depth    = ds_in["deptht"].values
#     if x_ind and not y_ind:
#         if len(x_ind) == 1:
#             lat      = ds_in["nav_lat"].values[:,x_ind[0]]
#             var2plot = np.squeeze(np.mean(ds_in[varname].values[:,:,:,x_ind[0]], axis=0))
#         elif len(x_ind) == 2:
#             lat      = ds_in["nav_lat"].values[:,int(np.round((x_ind[0]+x_ind[1])/2))]
#             var2plot = np.nanmean(ds_in[varname].values[:, :, :, x_ind[0]:x_ind[1]], axis=(0, 3))
#         lat_grid, depth_grid = np.meshgrid(lat, depth)
#         return [lat_grid, depth_grid], var2plot
#     elif y_ind and not x_ind:    
#         if len(y_ind) == 1:
#             lon      = ds_in["nav_lon"].values[y_ind[0],:]
#             var2plot = np.squeeze(np.mean(ds_in[varname].values[:,:,y_ind[0],:], axis=0))
#         elif len(y_ind) == 2:
#             lon = ds_in["nav_lon"].values[int(np.round((y_ind[0]+y_ind[1])/2)), :]
#             var2plot = np.nanmean(ds_in[varname].values[:, :, y_ind[0]:y_ind[1], :], axis=(0, 2))
#         lon_grid, depth_grid = np.meshgrid(lon, depth)
#         return [lon_grid, depth_grid], var2plot
#     else:
#         raise RuntimeError("Provide a value either for 'x' or 'y' variable, not for both\n"
#                            "    Exemple:\n"
#                            "        If x = [0, 100], you should have y = [] and vice-versa.")

# def compute_o2_bottom(file_ptrc_in, file_o2_bottom):
#     """Compute O2 at bottom"""
#     # Load data
#     ds_ptrc = xr.open_dataset(file_ptrc_in)

#     # Process 02 at bottom
#     o2_mean = np.mean(ds_ptrc["O2"], axis=0)
#     o2_bottom = np.zeros((ds_ptrc["O2"].shape[2],ds_ptrc["O2"].shape[3]))
#     for j in np.arange(0, ds_ptrc["O2"].shape[2]):
#         for i in np.arange(0, ds_ptrc["O2"].shape[3]):
#             ind_not_nan = np.where(~np.isnan(o2_mean[:,j,i]))[0]
#             if ind_not_nan.size == 0: # if ind is empty meaning only nans are found on this depth column
#                 o2_bottom[j,i] = np.nan
#             else:
#                 ind_bottom = ind_not_nan[-1]
#                 o2_bottom[j,i] = o2_mean[ind_bottom,j,i]

#     # Save variable O2_bottom to optimize next run of the same data (don't need to recompute it at next run)
#     ds_opti = xr.Dataset(
#         {'O2_bottom': (['y','x'], o2_bottom)}
#         )
#     os.makedirs('DATA', exist_ok=True)
#     ds_opti.to_netcdf(file_o2_bottom)

# def get_o2_bottom(file_o2_bottom):
#     """Load O2 bottom file generated by compute_o2_bottom method"""
#     ds_opti = xr.open_dataset(file_o2_bottom)
#     o2_bottom = ds_opti["O2_bottom"].values
#     return o2_bottom