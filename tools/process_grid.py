import numpy as np
import cartopy.crs as ccrs

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
