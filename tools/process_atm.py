import os
from PIL import Image
import glob

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs

from .process_grid import z_masked_overlap

# user = os.getenv()

class Atm():
    def __init__(self, work_path, filename_atm, filename_T, filename_msk, ranges, palettes):
        self.workpath = work_path
        self.format_ds_atm(filename_atm)
        self.format_ds_msk(filename_T, filename_msk)
        os.makedirs(self.workpath, exist_ok=True)
        self.ranges = ranges
        self.palettes = palettes
        print('Processing...')

    def format_ds_atm(self, filename_atm):
        ds = xr.open_dataset(filename_atm)
        self.lon_atm = ds["lon"]
        self.lat_atm = ds["lat"]
        self.u850 = np.ma.masked_where((ds["u850"] > 1e30) | (np.isnan(ds["u850"])), ds["u850"])
        self.v850 = np.ma.masked_where((ds["v850"] > 1e30) | (np.isnan(ds["v850"])), ds["v850"])
        self.z850 = np.ma.masked_where((ds["z850"] > 1e30) | (np.isnan(ds["z850"])), ds["z850"])
        self.precip_evap = (ds["precip"].values - ds["evap"].values)*86400
        self.t2m = ds["t2m"].values - 273.15
    
    def format_ds_msk(self, filename_T, filename_msk):
        ds_T   = xr.open_dataset(filename_T)
        ds_msk = xr.open_dataset(filename_msk)
        self.lon_msk = ds_T["nav_lon"].values
        self.lat_msk = ds_T["nav_lat"].values
        self.land_mask = np.squeeze(ds_msk['tmask'])[0,::]

    def gen_plot_param(self, nbContourLines):
        self.figXsize = 15
        self.figYsize = 10
        # Various plot's label sizes
        self.cbar_label_size = 18
        self.cbar_tick_size = 15
        self.title_font_size = 18
        self.xy_label_font_size = 18
        self.xy_ticks_font_size = 18
        self.plot_font_size = 15

        # Figure title
        self.figTitleMonth = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
        
        self.lonGrid, self.latGrid = np.meshgrid(self.lon_atm,self.lat_atm)
        self.latGrid[0,:] = 89.99 # trick so that z_masked_overlap works correctly and plt.contour can display correctly o
        self.latGrid[-1,:] = -89.99

        # Wind plot params
        #-----------------------------------------------------------------------------#
        # Map projection for plots
        if 'c_z850' in self.ranges:
            self.contour_z850 = self.ranges['c_z850']
            bounds = []
            for i in self.ranges['r_w850']:
                bounds = np.round(np.append(bounds, np.arange(i[0], i[1], i[2])),3)
        else:
            print('colormap auto WIND')

            norm_wind850 = np.sqrt(self.u850**2+self.v850**2)
            print(f"norm_wind850.min() : {norm_wind850.min()}")
            print(f"norm_wind850.max() : {norm_wind850.max()}")

            self.contour_z850 = np.round(np.linspace(self.z850.min(), self.z850.max(), nbContourLines), 1)
            bounds = np.round(np.linspace(norm_wind850.min(), norm_wind850.max(), 256), 1)

        self.cmap_wind = plt.get_cmap(self.palettes[0], len(bounds)-1)
        self.norm_wind = mpl.colors.BoundaryNorm(bounds, self.cmap_wind.N)

        self.figLegWind = 'Geopotential height 850 hPa (z850) (m)'
        self.cbar_title_wind = 'Wind 850hPa (u850, v850) (m.s$^{-1}$)'
        #-----------------------------------------------------------------------------#
        
        # Precip_evap plot params
        #-----------------------------------------------------------------------------#
        # Map projection for plots
        if 'c_p_e' in self.ranges:
            self.contour_precip_evap = self.ranges['c_p_e']
            bounds = []
            for i in self.ranges['r_p_e']:
                bounds = np.round(np.append(bounds, np.arange(i[0], i[1], i[2])),3)
        else:
            print('colormap auto P-E')
            mean_precip_evap = np.mean(self.precip_evap, axis=0)
            lim_precip_evap = np.abs((mean_precip_evap.min(),mean_precip_evap.max())).max()
            self.contour_precip_evap = np.round(np.linspace(-lim_precip_evap, lim_precip_evap, nbContourLines), 1)
            bounds = np.round(np.linspace(-lim_precip_evap, lim_precip_evap, 256), 1)

        self.cmap_precip_evap = plt.get_cmap(self.palettes[1], len(bounds)-1)
        self.norm_precip_pevap = mpl.colors.BoundaryNorm(bounds, self.cmap_precip_evap.N)

        # self.figLegprecipevap = 'Geopotential height 850 hPa (z850) (m)'
        self.cbar_title_precip_evap = 'precip - evap (mm.d-1)'
        #-----------------------------------------------------------------------------#

        # t2m plot params
        #-----------------------------------------------------------------------------#
        # Map projection for plots

        if 'c_t2m' in self.ranges:
            self.contour_t2m = self.ranges['c_t2m']
            bounds = []
            for i in self.ranges['r_t2m']:
                bounds = np.round(np.append(bounds, np.arange(i[0], i[1], i[2])),3)
        else:
            print('colormap auto T2M')
            self.contour_t2m = np.round(np.linspace(self.t2m.min(), self.t2m.max(), nbContourLines), 1)
            bounds = np.round(np.linspace(self.t2m.min(), self.t2m.max(), 256), 1)

        self.cmap_t2m = plt.get_cmap(self.palettes[2], len(bounds)-1)
        self.norm_t2m = mpl.colors.BoundaryNorm(bounds, self.cmap_t2m.N)

        self.cbar_title_t2m = 't2m (°)'
        #-----------------------------------------------------------------------------#

    def plot_wind(self, projDataIn, projDataOut, plot_limits):
        """"""
        for i in np.arange(0, self.u850.shape[0]):
            fig, ax = plt.subplots(figsize=(self.figXsize, self.figYsize), subplot_kw={'projection': projDataOut})

            # define limits of plot
            ax.set_extent([plot_limits[0][0], plot_limits[0][1], plot_limits[1][0], plot_limits[1][1]], projDataIn)

            # Display meridians and parallels
            ax.gridlines(draw_labels = True) # transform = ccrs.Geodetic(),

            X, Y, maskedZ = z_masked_overlap(ax, self.lonGrid, self.latGrid, self.z850[i,::], source_projection = ccrs.Geodetic())

            # Arrow plot
            color_array = np.sqrt((self.u850[i, ::2,::2])**2 + (self.v850[i, ::2,::2])**2)
            map1        = ax.quiver(self.lon_atm[::2], self.lat_atm[::2],
                                    self.u850[i, ::2, ::2],self.v850[i, ::2, ::2], 
                                    color_array, cmap=self.cmap_wind, norm = self.norm_wind, transform=projDataIn)
            cont1       = ax.contour(X, Y, maskedZ, self.contour_z850, transform = projDataOut, colors='r', linewidths=1.5)

            # Contour labels
            ax.clabel(cont1,fmt=' {:.1f} '.format,fontsize='x-large')

            # Compute coastline
            X2, Y2, maskedZ2 = z_masked_overlap(ax, self.lon_msk, self.lat_msk, self.land_mask, source_projection = ccrs.Geodetic())
            ax.contour(X2, Y2, maskedZ2, [0], transform = projDataOut, colors='k', linewidths=2)

            # Colorbar
            cbar = plt.colorbar(map1, orientation='horizontal', extend='both')
            cbar_title_wind = f'{self.cbar_title_wind} - {self.figTitleMonth[i]}'
            cbar.ax.set_title(cbar_title_wind, size=self.cbar_label_size)
            cbar.ax.tick_params(labelsize=self.cbar_tick_size)

            # Legend does not support <cartopy.mpl.contour.GeoContourSet object at 0x7fc2739f6f10> instances. So:
            lgd_line = mlines.Line2D([], [], color='red', marker='_', markersize=1, label=self.figLegWind)
            plt.legend(handles=[lgd_line], bbox_to_anchor=(
                0.95, 0.32), bbox_transform=plt.gcf().transFigure, prop={'size': self.plot_font_size})

            # Save figure
            filename = f'wind_speed_{str(i).zfill(3)}.png'
            pathFilename = os.path.join(self.workpath, filename)
            plt.savefig(pathFilename, format='png', bbox_inches='tight', facecolor='white')


    def plot_precip_evap(self, projDataIn, projDataOut, plot_limits):
        """"""
        for i in np.arange(0, self.u850.shape[0]):
            fig, ax = plt.subplots(figsize=(self.figXsize, self.figYsize), subplot_kw={
                'projection': projDataOut})
            
            # define limits of plot
            ax.set_extent([plot_limits[0][0], plot_limits[0][1], plot_limits[1][0], plot_limits[1][1]], projDataIn)
            
            # Display meridians and parallels
            ax.gridlines(draw_labels=True)  # transform = ccrs.Geodetic(),

            X, Y, maskedZ = z_masked_overlap(ax, self.lonGrid, self.latGrid, self.precip_evap[i,::], source_projection=ccrs.Geodetic())

            map1 = ax.pcolormesh(self.lon_atm, self.lat_atm, self.precip_evap[i,::], cmap=self.cmap_precip_evap, norm=self.norm_precip_pevap, transform=projDataIn)
            # Contour plot
            cont1 = ax.contour(X, Y, maskedZ, self.contour_precip_evap, transform=projDataOut, colors='r', linewidths=1.5)

            # Contour labelsfigLegWind
            ax.clabel(cont1, fmt=' {:.1f} '.format, fontsize='x-large')

            # Compute coastline
            X2, Y2, maskedZ2 = z_masked_overlap(ax, self.lon_msk, self.lat_msk, self.land_mask, source_projection = ccrs.Geodetic())
            ax.contour(X2, Y2, maskedZ2, [1], transform = projDataOut, colors='k', linewidths=2)

            # Colorbar
            cbar = plt.colorbar(map1, orientation='horizontal', extend='both')
            cbar_title_precip_evap = f'{self.cbar_title_precip_evap} - {self.figTitleMonth[i]}'
            cbar.ax.set_title(cbar_title_precip_evap, size=self.cbar_label_size)
            cbar.ax.tick_params(labelsize=self.cbar_tick_size)

            # Save figure
            filename = f'precip_pevap_{str(i).zfill(3)}.png'
            pathFilename = os.path.join(self.workpath, filename)
            plt.savefig(pathFilename, format='png', bbox_inches='tight', facecolor='white')

    def plot_t2m(self, projDataIn, projDataOut, plot_limits):
        """"""
        for i in np.arange(0, self.u850.shape[0]):
            fig, ax = plt.subplots(figsize=(self.figXsize, self.figYsize), subplot_kw={'projection': projDataOut})

            # define limits of plot
            ax.set_extent([plot_limits[0][0], plot_limits[0][1], plot_limits[1][0], plot_limits[1][1]], projDataIn)

            # Display meridians and parallels
            ax.gridlines(draw_labels=True)  # transform = ccrs.Geodetic(),

            X, Y, maskedZ = z_masked_overlap(
                ax, self.lonGrid, self.latGrid, self.t2m[i, ::], source_projection=ccrs.Geodetic())

            map1 = ax.pcolormesh(self.lon_atm, self.lat_atm, self.t2m[i, ::], cmap=self.cmap_t2m, norm=self.norm_t2m, transform=projDataIn)
            # Contour plot
            cont1 = ax.contour(X, Y, maskedZ, self.contour_t2m, transform=projDataOut, colors='r', linewidths=1.5)

            # Contour labels
            ax.clabel(cont1, fmt=' {:.1f} '.format, fontsize='x-large')

            # Compute coastline
            X2, Y2, maskedZ2 = z_masked_overlap(ax, self.lon_msk, self.lat_msk, self.land_mask, source_projection = ccrs.Geodetic())
            ax.contour(X2, Y2, maskedZ2, [1], transform = projDataOut, colors='k', linewidths=2)

            # Colorbar
            cbar = plt.colorbar(map1, orientation='horizontal', extend='both')
            cbar_title_t2m = f'{self.cbar_title_t2m} - {self.figTitleMonth[i]}'
            cbar.ax.set_title(cbar_title_t2m, size=self.cbar_label_size)
            cbar.ax.tick_params(labelsize=self.cbar_tick_size)

            # Save figure
            filename = f't2m_{str(i).zfill(3)}.png'
            pathFilename = os.path.join(self.workpath, filename)
            plt.savefig(pathFilename, format='png',
                        bbox_inches='tight', facecolor='white')

    def make_gif(self, timestep_frame):
        listFiles = glob.glob(os.path.join(self.workpath, '*.png'))
        listFiles.sort()
        file_name = listFiles[0].split('/')[-1].split('.')[0].split('_0')[0] + '.gif'
        self.file_name = file_name
        frames = [Image.open(image) for image in listFiles]
        frame_one = frames[0]
        frame_one.save(os.path.join(self.workpath, file_name), format="GIF", append_images=frames,
                       save_all=True, duration=timestep_frame, loop=0)
        os.system(f"rm -rf {os.path.join(self.workpath, '*.png')}")
        print(f'Gif {file_name} created')
