import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from math import cos, radians
import numpy as np

# Schiphol coordinates
center_lat, center_lon = 52.31, 4.77
radius_km = 350

deg_per_km_lat = 1 / 111
lat_radius = radius_km * deg_per_km_lat

deg_per_km_lon = 1 / (111 * cos(radians(center_lat)))
lon_radius = radius_km * deg_per_km_lon

lat_min = center_lat - lat_radius
lat_max = center_lat + lat_radius
lon_min = center_lon - lon_radius
lon_max = center_lon + lon_radius

def plot_trajectory_clickable(traffic_data, c_feature='altitude', label='altitude(m)', colormap='viridis', scale=None, log_cut_off=0.001):
    import matplotlib.lines as mlines

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    ax.set_aspect(1./ax.get_data_ratio())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    if scale is None:
        norm = mcolors.Normalize(vmin=traffic_data[c_feature].min(), vmax=traffic_data[c_feature].max())
    elif scale == 'log':
        norm = mcolors.LogNorm(vmin=traffic_data[c_feature].min() + log_cut_off, vmax=traffic_data[c_feature].max())
    else:
        norm = mcolors.Normalize(vmin=traffic_data[c_feature].min(), vmax=traffic_data[c_feature].max())

    cmap = plt.get_cmap(colormap)
    lines = []  # to store line objects
    line_map = {}  # map line to icao24

    for icao, group in traffic_data.groupby('icao24'):
        group = group.sort_values(by='timestamp')
        color = cmap(norm(group[c_feature].mean()))

        line, = ax.plot(
            group.longitude,
            group.latitude,
            transform=ccrs.PlateCarree(),
            color=color,
            linewidth=1.5,
            alpha=0.8,
            picker=5  # Enable picking with 5pt tolerance
        )
        lines.append(line)
        line_map[line] = icao

    # Create dummy ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label=label, shrink=0.75, pad=0.02)

    plt.title("Flight Trajectories ±300 km around Schiphol")

    # Event handler
    def on_pick(event):
        line = event.artist
        icao = line_map.get(line, 'Unknown')
        print(f"Clicked aircraft: {icao}")
        # Optional: show popup text on plot
        x_mouse, y_mouse = event.mouseevent.xdata, event.mouseevent.ydata
        ax.annotate(f'icao24: {icao}', xy=(x_mouse, y_mouse), xycoords='data',
                    bbox=dict(boxstyle="round", fc="w"), fontsize=9)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.show()


def plot_trajectory_clickable2(traffic_data, c_feature='altitude', label='altitude (m)',
                              colormap='turbo', scale=None, log_cut_off=0.001, color_dict=None, alpha=0.8,
                              vmin_plot=None, vmax_plot=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set map extent based on data or globals
    lon_min, lon_max = traffic_data.longitude.min(), traffic_data.longitude.max()
    lat_min, lat_max = traffic_data.latitude.min(), traffic_data.latitude.max()

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Make the map square (equal aspect in display coordinates)
    


    # Draw coastlines in black (no colors)
    # ax.coastlines(color='black', linewidth=1)
    ax.add_feature(cfeature.LAND, color='gainsboro')
    ax.add_feature(cfeature.OCEAN, color='whitesmoke')
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Normalization for color mapping
    vmin, vmax = traffic_data[c_feature].min(), traffic_data[c_feature].max()
    if scale == 'log':
        norm = mcolors.LogNorm(vmin=vmin + log_cut_off, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap(colormap)
    lines = []
    line_map = {}

    for icao, group in traffic_data.groupby('icao24'):
        group = group.sort_values(by='timestamp')

        if color_dict is not None:
            color = color_dict[icao]

        # Create list of line segments
        points = np.array([group.longitude.values, group.latitude.values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Use altitude for each segment (take value at the starting point)
        altitudes = group[c_feature].values[:-1]

        if color_dict is not None:
            lc = LineCollection(segments, colors=color, norm=norm, transform=ccrs.PlateCarree(),
                                linewidth=1.5, alpha=alpha, picker=5)
        else:
            lc = LineCollection(segments, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                                linewidth=1.5, alpha=alpha, picker=5)
        lc.set_array(altitudes)
        ax.add_collection(lc)

        lines.append(lc)
        line_map[lc] = icao

    # Colorbar
    if not color_dict:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        if vmin_plot is not None:
            sm.set_clim(vmin_plot, vmax_plot)   # add your numeric limits here
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label=label, shrink=0.75, pad=0.02)

    # Click event
    def on_pick(event):
        line = event.artist
        icao = line_map.get(line, 'Unknown')
        print(f"Clicked aircraft: {icao}")
        x_mouse, y_mouse = event.mouseevent.xdata, event.mouseevent.ydata
        ax.annotate(f'icao24: {icao}', xy=(x_mouse, y_mouse), xycoords='data',
                    bbox=dict(boxstyle="round", fc="w"), fontsize=9)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()

def plot_trajectory_line(traffic_data, c_feature='altitude', label='altitude(m)', colormap='viridis', scale=None, log_cut_off=0.001):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    ax.set_aspect(1./ax.get_data_ratio())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Normalize color scale
    if scale is None:
        norm = mcolors.Normalize(vmin=traffic_data[c_feature].min(), vmax=traffic_data[c_feature].max())
    elif scale == 'log':
        norm = mcolors.LogNorm(vmin=traffic_data[c_feature].min() + log_cut_off, vmax=traffic_data[c_feature].max())
    else:
        print(f'Not implemented scale: {scale}, using standard Normalize')
        norm = mcolors.Normalize(vmin=traffic_data[c_feature].min(), vmax=traffic_data[c_feature].max())

    cmap = plt.get_cmap(colormap)

    # Group by aircraft and plot line for each
    for icao, group in traffic_data.groupby('icao24'):
        group = group.sort_values(by='timestamp')  # Optional: sort by time
        color = cmap(norm(group[c_feature].mean()))  # color by mean altitude (or other feature)
        ax.plot(
            group.longitude,
            group.latitude,
            transform=ccrs.PlateCarree(),
            color=color,
            linewidth=1,
            alpha=0.7
        )

    # Create dummy ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy data
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label=label, shrink=0.75, pad=0.02)

    plt.title("Flight Trajectories ±300 km around Schiphol")
    plt.show()

def plot_trajectory(traffic_data, c_feature='altitude', label='altitude(m)', colormap='viridis', scale=None, log_cut_off=0.001):
    fig = plt.figure(figsize=(8, 8))  # Square figure
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set square extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Resize axes box to be square (normalized figure coordinates)
    ax.set_position([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height] in 0–1
    ax.set_aspect(1./ax.get_data_ratio())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Normalize color scale
    if scale is None:
        norm = mcolors.Normalize(vmin=traffic_data[c_feature].min(), vmax=traffic_data[c_feature].max())
    elif scale == 'log':
        norm = mcolors.LogNorm(vmin=traffic_data[c_feature].min() + log_cut_off, vmax=traffic_data[c_feature].max())
    else:
        print(f'Not implemented scale: {scale}, using standard Normalize')
        norm = mcolors.Normalize(vmin=traffic_data[c_feature].min(), vmax=traffic_data[c_feature].max())

    cmap = plt.get_cmap(colormap)
    cmap.set_bad('k')

    points = ax.scatter(
        traffic_data.longitude, traffic_data.latitude,
        c=traffic_data[c_feature], cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), s=2
    )

    # Place colorbar relative to figure
    cbar = fig.colorbar(points, ax=ax, orientation='vertical', label=label, shrink=0.75, pad=0.02)

    plt.title("Flight Trajectories ±300 km around Schiphol")
    plt.show()