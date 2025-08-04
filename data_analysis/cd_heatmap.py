import pandas as pd
from bluesky.tools.aero import ft, kts
from data_analysis.opensky_tools.conflict_detection import get_conflict_data
import numpy as np

data = pd.read_csv('output/experiment/flight_output.csv')

data = data.rename(
        columns={
            "time": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "heading": "track",
        }
    ).drop(columns=["geoaltitude"])

data = data.assign(
        altitude=data.baroaltitude / ft,
        vertical_rate=data.vertrate / ft * 60,
        groundspeed=data.velocity / kts,
    )

summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(data, r=5556, save=True, tlook=100, alt_cutoff=1220)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Parameters
lon_min, lon_max = 3, 8
lat_min, lat_max = 50.5, 54
grid_size = 0.05  # Adjustable resolution

# Create the bins
lon_bins = np.arange(lon_min, lon_max + grid_size, grid_size)
lat_bins = np.arange(lat_min, lat_max + grid_size, grid_size)

# 2D Histogram (counts per grid cell)
heatmap, xedges, yedges = np.histogram2d(conf_df['lon'], conf_df['lat'],
                                         bins=[lon_bins, lat_bins])

# Avoid log(0): set 0s to np.nan so they’re masked in the plot
heatmap[heatmap == 0] = np.nan

# Create plot
fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Plot heatmap with logarithmic normalization
norm = colors.LogNorm(vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))
mesh = ax.pcolormesh(xedges, yedges, heatmap.T, cmap='OrRd', norm=norm, transform=ccrs.PlateCarree())

# Add colorbar
plt.colorbar(mesh, ax=ax, orientation='vertical', label='Number of Conflicts (log scale)')

plt.title('Log-Scaled Conflict Density Over the Netherlands')
plt.show()

### OLD CODE USED FOR JUST SCATTERPLOTTING THE CONFLICTS WITH COLORS BASED ON TCPA ###

# import code
# code.interact(local=locals())

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # Example dataframe (replace with your real one)
# # df = pd.read_csv('your_data.csv')

# # Create plot
# fig, ax = plt.subplots(figsize=(10, 10),
#                        subplot_kw={'projection': ccrs.PlateCarree()})

# # Set extent to Netherlands
# ax.set_extent([3, 8, 50.5, 54], crs=ccrs.PlateCarree())  # [west, east, south, north]

# # Add features
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.RIVERS)

# conf_df['tcpa'] = conf_df['tcpa'].clip(lower=0,upper=100)
# # Scatter plot
# sc = ax.scatter(conf_df['lon'], conf_df['lat'], alpha=1, c=conf_df['tcpa'], cmap='OrRd_r',
#                 s=2, edgecolor=None, transform=ccrs.PlateCarree())

# # Add colorbar
# plt.colorbar(sc, ax=ax, orientation='vertical', label='TCPA')

# plt.title('TCPA Scatter Plot over the Netherlands')
# plt.show()

# import code
# code.interact(local=locals())# Parameters
