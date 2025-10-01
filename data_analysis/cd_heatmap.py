import pandas as pd
from bluesky.tools.aero import ft, kts
from data_analysis.opensky_tools.conflict_detection import get_conflict_data
import numpy as np

PLOT_INTRUSIONS = False
r_sep = 5556
vzh = 300
alt_cutoff = 1415 #1220
t_look=200
number_of_timesteps = 1000000

chunk_size = 10000

chunks = pd.read_csv('output/65ac_1_SA/flight_output.csv',chunksize=chunk_size)

conf_dfs = []
total_rows = 0
for chunk in chunks:
    rows_needed = number_of_timesteps - total_rows
    if rows_needed < 0:
        break
    total_rows += len(chunk)
    chunk = chunk.rename(
            columns={
                "time": "timestamp",
                "lat": "latitude",
                "lon": "longitude",
                "heading": "track",
            }
        ).drop(columns=["geoaltitude"])

    chunk = chunk.assign(
            altitude=chunk.baroaltitude / ft,
            vertical_rate=chunk.vertrate / ft * 60,
            groundspeed=chunk.velocity / kts,
        )

    summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(chunk[:rows_needed], r=r_sep, save=True, tlook=t_look, alt_cutoff=alt_cutoff)

    conf_dfs.append(conf_df)

conf_df = pd.concat(conf_dfs, ignore_index=True)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

if PLOT_INTRUSIONS:
    conf_df = conf_df[(conf_df['dalt'] < vzh) & (conf_df['alt'] > alt_cutoff) & (conf_df['dist']<r_sep)]
else:
    conf_df = conf_df[(conf_df['dalt'] < vzh) & (conf_df['alt'] > alt_cutoff)]

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
