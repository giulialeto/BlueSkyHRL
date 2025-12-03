import pandas as pd
from bluesky.tools.aero import ft, kts
from data_analysis.opensky_tools.conflict_detection import get_conflict_data
import numpy as np

PLOT_INTRUSIONS = True
METHOD = 'MA'
r_sep = 5556
vzh = 300
alt_cutoff = 1415 #1220
t_look=200
number_of_timesteps = 1000000

chunk_size = 10000

cache_file = "data_analysis/data/hist_int_sev.pkl"
conf_df = pd.read_pickle(cache_file)

# ### ADD DISTANCE FROM SCHIPHOL FOR FILTERING ###
# schiphol_lat = 52.3086
# schiphol_lon = 4.7639

# # Convert degrees to radians
# lat1 = np.radians(conf_df['lat'])
# lon1 = np.radians(conf_df['lon'])
# lat2 = np.radians(schiphol_lat)
# lon2 = np.radians(schiphol_lon)

# # Haversine formula
# R = 6371  # Earth's radius in km
# dlat = lat2 - lat1
# dlon = lon2 - lon1
# a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
# c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
# conf_df['distance_from_schiphol_km'] = R * c

# conf_df = conf_df[conf_df['distance_from_schiphol_km']<250]


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

conf_df = conf_df[conf_df["Method"]==METHOD] 
if PLOT_INTRUSIONS: 
    conf_df = conf_df[(conf_df['dalt'] < vzh) & (conf_df['alt'] > alt_cutoff) & (conf_df['dist']<r_sep)] 
else: 
    conf_df = conf_df[(conf_df['dalt'] < vzh) & (conf_df['alt'] > alt_cutoff)]

# Parameters
lon_min, lon_max = 0, 9
lat_min, lat_max = 49.5, 57

# Create figure
fig, ax = plt.subplots(figsize=(7, 7),
                       subplot_kw={'projection': ccrs.PlateCarree()})

# Set extent (Netherlands)
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.set_aspect('auto')
# Add map features

ax.add_feature(cfeature.LAND, color='gainsboro')
ax.add_feature(cfeature.OCEAN, color='whitesmoke')
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.RIVERS)

# Normalize and invert distance for point size
# Clip distances to avoid extreme outliers (optional)
dist_clipped = np.clip(conf_df['dist'], 100, r_sep)  # avoids 0 or huge values
size = 300000 / dist_clipped  # inversely proportional (adjust 3000 as needed)

# # Create a new axes on the right for the colorbar
# pos = ax.get_position()  # [x0, y0, width, height]
# cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])  # [x0, y0, width, height]

# # Scatter plot
# sc = ax.scatter(
#     conf_df['lon'], conf_df['lat'],
#     c=conf_df['dist'],
#     s=size,
#     cmap='turbo_r',  # good perceptual range; try 'plasma', 'viridis', or 'OrRd_r'
#     alpha=0.7,
#     edgecolor='none',
#     transform=ccrs.PlateCarree()
# )

# cbar = plt.colorbar(sc, cax=cax, orientation='vertical',
#                     fraction=0.046,  # width relative to figure
#                     pad=0.04,        # distance from map
#                     label='Separation Distance [m]')
sc = ax.scatter(
    conf_df['lon'], conf_df['lat'],
    c=conf_df['dist'],
    s=size,
    cmap='turbo_r',
    alpha=0.7,
    edgecolor='none',
    transform=ccrs.PlateCarree()
)

# Let Matplotlib handle placement
cbar = fig.colorbar(
    sc, ax=ax, orientation='vertical',
    fraction=0.046, pad=0.04,
    label='Separation Distance [m]'
)
plt.tight_layout()
plt.show()
plt.show()