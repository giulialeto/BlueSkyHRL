import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_analysis.opensky_tools.conflict_detection import get_conflict_data
from bluesky.tools.aero import ft, kts

# ---------- CONFIG ----------
TLOOK = 20  # s
ALT_CUTOFF = 1415  # m
R_SEP = 10000 # 9260 #m = 5NM
SAVE = False
CACHE_FILE = "data_analysis/data/hist_int_sev_dist_cutoff.pkl"  # <-- cache location
FORCE_RECOMPUTE = False # True = generate new data, False = use cache if available
CUTOFF_DISTANCE = 250

# FILES = [
#     "output/synthetic/15ac_1/flight_output.csv",
#     "output/synthetic/35ac_1/flight_output.csv",
#     "output/synthetic/65ac_1/flight_output.csv",
#     "output/synthetic/15ac_1_SA/flight_output.csv",
#     "output/synthetic/35ac_1_SA/flight_output.csv",
#     "output/synthetic/65ac_1_SA/flight_output.csv",
#     "output/synthetic/15ac_1_direct/flight_output.csv",
#     "output/synthetic/35ac_1_direct/flight_output.csv",
#     "output/synthetic/65ac_1_direct/flight_output.csv",
# ]
FILES = [
    "output/jan_2024/flight_output.csv",
    "output/march_2024/flight_output.csv",
    "output/july_2024/flight_output.csv",
    "output/jan_2024_SA/flight_output.csv",
    "output/march_2024_SA/flight_output.csv",
    "output/july_2024_SA/flight_output.csv",
    "output/jan_2024_direct/flight_output.csv",
    "output/march_2024_direct/flight_output.csv",
    "output/july_2024_direct/flight_output.csv",
]

TRAFFIC_LEVELS = ["Jan", "Mar", "Jul"] * 3
METHODS = ["MA"] * 3 + ["SA"] * 3 + ["direct"] * 3

CHUNK_SIZE = 10000
N_TIMESTEPS = 100_000_000

def compute_results(force_recompute=False, cache_file=CACHE_FILE):
    """Compute conflict/intrusion results and cache them."""
    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading cached results from {cache_file}")
        return pd.read_pickle(cache_file)

    results = []

    for file, level, method in zip(FILES, TRAFFIC_LEVELS, METHODS):
        print(f"Processing {file} ({method}, {level})")
        chunks = pd.read_csv(file, chunksize=CHUNK_SIZE)
        total_rows = total_conflicts = total_conflict_time = total_intrusions = total_intrusion_time = 0

        file_conf_dfs = []
        for chunk in chunks:
            rows_needed = N_TIMESTEPS - total_rows
            if rows_needed < 0:
                break
            total_rows += len(chunk)

            chunk = chunk.rename(columns={
                "time": "timestamp",
                "lat": "latitude",
                "lon": "longitude",
                "heading": "track",
            }).drop(columns=["geoaltitude"])

            chunk = chunk.assign(
                altitude=chunk.baroaltitude / ft,
                vertical_rate=chunk.vertrate / ft * 60,
                groundspeed=chunk.velocity / kts,
            )

            summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(
                chunk[:rows_needed],
                r=R_SEP,
                save=SAVE,
                tlook=TLOOK,
                alt_cutoff=ALT_CUTOFF,
                cutoff_dist=CUTOFF_DISTANCE
            )

            total_conflicts += len(conflict_counts)
            total_conflict_time += conflict_counts["conflicts"].sum()
            total_intrusions += len(intrusion_counts)
            total_intrusion_time += intrusion_counts["intrusions"].sum()

            conf_df['Method'] = method
            conf_df['Traffic Density'] = level
            file_conf_dfs.append(conf_df)

        results.append(pd.concat(file_conf_dfs, ignore_index=True))

    df_results = pd.concat(results, ignore_index=True)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df_results.to_pickle(cache_file)
    print(f"Results saved to {cache_file}")
    return df_results

def plot_intrusion_severity(df, dist=R_SEP):
    # Define linestyle mapping per method
    # color_map = {
    #     "MA": "red",
    #     "SA": "blue",
    #     "direct": "orange"
    # }

    # Filter to only distances below separation
    plot_df = df[(df["dist"] < dist) & (df["alt"] > ALT_CUTOFF)] 

    # group columns that identify a conflict pair
    plot_df["tcpa_round"] = plot_df["tcpa"].round(1)

    group_cols = ["tcpa_round", "dist", "Method", "Traffic Density"]

    # function to build the pair ID
    def build_id(group):
        calls = sorted(group["callsign"].unique())
        return "_".join(calls)
    
    # 1. Group by tcpa/dist/etc. and assign a pair ID
    tmp = (
        plot_df.groupby(group_cols)
        .apply(lambda g: pd.Series({"ID": build_id(g)}))
        .reset_index()
    )

    # 2. Now group by the ID and pick the row with the MINIMUM distance
    result = (
        tmp.loc[tmp.groupby(["ID","Method","Traffic Density"])["dist"].idxmin()]
        .reset_index(drop=True)
    )
    plot_df = result

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=plot_df,
        x="Traffic Density",
        y="dist",
        hue="Method",
        order=["low", "medium", "high", "jan", "mar", "jul"],                   # x-axis order
        hue_order=["MA", "SA", "direct"] # your same dict works fine here
    )
    sns.despine()
    plt.axhline(y=5556, color='black', linestyle='-')
    plt.ylabel("Distance (m)")
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ["MA", "SA", "Base"]  # your desired legend text
    plt.legend(handles, new_labels, title="Method",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0)
    plt.tight_layout()
    plt.xticks(["low", "medium", "high", "jan", "mar", "jul"], ["Low", "Medium", "High", "Jan", "Mar", "Jul"])
    plt.show()

cache_file = "data_analysis/data/int_sev.pkl"
df1 = compute_results(force_recompute=FORCE_RECOMPUTE, cache_file=cache_file)
cache_file = "data_analysis/data/hist_int_sev.pkl"
df2 = compute_results(force_recompute=FORCE_RECOMPUTE, cache_file=cache_file)

### ADD DISTANCE FROM SCHIPHOL FOR FILTERING ###
schiphol_lat = 52.3086
schiphol_lon = 4.7639

# Convert degrees to radians
lat1 = np.radians(df2['lat'])
lon1 = np.radians(df2['lon'])
lat2 = np.radians(schiphol_lat)
lon2 = np.radians(schiphol_lon)

# Haversine formula
R = 6371  # Earth's radius in km
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
df2['distance_from_schiphol_km'] = R * c

# df2 = df2[df2['distance_from_schiphol_km']<250]

df = pd.concat([df1,df2])
plot_intrusion_severity(df)