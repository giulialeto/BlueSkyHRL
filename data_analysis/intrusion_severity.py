import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_analysis.opensky_tools.conflict_detection import get_conflict_data
from bluesky.tools.aero import ft, kts

tlook = 20 #s
alt_cutoff = 1415 #m
r_sep = 9260 #m = 5NM
save = False

file1 = "output/15ac_1/flight_output.csv"
file2 = "output/35ac_1/flight_output.csv"
file3 = "output/65ac_1/flight_output.csv"

file4 = "output/15ac_1_SA/flight_output.csv"
file5 = "output/35ac_1_SA/flight_output.csv"
file6 = "output/65ac_1_SA/flight_output.csv"

file7 = "output/15ac_1_direct/flight_output.csv"
file8 = "output/35ac_1_direct/flight_output.csv"
file9 = "output/65ac_1_direct/flight_output.csv"

files = [file1, file2, file3,file4,file5,file6,file7,file8,file9]
traffic_levels = ["Low", "Medium", "High","Low", "Medium", "High","Low", "Medium", "High"]
methods = ['MA','MA','MA','SA','SA','SA','direct','direct','direct']

number_of_timesteps = 100000
chunk_size = 10000

results = []

for file, level, method in zip(files, traffic_levels,methods):
    chunks = pd.read_csv(file,chunksize=chunk_size)
    total_rows = 0
    total_conflicts = 0
    total_intrusions = 0
    
    file_conf_dfs = []
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

        summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(chunk[:rows_needed], r=r_sep, save=save, tlook=tlook, alt_cutoff=alt_cutoff)
        total_conflicts += len(conflict_counts)
        total_intrusions += len(intrusion_counts)
        
        conf_df['Method'] = method
        conf_df['Traffic Density'] = level

        file_conf_dfs.append(conf_df)

    results.append(pd.concat(file_conf_dfs, ignore_index=True))

df_results = pd.concat(results, ignore_index=True)

# Define linestyle mapping per method
color_map = {
    "MA": "red",
    "SA": "blue",
    "direct": "orange"
}

# Filter to only distances below separation
plot_df = df_results[(df_results["dist"] < r_sep) & (df_results["alt"] > alt_cutoff)]

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=plot_df,
    x="Traffic Density",
    y="dist",
    hue="Method",
    palette=color_map  # your same dict works fine here
)

plt.axhline(y=5556, color='black', linestyle='-')
plt.ylabel("Distance (m)")
plt.title("Intrusion severity by method & traffic density")
plt.legend(title="Method")
plt.show()