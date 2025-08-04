import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_analysis.opensky_tools.conflict_detection import get_conflict_data
from bluesky.tools.aero import ft, kts

tlook = 100 #s
alt_cutoff = 1220 #m
r_sep = 5556 #m
save = False

file1 = "output/15ac_2/flight_output.csv"
file2 = "output/35ac_1/flight_output.csv"
file3 = "output/65ac_1/flight_output.csv"

files = [file1, file2, file3]
traffic_levels = ["Low", "Medium", "High"]

number_of_timesteps = 100000
chunk_size = 10000

results = []

for file, level in zip(files, traffic_levels):
    chunks = pd.read_csv(file,chunksize=chunk_size)
    total_rows = 0
    total_conflicts = 0
    total_intrusions = 0
    
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
        
    results.append({
        "Traffic Density": level,
        "Conflicts": total_conflicts,
        "Intrusions": total_intrusions
    })

df_results = pd.DataFrame(results)

fig, ax1 = plt.subplots(figsize=(8, 6))

# Left Y-axis: Conflicts
color_conf = "tab:blue"
ax1.set_xlabel("Traffic Density")
ax1.set_ylabel("Conflicts", color=color_conf)
ax1.plot(df_results["Traffic Density"], df_results["Conflicts"], 
         color=color_conf, marker="o", label="Conflicts")
ax1.tick_params(axis="y", labelcolor=color_conf)
ax1.set_ylim(bottom=0) 

# Right Y-axis: Intrusions
ax2 = ax1.twinx()
color_intr = "tab:red"
ax2.set_ylabel("Intrusions", color=color_intr)
ax2.plot(df_results["Traffic Density"], df_results["Intrusions"], 
         color=color_intr, marker="s", linestyle="--", label="Intrusions")
ax2.tick_params(axis="y", labelcolor=color_intr)
ax2.set_ylim(bottom=0) 

# Title and layout
plt.title("Conflicts and Intrusions by Traffic Density")
fig.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()
