import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_analysis.opensky_tools.conflict_detection import get_conflict_data
from bluesky.tools.aero import ft, kts

tlook = 500 #s
alt_cutoff = 1415 #m
r_sep = 5556 #m
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

number_of_timesteps = 10000000
chunk_size = 10000

results = []

for file, level, method in zip(files, traffic_levels,methods):
    chunks = pd.read_csv(file,chunksize=chunk_size)
    total_rows = 0
    total_conflicts = 0
    total_conflict_time = 0
    total_intrusions = 0
    total_intrusion_time = 0

    
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
        total_conflict_time += conflict_counts['conflicts'].sum()
        total_intrusions += len(intrusion_counts)
        total_intrusion_time += intrusion_counts['intrusions'].sum()

    results.append({
        "Traffic Density": level,
        "Method": method,
        "Conflicts": total_conflicts,
        "Intrusions": total_intrusions,
        "Conflict Time": total_conflict_time,
        "Intrusion Time": total_intrusion_time
    })

df_results = pd.DataFrame(results)

# Define linestyle mapping per method
linestyle_map = {
    "MA": "-",
    "SA": "--",
    "direct": ":"
}

fig, ax1 = plt.subplots(figsize=(8, 6))

color_conf = "tab:blue"
color_intr = "tab:red"

# Create the second y-axis only once
ax2 = ax1.twinx()

# Plot both Conflicts and Intrusions for each method
for method in df_results["Method"].unique():
    subset = df_results[df_results["Method"] == method]

    # Conflicts (left y-axis)
    ax1.plot(subset["Traffic Density"], subset["Conflicts"],
             color=color_conf, linestyle=linestyle_map[method],
             label=f"Conflicts ({method})")

    # Intrusions (right y-axis)
    ax2.plot(subset["Traffic Density"], subset["Intrusions"],
             color=color_intr, linestyle=linestyle_map[method],
             label=f"Intrusions ({method})")

# Labels
ax1.set_xlabel("Traffic Density")
ax1.set_ylabel("Conflicts", color=color_conf)
ax1.tick_params(axis="y", labelcolor=color_conf)
ax1.set_ylim(bottom=0) 

ax2.set_ylabel("Intrusions", color=color_intr)
ax2.tick_params(axis="y", labelcolor=color_intr)
ax2.set_ylim(bottom=0) 

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", bbox_to_anchor=(1.05, 1))

# Title and layout
plt.title("Conflicts and Intrusions by Traffic Density & Method")
fig.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()


fig, ax1 = plt.subplots(figsize=(8, 6))

color_conf = "tab:blue"
color_intr = "tab:red"

# Create the second y-axis only once
ax2 = ax1.twinx()

# Plot both Conflicts and Intrusions for each method
for method in df_results["Method"].unique():
    subset = df_results[df_results["Method"] == method]

    # Conflicts (left y-axis)
    ax1.plot(subset["Traffic Density"], subset["Conflict Time"],
             color=color_conf, linestyle=linestyle_map[method],
             label=f"Conflicts ({method})")

    # Intrusions (right y-axis)
    ax2.plot(subset["Traffic Density"], subset["Intrusion Time"],
             color=color_intr, linestyle=linestyle_map[method],
             label=f"Intrusions ({method})")

# Labels
ax1.set_xlabel("Traffic Density")
ax1.set_ylabel("Conflicts Time", color=color_conf)
ax1.tick_params(axis="y", labelcolor=color_conf)
ax1.set_ylim(bottom=0) 

ax2.set_ylabel("Intrusion Time", color=color_intr)
ax2.tick_params(axis="y", labelcolor=color_intr)
ax2.set_ylim(bottom=0) 

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", bbox_to_anchor=(1.05, 1))

# Title and layout
plt.title("Time in Conflict and Intrusion by Traffic Density & Method")
fig.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()

import code
code.interact(local=locals())