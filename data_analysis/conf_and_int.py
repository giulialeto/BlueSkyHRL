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

file1 = "output/15ac_1/flight_output.csv"
file2 = "output/15ac_2/flight_output.csv"
file3 = "output/experiment/flight_output.csv"

files = [file1, file2, file3]
traffic_levels = ["Low", "Medium", "High"]

number_of_timesteps = 30000

results = []

for file, level in zip(files, traffic_levels):
    data = pd.read_csv(file)
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

    summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(data[:number_of_timesteps], r=r_sep, save=save, tlook=tlook, alt_cutoff=alt_cutoff)

    results.append({
        "Traffic Density": level,
        "Conflicts": len(conflict_counts),
        "Intrusions": len(intrusion_counts)
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

# Right Y-axis: Intrusions
ax2 = ax1.twinx()
color_intr = "tab:red"
ax2.set_ylabel("Intrusions", color=color_intr)
ax2.plot(df_results["Traffic Density"], df_results["Intrusions"], 
         color=color_intr, marker="s", linestyle="--", label="Intrusions")
ax2.tick_params(axis="y", labelcolor=color_intr)

# Title and layout
plt.title("Conflicts and Intrusions by Traffic Density")
fig.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()
