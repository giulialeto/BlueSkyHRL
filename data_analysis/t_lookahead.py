"""
This script should make a plot with on the y axis the number of conflicts, the x axis the look_ahead time, and with the color the different traffic densities.
This can then show at which the look_ahead time most of the conflicts are resolved by the model. e.g. if there exist virtually no conflicts with t_look at 200 seconds, 
but a lot at 250, this would show that the model solves most conflicts at t_look = 250-200
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_analysis.opensky_tools.conflict_detection import get_conflict_data
from bluesky.tools.aero import ft, kts

tlook = [10,20,50,75,100,150,200,300,400,500,1000] #s
# tlook = [100,200,500]
alt_cutoff = 1415 #m
r_sep = 5556 #m
save = False

file1 = "output/15ac_1/flight_output.csv"
file2 = "output/35ac_1/flight_output.csv"
file3 = "output/65ac_1/flight_output.csv"

files = [file1,file2,file3]
traffic_levels = ["Low", "Medium", "High"]

number_of_timesteps = 100000
chunk_size = 10000
results = []
# for file, density in zip(files,traffic_levels):
#     data = pd.read_csv(file)
#     data = data.rename(
#             columns={
#                 "time": "timestamp",
#                 "lat": "latitude",
#                 "lon": "longitude",
#                 "heading": "track",
#             }
#         ).drop(columns=["geoaltitude"])

#     data = data.assign(
#             altitude=data.baroaltitude / ft,
#             vertical_rate=data.vertrate / ft * 60,
#             groundspeed=data.velocity / kts,
#         )

    

#     for look_ahead in tlook:    

#         summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(data[:number_of_timesteps], r=r_sep, save=save, tlook=look_ahead, alt_cutoff=alt_cutoff)

#         results.append({
#             "T-Lookahead": look_ahead,
#             "Conflicts": len(conflict_counts),
#             "Intrusions": len(intrusion_counts),
#             "Density": density
#         })

# df_results = pd.DataFrame(results)

for look_ahead in tlook:    
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
            
            summary, conf_df, conflict_counts, intrusion_counts = get_conflict_data(chunk[:rows_needed], r=r_sep, save=save, tlook=look_ahead, alt_cutoff=alt_cutoff)
            total_conflicts += len(conflict_counts)

        results.append({
            "T-Lookahead": look_ahead,
            "Conflicts": total_conflicts,
            "Density": level
        })


df_results = pd.DataFrame(results)

# Set a nice style
sns.set(style="whitegrid")

# Plot Conflicts
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df_results,
    x="T-Lookahead",
    y="Conflicts",
    hue="Density",
    marker="o",
    palette="Set1"
)
plt.xlabel("Lookahead Time (s)")
plt.ylabel("Number of Conflicts")
plt.title("Conflicts as a Function of Lookahead Time")
plt.legend(title="Traffic Density")
plt.tight_layout()
plt.show()
