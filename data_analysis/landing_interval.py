import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

time_cutoff = 500 #maximum number of seconds between landings for interest
threshold = 60 #minimum interval between landings for the runways

file1 = "output/15ac_1/delete.csv"
file2 = "output/35ac_1/delete.csv"
file3 = "output/65ac_1/delete.csv"

file4 = "output/15ac_1_SA/delete.csv"
file5 = "output/35ac_1_SA/delete.csv"
file6 = "output/65ac_1_SA/delete.csv"

file7 = "output/15ac_1_direct/delete.csv"
file8 = "output/35ac_1_direct/delete.csv"
file9 = "output/65ac_1_direct/delete.csv"

files = [file1, file2, file3,file4,file5,file6,file7,file8,file9]
traffic_levels = ["Low", "Medium", "High","Low", "Medium", "High","Low", "Medium", "High"]
methods = ['MA','MA','MA','SA','SA','SA','direct','direct','direct']

number_of_timesteps = 10000
chunk_size = 10000

results = []

for file, level, method in zip(files, traffic_levels,methods):
    chunks = pd.read_csv(file,chunksize=chunk_size)
    total_rows = 0

    file_dfs = []
    for chunk in chunks:
        rows_needed = number_of_timesteps - total_rows
        if rows_needed < 0:
            break
        total_rows += len(chunk)
        
        chunk['Method'] = method
        chunk['Traffic Density'] = level

        file_dfs.append(chunk[:rows_needed])
    
    landing_data = pd.concat(file_dfs, ignore_index=True)
    # Step 1: Sort by runway and landing time
    landing_data_sorted = landing_data.sort_values(by=['runway', 'landing_time'])

    # Step 2: Group by runway and calculate the time difference
    landing_data_sorted['time_diff'] = landing_data_sorted.groupby('runway')['landing_time'].diff()

    # Step 3: Drop NaNs (first landing on each runway has no previous time)
    landing_data_diff = landing_data_sorted.dropna(subset=['time_diff'])
    results.append(landing_data_diff)

# Define linestyle mapping per method
color_map = {
    "MA": "red",
    "SA": "blue",
    "direct": "orange"
}

order = ["Low", "Medium", "High"]

landing_data = pd.concat(results, ignore_index=True)

plt.figure(figsize=(8, 6))
ax = sns.boxplot(
    x='Traffic Density',
    y='time_diff',
    hue="Method",
    data=landing_data[landing_data["time_diff"] < time_cutoff],
    palette=color_map,
    order=order
)

# Count values below threshold
counts = (
    landing_data[landing_data["time_diff"] < threshold]
    .groupby(["Traffic Density", "Method"])
    .size()
    .reset_index(name="count")
)

# Compute medians for positioning
medians = (
    landing_data[landing_data["time_diff"] < time_cutoff]
    .groupby(["Traffic Density", "Method"])["time_diff"]
    .median()
    .reset_index(name="median_time")
)


num_methods = landing_data["Method"].nunique()

# Calculate positions based on same order as the plot
positions = []
for i, level in enumerate(order):
    for j, method in enumerate(sorted(landing_data["Method"].unique())):
        xpos = i + (j - (num_methods - 1)/2) * 0.275

        match_count = counts[
            (counts["Traffic Density"] == level) &
            (counts["Method"] == method)
        ]
        match_medians = medians[
            (medians["Traffic Density"] == level) &
            (medians["Method"] == method)
        ]

        if not match_count.empty and not match_medians.empty:
            count_value = match_count["count"].iloc[0]
            y_medians = match_medians["median_time"].iloc[0]
            ax.text(
                xpos, y_medians + 2,  # 2 units above the mean
                str(count_value),
                ha='center', va='bottom',
                fontsize=12, color='black', fontweight='bold'
            )

#         positions.append((level, method, i + (j - (num_methods-1)/2) * 0.2))

# # Annotate counts
# for (level, method, xpos) in positions:
#     match = counts[
#         (counts["Traffic Density"] == level) &
#         (counts["Method"] == method)
#     ]
#     if not match.empty:
#         count_value = match["count"].iloc[0]
#         ax.text(
#             xpos, threshold/2,
#             str(count_value),
#             ha='center', va='center',
#             fontsize=8, color='black', fontweight='bold'
#         )

# Threshold line
plt.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'{threshold} sec threshold')
plt.ylim(bottom=0)
plt.title('Time Between Sequential Landings per Runway')
plt.xlabel('Traffic Density')
plt.ylabel('Time Between Landings (s)')
plt.tight_layout()
plt.show()