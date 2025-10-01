import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file1 = "output/15ac_1/summary.csv"
file2 = "output/35ac_1/summary.csv"
file3 = "output/65ac_1/summary.csv"

file4 = "output/15ac_1_SA/summary.csv"
file5 = "output/35ac_1_SA/summary.csv"
file6 = "output/65ac_1_SA/summary.csv"

file7 = "output/15ac_1_direct/summary.csv"
file8 = "output/35ac_1_direct/summary.csv"
file9 = "output/65ac_1_direct/summary.csv"

files = [file1, file2, file3,file4,file5,file6,file7,file8,file9]
traffic_levels = ["Low", "Medium", "High","Low", "Medium", "High","Low", "Medium", "High"]
methods = ['MA','MA','MA','SA','SA','SA','direct','direct','direct']

results = []

for file, level, method in zip(files, traffic_levels,methods):
    data = pd.read_csv(file)
    data['Traffic Density'] = level
    data['Method'] = method

    results.append(data)

combined_data = pd.concat(results,ignore_index=True)
sns.set(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot Conflicts on left y-axis
color_conf = "tab:blue"
sns.lineplot(
    data=combined_data,
    x="Traffic Density", y="total_noise", ax=ax1,
    style="Method", color=color_conf
)
ax1.set_ylabel("Noise Emissions", color=color_conf)
ax1.tick_params(axis='y', labelcolor=color_conf)
ax1.set_xlabel("Traffic Density")

# Plot Intrusions on right y-axis
ax2 = ax1.twinx()
color_intr = "tab:red"
sns.lineplot(
    data=combined_data,
    x="Traffic Density", y="total_fuel", ax=ax2,
    style="Method", color=color_intr
)
ax2.set_ylabel("Fuel Emissions", color=color_intr)
ax2.tick_params(axis='y', labelcolor=color_intr)

# Title and layout
plt.title("Noise and Fuel Emissions by Traffic Density")
fig.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()