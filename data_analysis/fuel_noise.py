import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file1 = "output/15ac_2/summary.csv"
file2 = "output/experiment/summary.csv"
file3 = "output/experiment/summary.csv"

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)

# Assign density categories
data1['Density'] = 'low'
data2['Density'] = 'medium'
data3['Density'] = 'high'

# Combine all data
combined_data = pd.concat([data1, data2, data3], ignore_index=True)

sns.set(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot Conflicts on left y-axis
color_conf = "tab:blue"
sns.lineplot(
    data=combined_data,
    x="Density", y="total_noise", ax=ax1,
    marker="o", color=color_conf, label="Noise Emissions"
)
ax1.set_ylabel("Noise Emissions", color=color_conf)
ax1.tick_params(axis='y', labelcolor=color_conf)
ax1.set_xlabel("Traffic Density")

# Plot Intrusions on right y-axis
ax2 = ax1.twinx()
color_intr = "tab:red"
sns.lineplot(
    data=combined_data,
    x="Density", y="total_fuel", ax=ax2,
    marker="s", linestyle="--", color=color_intr, label="Fuel Emissions"
)
ax2.set_ylabel("Fuel Emissions", color=color_intr)
ax2.tick_params(axis='y', labelcolor=color_intr)

# Title and layout
plt.title("Noise and Fuel Emissions by Traffic Density")
fig.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()