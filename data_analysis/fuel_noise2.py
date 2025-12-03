import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILES = [
    "output/synthetic/15ac_1/summary.csv",
    "output/synthetic/35ac_1/summary.csv",
    "output/synthetic/65ac_1/summary.csv",
    "output/synthetic/15ac_1_SA/summary.csv",
    "output/synthetic/35ac_1_SA/summary.csv",
    "output/synthetic/65ac_1_SA/summary.csv",
    "output/synthetic/15ac_1_direct/summary.csv",
    "output/synthetic/35ac_1_direct/summary.csv",
    "output/synthetic/65ac_1_direct/summary.csv",
]

TRAFFIC_LEVELS = ["Low", "Medium", "High"] * 3
METHODS = ["MA"] * 3 + ["SA"] * 3 + ["direct"] * 3

results = []
for file, level, method in zip(FILES, TRAFFIC_LEVELS, METHODS):
    data = pd.read_csv(file)
    data["Traffic Density"] = level
    data["Method"] = method
    results.append(data)

combined_data = pd.concat(results, ignore_index=True)

# --- Compute mean values for 'direct' per traffic level ---
direct_means = (
    combined_data[combined_data["Method"] == "direct"]
    .groupby("Traffic Density")[["total_noise", "total_fuel"]]
    .mean()
    .rename(columns=lambda c: f"{c}_direct_mean")
)

# Merge means back to combined_data
combined_data = combined_data.merge(direct_means, on="Traffic Density", how="left")

# --- Compute percentage change relative to direct mean ---
combined_data["noise_pct_change"] = (
    (combined_data["total_noise"] - combined_data["total_noise_direct_mean"])
    / combined_data["total_noise_direct_mean"]
    * 100
)
combined_data["fuel_pct_change"] = (
    (combined_data["total_fuel"] - combined_data["total_fuel_direct_mean"])
    / combined_data["total_fuel_direct_mean"]
    * 100
)

# --- Keep only MA and SA for plotting ---
plot_data = combined_data[combined_data["Method"].isin(["MA", "SA"])]

sns.set(style="whitegrid")

# --- Plot 1: Noise ---
plt.figure(figsize=(5, 3.5))
sns.lineplot(
    data=plot_data,
    x="Traffic Density",
    y="noise_pct_change",
    hue="Method",
    marker="o",
    linewidth=2,
    legend=False
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.ylabel("Percent Change (%)")
plt.xlabel("Traffic Density")

plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# --- Plot 2: Fuel ---
plt.figure(figsize=(5, 3.5))
sns.lineplot(
    data=plot_data,
    x="Traffic Density",
    y="fuel_pct_change",
    hue="Method",
    marker="s",
    linewidth=2,
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.ylabel("Percent Change (%)")
plt.xlabel("Traffic Density")
plt.legend(loc='upper right')
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()