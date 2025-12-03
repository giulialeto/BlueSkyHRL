import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## HAVE TO FILTER FOR INVALID FLIGHTS

# file1 = "output/15ac_2/summary.csv"
# file2 = "output/experiment/summary.csv"
# file3 = "output/experiment/summary.csv"

# data1 = pd.read_csv(file1)
# data2 = pd.read_csv(file2)
# data3 = pd.read_csv(file3)

# # Assign density categories
# data1['density'] = 'low'
# data2['density'] = 'medium'
# data3['density'] = 'high'

# # Combine all data
# combined_data = pd.concat([data1, data2, data3], ignore_index=True)

# # Plot
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='density', y='flight_time', data=combined_data, order=['low', 'medium', 'high'])

# # Set y-axis to start at 0
# plt.ylim(bottom=0)

# plt.title('Boxplot of the flight time for the aircraft')
# plt.xlabel('Traffic Density')
# plt.ylabel('Flight Time')
# plt.tight_layout()
# plt.show()

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
traffic_levels = ["Low", "Medium", "High","Low", "Medium", "High","Low", "Medium", "High"]
methods = ['MA','MA','MA','SA','SA','SA','direct','direct','direct']

results = []

for file, level, method in zip(FILES, traffic_levels,methods):
    data = pd.read_csv(file)
    data['Traffic Density'] = level
    data['Method'] = method

    results.append(data)

combined_data = pd.concat(results,ignore_index=True)

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Traffic Density', y='flight_time', hue='Method',data=combined_data, order=['Low', 'Medium', 'High'])

# Set y-axis to start at 0
plt.ylim(bottom=1000)
plt.yscale('log')

plt.title('Boxplot of the flight time for the aircraft')
plt.xlabel('Traffic Density')
plt.ylabel('Flight Time')
plt.tight_layout()
plt.show()


def lorenz_curve(flighttime):
    """Return Lorenz curve coordinates for an array of flight times."""
    sorted_vals = np.sort(flighttime)
    cumvals = np.cumsum(sorted_vals)
    cumvals = np.insert(cumvals, 0, 0)  # start at 0
    cumvals = cumvals / cumvals[-1]     # normalize to [0, 1]
    x = np.linspace(0, 1, len(cumvals))
    return x, cumvals

methods = ["MA","SA","direct"]
labels = ["MA","SA","Base"]
plt.figure(figsize=(5, 5))

combined_data = combined_data[combined_data["Traffic Density"] == "High"]

for (method, group), label in zip(combined_data.groupby("Method"),labels):
    x, y = lorenz_curve(group["flight_time"].values)
    plt.plot(x, y, label=label)

for method, group in combined_data.groupby("Method"):
    times = group["flight_time"].values
    
    mean_ft = np.mean(times)
    median_ft = np.median(times)
    
    # sort descending to get the longest flights
    sorted_times = np.sort(times)[::-1]
    n = len(times)
    
    # top 5% and 1% thresholds
    top5 = sorted_times[:int(0.05 * n)]
    top1 = sorted_times[:int(0.01 * n)]
    
    total_top5 = np.sum(top5)
    total_top1 = np.sum(top1)
    total_all = np.sum(times)
    
    # share of total time contributed by top 5% and 1%
    share_top5 = total_top5 / total_all
    share_top1 = total_top1 / total_all

    print(f"{method} mean: {mean_ft}")
    print(f"{method} median: {median_ft}")
    print(f"{method} top5%: {top5}")
    print(f"{method} top1%: {top1}")

# Line of perfect equality
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Perfect fairness')

plt.xlabel('Cumulative share of aircraft')
plt.ylabel('Cumulative share of total flight time')
plt.legend()
plt.grid(True)
plt.show()