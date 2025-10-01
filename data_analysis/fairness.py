import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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