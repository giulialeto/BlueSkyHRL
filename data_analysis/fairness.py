import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file1 = "output/15ac_2/summary.csv"
file2 = "output/experiment/summary.csv"
file3 = "output/experiment/summary.csv"

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)

# Assign density categories
data1['density'] = 'low'
data2['density'] = 'medium'
data3['density'] = 'high'

# Combine all data
combined_data = pd.concat([data1, data2, data3], ignore_index=True)

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='density', y='flight_time', data=combined_data, order=['low', 'medium', 'high'])

# Set y-axis to start at 0
plt.ylim(bottom=0)

plt.title('Boxplot of the flight time for the aircraft')
plt.xlabel('Traffic Density')
plt.ylabel('Flight Time')
plt.tight_layout()
plt.show()