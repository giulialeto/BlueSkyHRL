import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

landing_data = pd.read_csv("output/experiment/delete.csv")

# Step 1: Sort by runway and landing time
landing_data_sorted = landing_data.sort_values(by=['runway', 'landing_time'])

# Step 2: Group by runway and calculate the time difference
landing_data_sorted['time_diff'] = landing_data_sorted.groupby('runway')['landing_time'].diff()

# Step 3: Drop NaNs (first landing on each runway has no previous time)
landing_data_diff = landing_data_sorted.dropna(subset=['time_diff'])



# Step 4: Plot the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='runway', y='time_diff', data=landing_data_diff)

# Set y-axis to start at 0
plt.ylim(bottom=0)

# Add horizontal line at 77 seconds
plt.axhline(60, color='red', linestyle='--', linewidth=1.5, label='77 sec threshold')

plt.title('Time Between Sequential Landings per Runway')
plt.xlabel('Runway')
plt.ylabel('Time Between Landings')
plt.tight_layout()
plt.show()
