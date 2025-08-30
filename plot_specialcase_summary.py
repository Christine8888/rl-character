import json
import matplotlib.pyplot as plt
import numpy as np

# Read the data
with open('/workspace/rl-character/christine_experiments/20250827_testcases/gold_sft/hacks_0828/specialcase_train/specialcase_classifier_summary.json', 'r') as f:
    data = json.load(f)

# Create array for all values 0-100
x_values = list(range(0, 101))
y_values = []

# Extract percentages from the distribution
distribution = data['distribution']

# Fill in the percentages for each grade
for grade in x_values:
    if str(grade) in distribution:
        y_values.append(distribution[str(grade)]['percentage'])
    else:
        y_values.append(0.0)

# Create the bar plot
plt.figure(figsize=(10, 4))
bars = plt.bar(x_values, y_values, width=0.8, alpha=0.7)

# Set labels
plt.xlabel('Sonnet 4 grade (higher = more egregious special-casing)')
plt.ylabel('% of hacking data')

# Set x-axis to show all values 0-100
plt.xlim(-0.5, 100.5)
plt.xticks(range(0, 101, 5))  # Show every 5th tick for readability

# Show grid for better readability
plt.grid(True, alpha=0.3, axis='y')

# Adjust layout and save
plt.tight_layout()
plt.savefig('/workspace/rl-character/specialcase_classifier_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()