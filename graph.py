import matplotlib.pyplot as plt
import numpy as np

# Performance metrics for models (accuracy in percentages)
categories = ['Genre', 'Directors', 'Release Date', 'Runtime']
ann_scores = [88, 87, 89, 100]  # Replace with actual ANN scores
best_ml_scores = [92, 90, 91, 100]  # Replace with actual ML model scores

# Bar width and positions
bar_width = 0.35
x_indexes = np.arange(len(categories))

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(x_indexes - bar_width/2, ann_scores, width=bar_width, color='y', label='ANN (DL)')
plt.bar(x_indexes + bar_width/2, best_ml_scores, width=bar_width, color='b', label='Best ML')

# Labels and title
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('ANN (DL) vs Best ML Performance Comparison', fontsize=14)
plt.xticks(ticks=x_indexes, labels=categories)
plt.ylim(80, 105)  # Adjust as needed
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()
