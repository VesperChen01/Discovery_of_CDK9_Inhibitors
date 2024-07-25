import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data preparation
data = {
    'Fingerprints': ['RDkit', 'Atom pairs', 'Morgan', 'MACCS'],
    'Mean R2': [0.539, 0.512, 0.523, 0.479],
    'Mean RMSE': [0.670, 0.689, 0.679, 0.712]
}

df = pd.DataFrame(data)

# Define the position of bars
bar_width = 0.4
index = np.arange(len(df['Fingerprints']))

# Plotting bar chart without error bars
plt.figure(figsize=(10, 6))

# Bar chart for Mean R2
plt.bar(index, df['Mean R2'], color='b', alpha=0.6, label='Mean R2', width=bar_width)

# Bar chart for Mean RMSE
plt.bar(index + bar_width, df['Mean RMSE'], color='r', alpha=0.6, label='Mean RMSE', width=bar_width)

# Adding title and labels
plt.title('Five-Fold Cross-Validation Results')
plt.xlabel('Fingerprints')
plt.ylabel('Metrics')
plt.xticks(index + bar_width / 2, df['Fingerprints'])
plt.legend()

# Display plot
plt.grid(True)
plt.show()
