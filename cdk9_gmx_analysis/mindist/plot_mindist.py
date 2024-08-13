import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv('/home/zheyuan/cxl/cdk9/gmx_500/1128096-91-2/protocolGromacs-master/mindist/cdk9_mindist.csv')

# Base path for saving plots
base_save_path = '/home/zheyuan/cxl/cdk9/gmx_500/1128096-91-2/protocolGromacs-master/mindist/'

# Ensure the directory exists
os.makedirs(base_save_path, exist_ok=True)

# Create individual plots for each amino acid
for i in range(1, len(data.columns), 2):  # Start from 1 and increment by 2 to get odd indices (times)
    times = data.columns[i]
    distances = data.columns[i + 1]
    amino_acid = distances  # The header of the distance column is the amino acid name

    # Create a new figure for each amino acid
    plt.figure()
    plt.plot(data[times], data[distances], label=amino_acid, marker='', linestyle='-')  # No markers, just line

    # Set title and labels specific to the amino acid
    plt.title(f'The distance of compound 2 to {amino_acid}')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.legend(title=amino_acid)

    # Save the plot as a PNG file
    save_path = os.path.join(base_save_path, f'{amino_acid}.png',dpi=600)
    plt.savefig(save_path, dpi=600)  # Save each plot with the amino acid name
    plt.close()  # Close the plot to free up memory

# Uncomment the line below if you want to display all plots in the notebook or script output instead of saving
# plt.show()