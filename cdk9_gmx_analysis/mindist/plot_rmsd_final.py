
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys

# # Function to read the CSV and determine the data columns
# def read_and_plot_csv(filename, xlabel, ylabel):
#     # Read the CSV into a DataFrame
#     df = pd.read_csv(filename)

#     # Prepare the plot
#     plt.figure(figsize=(12, 6))

#     # Plot each pair of data columns
#     for i in range(1, len(df.columns), 2):
#         plt.plot(df.iloc[:, i - 1], df.iloc[:, i], label=df.columns[i])
    
#     # Labeling the axes
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title('RMSD Analysis')
#     plt.legend(loc='upper right', bbox_to_anchor=(0.98, 1))

#     # Save and show the plot
#     plt.tight_layout()
#     plt.savefig(filename.replace('.csv', '.png'))
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Incorrect usage, correct format:")
#         print("python script.py <path_to_csv> <xlabel> <ylabel>")
#         sys.exit(1)
#     read_and_plot_csv(sys.argv[1], sys.argv[2], sys.argv[3])

import pandas as pd
import matplotlib.pyplot as plt
import sys

# Function to read the CSV and plot each amino acid separately
def read_and_plot_csv(filename, xlabel, ylabel):
    # Read the CSV into a DataFrame
    df = pd.read_csv(filename)

    for i in range(1, len(df.columns), 2):
        plt.figure(figsize=(12, 6))  # Create a new figure for each plot
        plt.plot(df.iloc[:, i - 1], df.iloc[:, i], label=df.columns[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'RMSD Analysis - {df.columns[i]}')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        # Save each plot as a PNG file with 300 DPI
        output_filename = f"{filename.replace('.csv', '')}_{df.columns[i]}.png"
        plt.savefig(output_filename, dpi=300)
        plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Incorrect usage, correct format:")
        print("python script.py <path_to_csv> <xlabel> <ylabel>")
        sys.exit(1)
    read_and_plot_csv(sys.argv[1], sys.argv[2], sys.argv[3])
