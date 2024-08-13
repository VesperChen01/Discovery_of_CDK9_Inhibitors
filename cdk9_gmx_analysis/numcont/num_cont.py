import pandas as pd

# Load the data from CSV
data = pd.read_csv('/home/zheyuan/cxl/cdk9/gmx_500/1128096-91-2/protocolGromacs-master/numcont/cdk9_numcont.csv')

# Create a dictionary to store counts of zeros for each distance column
zero_counts = {}

# Iterate over each column, checking only even indices for distance columns
for i in range(1, len(data.columns), 2):
    distances = data.columns[i]
    # Count zeros in the column
    zero_counts[distances] = (data[distances] == 0).sum()

# Display the counts
for amino_acid, count in zero_counts.items():
    print(f'{amino_acid}: {count}')

# Optionally, you can add a new row to the DataFrame with these counts
new_row = [None] * len(data.columns)  # Start with a list of None
for i, col in enumerate(data.columns):
    if i % 2 == 1:  # Even index in zero-based, odd in one-based (distance columns)
        new_row[i] = zero_counts[col]
        
# Append the new row to the DataFrame
data.loc[len(data)] = new_row

# Save the modified DataFrame back to CSV if needed
data.to_csv('/home/zheyuan/cxl/cdk9/gmx_500/1128096-91-2/protocolGromacs-master/numcont/file.csv', index=False)
