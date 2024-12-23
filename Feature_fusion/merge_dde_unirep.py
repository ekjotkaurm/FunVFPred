import pandas as pd

# List of CSV files to merge
csv_files = [
    "DDE.csv",
    "UNIREP.csv"
]

# Initialize merged_df with the first file
merged_df = pd.read_csv(csv_files[0])

for file in csv_files[1:]:
    temp_df = pd.read_csv(file)

    # Ensure protein_ids exists in both dataframes and is used for joining
    if 'protein_ids' in merged_df.columns and 'protein_ids' in temp_df.columns:
        # Merge on 'protein_ids' to ensure the correct rows are combined
        merged_df = pd.merge(merged_df, temp_df, on='protein_ids', how='outer')
    else:
        print(f"protein_ids column missing in either merged_df or {file}")
        continue

# Handle duplicate label columns
label_cols = [col for col in merged_df.columns if 'label' in col]
if len(label_cols) > 1:
    # Create a single 'label' column
    merged_df['label'] = merged_df[label_cols[0]].combine_first(merged_df[label_cols[1]])
    merged_df.drop(columns=label_cols, inplace=True)

# Fill any missing values (e.g., fill with zeros or appropriate value)
merged_df.fillna(0, inplace=True)

# Save the merged dataframe to a new CSV file
output_path = "MERGED_DDE_UNIREP.csv"
merged_df.to_csv(output_path, index=False)
print(f"Merged CSV saved to '{output_path}'")
