import pandas as pd

# List of CSV files to merge
csv_files = [
    "/FILES/AAC.csv",
    "/FILES/DDE.csv",
    "/FILES/UNIREP.csv"
]

# Initialize an empty list to store individual dataframes
dataframes = []

for file in csv_files:
    # Read the CSV file
    temp_df = pd.read_csv(file)

    # Ensure 'protein_ids' and 'label' are present
    if 'protein_ids' not in temp_df.columns or 'label' not in temp_df.columns:
        print(f"File {file} is missing 'protein_ids' or 'label' columns. Skipping.")
        continue

    # Drop duplicate label column to prevent redundancy during merging
    temp_df = temp_df.drop(columns=['label'])
    dataframes.append(temp_df)

# Merge all dataframes on 'protein_ids'
merged_df = dataframes[0]
for temp_df in dataframes[1:]:
    merged_df = pd.merge(merged_df, temp_df, on='protein_ids', how='outer')

# Add the label column from the first file
label_df = pd.read_csv(csv_files[0])[['protein_ids', 'label']]
merged_df = pd.merge(merged_df, label_df, on='protein_ids', how='inner')

# Save the merged dataframe to a new CSV file
output_path = "MERGED_AAC_DDE_UNIREP_AF.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged CSV saved to '{output_path}'")
