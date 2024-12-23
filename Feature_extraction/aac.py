import pandas as pd
from Bio import SeqIO
import numpy as np

def compute_aac(sequence):
    """Compute the amino acid composition (AAC) for a protein sequence."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac_vector = np.zeros(len(amino_acids))

    # Count occurrences of each amino acid
    for amino_acid in sequence:
        if amino_acid in amino_acids:
            aac_vector[amino_acids.index(amino_acid)] += 1

    # Convert counts to proportions
    aac_vector /= len(sequence)
    return aac_vector

def extract_features(fasta_file, label_file):
    """Extract AAC features from FASTA file and combine with labels from CSV."""
    # Read labels
    labels_df = pd.read_csv(label_file)

    # Initialize lists to hold features and corresponding labels
    aac_features = []
    sequence_ids = []

    # Read protein sequences from FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        sequence_id = record.id

        # Compute AAC features
        aac_vector = compute_aac(sequence)

        # Append features and sequence ID
        aac_features.append(aac_vector)
        sequence_ids.append(sequence_id)

    # Convert lists to DataFrames
    aac_df = pd.DataFrame(aac_features, columns=[f'AAC_{i+1}' for i in range(20)])  # AAC_1 to AAC_20


    # Create a DataFrame for sequence IDs
    sequence_id_df = pd.DataFrame(sequence_ids, columns=['protein_ids'])  # Rename for merging

    # Combine AAC features with sequence IDs
    combined_df = pd.concat([sequence_id_df, aac_df], axis=1)

    # Combine features and labels
    combined_df = combined_df.merge(labels_df, on='protein_ids', how='inner')  # Merge with labels

    return combined_df

# Example usage
fasta_file = 'input.fasta'  # Replace with your FASTA file
label_file = 'labels.csv'            # Replace with your label CSV file
features_df = extract_features(fasta_file, label_file)

# Save features to CSV
features_df.to_csv('AAC.csv', index=False)

print("Features extracted and saved to 'AAC.csv'.")
