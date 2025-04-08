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

def compute_dde(sequence):
    """Compute the dipeptide deviation from expected mean (DDE) for a protein sequence."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    num_amino_acids = len(amino_acids)

    # Create an array for dipeptide counts
    dde_vector = np.zeros((num_amino_acids, num_amino_acids))

    # Count dipeptides
    for i in range(len(sequence) - 1):
        aa1 = sequence[i]
        aa2 = sequence[i + 1]
        if aa1 in amino_acids and aa2 in amino_acids:
            dde_vector[amino_acids.index(aa1)][amino_acids.index(aa2)] += 1

    # Calculate total dipeptides
    total_dipeptides = np.sum(dde_vector)

    # Convert counts to proportions
    if total_dipeptides > 0:
        dde_vector /= total_dipeptides

    # Compute expected frequencies based on AAC
    expected_freq = np.outer(compute_aac(sequence), compute_aac(sequence))

    # Calculate deviation from expected mean
    dde_deviation = dde_vector.flatten() - expected_freq.flatten()
    return dde_deviation

def extract_features(fasta_file, label_file):
    """Extract DDE features from FASTA file and combine with labels from CSV."""
    # Read labels
    labels_df = pd.read_csv(label_file)

    # Initialize lists to hold features and corresponding labels
    dde_features = []
    sequence_ids = []

    # Read protein sequences from FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        sequence_id = record.id

        # Compute DDE features
        dde_vector = compute_dde(sequence)

        # Append features and sequence ID
        dde_features.append(dde_vector)
        sequence_ids.append(sequence_id)

    # Convert lists to DataFrames
    dde_df = pd.DataFrame(dde_features, columns=[f'DDE_{i+1}' for i in range(400)])  # DDE_1 to DDE_400
    sequence_id_df = pd.DataFrame(sequence_ids, columns=['protein_ids'])  # Create DataFrame for sequence IDs

    # Combine DDE features with sequence IDs
    combined_df = pd.concat([sequence_id_df, dde_df], axis=1)

    # Combine features and labels
    combined_df = combined_df.merge(labels_df, on='protein_ids', how='inner')  # Merge with labels

    return combined_df

# Example usage
fasta_file = '/DATA/input.fasta'  # Replace with your FASTA file
label_file = '/DATA/labels.csv'            # Replace with your label CSV file
features_df = extract_features(fasta_file, label_file)

# Save features to CSV
features_df.to_csv('DDE.csv', index=False)

print("Features extracted and saved to 'DDE.csv'.")
