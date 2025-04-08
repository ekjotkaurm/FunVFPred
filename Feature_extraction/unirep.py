import pandas as pd
from Bio import SeqIO
import torch
from tape import UniRepModel, TAPETokenizer
import numpy as np

# Load protein sequences from FASTA file
def load_sequences(fasta_file):
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    return ids, sequences

# UniRep feature extraction
def extract_unirep_features(sequences, max_length=1024):
    model = UniRepModel.from_pretrained("babbler-1900").eval()  # Load the UniRep model
    tokenizer = TAPETokenizer(vocab="unirep")  # Use UniRep-compatible tokenizer
    embeddings = []

    with torch.no_grad():
        for seq in sequences:
            # Tokenize and encode the sequence
            input_ids = torch.tensor([tokenizer.encode(seq[:max_length])])
            output = model(input_ids)[0]  # Get the last hidden state
            # Average the hidden states to get a single vector representation per sequence
            embeddings.append(output.mean(dim=1).squeeze().numpy())

    return np.array(embeddings)

# Load label file
def load_labels(label_file):
    labels = pd.read_csv(label_file)
    return labels

# Main function to process and save all features
def process_and_save_features(fasta_file, label_file, output_file, max_length=1024):
    # Load sequences and IDs
    ids, sequences = load_sequences(fasta_file)

    # Load labels
    labels = load_labels(label_file)

    print("Extracting UniRep features...")
    unirep_features = extract_unirep_features(sequences, max_length=max_length)

    # Create a DataFrame for UniRep features
    unirep_df = pd.DataFrame(unirep_features, columns=[f"unirep_{i}" for i in range(unirep_features.shape[1])])

    # Add protein IDs to the DataFrame
    unirep_df["protein_ids"] = ids

    # Merge with labels
    final_df = pd.merge(unirep_df, labels, left_on="protein_ids", right_on="protein_ids", how="inner")

    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

# Run the feature extraction and saving process
fasta_file = "/DATA/input.fasta"  # Path to your FASTA file
label_file = "/DATA/labels.csv"        # Path to your labels CSV file
output_file = "UNIREP.csv"  # Output CSV file with features

process_and_save_features(fasta_file, label_file, output_file)    
