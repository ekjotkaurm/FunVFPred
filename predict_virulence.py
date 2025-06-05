import pandas as pd
import numpy as np
from Bio import SeqIO
import torch
from tape import UniRepModel, TAPETokenizer
import joblib

# ==============================
# AAC Feature Extraction
# ==============================
def compute_aac(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac_vector = np.zeros(len(amino_acids))
    for aa in sequence:
        if aa in amino_acids:
            aac_vector[amino_acids.index(aa)] += 1
    aac_vector /= len(sequence)
    return aac_vector

# ==============================
# DDE Feature Extraction
# ==============================
def compute_dde(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    num_amino_acids = len(amino_acids)
    dde_vector = np.zeros((num_amino_acids, num_amino_acids))
    for i in range(len(sequence) - 1):
        aa1 = sequence[i]
        aa2 = sequence[i + 1]
        if aa1 in amino_acids and aa2 in amino_acids:
            dde_vector[amino_acids.index(aa1)][amino_acids.index(aa2)] += 1
    total = np.sum(dde_vector)
    if total > 0:
        dde_vector /= total
    expected_freq = np.outer(compute_aac(sequence), compute_aac(sequence))
    dde_deviation = dde_vector.flatten() - expected_freq.flatten()
    return dde_deviation

# ==============================
# UniRep Feature Extraction
# ==============================
def extract_unirep_features(sequences, max_length=1024):
    print("Loading UniRep model...")
    model = UniRepModel.from_pretrained("babbler-1900").eval()
    tokenizer = TAPETokenizer(vocab="unirep")
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            input_ids = torch.tensor([tokenizer.encode(seq[:max_length])])
            output = model(input_ids)[0]
            embeddings.append(output.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# ==============================
# Prediction Pipeline
# ==============================
def predict_from_fasta(fasta_file, model_file="rf_model.joblib"):
    # Load protein sequences
    sequences, ids = [], []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)

    print(f"Total proteins to predict: {len(sequences)}")

    # Feature extraction
    print("Extracting AAC...")
    aac = np.array([compute_aac(seq) for seq in sequences])

    print("Extracting DDE...")
    dde = np.array([compute_dde(seq) for seq in sequences])

    print("Extracting UniRep...")
    unirep = extract_unirep_features(sequences)

    # Merge features
    all_features = np.concatenate([aac, dde, unirep], axis=1)

    # Load model
    print("Loading trained model...")
    model = joblib.load(model_file)

    # Predict
    predictions = model.predict(all_features)
    probabilities = model.predict_proba(all_features)[:, 1]

    # Create output DataFrame
    result_df = pd.DataFrame({
        "protein_ids": ids,
        "prediction": predictions,
        "probability": probabilities
    })
    result_df["prediction"] = result_df["prediction"].map({1: "Virulent", 0: "Non-Virulent"})

    # Save
    result_df.to_csv("virulence_predictions.csv", index=False)
    print("Prediction results saved to 'virulence_predictions.csv'")

    return result_df

# ==============================
# Run if called directly
# ==============================
if __name__ == "__main__":
    fasta_path = "proteins.fasta"  # Replace with your FASTA file
    predict_from_fasta(fasta_path)
