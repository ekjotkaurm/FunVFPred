import pandas as pd
import numpy as np
from Bio import SeqIO
import torch
from tape import UniRepModel, TAPETokenizer
import joblib

# ==========================
# AAC Feature Extraction
# ==========================
def compute_aac(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac_vector = np.zeros(len(amino_acids))
    for aa in sequence:
        if aa in amino_acids:
            aac_vector[amino_acids.index(aa)] += 1
    aac_vector /= len(sequence)
    return aac_vector

# ==========================
# DDE Feature Extraction
# ==========================
def compute_dde(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dde_vector = np.zeros((20, 20))
    for i in range(len(sequence) - 1):
        aa1 = sequence[i]
        aa2 = sequence[i + 1]
        if aa1 in amino_acids and aa2 in amino_acids:
            i1 = amino_acids.index(aa1)
            i2 = amino_acids.index(aa2)
            dde_vector[i1][i2] += 1
    total = np.sum(dde_vector)
    if total > 0:
        dde_vector /= total
    expected = np.outer(compute_aac(sequence), compute_aac(sequence))
    deviation = dde_vector.flatten() - expected.flatten()
    return deviation

# ==========================
# UniRep Feature Extraction
# ==========================
def extract_unirep_features(sequences):
    print("Loading UniRep model...")
    model = UniRepModel.from_pretrained("babbler-1900").eval()
    tokenizer = TAPETokenizer(vocab="unirep")
    embeddings = []
    with torch.no_grad():
        for seq in sequences:
            encoded = torch.tensor([tokenizer.encode(seq[:1024])])  # truncate if >1024
            output = model(encoded)[0]
            mean_embedding = output.mean(dim=1).squeeze().numpy()
            embeddings.append(mean_embedding)
    return np.array(embeddings)

# ==========================
# Prediction Pipeline
# ==========================
def predict_from_fasta(fasta_file, model_path="rf_model.joblib", output_file="virulence_predictions.csv"):
    # Step 1: Read sequences
    sequences = []
    protein_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        protein_ids.append(record.id)

    print(f"Loaded {len(sequences)} sequences from {fasta_file}")

    # Step 2: Extract Features
    print("Extracting AAC features...")
    aac_features = np.array([compute_aac(seq) for seq in sequences])

    print("Extracting DDE features...")
    dde_features = np.array([compute_dde(seq) for seq in sequences])

    print("Extracting UniRep features...")
    unirep_features = extract_unirep_features(sequences)

    print("Combining all features...")
    full_features = np.concatenate([aac_features, dde_features, unirep_features], axis=1)

    # Step 3: Load Trained Model
    print("Loading trained model...")
    model = joblib.load(model_path)

    # Step 4: Predict
    print("Predicting virulence...")
    preds = model.predict(full_features)
    pred_labels = ['Virulent' if p == 1 else 'Non-Virulent' for p in preds]

    # Step 5: Output Results
    results = pd.DataFrame({
        'protein_ids': protein_ids,
        'prediction': pred_labels
    })

    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'\n")

    # Print nicely
    print("=== Prediction Results ===")
    print(results.to_string(index=False))

# ==========================
# Run Script
# ==========================
if __name__ == "__main__":
    fasta_input = "unknown_sequences.fasta"  # replace this with your file path
    predict_from_fasta(fasta_input)
