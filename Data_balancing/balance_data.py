from Bio import SeqIO
import random


# Input files
positive_file = "CDHIT100_pos.fasta"
negative_file = "CDHIT100_neg.fasta"
balanced_negative_file = "balanced_neg.fasta"


# Load positive proteins
positive_proteins = [record for record in SeqIO.parse(positive_file, "fasta")]

# Load all negative proteins
negative_proteins = [record for record in SeqIO.parse(negative_file, "fasta")]

# Randomly sample 65 negative proteins
random.seed(42)  # For reproducibility
sampled_negatives = random.sample(negative_proteins, len(positive_proteins))

# Write the sampled negatives to a new file
with open(balanced_negative_file, "w") as out_f:
    SeqIO.write(sampled_negatives, out_f, "fasta")

# Save the positive proteins into another file for further use
balanced_positive_file = "balanced_pos.fasta"
with open(balanced_positive_file, "w") as out_f:
    SeqIO.write(positive_proteins, out_f, "fasta")

print(f"Balanced negative proteins written to: {balanced_negative_file}")
print(f"Balanced positive proteins written to: {balanced_positive_file}")
