import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "AAC.csv"  # Replace with the path to your file
data = pd.read_csv(file_path)

# Separate positive and negative samples
positive = data[data['label'] == 1]
negative = data[data['label'] == 0]

# Split positive samples
pos_train, pos_temp = train_test_split(positive, test_size=0.3, random_state=42, stratify=positive['label'])
pos_test, pos_val = train_test_split(pos_temp, test_size=1/3, random_state=42, stratify=pos_temp['label'])

# Split negative samples
neg_train, neg_temp = train_test_split(negative, test_size=0.3, random_state=42, stratify=negative['label'])
neg_test, neg_val = train_test_split(neg_temp, test_size=1/3, random_state=42, stratify=neg_temp['label'])

# Combine train, test, and validation sets
train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42)  # Shuffle the dataset
test = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=42)
val = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=42)

# Save to CSV files
train.to_csv("train_AAC.csv", index=False)
test.to_csv("test_AAC.csv", index=False)
val.to_csv("validation_AAC.csv", index=False)

print("Files saved successfully as train_AAC.csv, test_AAC.csv, and validation_AAC.csv")
