import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix,  matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

# Load the datasets
train_data = pd.read_csv('train_AAC.csv')
test_data = pd.read_csv('test_AAC.csv')
validate_data = pd.read_csv('validation_AAC.csv')

# Drop the 'protein_ids' column if present
train_data = train_data.drop(columns=['protein_ids'], errors='ignore')
test_data = test_data.drop(columns=['protein_ids'], errors='ignore')
validate_data = validate_data.drop(columns=['protein_ids'], errors='ignore')

# Separate features and labels
X_train = train_data.drop(columns=['label'])  # Replace 'label' with the target column
y_train = train_data['label']

X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

X_validate = validate_data.drop(columns=['label'])
y_validate = validate_data['label']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validate_scaled = scaler.transform(X_validate)

# Build the ANN model (with one hidden layer)
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input layer + 1 hidden layer
    Dropout(0.3),  # Dropout layer to reduce overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
ann_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the model
history = ann_model.fit(
    X_train_scaled, y_train,
    epochs=100, batch_size=32,
    validation_data=(X_validate_scaled, y_validate),
    verbose=1
)

# === Evaluate on Test Set ===
y_test_prob = ann_model.predict(X_test_scaled).flatten()
y_test_pred = (y_test_prob >= 0.5).astype(int)

accuracy_test = accuracy_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_prob)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Calculate precision, recall, F1 score for the test set
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

cm_test = confusion_matrix(y_test, y_test_pred)
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
sensitivity_test = tp_test / (tp_test + fn_test)
specificity_test = tn_test / (tn_test + fp_test)

# === Evaluate on Validation Set ===
y_validate_prob = ann_model.predict(X_validate_scaled).flatten()
y_validate_pred = (y_validate_prob >= 0.5).astype(int)

accuracy_validate = accuracy_score(y_validate, y_validate_pred)
roc_auc_validate = roc_auc_score(y_validate, y_validate_prob)
mcc_validate = matthews_corrcoef(y_validate, y_validate_pred)

# Calculate precision, recall, F1 score for the validation set
precision_validate = precision_score(y_validate, y_validate_pred)
recall_validate = recall_score(y_validate, y_validate_pred)
f1_validate = f1_score(y_validate, y_validate_pred)

cm_validate = confusion_matrix(y_validate, y_validate_pred)
tn_validate, fp_validate, fn_validate, tp_validate = cm_validate.ravel()
sensitivity_validate = tp_validate / (tp_validate + fn_validate)
specificity_validate = tn_validate / (tn_validate + fp_validate)

# Save Test and Validation Results
with open('ANN_aac_test_results.txt', 'w') as f:
    f.write("=== Test Set Results ===\n")
    f.write(f"Accuracy: {accuracy_test:.4f}\n")
    f.write(f"ROC-AUC Score: {roc_auc_test:.4f}\n")
    f.write(f"MCC: {mcc_test:.4f}\n")
    f.write(f"Precision: {precision_test:.4f}\n")
    f.write(f"Recall: {recall_test:.4f}\n")
    f.write(f"F1 Score: {f1_test:.4f}\n")
    f.write(f"Sensitivity: {sensitivity_test:.4f}\n")
    f.write(f"Specificity: {specificity_test:.4f}\n")

with open('ANN_aac_validation_results.txt', 'w') as f:
    f.write("=== Validation Set Results ===\n")
    f.write(f"Accuracy: {accuracy_validate:.4f}\n")
    f.write(f"ROC-AUC Score: {roc_auc_validate:.4f}\n")
    f.write(f"MCC: {mcc_validate:.4f}\n")
    f.write(f"Precision: {precision_validate:.4f}\n")
    f.write(f"Recall: {recall_validate:.4f}\n")
    f.write(f"F1 Score: {f1_validate:.4f}\n")
    f.write(f"Sensitivity: {sensitivity_validate:.4f}\n")
    f.write(f"Specificity: {specificity_validate:.4f}\n")

# Print Results
print("=== Test Set Results ===")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"ROC-AUC Score: {roc_auc_test:.4f}")
print(f"MCC: {mcc_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print(f"Sensitivity: {sensitivity_test:.4f}")
print(f"Specificity: {specificity_test:.4f}")

print("\n=== Validation Set Results ===")
print(f"Accuracy: {accuracy_validate:.4f}")
print(f"ROC-AUC Score: {roc_auc_validate:.4f}")
print(f"MCC: {mcc_validate:.4f}")
print(f"Precision: {precision_validate:.4f}")
print(f"Recall: {recall_validate:.4f}")
print(f"F1 Score: {f1_validate:.4f}")
print(f"Sensitivity: {sensitivity_validate:.4f}")
print(f"Specificity: {specificity_validate:.4f}")

# === Confusion Matrix ===
def plot_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.show()

# Test set confusion matrix
plot_confusion_matrix(cm_test, ['Negative', 'Positive'], "ANN_aac_test_confusion_matrix.png")

# Validation set confusion matrix
plot_confusion_matrix(cm_validate, ['Negative', 'Positive'], "ANN_aac_validation_confusion_matrix.png")

# === ROC Curve ===
def plot_roc_curve(y_true, y_prob, filename):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Plot ROC curve for test and validation sets
plot_roc_curve(y_test, y_test_prob, "ANN_AAC_test_roc_curve.png")
plot_roc_curve(y_validate, y_validate_prob, "ANN_AAC_validation_roc_curve.png")
