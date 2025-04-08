import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    matthews_corrcoef, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

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

# Build the DNN model
dnn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
dnn_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the model
history = dnn_model.fit(
    X_train_scaled, y_train,
    epochs=100, batch_size=16,
    validation_data=(X_validate_scaled, y_validate),
    verbose=1
)

# === Evaluate on Test Set ===
y_test_prob = dnn_model.predict(X_test_scaled).flatten()
y_test_pred = (y_test_prob >= 0.5).astype(int)

accuracy_test = accuracy_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_prob)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

cm_test = confusion_matrix(y_test, y_test_pred)
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
sensitivity_test = tp_test / (tp_test + fn_test)
specificity_test = tn_test / (tn_test + fp_test)

# Save Test Results
with open('DNN_aac_test_results.txt', 'w') as f:
    f.write("=== Test Set Results ===\n")
    f.write(f"Accuracy: {accuracy_test:.4f}\n")
    f.write(f"ROC-AUC Score: {roc_auc_test:.4f}\n")
    f.write(f"MCC: {mcc_test:.4f}\n")
    f.write(f"Sensitivity: {sensitivity_test:.4f}\n")
    f.write(f"Specificity: {specificity_test:.4f}\n")

# === Evaluate on Validation Set ===
y_validate_prob = dnn_model.predict(X_validate_scaled).flatten()
y_validate_pred = (y_validate_prob >= 0.5).astype(int)

accuracy_validate = accuracy_score(y_validate, y_validate_pred)
roc_auc_validate = roc_auc_score(y_validate, y_validate_prob)
mcc_validate = matthews_corrcoef(y_validate, y_validate_pred)

cm_validate = confusion_matrix(y_validate, y_validate_pred)
tn_validate, fp_validate, fn_validate, tp_validate = cm_validate.ravel()
sensitivity_validate = tp_validate / (tp_validate + fn_validate)
specificity_validate = tn_validate / (tn_validate + fp_validate)

# Save Validation Results
with open('DNN_aac_validation_results.txt', 'w') as f:
    f.write("=== Validation Set Results ===\n")
    f.write(f"Accuracy: {accuracy_validate:.4f}\n")
    f.write(f"ROC-AUC Score: {roc_auc_validate:.4f}\n")
    f.write(f"MCC: {mcc_validate:.4f}\n")
    f.write(f"Sensitivity: {sensitivity_validate:.4f}\n")
    f.write(f"Specificity: {specificity_validate:.4f}\n")

# Print Results
print("=== Test Set Results ===")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"ROC-AUC Score: {roc_auc_test:.4f}")
print(f"MCC: {mcc_test:.4f}")
print(f"Sensitivity: {sensitivity_test:.4f}")
print(f"Specificity: {specificity_test:.4f}")

print("\n=== Validation Set Results ===")
print(f"Accuracy: {accuracy_validate:.4f}")
print(f"ROC-AUC Score: {roc_auc_validate:.4f}")
print(f"MCC: {mcc_validate:.4f}")
print(f"Sensitivity: {sensitivity_validate:.4f}")
print(f"Specificity: {specificity_validate:.4f}")

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('DNN_aac_training_history.csv', index=False)
print("\nTraining history saved as 'DNN_aac_training_history.csv'.")

# === 1. Confusion Matrices ===
def plot_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.show()
    print(f"Confusion Matrix saved as {filename}.")

# Test set confusion matrix
plot_confusion_matrix(cm_test, ['Negative', 'Positive'], "DNN_aac_test_confusion_matrix.png")

# Validation set confusion matrix
plot_confusion_matrix(cm_validate, ['Negative', 'Positive'], "DNN_aac_validation_confusion_matrix.png")

# === 2. ROC Curve ===
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
    print(f"ROC Curve saved as {filename}.")

# Plot ROC curve for test and validation sets
plot_roc_curve(y_test, y_test_prob, "DNN_aac_test_roc_curve.png")
plot_roc_curve(y_validate, y_validate_prob, "DNN_aac_validation_roc_curve.png")

# === 3. Accuracy and Loss vs Epochs Plots ===
def plot_accuracy_loss(history, filename_accuracy, filename_loss):
    # Accuracy plot for training and validation
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename_accuracy)
    plt.show()

    # Loss plot for training and validation
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename_loss)
    plt.show()

# Plot accuracy vs epochs and loss vs epochs for training and validation sets
plot_accuracy_loss(history, "DNN_aac_validation_accuracy_vs_epochs.png", "DNN_aac_validation_loss_vs_epochs.png")

# === 4. Accuracy and Loss vs Epochs Plots ===
def plot_accuracy_loss(history, test_accuracy, test_loss, filename_accuracy, filename_loss):
    # Accuracy plot for training, validation, and test
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot([test_accuracy] * len(history.history['accuracy']), label='Test Accuracy', linestyle='--')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename_accuracy)
    plt.show()

    # Loss plot for training, validation, and test
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot([test_loss] * len(history.history['loss']), label='Test Loss', linestyle='--')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename_loss)
    plt.show()

# Calculate the final test accuracy and loss
test_accuracy = accuracy_score(y_test, y_test_pred)
test_loss = tf.keras.losses.binary_crossentropy(y_test, y_test_prob).numpy().mean()

# Plot accuracy vs epochs and loss vs epochs for training, validation, and test sets
plot_accuracy_loss(history, test_accuracy, test_loss, "DNN_aac_test_accuracy_vs_epochs.png", "DNN_aac_test_loss_vs_epochs.png")
