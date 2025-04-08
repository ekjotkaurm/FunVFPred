import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# Load the datasets
train_data = pd.read_csv('train_AAC.csv')
test_data = pd.read_csv('test_AAC.csv')
validate_data = pd.read_csv('validation_AAC.csv')

# Drop the 'protein_ids' column (if exists)
train_data = train_data.drop(columns=['protein_ids'])
test_data = test_data.drop(columns=['protein_ids'])
validate_data = validate_data.drop(columns=['protein_ids'])

# Separate features and labels
X_train = train_data.drop(columns=['label'])  # Replace 'label' with the actual column name for your target
y_train = train_data['label']

X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

X_validate = validate_data.drop(columns=['label'])
y_validate = validate_data['label']

# === 1. Class Imbalance Handling ===
# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# === 2. Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validate_scaled = scaler.transform(X_validate)

# === 3. Define the MLP Model ===
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # First hidden layer with 128 neurons
    Dense(32, activation='relu'),                               # Second hidden layer with 64 neurons
    Dense(16, activation='relu'),                               # Third hidden layer with 32 neurons
    Dense(1, activation='sigmoid')                              # Output layer (binary classification)
])

# === 4. Compile the Model ===
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# === 5. Train the Model ===
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, class_weight=class_weight_dict, verbose=2)

# === 6. Evaluate Model on Test Set ===
y_test_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")  # Predict binary classes
y_test_prob = model.predict(X_test_scaled)  # Probability for ROC-AUC

# === 7. Test Set Evaluation Metrics ===
print("=== Test Set Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_test_prob))

# === 8. Evaluate Model on Validation Set ===
y_validate_pred = (model.predict(X_validate_scaled) > 0.5).astype("int32")
y_validate_prob = model.predict(X_validate_scaled)

# === 9. Validation Set Evaluation Metrics ===
print("\n=== Validation Set Evaluation ===")
print("Accuracy:", accuracy_score(y_validate, y_validate_pred))
print("Classification Report:\n", classification_report(y_validate, y_validate_pred))
print("ROC-AUC Score:", roc_auc_score(y_validate, y_validate_prob))

# === 10. ROC Curve ===
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
    plt.show()  # Display the plot
    plt.close()
    print(f"ROC Curve saved as {filename}.")

# Plot and save ROC curves
plot_roc_curve(y_test, y_test_prob, "MLP_aac_roc_curve_test_set.png")
plot_roc_curve(y_validate, y_validate_prob, "MLP_aac_roc_curve_validation_set.png")

# === 11. Confusion Matrix ===
def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Negative', 'Positive']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.show()  # Display the plot
    plt.close()
    print(f"Confusion Matrix saved as {filename}.")

# Plot and save confusion matrices
plot_confusion_matrix(y_test, y_test_pred, "MLP_aac_confusion_matrix_test_set.png")
plot_confusion_matrix(y_validate, y_validate_pred, "MLP_aac_confusion_matrix_validation_set.png")

# === 12. Classification Report ===
def save_classification_report(y_true, y_pred, filename):
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    with open(filename, "w") as f:
        f.write("Classification Report\n")
        f.write(report)
    print(f"Classification Report saved as {filename}.")
    print(f"\nClassification Report:\n{report}")

# Save classification reports
save_classification_report(y_test, y_test_pred, "MLP_aac_classification_report_test_set.txt")
save_classification_report(y_validate, y_validate_pred, "MLP_aac_classification_report_validation_set.txt")

# === 13. Precision, Recall, and F1 Score ===
def save_precision_recall_f1(y_true, y_pred, filename):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    with open(filename, "w") as f:
        f.write("Precision, Recall, F1 Score\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
    print(f"Precision, Recall, and F1 Score saved as {filename}.")
    print(f"\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Save precision, recall, and F1 score
save_precision_recall_f1(y_test, y_test_pred, "MLP_aac_precision_recall_f1_test_set.txt")
save_precision_recall_f1(y_validate, y_validate_pred, "MLP_aac_precision_recall_f1_validation_set.txt")

# === 14. TP, TN, FP, FN ===
def save_tp_tn_fp_fn(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nResults saved in {filename}:")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    with open(filename, "w") as f:
        f.write("Confusion Matrix Analysis\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
    print(f"Confusion matrix analysis saved as '{filename}'.")

# Save TP, TN, FP, FN for test and validation sets
save_tp_tn_fp_fn(y_test, y_test_pred, "MLP_aac_tp_tn_fp_fn_test_set.txt")
save_tp_tn_fp_fn(y_validate, y_validate_pred, "MLP_aac_tp_tn_fp_fn_validation_set.txt")

from sklearn.metrics import matthews_corrcoef

# ===  Calculate MCC, Sensitivity, and Specificity ===
def calculate_metrics(y_true, y_pred, filename):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # Sensitivity (Recall or True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Save results to file
    with open(filename, "w") as f:
        f.write("Metrics Results\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}\n")
        f.write(f"Sensitivity (True Positive Rate): {sensitivity:.2f}\n")
        f.write(f"Specificity (True Negative Rate): {specificity:.2f}\n")

    # Print results
    print(f"\nMetrics saved in {filename}:")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}")
    print(f"Specificity (True Negative Rate): {specificity:.2f}")

# ===  Calculate and Save Metrics for Test Set ===
calculate_metrics(y_test, y_test_pred, "MLP_aac_metrics_test_set.txt")

# ===  Calculate and Save Metrics for Validation Set ===
calculate_metrics(y_validate, y_validate_pred, "MLP_aac_metrics_validation_set.txt")
