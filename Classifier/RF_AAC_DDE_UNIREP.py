import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  #added to enable model saving

# === 1. Load the datasets ===
train_data = pd.read_csv('train_aac_dde_unirep.csv')
test_data = pd.read_csv('test_aac_dde_unirep.csv')
validate_data = pd.read_csv('validation_aac_dde_unirep.csv')

# Drop 'protein_ids' column if it exists
train_data = train_data.drop(columns=['protein_ids'], errors='ignore')
test_data = test_data.drop(columns=['protein_ids'], errors='ignore')
validate_data = validate_data.drop(columns=['protein_ids'], errors='ignore')

# Separate features and labels
X_train = train_data.drop(columns=['label'])  # Features
y_train = train_data['label']                # Labels

X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

X_validate = validate_data.drop(columns=['label'])
y_validate = validate_data['label']

# === 2. Build and Train Random Forest Model ===
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

# === Save the trained model ===
joblib.dump(rf_model, "rf_model.joblib")
print("Random Forest model saved as 'rf_model.joblib'")

# === 3. Predictions ===
# Test set predictions
y_test_prob = rf_model.predict_proba(X_test)[:, 1]
y_test_pred = rf_model.predict(X_test)

# Validation set predictions
y_validate_prob = rf_model.predict_proba(X_validate)[:, 1]
y_validate_pred = rf_model.predict(X_validate)

# === 4. Evaluation Metrics ===
def calculate_metrics(y_true, y_pred, y_prob, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = classification_report(y_true, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_true, y_pred, output_dict=True)['1']['recall']
    f1_score = classification_report(y_true, y_pred, output_dict=True)['1']['f1-score']

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Save metrics as CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC-AUC', 'MCC', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity'],
        'Value': [accuracy, roc_auc, mcc, precision, recall, f1_score, sensitivity, specificity]
    })
    filename = f'RF_AAC_DDE_UNIREP_{dataset_name}_metrics.csv'
    results_df.to_csv(filename, index=False)
    print(f"Metrics saved as {filename}")

    # Display metrics on the screen
    print(f"\n=== {dataset_name} Set Metrics ===")
    print(results_df.to_string(index=False))

    return accuracy, roc_auc, mcc, precision, recall, f1_score, cm

# Calculate metrics for test and validation sets
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob, "Test")
validation_metrics = calculate_metrics(y_validate, y_validate_pred, y_validate_prob, "Validation")

# === 5. Plotting Functions ===
# Confusion Matrix
def plot_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.show()
    print(f"Confusion Matrix saved as {filename}.")

# Plot confusion matrices
plot_confusion_matrix(test_metrics[-1], ['Negative', 'Positive'], "RF_AAC_DDE_UNIREP_test_confusion_matrix.png")
plot_confusion_matrix(validation_metrics[-1], ['Negative', 'Positive'], "RF_AAC_DDE_UNIREP_validation_confusion_matrix.png")

# ROC Curve
def plot_roc_curve(y_true, y_prob, filename, dataset_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{dataset_name} AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f"ROC Curve - {dataset_name} Set")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    print(f"ROC Curve saved as {filename}.")

# Plot ROC curves
plot_roc_curve(y_test, y_test_prob, "RF_AAC_DDE_UNIREP_test_roc_curve.png", "Test")
plot_roc_curve(y_validate, y_validate_prob, "RF_AAC_DDE_UNIREP_validation_roc_curve.png", "Validation")

# === 6. Save Classification Reports ===
def save_classification_report(y_true, y_pred, dataset_name):
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    with open(f'RF_AAC_DDE_UNIREP_{dataset_name}_classification_report.txt', 'w') as f:
        f.write(report)
    print(f"Classification report saved as RF_AAC_DDE_UNIREP_{dataset_name}_classification_report.txt.")

# Save classification reports
save_classification_report(y_test, y_test_pred, "Test")
save_classification_report(y_validate, y_validate_pred, "Validation")
