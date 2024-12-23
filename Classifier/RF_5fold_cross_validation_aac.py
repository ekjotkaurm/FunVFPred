import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load the dataset ===
data = pd.read_csv('train_AAC.csv')
data = data.drop(columns=['protein_ids'], errors='ignore')
X = data.drop(columns=['label'])  # Features
y = data['label']  # Labels

# Split the data into training (70%) and a temporary dataset (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Further split the 30% temporary dataset into test (20/30 = 20%) and validation (10/30 = 10%)
test_size = 20 / 30  # Proportion of the 30% temporary dataset reserved for testing
X_test, X_validate, y_test, y_validate = train_test_split(X_temp, y_temp, test_size=test_size, random_state=42, stratify=y_temp)

# Check splits
print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_validate)}")
print(f"Test set size: {len(y_test)}")

# === 3. Perform 5-Fold Cross-Validation on the training set ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# Store metrics across folds
all_fold_metrics = []

for train_index, val_index in kf.split(X_train):
    print(f"\n=== Fold {fold} ===")

    # Split data into training and validation for each fold
    X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

    # === Train the Random Forest Model ===
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_cv, y_train_cv)

    # === Make predictions on validation fold ===
    y_val_cv_prob = rf_model.predict_proba(X_val_cv)[:, 1]
    y_val_cv_pred = rf_model.predict(X_val_cv)

    # === Evaluate metrics for the fold ===
    def calculate_metrics(y_true, y_pred, y_prob, fold):
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        mcc = matthews_corrcoef(y_true, y_pred)
        precision = classification_report(y_true, y_pred, output_dict=True)['1']['precision']
        recall = classification_report(y_true, y_pred, output_dict=True)['1']['recall']
        f1_score = classification_report(y_true, y_pred, output_dict=True)['1']['f1-score']

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Save metrics as CSV
        results_df = pd.DataFrame({
            'Metric': ['Accuracy', 'ROC-AUC', 'MCC', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity'],
            'Value': [accuracy, roc_auc, mcc, precision, recall, f1_score, sensitivity, specificity]
        })
        filename = f'RF_fold{fold}_metrics.csv'
        results_df.to_csv(filename, index=False)
        print(f"Metrics saved as {filename}")

        # Display metrics on the screen
        print(f"\n=== Fold {fold} Metrics ===")
        print(results_df.to_string(index=False))

        return accuracy, roc_auc, mcc, precision, recall, f1_score, sensitivity, specificity, cm

    # Calculate metrics for this fold
    fold_metrics = calculate_metrics(y_val_cv, y_val_cv_pred, y_val_cv_prob, fold)
    all_fold_metrics.append(fold_metrics[:-1])  # Save metrics excluding confusion matrix
    fold += 1

# === 4. Train on entire training data after cross-validation ===
rf_final_model = RandomForestClassifier(
    n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
)
rf_final_model.fit(X_train, y_train)  # Train final model on entire training set

# === 5. Evaluate on validation set ===
y_validate_prob = rf_final_model.predict_proba(X_validate)[:, 1]
y_validate_pred = rf_final_model.predict(X_validate)

# === 6. Evaluate on the test set ===
y_test_prob = rf_final_model.predict_proba(X_test)[:, 1]
y_test_pred = rf_final_model.predict(X_test)

# === 7. Compute metrics for validation and test sets ===
def calculate_final_metrics(y_true, y_pred, y_prob, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Extract precision, recall, f1-score from classification_report
    report = classification_report(y_true, y_pred, output_dict=True)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1_score = report['1']['f1-score']

    # Confusion matrix for sensitivity and specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Print metrics
    print(f"\nMetrics for {dataset_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"MCC: {mcc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")

    return accuracy, roc_auc, mcc, precision, recall, f1_score, sensitivity, specificity

# Metrics for validation set
print("\nEvaluating metrics on the validation set...")
validate_metrics = calculate_final_metrics(y_validate, y_validate_pred, y_validate_prob, "Validation Set")

# Metrics for test set
print("\nEvaluating metrics on the test set...")
test_metrics = calculate_final_metrics(y_test, y_test_pred, y_test_prob, "Test Set")

# === 8. Save final models and metrics ===
# Save metrics summary across folds
overall_metrics = pd.DataFrame(all_fold_metrics, columns=['Accuracy', 'ROC-AUC', 'MCC', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity'])
overall_metrics.loc['Mean'] = overall_metrics.mean()
overall_metrics.loc['Std'] = overall_metrics.std()
overall_metrics.to_csv('RF_AAC_5Fold_Training_Metrics.csv', index=True)
print("\nFinal cross-validation metrics saved to RF_AAC_5Fold_Training_Metrics.csv")
