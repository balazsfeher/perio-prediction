# Machine learning predicts patient clinical responses to periodontal treatment 
# Submitted to the Journal of Periodontology
#
# Balazs Feher, Eduardo H. de Souza Oliveira, Poliana Duarte, Andreas A. Werdich, William V. Giannobile, Magda Feres
# Harvard School of Dental Medicine
# balazs_feher@hsdm.harvard.edu
# 
# Last updated February 3, 2025

import os
from pathlib import Path

import pip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, precision_recall_curve, average_precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.utils import resample
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import mutual_info_classif

# PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
Path(output_dir).mkdir(parents=True, exist_ok=True)

# TRAINING/INTERNAL TESTING DATASET
df = pd.read_csv('synthdata_internal.csv') # Synthetic dataset generated from South American data

# PREDICTORS AT BASELINE, OUTCOME AT FOLLOW-UP
X = df[['Pt_Age', 'Pt_Gender', # Demographic
        'Tx_AMX', 'Tx_MTZ', 'Tx_Duration', # Treatment-related
        'N_PDmax4', 'N_PDmin5', 'N_CALmax4', 'N_CALmin5', 'BoP', 'PI', # Clinical
        'MB_A', 'MB_P', 'MB_Y', 'MB_G', 'MB_O', 'MB_R', 'MB_M']] # Microbiological
y = df['Endpoint_1Y'] # Clinical endpoint at the 1-year follow-up (binary, see Feres et al., 2020)

# MODEL DEFINITION
RF_model = RandomForestClassifier(
    n_estimators=200, bootstrap=True, oob_score=True, max_depth=5,
    max_features=10, max_leaf_nodes=20, min_samples_leaf=8, 
    min_samples_split=10, random_state=1
)

# MODEL TRAINING
RF_model.fit(X, y)

# OUT-OF-BAG SAMPLES AND PLOT (Figure 2D)
n_samples = len(X)
for i, estimator in enumerate(RF_model.estimators_):
    random_state = estimator.random_state
    bootstrap_indices = resample(range(n_samples), replace=True, n_samples=n_samples, random_state=random_state)
    oob_indices = np.setdiff1d(range(n_samples), bootstrap_indices) 
    print(f"Tree {i + 1}: {len(oob_indices)} OOB samples")

oob_errors = []
for i in range(1, len(RF_model.estimators_) + 1):
    sub_forest = RandomForestClassifier(
        n_estimators=i,
        bootstrap=True,
        oob_score=True,
        max_depth=5,
        max_features=10,
        max_leaf_nodes=20,
        min_samples_leaf=8,
        min_samples_split=10,
        random_state=1
    )
    sub_forest.fit(X, y)
    oob_errors.append(1 - sub_forest.oob_score_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(RF_model.estimators_) + 1), oob_errors, marker='o', linestyle='-', label='OOB Error')
plt.title('Out-of-bag error plot')
plt.xlabel('Trees')
plt.ylabel('Out-of-bag error')
plt.grid(True)
plt.legend()
plt.show()

# MODEL PREDICTIONS
y_pred = RF_model.predict(X)
y_proba = RF_model.predict_proba(X)[:, 1]

# CALIBRATION CURVE (Figure 2E)
prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
internal_model = LinearRegression().fit(prob_pred.reshape(-1, 1), prob_true)
internal_slope = internal_model.coef_[0]
print(f"Calibration Slope (Internal Dataset): {internal_slope:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label="RF")
plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
plt.title("Calibration curve")
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.legend()
plt.grid()
plt.show()

# PERFORMANCE METRICS
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_proba)
auprc = average_precision_score(y, y_proba)
conf_matrix = confusion_matrix(y, y_pred)
oob_score = RF_model.oob_score_

print('Internal Testing:')
print(f'OOB Score: {oob_score:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1-Score: {f1:.3f}')
print(f'AUROC: {roc_auc:.3f}')
print(f'AUPRC: {auprc:.3f}')

# INTERNAL DATASET PREDICTION DISTRIBUTION (Figure 2F)
cases_internal = y_proba[y == 1]  
controls_internal = y_proba[y == 0]  

plt.figure(figsize=(10, 6))
sns.kdeplot(cases_internal, label='Endpoint achieved', color='blue', shade=True, bw_adjust=0.5)
sns.kdeplot(controls_internal, label='Endpoint not achieved', color='deepskyblue', shade=True, bw_adjust=0.5)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision threshold (0.5)')
plt.title('Prediction distribution')
plt.xlabel('Predicted probability')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# CONFUSION MATRIX (Figure 2C)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Endpoint not achieved', 'Endpoint achieved'])
disp.plot(cmap='YlOrRd')
plt.title('Confusion Matrix')
plt.xlabel('Predicted outcome')
plt.ylabel('True outcome')
plt.show()

# ROC CURVE (Figure 2A)
RocCurveDisplay.from_estimator(RF_model, X, y, name="RF")
plt.title('Receiver operating characteristics curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()

# PR CURVE (Figure 2B)
disp_pr = PrecisionRecallDisplay.from_estimator(RF_model, X, y, name="RF")
plt.title('Precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
precision_gain = precisions / (1 - precisions + 1e-6)
recall_gain = recalls / (1 - recalls + 1e-6)

plt.figure(figsize=(8, 6))
plt.plot(recall_gain[:-1], precision_gain[:-1], label="Precision-Recall Gain Curve")
plt.title("Precision-Recall Gain Curve")
plt.xlabel("Recall Gain")
plt.ylabel("Precision Gain")
plt.grid()
plt.legend()
plt.show()

# RELATIVE FEATURE IMPORTANCES (Figure 3)
importances = RF_model.feature_importances_
feature_names = X.columns

importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importances_df = importances_df.sort_values(by='Importance', ascending=False)
print(importances_df)

plt.figure(figsize=(10, 6))
plt.bar(importances_df['Feature'], importances_df['Importance'], color='green')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Relative Feature Importances')
plt.xticks(rotation=45)
plt.show()

# PARTIAL DEPENDENCE PLOTS (Figure 4)
features_to_plot = ['Tx_MTZ', 'N_PDmin5', 'N_CALmin5', 'Tx_Duration']

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
pdp_disp = PartialDependenceDisplay.from_estimator(RF_model, X, features_to_plot, grid_resolution=50, ax=ax)
ax[0, 0].set_title(r'Dependence on metronidazole dosage')
ax[0, 1].set_title(r'Dependence on deep sites')
ax[1, 0].set_title(r'Dependence on attachment loss')
ax[1, 1].set_title(r'Dependence on antibiotic duration')
ax[0, 0].set_xlabel(r'Metronidazole dosage, $\it{g \cdot d^{-1}}$')
ax[0, 1].set_xlabel('Sites with PPD ≥ 5 mm')
ax[1, 0].set_xlabel('Sites with CAL ≥ 5 mm')
ax[1, 1].set_xlabel(r'Antibiotic duration, $\it{d}$')
ax[0, 0].set_ylabel(r'Endpoint probability')
ax[0, 1].set_ylabel(r'Endpoint probability')
ax[1, 0].set_ylabel(r'Endpoint probability')
ax[1, 1].set_ylabel(r'Endpoint probability')
plt.subplots_adjust(hspace=0.5)
plt.show()

# 2D PARTIAL DEPENDENCE PLOTS (Figure 4)
features_to_plot_2d = [('Tx_MTZ', 'N_PDmin5')]  # Pair of features for 2D plot

fig, ax = plt.subplots(figsize=(8, 6))
pdp2d1 = PartialDependenceDisplay.from_estimator(RF_model, X, features_to_plot_2d, grid_resolution=50, ax=ax)
ax.set_title('Dependence on metronidazole dosage and deep sites')
ax.set_xlabel(r'Metronidazole dosage, $\it{g \cdot d^{-1}}$')
ax.set_ylabel('Sites with PPD ≥ 5 mm')
plt.show()

features_to_plot_2d_3 = [('Tx_Duration', 'Tx_MTZ')]  # Pair of features for 2D plot

fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(RF_model, X, features_to_plot_2d_3, grid_resolution=50, ax=ax)
plt.title('Dependence on antibiotic duration and metronidazole dosage')
plt.show()

# EXTERNAL TESTING DATASET
df_val = pd.read_csv('valdata_full_new.csv') # North America/Europe

# PREDICTORS AT BASELINE, OUTCOME AT FOLLOW-UP 
X_val = df_val[['Pt_Age', 'Pt_Gender',  
                'Tx_AMX', 'Tx_MTZ', 'Tx_Duration', 
                'N_PDmax4', 'N_PDmin5', 'N_CALmax4', 'N_CALmin5', 'BoP', 'PI',
                'MB_A', 'MB_P', 'MB_Y', 'MB_G', 'MB_O', 'MB_R', 'MB_M']]
y_val = df_val['Endpoint_1Y']

# MODEL PREDICTIONS ON EXTERNAL DATASET
y_val_pred = RF_model.predict(X_val)
y_val_proba = RF_model.predict_proba(X_val)[:, 1]

# CALIBRATION CURVE ON EXTERNAL DATASET (Figure 5D)
val_prob_true, val_prob_pred = calibration_curve(y_val, y_val_proba, n_bins=10)
external_model = LinearRegression().fit(val_prob_pred.reshape(-1, 1), val_prob_true)
external_slope = external_model.coef_[0]
print(f"Calibration Slope (External Dataset): {external_slope:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(val_prob_pred, val_prob_true, marker='o', label="RF (ext. testing)")
plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
plt.title("Calibration curve")
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.legend()
plt.grid()
plt.show()

# EXTERNAL DATASET PREDICTION DISTRIBUTION (Figure 5E)
cases_external = y_val_proba[y_val == 1]  
controls_external = y_val_proba[y_val == 0]

plt.figure(figsize=(10, 6))
sns.kdeplot(cases_external, label='Endpoint achieved)', color='maroon', shade=True, bw_adjust=0.5)
sns.kdeplot(controls_external, label='Endpoint not achieved)', color='lightcoral', shade=True, bw_adjust=0.5)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision threshold (0.5)')
plt.title('Prediction distribution')
plt.xlabel('Predicted probability')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# PERFORMANCE METRICS ON EXTERNAL DATASET
precision_val = precision_score(y_val, y_val_pred)
recall_val = recall_score(y_val, y_val_pred)
f1_val = f1_score(y_val, y_val_pred)
roc_auc_val = roc_auc_score(y_val, y_val_proba)
auprc_val = average_precision_score(y_val, y_val_proba) 
conf_matrix_val = confusion_matrix(y_val, y_val_pred)

print('External Testing:')
print(f'Precision: {precision_val:.3f}')
print(f'Recall: {recall_val:.3f}')
print(f'F1-Score: {f1_val:.3f}')
print(f'AUROC: {roc_auc_val:.3f}')
print(f'AUPRC: {auprc_val:.3f}')

# CONFUSION MATRIX ON EXTERNAL DATASET (Figure 5C)
disp_val = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_val, display_labels=['Endpoint not achieved', 'Endpoint achieved'])
disp_val.plot(cmap='YlOrRd')
plt.title('Confusion Matrix (External Dataset)')
plt.xlabel('Predicted outcome')
plt.ylabel('True outcome')
plt.show()

# ROC CURVE ON EXTERNAL DATASET (Figure 5A)
RocCurveDisplay.from_estimator(RF_model, X_val, y_val, name="RF (ext. testing)")
plt.title('Receiver operating characteristics curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()

# PR CURVE ON EXTERNAL DATASET (Figure 5B)
disp_pr_val = PrecisionRecallDisplay.from_estimator(RF_model, X_val, y_val, name="RF (ext. testing)")
plt.title('Precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# SUBGROUP ANALYSIS (Supplementary Tables 2–3)
def perform_quartile_analysis_with_ranges(X, y, dataset_name):
    # Quartiles are built based on the number of sites with CAL ≥ 5 mm based on clinical relevance
    quartiles, bins = pd.qcut(X['N_CALmin5'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], retbins=True)
    X['N_CALmin5_Quartile'] = quartiles

    print(f"{dataset_name} Dataset - N_CALmin5 Quartile Boundaries:")
    for i in range(len(bins) - 1):
        print(f"Q{i+1}: {bins[i]:.2f} to {bins[i+1]:.2f}")
    
    subgroup_results = []
    
    for quartile in X['N_CALmin5_Quartile'].unique():
        X_subgroup = X[X['N_CALmin5_Quartile'] == quartile]
        y_subgroup = y[X['N_CALmin5_Quartile'] == quartile]
        
        if len(y_subgroup) > 0:
            y_pred_subgroup = RF_model.predict(X_subgroup.drop(columns=['N_CALmin5_Quartile']))
            y_proba_subgroup = RF_model.predict_proba(X_subgroup.drop(columns=['N_CALmin5_Quartile']))[:, 1]
            
            precision = precision_score(y_subgroup, y_pred_subgroup)
            recall = recall_score(y_subgroup, y_pred_subgroup)
            f1 = f1_score(y_subgroup, y_pred_subgroup)
            roc_auc = roc_auc_score(y_subgroup, y_proba_subgroup) if len(np.unique(y_subgroup)) > 1 else np.nan
            auprc = average_precision_score(y_subgroup, y_proba_subgroup)
            
            subgroup_results.append({
                'Dataset': dataset_name,
                'Quartile': quartile,
                'Size': len(y_subgroup),
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'AUPRC': auprc,
            })
    
    return pd.DataFrame(subgroup_results)

internal_results = perform_quartile_analysis_with_ranges(X.copy(), y, dataset_name='Internal')
external_results = perform_quartile_analysis_with_ranges(X_val.copy(), y_val, dataset_name='External')
combined_results = pd.concat([internal_results, external_results])
combined_results.to_csv('quartile_analysis_results.csv', index=False)
print(combined_results)