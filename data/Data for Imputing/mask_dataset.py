import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Import the imputation methods (ensure these modules are accessible in your PYTHONPATH)
from knn_imputation import impute_knn      # your module for KNN imputation
from mice_imputation import impute_mice    # your module for MICE imputation
from random_forest_imputation import impute_random_forest  # your module for RandomForest imputation

# ----------------------------
# Step 1: Load and merge datasets for train and test
# ----------------------------
try:
    # Load categorical datasets
    train_cat = pd.read_excel("/Users/noeliagarciaw/Desktop/wids/wids_project/data/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx")
    test_cat  = pd.read_excel("/Users/noeliagarciaw/Desktop/wids/wids_project/data/TEST/TEST_CATEGORICAL.xlsx")
    
    # Load quantitative datasets
    train_quant = pd.read_excel("/Users/noeliagarciaw/Desktop/wids/wids_project/data/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx")
    test_quant  = pd.read_excel("/Users/noeliagarciaw/Desktop/wids/wids_project/data/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
except Exception as e:
    print("Error loading Excel files:", e)
    exit()

# Merge categorical and quantitative data by participant_id for train and test respectively
train_data = pd.merge(train_cat, train_quant, on="participant_id", how="outer")
test_data  = pd.merge(test_cat, test_quant, on="participant_id", how="outer")

print("Merged Train Data (first 5 rows):")
print(train_data.head())
print("\nMerged Test Data (first 5 rows):")
print(test_data.head())

# ----------------------------
# Step 2: Define manual missing percentages (values are already in percentage)
# ----------------------------
# For example, a value of 0.328947 means 0.328947% of the rows should be masked.
manual_missing = {
    "participant_id": 0.000000,
    "EHQ_EHQ_Total": 0.328947,
    "ColorVision_CV_Score": 2.960526,
    "APQ_P_APQ_P_CP": 4.934211,
    "APQ_P_APQ_P_ID": 4.934211,
    "APQ_P_APQ_P_INV": 4.934211,
    "APQ_P_APQ_P_OPD": 4.934211,
    "APQ_P_APQ_P_PM": 4.934211,
    "APQ_P_APQ_P_PP": 4.934211,
    "SDQ_SDQ_Conduct_Problems": 9.868421,
    "SDQ_SDQ_Difficulties_Total": 9.868421,
    "SDQ_SDQ_Emotional_Problems": 9.868421,
    "SDQ_SDQ_Externalizing": 9.868421,
    "SDQ_SDQ_Generating_Impact": 9.868421,
    "SDQ_SDQ_Hyperactivity": 9.868421,
    "SDQ_SDQ_Internalizing": 9.868421,
    "SDQ_SDQ_Peer_Problems": 9.868421,
    "SDQ_SDQ_Prosocial": 9.868421,
    "MRI_Track_Age_at_Scan": 0.000000
}

print("\nManual Missing Percentages for Quantitative Data:")
for col, perc in manual_missing.items():
    print(f"{col}: {perc:.6f}%")

# ----------------------------
# Step 3: Mask the merged train dataset using manual_missing percentages
# ----------------------------
masked_train = train_data.copy()
mask_dict = {}
n = len(train_data)

for col in train_data.columns:
    if col == 'participant_id':
        continue
    # Use the provided percentage and convert it to a fraction by dividing by 100.
    fraction = manual_missing.get(col, 0) / 100
    k = int(round(n * fraction))  # number of indices to mask exactly
    # Reset seed for reproducibility based on the column name
    seed = 42 + sum(ord(c) for c in col)
    np.random.seed(seed)
    indices = np.random.choice(n, size=k, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    masked_train.loc[mask, col] = np.nan
    mask_dict[col] = mask
    print(f"Column '{col}': {np.mean(mask)*100:.6f}% masked in train dataset.")

# ----------------------------
# Step 4: Save the masked merged train dataset to a new file
# ----------------------------
masked_file_path = "/Users/noeliagarciaw/Desktop/wids/wids_project/data/Data for Imputing/asked_file_path.xlsx"
try:
    masked_train.to_excel(masked_file_path, index=False)
    print(f"\nMasked train dataset saved to: {masked_file_path}")
except Exception as e:
    print("Error saving masked dataset:", e)

# ----------------------------
# Step 5: Apply the imputation methods on the masked train data
# ----------------------------
print("\nApplying KNN imputation...")
imputed_knn = impute_knn(masked_train)

print("Applying MICE imputation...")
imputed_mice = impute_mice(masked_train)

print("Applying RandomForest imputation...")
imputed_rf = impute_random_forest(masked_train)

# ----------------------------
# Step 6: Evaluate imputation accuracy (compute RMSE on masked cells inline)
# ----------------------------
rmse_knn = {}
rmse_mice = {}
rmse_rf = {}

print("\nImputation Accuracy (RMSE per column):")

print("\nKNN Imputation:")
for col, mask in mask_dict.items():
    if mask.sum() > 0:
        true_values = train_data[col][mask]
        imputed_values = imputed_knn[col][mask]
        rmse = sqrt(mean_squared_error(true_values, imputed_values))
        rmse_knn[col] = rmse
        print(f"{col}: {rmse:.4f}")

print("\nMICE Imputation:")
for col, mask in mask_dict.items():
    if mask.sum() > 0:
        true_values = train_data[col][mask]
        imputed_values = imputed_mice[col][mask]
        rmse = sqrt(mean_squared_error(true_values, imputed_values))
        rmse_mice[col] = rmse
        print(f"{col}: {rmse:.4f}")

print("\nRandomForest Imputation:")
for col, mask in mask_dict.items():
    if mask.sum() > 0:
        true_values = train_data[col][mask]
        imputed_values = imputed_rf[col][mask]
        rmse = sqrt(mean_squared_error(true_values, imputed_values))
        rmse_rf[col] = rmse
        print(f"{col}: {rmse:.4f}")

# ----------------------------
# Step 7: Compute Bias for Each Numeric Column (Imputed - True)
# ----------------------------
numeric_cols = [col for col in train_data.columns if col != 'participant_id' and pd.api.types.is_numeric_dtype(train_data[col])]

bias_knn = {}
bias_mice = {}
bias_rf = {}

for col in numeric_cols:
    mask = mask_dict[col]
    if mask.sum() > 0:
        bias_knn[col] = (imputed_knn[col][mask] - train_data[col][mask]).mean()
        bias_mice[col] = (imputed_mice[col][mask] - train_data[col][mask]).mean()
        bias_rf[col] = (imputed_rf[col][mask] - train_data[col][mask]).mean()
    else:
        bias_knn[col] = np.nan
        bias_mice[col] = np.nan
        bias_rf[col] = np.nan

print("\nAverage Bias (Imputed - True) for Numeric Columns:")
for col in numeric_cols:
    print(f"{col}: KNN = {bias_knn[col]:.4f}, MICE = {bias_mice[col]:.4f}, RandomForest = {bias_rf[col]:.4f}")

# ----------------------------
# Step 8: Visualize Bias for Each Numeric Column in One Grouped Bar Chart
# ----------------------------
cols = numeric_cols
knn_biases = [bias_knn[col] for col in cols]
mice_biases = [bias_mice[col] for col in cols]
rf_biases = [bias_rf[col] for col in cols]

x = np.arange(len(cols))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, knn_biases, width, label="KNN", color='blue')
plt.bar(x, mice_biases, width, label="MICE", color='orange')
plt.bar(x + width, rf_biases, width, label="RandomForest", color='green')

plt.xlabel("Numeric Columns")
plt.ylabel("Average Bias (Imputed - True)")
plt.title("Average Bias for Each Numeric Column by Imputation Method")
plt.xticks(x, cols, rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=0.8)
plt.legend(title="Method")
plt.tight_layout()
plt.show()


# ----------------------------
# Step 9: Determine Best Imputation Method Based on Bias
# ----------------------------

avg_bias_knn = np.nanmean([abs(bias_knn[col]) for col in bias_knn])
avg_bias_mice = np.nanmean([abs(bias_mice[col]) for col in bias_mice])
avg_bias_rf = np.nanmean([abs(bias_rf[col]) for col in bias_rf])

bias_scores = {
    'KNN': avg_bias_knn,
    'MICE': avg_bias_mice,
    'RandomForest': avg_bias_rf
}
best_method = min(bias_scores, key=bias_scores.get)

print("\nAverage Absolute Bias per Method:")
for method, bias in bias_scores.items():
    print(f"{method}: {bias:.6f}")
print(f"\n\u2705 Best method based on lowest average absolute bias: {best_method}")

# ----------------------------
# Step 10: Impute Full Train and Test Using Best Method
# ----------------------------

print("\nApplying best method to full train and test datasets...")

if best_method == "KNN":
    final_train = impute_knn(masked_train)
    final_test = impute_knn(test_data)
elif best_method == "MICE":
    final_train = impute_mice(masked_train)
    final_test = impute_mice(test_data)
else:
    final_train = impute_random_forest(masked_train)
    final_test = impute_random_forest(test_data)

# ----------------------------
# Step 11: Save Final Imputed Train and Test Datasets
# ----------------------------

train_out_path = "/Users/noeliagarciaw/Desktop/wids/wids_project/data/Data for Imputing/train_out_path.xlsx"
test_out_path  = "/Users/noeliagarciaw/Desktop/wids/wids_project/data/Data for Imputing/test_out_path.xlsx"

try:
    final_train.to_excel(train_out_path, index=False)
    final_test.to_excel(test_out_path, index=False)
    print(f"\n\u2705 Final imputed train data saved to: {train_out_path}")
    print(f"\u2705 Final imputed test data saved to: {test_out_path}")
except Exception as e:
    print("\u274C Error saving final datasets:", e)