import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

# Import your imputation methods (you should create these modules separately)
from knn_imputation import impute_knn      # (your module for KNN imputation)
from mice_imputation import impute_mice    # (your module for MICE imputation)
from missing_forest_imputation import impute_missing_forest  # (see code below)

def mask_entire_dataset(df, missing_fraction=0.2, random_state=42):
    """
    Randomly masks a fraction of cells in all columns (except 'participant_id').
    
    Parameters:
      df              : Input complete DataFrame.
      missing_fraction: Fraction (0 to 1) of cells to mask in each column.
      random_state    : Seed for reproducibility.
      
    Returns:
      df_masked : DataFrame with NaNs inserted.
      mask_dict : Dictionary with a boolean mask for each column.
    """
    np.random.seed(random_state)
    df_masked = df.copy()
    mask_dict = {}
    
    for col in df.columns:
        if col == 'participant_id':
            continue
        mask = np.random.rand(len(df)) < missing_fraction
        df_masked.loc[mask, col] = np.nan
        mask_dict[col] = mask
        
    return df_masked, mask_dict

def evaluate_imputation(original_df, imputed_df, mask_dict):
    """
    Computes RMSE between original and imputed values for the masked cells in each column.
    """
    results = {}
    for col, mask in mask_dict.items():
        true_values = original_df.loc[mask, col]
        imputed_values = imputed_df.loc[mask, col]
        if len(true_values) > 0:
            rmse = sqrt(mean_squared_error(true_values, imputed_values))
            results[col] = rmse
    return results

def main():
    # ----------------------------
    # Step 1: Load and merge categorical datasets
    # ----------------------------
    try:
        train_cat = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/Data for Imputing/Categorical Data for Imputing/TRAIN_CATEGORICAL_METADATA.xlsx")
        test_cat  = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/Data for Imputing/Categorical Data for Imputing/TEST_CATEGORICAL.xlsx")
    except Exception as e:
        print("Error loading Excel files:", e)
        return

    merged_cat = pd.merge(train_cat, test_cat, on="participant_id", how="outer")
    print("Merged Categorical Data (first 5 rows):")
    print(merged_cat.head())
    
    # ----------------------------
    # Step 2: Mask the merged dataset
    # ----------------------------
    missing_fraction = 0.2  # 20% missing in each column
    masked_df, mask_dict = mask_entire_dataset(merged_cat, missing_fraction, random_state=42)
    
    print("\nMasked Dataset (first 5 rows):")
    print(masked_df.head())
    for col, mask in mask_dict.items():
        print(f"Column '{col}': {np.mean(mask)*100:.1f}% masked.")
    
    # ----------------------------
    # Step 3: Apply imputation methods
    # ----------------------------
    print("\nApplying KNN imputation...")
    imputed_knn = impute_knn(masked_df)
    
    print("Applying MICE imputation...")
    imputed_mice = impute_mice(masked_df)
    
    print("Applying MissForest imputation...")
    imputed_mf = impute_missing_forest(masked_df)
    
    # ----------------------------
    # Step 4: Evaluate imputation accuracy (RMSE on masked cells)
    # ----------------------------
    rmse_knn = evaluate_imputation(merged_cat, imputed_knn, mask_dict)
    rmse_mice = evaluate_imputation(merged_cat, imputed_mice, mask_dict)
    rmse_mf   = evaluate_imputation(merged_cat, imputed_mf, mask_dict)
    
    print("\nImputation Accuracy (RMSE per column):")
    print("\nKNN Imputation:")
    for col, rmse in rmse_knn.items():
        print(f"{col}: {rmse:.4f}")
    
    print("\nMICE Imputation:")
    for col, rmse in rmse_mice.items():
        print(f"{col}: {rmse:.4f}")
    
    print("\nMissForest Imputation:")
    for col, rmse in rmse_mf.items():
        print(f"{col}: {rmse:.4f}")

if __name__ == '__main__':
    main()
