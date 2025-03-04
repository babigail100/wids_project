import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

# 1. Read in the data
train_cat = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/Data for Imputing/Categorical Data for Imputing/TRAIN_CATEGORICAL_METADATA.xlsx")
test_cat = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/Data for Imputing/Categorical Data for Imputing/TEST_CATEGORICAL.xlsx")
train_quant = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx")
test_quant = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/Data for Imputing/Quanitative Data for Imputing/TEST_QUANTITATIVE_METADATA.xlsx")


# 2. Quick Numerical Missing Check
print("=== Missing Values: Train Categorical ===")
print(train_cat.isnull().sum(), "\n")

print("=== Missing Values: Test Categorical ===")
print(test_cat.isnull().sum(), "\n")

print("=== Missing Values: Train Quantitative ===")
print(train_quant.isnull().sum(), "\n")

print("=== Missing Values: Test Quantitative ===")
print(test_quant.isnull().sum(), "\n")


# 3. Visual Inspection of Missingness Using Missingno
# Categorical Data Visuals
msno.matrix(train_cat, figsize=(10,4), fontsize=8, color=(0.2, 0.6, 0.9))
plt.title("Missingness Matrix - Train Categorical")
plt.show()

msno.matrix(test_cat, figsize=(10,4), fontsize=8, color=(0.2, 0.6, 0.9))
plt.title("Missingness Matrix - Test Categorical")
plt.show()

msno.bar(train_cat, figsize=(10,4), fontsize=8, color='green')
plt.title("Missingness Bar - Train Categorical")
plt.show()

msno.bar(test_cat, figsize=(10,4), fontsize=8, color='green')
plt.title("Missingness Bar - Test Categorical")
plt.show()

# Quantitative Data Visuals
msno.matrix(train_quant, figsize=(10,4), fontsize=8, color=(0.9, 0.3, 0.3))
plt.title("Missingness Matrix - Train Quantitative")
plt.show()

msno.matrix(test_quant, figsize=(10,4), fontsize=8, color=(0.9, 0.3, 0.3))
plt.title("Missingness Matrix - Test Quantitative")
plt.show()

msno.bar(train_quant, figsize=(10,4), fontsize=8, color='orange')
plt.title("Missingness Bar - Train Quantitative")
plt.show()

msno.bar(test_quant, figsize=(10,4), fontsize=8, color='orange')
plt.title("Missingness Bar - Test Quantitative")
plt.show()


# 4. Compare Missingness Between Train and Test Sets
# Categorical Data Comparison
common_cat_cols = set(train_cat.columns).intersection(set(test_cat.columns))
cat_missing_comparison = []
for col in sorted(common_cat_cols):
    train_missing = train_cat[col].isnull().sum()
    test_missing = test_cat[col].isnull().sum()
    cat_missing_comparison.append([col, train_missing, test_missing])
    
cat_missing_df = pd.DataFrame(cat_missing_comparison, 
                              columns=["Column", "Train_Cat_Missing", "Test_Cat_Missing"])
print("=== Comparison of Missing Values (Categorical) ===")
print(cat_missing_df)

if not cat_missing_df.empty:
    cat_missing_df_melted = cat_missing_df.melt(id_vars="Column", 
                                                value_vars=["Train_Cat_Missing", "Test_Cat_Missing"], 
                                                var_name="Dataset", value_name="MissingCount")
    plt.figure(figsize=(12,6))
    sns.barplot(data=cat_missing_df_melted, x="Column", y="MissingCount", hue="Dataset")
    plt.xticks(rotation=90)
    plt.title("Missing Count Comparison (Categorical: Train vs Test)")
    plt.tight_layout()
    plt.show()

# Quantitative Data Comparison
common_quant_cols = set(train_quant.columns).intersection(set(test_quant.columns))
quant_missing_comparison = []
for col in sorted(common_quant_cols):
    train_missing = train_quant[col].isnull().sum()
    test_missing = test_quant[col].isnull().sum()
    quant_missing_comparison.append([col, train_missing, test_missing])
    
quant_missing_df = pd.DataFrame(quant_missing_comparison, 
                                columns=["Column", "Train_Quant_Missing", "Test_Quant_Missing"])
print("\n=== Comparison of Missing Values (Quantitative) ===")
print(quant_missing_df)

if not quant_missing_df.empty:
    quant_missing_df_melted = quant_missing_df.melt(id_vars="Column", 
                                                    value_vars=["Train_Quant_Missing", "Test_Quant_Missing"], 
                                                    var_name="Dataset", value_name="MissingCount")
    plt.figure(figsize=(12,6))
    sns.barplot(data=quant_missing_df_melted, x="Column", y="MissingCount", hue="Dataset")
    plt.xticks(rotation=90)
    plt.title("Missing Count Comparison (Quantitative: Train vs Test)")
    plt.tight_layout()
    plt.show()

#4. Merge Train and Test Datasets Separately by 'participant_id'
# To avoid column name clashes and to easily differentiate the data sources,
# we add a prefix for columns (except 'participant_id').

def rename_columns(df, prefix):
    new_cols = {col: f"{prefix}_{col}" for col in df.columns if col != "participant_id"}
    return df.rename(columns=new_cols)

# Rename columns for train and test datasets separately
train_cat_renamed = rename_columns(train_cat, "train_cat")
train_quant_renamed = rename_columns(train_quant, "train_quant")
test_cat_renamed = rename_columns(test_cat, "test_cat")
test_quant_renamed = rename_columns(test_quant, "test_quant")

# Merge train datasets (categorical with quantitative) on 'participant_id'
merged_train = pd.merge(train_cat_renamed, train_quant_renamed, on="participant_id", how="outer")

# Merge test datasets (categorical with quantitative) on 'participant_id'
merged_test = pd.merge(test_cat_renamed, test_quant_renamed, on="participant_id", how="outer")

print("\n=== Merged Train Data Missing Values Summary ===")
print(merged_train.isnull().sum())

print("\n=== Merged Test Data Missing Values Summary ===")
print(merged_test.isnull().sum())

# Visualize missingness for merged datasets
msno.matrix(merged_train, figsize=(12,6), fontsize=8, color=(0.3, 0.7, 0.5))
plt.title("Missingness Matrix - Merged Train Data")
plt.show()

msno.matrix(merged_test, figsize=(12,6), fontsize=8, color=(0.7, 0.3, 0.5))
plt.title("Missingness Matrix - Merged Test Data")
plt.show()


# 5. Analyze Missingness by Participant for Each Merged Dataset
# For each merged dataset, we separate the columns back into categorical and quantitative groups
# based on the prefixes we added.

# For Train Data:
train_cat_cols = [col for col in merged_train.columns if col.startswith("train_cat_")]
train_quant_cols = [col for col in merged_train.columns if col.startswith("train_quant_")]

# Calculate missing counts per participant
merged_train['cat_missing_count'] = merged_train[train_cat_cols].isnull().sum(axis=1)
merged_train['quant_missing_count'] = merged_train[train_quant_cols].isnull().sum(axis=1)

# Boolean indicators for missingness
merged_train['cat_missing'] = merged_train['cat_missing_count'] > 0
merged_train['quant_missing'] = merged_train['quant_missing_count'] > 0

print("\n=== Participants in Merged Train Data with Missing Data in Both Categorical and Quantitative Fields ===")
missing_both_train = merged_train[(merged_train['cat_missing']) & (merged_train['quant_missing'])]
print(missing_both_train[['participant_id', 'cat_missing_count', 'quant_missing_count']])


# For Test Data:
test_cat_cols = [col for col in merged_test.columns if col.startswith("test_cat_")]
test_quant_cols = [col for col in merged_test.columns if col.startswith("test_quant_")]

merged_test['cat_missing_count'] = merged_test[test_cat_cols].isnull().sum(axis=1)
merged_test['quant_missing_count'] = merged_test[test_quant_cols].isnull().sum(axis=1)

merged_test['cat_missing'] = merged_test['cat_missing_count'] > 0
merged_test['quant_missing'] = merged_test['quant_missing_count'] > 0

print("\n=== Participants in Merged Test Data with Missing Data in Both Categorical and Quantitative Fields ===")
missing_both_test = merged_test[(merged_test['cat_missing']) & (merged_test['quant_missing'])]
print(missing_both_test[['participant_id', 'cat_missing_count', 'quant_missing_count']])


# Plot Only Columns with Missingness ---
# For Merged Train Data:
cols_with_missing_train = merged_train.columns[merged_train.isnull().sum() > 0]
merged_train_missing_only = merged_train[cols_with_missing_train]

msno.matrix(merged_train_missing_only, figsize=(12,6), fontsize=8, color=(0.3, 0.7, 0.5))
plt.title("Missingness Matrix - Merged Train Data (Only Columns with Missingness)")
plt.show()

# For Merged Test Data:
cols_with_missing_test = merged_test.columns[merged_test.isnull().sum() > 0]
merged_test_missing_only = merged_test[cols_with_missing_test]

msno.matrix(merged_test_missing_only, figsize=(12,6), fontsize=8, color=(0.7, 0.3, 0.5))
plt.title("Missingness Matrix - Merged Test Data (Only Columns with Missingness)")
plt.show()


# 5. Analyze Missingness by Participant for Each Merged Dataset
# For Train Data:
train_cat_cols = [col for col in merged_train.columns if col.startswith("train_cat_")]
train_quant_cols = [col for col in merged_train.columns if col.startswith("train_quant_")]

# Calculate missing counts per participant
merged_train['cat_missing_count'] = merged_train[train_cat_cols].isnull().sum(axis=1)
merged_train['quant_missing_count'] = merged_train[train_quant_cols].isnull().sum(axis=1)

# Boolean indicators for missingness
merged_train['cat_missing'] = merged_train['cat_missing_count'] > 0
merged_train['quant_missing'] = merged_train['quant_missing_count'] > 0

print("\n=== Participants in Merged Train Data with Missing Data in Both Categorical and Quantitative Fields ===")
missing_both_train = merged_train[(merged_train['cat_missing']) & (merged_train['quant_missing'])]
print(missing_both_train[['participant_id', 'cat_missing_count', 'quant_missing_count']])

# For Test Data:
test_cat_cols = [col for col in merged_test.columns if col.startswith("test_cat_")]
test_quant_cols = [col for col in merged_test.columns if col.startswith("test_quant_")]

merged_test['cat_missing_count'] = merged_test[test_cat_cols].isnull().sum(axis=1)
merged_test['quant_missing_count'] = merged_test[test_quant_cols].isnull().sum(axis=1)

merged_test['cat_missing'] = merged_test['cat_missing_count'] > 0
merged_test['quant_missing'] = merged_test['quant_missing_count'] > 0

print("\n=== Participants in Merged Test Data with Missing Data in Both Categorical and Quantitative Fields ===")
missing_both_test = merged_test[(merged_test['cat_missing']) & (merged_test['quant_missing'])]
print(missing_both_test[['participant_id', 'cat_missing_count', 'quant_missing_count']])

# --- Extra Chart: Plot Only Rows with Missing Values ---
# Filter rows that have any missing values
merged_train_missing_rows = merged_train[merged_train.isnull().any(axis=1)]
merged_test_missing_rows  = merged_test[merged_test.isnull().any(axis=1)]

# Plot missingness for only those rows in the merged train dataset
msno.matrix(merged_train_missing_rows, figsize=(12,6), fontsize=8, color=(0.3, 0.7, 0.5))
plt.title("Merged Train Data (Rows with Missingness)")
plt.show()

# Plot missingness for only those rows in the merged test dataset
msno.matrix(merged_test_missing_rows, figsize=(12,6), fontsize=8, color=(0.7, 0.3, 0.5))
plt.title("Merged Test Data (Rows with Missingness)")
plt.show()