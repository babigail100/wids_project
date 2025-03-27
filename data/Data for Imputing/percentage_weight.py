import pandas as pd

# Define the file paths for the categorical and quantitative datasets
cat_file_path = '/Users/noeliagarciaw/Desktop/wids/wids_project/data/TEST/TEST_CATEGORICAL.xlsx'
quant_file_path = '/Users/noeliagarciaw/Desktop/wids/wids_project/data/TEST/TEST_QUANTITATIVE_METADATA.xlsx'

# Load the datasets into DataFrames
cat_df = pd.read_excel(cat_file_path)
quant_df = pd.read_excel(quant_file_path)

# Calculate the missing value percentage for each column in the categorical dataset
missing_percent_cat = cat_df.isna().mean() * 100
print("Missing Value Percentage for Categorical Data:")
print(missing_percent_cat)

# Calculate the missing value percentage for each column in the quantitative dataset
missing_percent_quant = quant_df.isna().mean() * 100
print("\nMissing Value Percentage for Quantitative Data:")
print(missing_percent_quant)
