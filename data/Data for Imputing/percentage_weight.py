import pandas as pd

def main():
    # -------------------------------------------
    # Step 1: Load the Train Categorical and Quantitative Datasets
    # -------------------------------------------
    try:
        train_cat = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/TEST/TEST_CATEGORICAL.xlsx")
        train_quant = pd.read_excel("/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
    except Exception as e:
        print("Error loading Excel files:", e)
        return

    # -------------------------------------------
    # Step 2: Merge Datasets on 'participant_id'
    # -------------------------------------------
    merged_train = pd.merge(train_cat, train_quant, on="participant_id", how="outer")
    print("Merged Train Data (first 5 rows):")
    print(merged_train.head())
    
    # -------------------------------------------
    # Step 3: Calculate Missing Data Percentages
    # -------------------------------------------
    # Percentage of missing values for each column:
    missing_perc = merged_train.isnull().mean() * 100
    print("\nMissing Percentage per Column:")
    print(missing_perc)
    
    # Overall missing percentage:
    total_cells = merged_train.shape[0] * merged_train.shape[1]
    total_missing = merged_train.isnull().sum().sum()
    overall_missing_perc = (total_missing / total_cells) * 100
    print("\nOverall Missing Percentage: {:.2f}%".format(overall_missing_perc))

if __name__ == '__main__':
    main()
