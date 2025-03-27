import pandas as pd
from sklearn.impute import KNNImputer

def impute_knn(df):
    """
    Impute missing values using KNN imputation.
    
    Parameters:
      df : pandas DataFrame with missing values.
           Assumes there is a 'participant_id' column that should not be imputed.
           
    Returns:
      imputed_df : pandas DataFrame with imputed values and the original 'participant_id' column.
    """
    # Preserve the participant_id column
    ids = df['participant_id']
    # Exclude 'participant_id' from the imputation
    data_to_impute = df.drop(columns=['participant_id'])
    
    # Initialize KNN imputer (adjust n_neighbors if desired)
    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(data_to_impute)
    
    # Convert the imputed array back into a DataFrame
    imputed_data = pd.DataFrame(imputed_array, columns=data_to_impute.columns)
    # Reattach the participant_id column
    imputed_data.insert(0, 'participant_id', ids.values)
    
    return imputed_data

