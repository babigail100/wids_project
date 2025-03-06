
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def impute_missing_forest_no_missingpy(df):
    """
    Impute missing values using an IterativeImputer with a RandomForestRegressor.

    Parameters:
      df : DataFrame with missing values. Assumes a 'participant_id' column that should not be imputed.

    Returns:
      imputed_df : DataFrame with imputed values and the original 'participant_id' column.
    """
    # Preserve the participant_id column
    ids = df['participant_id']
    # Exclude participant_id for imputation
    data_to_impute = df.drop(columns=['participant_id'])

    # Initialize the RandomForestRegressor as the estimator for the iterative imputer
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Setup the IterativeImputer
    imputer = IterativeImputer(estimator=estimator, max_iter=10, random_state=42)
    imputed_array = imputer.fit_transform(data_to_impute)

    # Convert the imputed array back to a DataFrame
    imputed_data = pd.DataFrame(imputed_array, columns=data_to_impute.columns)
    # Reattach the participant_id column
    imputed_data.insert(0, 'participant_id', ids.values)

    return imputed_data    