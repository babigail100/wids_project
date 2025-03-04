import pandas as pd
from missingpy import MissForest

   
# Preserve the participant_id column
ids = df['participant_id']
 # Exclude participant_id for imputation
data_to_impute = df.drop(columns=['participant_id'])
    
# Initialize MissForest imputer
imputer = MissForest(max_iter=10, n_estimators=100, random_state=42)
imputed_array = imputer.fit_transform(data_to_impute)
    
# Convert the imputed array back to a DataFrame
imputed_data = pd.DataFrame(imputed_array, columns=data_to_impute.columns)
# Reattach the participant_id column
imputed_data.insert(0, 'participant_id', ids.values)
    