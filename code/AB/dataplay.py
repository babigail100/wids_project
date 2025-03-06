import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

test_df = pd.read_excel(r".\data\TEST\TEST_CATEGORICAL.xlsx").merge(
    pd.read_excel(r".\data\TEST\TEST_QUANTITATIVE_METADATA.xlsx"), on='participant_id',how='left').merge(
    pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"), on='participant_id',how='left')

test_df.head()

train_df = pd.read_excel(r".\data\TRAIN\TRAIN_CATEGORICAL_METADATA.xlsx").merge(
    pd.read_excel(r".\data\TRAIN\TRAIN_QUANTITATIVE_METADATA.xlsx"), on='participant_id',how='left').merge(
    pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx"),on='participant_id',how='left').merge(
    pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"), on='participant_id',how='left')

train_df.head()

train_fmri = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")

train_fmri.head()

'''
PREPROCESSING:
feature selection/dimensionality reduction necessary:
- PCA, ICA
normalization?
missing values?
one-hot encode categorical variables

MODEL DESIGN:
multi-outcome-> neural networks, multi-output random forest
regularization to prevent overfit (l1, l2)

OTHER EXPLORATION:
visualization
calculate correlation between non-fMRI features and ADHD/sex


fMRI: measures brain activity by detecting changes in blood flow via Blood Oxygen 
    Level Dependent (BOLD) as time-series measurements across brain regions
connectome matrices: represent connectivity between regions of the brain; each element
    quantifies a relationship (correlation) between two regions over time
matrix structure: rows/columns are brain regions or nodes; values are pairwise correlations 
    between BOLD signalls from two regions
- our data: vectorized form of symmetric connectome matrix
    a 200-region matrix would have (200 choose 2) = 19,900 unique connections
    A symmetric matrix means the correlation between Region A and Region B is the same as between Region B and Region A
    Reduce the dimensionality of the data by working only with the upper triangle

fMRI connectome matrices:
- reflect brain region correlations
- A "functional MRI connectome matrix" is a square matrix that represents the 
    strength of functional connections between different brain regions, derived from 
    data acquired using functional magnetic resonance imaging (fMRI), where each 
    cell in the matrix represents the correlation between the activity time series of 
    two brain regions, essentially showing how strongly they are functionally 
    connected to each other; essentially creating a "map" of the brain's functional network
- Structure:
    Each row and column of the matrix corresponds to a specific brain region (defined by 
    a brain atlas), and the value at the intersection of a row and column indicates the 
    strength of the functional connection between those two regions

Questions:
What atlas or parcellation was used to generate the connectome (e.g., AAL, Schaefer)?
Are the connectome values preprocessed (e.g., detrended, denoised, filtered)?
'''

# Extract one participant's flattened vector 
flattened_vector = train_fmri.iloc[4, 1:].values  # Skip participant ID
flattened_vector

full_matrix = np.zeros((200,200))
full_matrix[np.triu_indices(200,1)]=flattened_vector
full_matrix += full_matrix.T
np.fill_diagonal(full_matrix,1) # Fill the upper triangle including the diagonal
#plt.figure(figsize=(8, 6))
sns.heatmap(full_matrix, cmap='coolwarm', square=True)
plt.title("Connectome Matrix Heatmap")
plt.show()

max(flattened_vector)
min(flattened_vector)



### RANDOM FOREST APRIORI

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# identify data type for each column
nums = ['EHQ_EHQ_Total','ColorVision_CV_Score','APQ_P_APQ_P_CP','APQ_P_APQ_P_ID',
        'APQ_P_APQ_P_INV','APQ_P_APQ_P_OPD','APQ_P_APQ_P_PM','APQ_P_APQ_P_PP',
        'SDQ_SDQ_Conduct_Problems','SDQ_SDQ_Difficulties_Total','SDQ_SDQ_Emotional_Problems',
        'SDQ_SDQ_Externalizing','SDQ_SDQ_Generating_Impact','SDQ_SDQ_Hyperactivity',
        'SDQ_SDQ_Internalizing','SDQ_SDQ_Peer_Problems','SDQ_SDQ_Prosocial','MRI_Track_Age_at_Scan']

cats = ['Basic_Demos_Enroll_Year','Basic_Demos_Study_Site','PreInt_Demos_Fam_Child_Ethnicity',
        'PreInt_Demos_Fam_Child_Race','MRI_Track_Scan_Location','Barratt_Barratt_P1_Edu',
        'Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Edu','Barratt_Barratt_P2_Occ']

targs = ['ADHD_Outcome','Sex_F'] #both are categorical; 0 or 1

participant_ids = test_df['participant_id']
X_train = train_df.drop(columns=targs + ['participant_id'])
y_train = train_df[targs]
X_test = test_df.drop(columns=['participant_id'])

# Preprocessing
X_train = pd.get_dummies(X_train, columns=cats, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cats, drop_first=True)

X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_test

# Build and fit model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

results = pd.DataFrame({
                        "participant_id": participant_ids,
                        "ADHD_Outcome": y_pred[:,0],
                        "Sex_F": y_pred[:,1]
})

# Save to a CSV file
#results.to_csv("RF_apriori_results.csv", index=False)

'''
if X_train.isnull().values.any() or X_test.isnull().values.any():
    print("Warning: Missing values detected after conversion. Filling with zeros.")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
'''
# sum of the nulls
train_df.isna().sum()[train_df.isna().sum() > 0]
test_df.isna().sum()[test_df.isna().sum() > 0]


'''
MODEL OPTIONS:
- Random Forest/XGBoost after PCA
    * Bagging vs boosting: 
    		○ Bagging: creating many copies of training data and applying weak 
            learner to each copy to obtain multiple weak models to combine; 
            bootstrapped trees are independent from each other
            ○ Boosting: using original training data and iteratively creating 
            multiple models by using a weak learner, with each new model 
            trying to fix the errors that previous models make; each tree 
            grown using info from previous tree
            ○ Boosting may potentially give us more accurate classifications

- Neural Networks
    * CNN, RNN, Transformer for fMRI, Dense Network for tabular data; Concatenate both 
    in a "multimodal deep learning framework"
    * autoencoders instead of PCA? --PCA is linear
    * may work without PCA (but we'll still compare)
    * risk of overfitting (1200 records is relatively small)
    * regularization (dropout, weight decay, early stopping)
    * computationally expensive

- PyCaret model selection?
'''