import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score

# load and merge data
q_train = pd.read_excel(r".\data\TRAIN\TRAIN_QUANTITATIVE_METADATA.xlsx")
c_train = pd.read_excel(r".\data\TRAIN\TRAIN_CATEGORICAL_METADATA.xlsx")
s_train = pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx")
f_train = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
train_df = q_train.merge(c_train, on='participant_id', how='left').merge(s_train, on='participant_id', how='left').merge(f_train, on='participant_id', how='left')

q_test = pd.read_excel(r".\data\TEST\TEST_QUANTITATIVE_METADATA.xlsx")
c_test = pd.read_excel(r".\data\TEST\TEST_CATEGORICAL.xlsx")
f_test = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_df = q_test.merge(c_test, on='participant_id', how='left').merge(f_test, on='participant_id', how='left')

# identify data type for each column
nums = ['EHQ_EHQ_Total','ColorVision_CV_Score','APQ_P_APQ_P_CP','APQ_P_APQ_P_ID',
        'APQ_P_APQ_P_INV','APQ_P_APQ_P_OPD','APQ_P_APQ_P_PM','APQ_P_APQ_P_PP',
        'SDQ_SDQ_Conduct_Problems','SDQ_SDQ_Difficulties_Total','SDQ_SDQ_Emotional_Problems',
        'SDQ_SDQ_Externalizing','SDQ_SDQ_Generating_Impact','SDQ_SDQ_Hyperactivity',
        'SDQ_SDQ_Internalizing','SDQ_SDQ_Peer_Problems','SDQ_SDQ_Prosocial','MRI_Track_Age_at_Scan']

cats = ['Basic_Demos_Enroll_Year','Basic_Demos_Study_Site','PreInt_Demos_Fam_Child_Ethnicity',
        'PreInt_Demos_Fam_Child_Race','MRI_Track_Scan_Location','Barratt_Barratt_P1_Edu',
        'Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Edu','Barratt_Barratt_P2_Occ']

targs = ['ADHD_Outcome','Sex_F'] #binary targets

# save participant ids to reference in the results
participant_ids = test_df['participant_id']
X_train = train_df.drop(columns=targs + ['participant_id'])
y_train = train_df[targs]
X_test = test_df.drop(columns=['participant_id'])

# OHE categorical variables (may be optional depending on loaded data)
X_train = pd.get_dummies(X_train, columns=cats, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cats, drop_first=True)

# ensure test set has same columns as training set
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Define the base SVM model
base_svm = SVC(probability=True, random_state=42)

# Wrap it in MultiOutputClassifier
msvm = MultiOutputClassifier(base_svm)

# Define hyperparameter grid
param_grid = {
    "estimator__C": [0.1, 1, 10],  # Regularization parameter
    "estimator__gamma": ["scale", "auto"],  # Kernel coefficient
}

# Define weighted F1 scoring
def weighted_f1(y_true, y_pred):
    # Double weight for female participants (Sex_F = 1)
    sample_weights = np.where(y_true[:, 1] == 1, 2, 1)
    
    return f1_score(y_true, y_pred, average="weighted", sample_weight=sample_weights)

weighted_f1_scorer = make_scorer(weighted_f1)

# Set up GridSearchCV
grid_search = GridSearchCV(msvm, param_grid, scoring=weighted_f1_scorer, cv=10, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Parameters:", best_params)
print("Best Weighted F1-Score:", grid_search.best_score_)

# Evaluate with cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, scoring=weighted_f1_scorer, cv=5)
print("Cross-validated Weighted F1-Score:", cv_scores.mean())
