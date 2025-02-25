import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

# load and merge data
q_train = pd.read_excel(r".\data\TRAIN\TRAIN_QUANTITATIVE_METADATA.xlsx")
c_train = pd.read_excel(r".\data\TRAIN\TRAIN_CATEGORICAL_METADATA.xlsx")
s_train = pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx")
f_train = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
train_df = q_train.merge(c_train, on='participant_id', how='left')\
                  .merge(s_train, on='participant_id', how='left')\
                  .merge(f_train, on='participant_id', how='left')

q_test = pd.read_excel(r".\data\TEST\TEST_QUANTITATIVE_METADATA.xlsx")
c_test = pd.read_excel(r".\data\TEST\TEST_CATEGORICAL.xlsx")
f_test = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_df = q_test.merge(c_test, on='participant_id', how='left')\
                .merge(f_test, on='participant_id', how='left')

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

# Define weighted F1-score metric
weighted_f1 = make_scorer(f1_score, average='weighted')

# CV and hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf, param_grid, scoring=weighted_f1, cv=cv, n_jobs=-1)

# Fit model using cross-validation
grid_search.fit(X_train, y_train)

# Best model from tuning
best_rf = grid_search.best_estimator_

# Evaluate on training set using cross-validation
cv_scores = cross_val_score(best_rf, X_train, y_train, scoring=weighted_f1, cv=cv)
print(f"Mean Weighted F1-Score (CV): {cv_scores.mean():.4f}")

# Make final predictions
y_pred = best_rf.predict(X_test)

# Save results
results = pd.DataFrame({
    "participant_id": participant_ids,
    "ADHD_Outcome": y_pred[:, 0],
    "Sex_F": y_pred[:, 1]
})
#results.to_csv("RF_results.csv", index=False)