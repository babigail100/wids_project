import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, classification_report
from itertools import product

# load and merge data

# original data
#q_train = pd.read_excel(r".\data\TRAIN\TRAIN_QUANTITATIVE_METADATA.xlsx")
#c_train = pd.read_excel(r".\data\TRAIN\TRAIN_CATEGORICAL_METADATA.xlsx")
#s_train = pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx")
#f_train = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
#train_df = q_train.merge(c_train, on='participant_id', how='left').merge(s_train, on='participant_id', how='left').merge(f_train, on='participant_id', how='left')

#q_test = pd.read_excel(r".\data\TEST\TEST_QUANTITATIVE_METADATA.xlsx")
#c_test = pd.read_excel(r".\data\TEST\TEST_CATEGORICAL.xlsx")
#f_test = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
#test_df = q_test.merge(c_test, on='participant_id', how='left').merge(f_test, on='participant_id', how='left')

# training data
imp_train = pd.read_excel(r".\data\imputed_data\train_out_path.xlsx")
fmri_train = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv") # this dataset cannot be stored in GitHub; found in Kaggle
s_train = pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx")
train_df = imp_train.merge(s_train, on='participant_id',how='left').merge(fmri_train, on='participant_id',how='left')

# testing data
imp_test = pd.read_excel(r".\data\imputed_data\test_out_path.xlsx")
fmri_test = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_df = imp_test.merge(fmri_test, on='participant_id',how='left')

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
#X_train = pd.get_dummies(X_train, columns=cats, drop_first=True)
#X_test = pd.get_dummies(X_test, columns=cats, drop_first=True)

# amateur imputation for sake of writing code
#means = pd.concat([X_train, X_test]).mean()
#X_train.fillna(means, inplace=True)
#X_test.fillna(means, inplace=True)

# train and validation set for finding optimal model
X_train_fr, X_test_fr, y_train_fr, y_test_fr = train_test_split(X_train, y_train, test_size=.2, random_state=42)
 
# ensure test set has same columns as training set
# X_train = X_train.apply(pd.to_numeric, errors='coerce')
# X_test = X_test.apply(pd.to_numeric, errors='coerce')
# X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

########################################################################################################
# Observe Sex_F probability distribution
########################################################################################################

svm = SVC(probability=True, random_state = 42, class_weight = 'balanced')
clf = MultiOutputClassifier(svm, n_jobs = -1)
clf.fit(X_train_fr, y_train_fr)

# Get predicted probabilities for each class
y_pred_probs = clf.predict_proba(X_test_fr)  
# This returns a list of arrays (one per target)
# The order is 0: ADHD_Outcome, 1: Sex_F
# y_pred_probs[1] gives you the Sex_F predictions - the first column is the probability
#    of Male, the second column is the probability of Female

# Extract probabilities for Sex_F = 1
sex_probs = y_pred_probs[1][:, 1]  # Get probability of Sex_F = 1

# Visulaize probabilty of Female distribution
sns.histplot(sex_probs, bins=20, kde=True)
plt.xlabel("Probability of Sex_F = 1")
plt.ylabel("Count")
plt.title("Distribution of Sex_F = 1 Probabilities")
plt.show()

custom_threshold = 0.30
y_pred_adjusted = (sex_probs > custom_threshold).astype(int)
np.mean(y_pred_adjusted)

#########################################################################################################
# Use different test model to find optimal cutoff values
#########################################################################################################
svm = SVC(probability=True, random_state = 42)
clf = MultiOutputClassifier(svm, n_jobs = -1)
sample_weight = np.ones(len(y_train_fr))  # Default weight = 1 for all rows
sample_weight[(y_train_fr["ADHD_Outcome"] == 1) & (y_train_fr["Sex_F"] == 1)] = 2
clf.fit(X_train_fr, y_train_fr, sample_weight=sample_weight)

y_true = y_test_fr.to_numpy()  # Convert y_test to NumPy array for consistency
y_true_adhd = y_true[:, 0]  # True labels for ADHD
y_true_sex = y_true[:, 1]   # True labels for Sex_F

y_pred_probs = clf.predict_proba(X_test_fr)  

# Define ranges of thresholds to test for ADHD and Sex_F separately
thresholds_adhd = np.linspace(0, 1, 50)  # 20 thresholds for ADHD
thresholds_sex = np.linspace(0, 1, 50)   # 20 thresholds for Sex_F

# Store results
results = []

# Loop over all combinations of ADHD and Sex_F thresholds
for threshold_adhd, threshold_sex in product(thresholds_adhd, thresholds_sex):
    # Convert probabilities to binary predictions using separate thresholds
    y_pred_adjusted = np.column_stack([
        (y_pred_probs[0][:, 1] > threshold_adhd).astype(int),  # ADHD
        (y_pred_probs[1][:, 1] > threshold_sex).astype(int)    # Sex_F
    ])

    # Extract predictions
    classification_adhd = y_pred_adjusted[:, 0]
    classification_sex = y_pred_adjusted[:, 1]

    # Define sample weights (double weight for ADHD=1 & Sex_F=1 cases)
    sample_weight = np.ones(len(y_true))
    sample_weight[(y_true[:, 0] == 1) & (y_true[:, 1] == 1)] = 2

    # Compute F1 scores
    f1_adhd = f1_score(y_true_adhd, classification_adhd, average="macro", sample_weight=sample_weight)
    f1_sex = f1_score(y_true_sex, classification_sex, average="macro", sample_weight=sample_weight)

    # Final leaderboard score (average of both F1 scores)
    final_f1_score = (f1_adhd + f1_sex) / 2

    # Store results
    results.append((threshold_adhd, threshold_sex, final_f1_score))

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results, columns=["Threshold_ADHD", "Threshold_Sex", "Final_F1_Score"])

# Plot heatmap of F1 scores for different threshold pairs
pivot_table = results_df.pivot(index="Threshold_ADHD", columns="Threshold_Sex", values="Final_F1_Score")

plt.figure(figsize=(10, 6))
plt.imshow(pivot_table, aspect="auto", cmap="viridis", origin="lower")
plt.colorbar(label="Final F1 Score")
plt.xlabel("Threshold for Sex_F")
plt.ylabel("Threshold for ADHD")
plt.title("Final F1 Score Across Different Threshold Combinations")

# Set tick labels
plt.xticks(ticks=np.linspace(0, len(thresholds_sex) - 1, 5), labels=np.round(thresholds_sex[::len(thresholds_sex)//5], 2))
plt.yticks(ticks=np.linspace(0, len(thresholds_adhd) - 1, 5), labels=np.round(thresholds_adhd[::len(thresholds_adhd)//5], 2))

plt.show()


# Find the optimal pair of thresholds that maximize the F1 score
optimal_thresholds = results_df.loc[results_df["Final_F1_Score"].idxmax()]

# Print the optimal thresholds and corresponding F1 score
print("Optimal ADHD Threshold:", optimal_thresholds["Threshold_ADHD"])
print("Optimal Sex_F Threshold:", optimal_thresholds["Threshold_Sex"])
print("Maximized Final F1 Score:", optimal_thresholds["Final_F1_Score"])

#########################################################################################################
# Apply cutoff values to real train data and predict on real test data
#########################################################################################################
# train model
base_svm = SVC(probability=True, random_state=42)
msvm = MultiOutputClassifier(base_svm, n_jobs=-1)
sample_weight = np.ones(len(y_train))
sample_weight[(y_train["ADHD_Outcome"] == 1) & (y_train["Sex_F"] == 1)] = 2
msvm.fit(X_train, y_train, sample_weight=sample_weight)

# Get prediction probabilities
y_pred_probs = msvm.predict_proba(X_test)

# Adjust predictions based on predefined thresholds
y_pred_adjusted = np.column_stack([
    (y_pred_probs[0][:, 1] > optimal_thresholds["Threshold_ADHD"]).astype(int),  # ADHD
    (y_pred_probs[1][:, 1] > optimal_thresholds["Threshold_Sex"]).astype(int)    # Sex_F
])

classification_adhd = y_pred_adjusted[:, 0]
classification_sex = y_pred_adjusted[:, 1]

# Save results
results = pd.DataFrame({
    "participant_id": participant_ids,
    "ADHD_Outcome": classification_adhd,
    "Sex_F": classification_sex
})

results.to_csv("SVM_results_0327.csv", index=False)
'''
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
'''