#################################################################################################
# Import libraries and load data
#################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support
from itertools import product

from imblearn.over_sampling import RandomOverSampler, SMOTE

q_train = pd.read_excel(r"C:\Users\a01406508\Documents\WiDS\wids_project\data\TRAIN\TRAIN_QUANTITATIVE_METADATA.xlsx")
c_train = pd.read_excel(r"C:\Users\a01406508\Documents\WiDS\wids_project\data\TRAIN\TRAIN_CATEGORICAL_METADATA.xlsx")
s_train = pd.read_excel(r"C:\Users\a01406508\Documents\WiDS\wids_project\data\TRAIN\TRAINING_SOLUTIONS.xlsx")
f_train = pd.read_csv(r"C:\Users\a01406508\Downloads\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")

q_test = pd.read_excel(r"C:\Users\a01406508\Documents\WiDS\wids_project\data\TEST\TEST_QUANTITATIVE_METADATA.xlsx")
c_test = pd.read_excel(r"C:\Users\a01406508\Documents\WiDS\wids_project\data\TEST\TEST_CATEGORICAL.xlsx")
f_test = pd.read_csv(r"C:\Users\a01406508\Downloads\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")

#################################################################################################
# Merge data sets, clean data, split into train/test sets
#################################################################################################
train_df = q_train.merge(c_train, on='participant_id', how='left').merge(s_train, on='participant_id', how='left').merge(f_train, on='participant_id', how='left')
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
# X_train = X_train.apply(pd.to_numeric, errors='coerce')
# X_test = X_test.apply(pd.to_numeric, errors='coerce')
# X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_train_fr, X_test_fr, y_train_fr, y_test_fr = train_test_split(X_train, y_train, test_size=.2, random_state=42)
 
#################################################################################################
# Test 1    RF splits using balanced subsample
#################################################################################################

rf = RandomForestClassifier(n_estimators = 100, 
                            random_state = 42, 
                            class_weight = 'balanced_subsample')
clf = MultiOutputClassifier(rf, n_jobs = -1)
clf.fit(X_train_fr, y_train_fr)

y_pred = clf.predict(X_test_fr)
y_pred_df = pd.DataFrame(y_pred, columns = ['ADHD_Outcome', 'Sex_F'])
y_pred_df.describe()  
# Sex_F mean = 0 <- not what we want

#################################################################################################
# Test 2    RF splits using custom weights
#################################################################################################

# Male (0) gets weight = 1, Female (1) gets twice the weight of Male
rf = RandomForestClassifier(n_estimators = 100, 
                            random_state = 42, 
                            class_weight = {0: 1, 1: 2} )  # twice weight
clf = MultiOutputClassifier(rf, n_jobs = -1)
clf.fit(X_train_fr, y_train_fr)

y_pred = clf.predict(X_test_fr)
y_pred_df = pd.DataFrame(y_pred, columns = ['ADHD_Outcome', 'Sex_F'])
y_pred_df.describe()  
# weight: 20, Sex_F mean = 0.041152 <- not what we want
# weight: 100, Sex_F mean = 0.069959 <- not what we want
# weight: 200, Sex_F mean = 0.069959 <- not what we want

#################################################################################################
# Test 3    Fitting model custom data weights
#################################################################################################

rf = RandomForestClassifier(n_estimators = 100, 
                            random_state = 42)
clf = MultiOutputClassifier(rf, n_jobs = -1)

sample_weight = np.ones(len(y_train_fr))  # Default weight = 1 for all rows
sample_weight[(y_train_fr["ADHD_Outcome"] == 1) & (y_train_fr["Sex_F"] == 1)] = 2

clf.fit(X_train_fr, y_train_fr, sample_weight=sample_weight)

y_pred = clf.predict(X_test_fr)
y_pred_df = pd.DataFrame(y_pred, columns = ['ADHD_Outcome', 'Sex_F'])
y_pred_df.describe()
# weight: 2, Sex_F mean = 0 <- not what we want
# weight: 100, Sex_F mean = 0.012346 <- not what we want



#################################################################################################
# Digging into Test 1
#################################################################################################

#################################################################################################
# Look at predicted probabilities of being Female -----------------------------------------------

rf = RandomForestClassifier(n_estimators = 100, 
                            random_state = 42, 
                            class_weight = 'balanced_subsample')
clf = MultiOutputClassifier(rf, n_jobs = -1)
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
# Since they are all less than 0.50, they are all getting classified as Male
# We will need to lower the 0.50 cutoff value

#################################################################################################
# Example of the effects of adjusting the cutoff value ------------------------------------------

# Set custom threshold (e.g., 0.40 instead of 0.50)
custom_threshold = 0.30
y_pred_adjusted = (sex_probs > custom_threshold).astype(int)

# Check how many 1s are predicted now
np.mean(y_pred_adjusted)
# 0.10699 with custom_threshold = 0.40
# 0.82304 with custom_threshold = 0.30

#################################################################################################
# Use different test models and find optimal cutoff values
#################################################################################################

# Test 1
# rf = RandomForestClassifier(n_estimators = 100, 
#                             random_state = 42, 
#                             class_weight = 'balanced_subsample')
# clf = MultiOutputClassifier(rf, n_jobs = -1)
# clf.fit(X_train_fr, y_train_fr)

# Test 2
# rf = RandomForestClassifier(n_estimators = 100, 
#                             random_state = 42, 
#                             class_weight = {0: 1, 1: 2} )  # twice weight
# clf = MultiOutputClassifier(rf, n_jobs = -1)
# clf.fit(X_train_fr, y_train_fr)

# Test 3
rf = RandomForestClassifier(n_estimators = 100, 
                            random_state = 42)
clf = MultiOutputClassifier(rf, n_jobs = -1)
sample_weight = np.ones(len(y_train_fr))  # Default weight = 1 for all rows
sample_weight[(y_train_fr["ADHD_Outcome"] == 1) & (y_train_fr["Sex_F"] == 1)] = 2
clf.fit(X_train_fr, y_train_fr, sample_weight=sample_weight)

#################################################################################################
# Find optimal cutoff value that maximizes the weighted F1 score --------------------------------

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

# Find the optimal pair of thresholds that maximize the F1 score
optimal_thresholds = results_df.loc[results_df["Final_F1_Score"].idxmax()]

# Display the optimal thresholds and corresponding F1 score
print("Optimal ADHD Threshold:", optimal_thresholds["Threshold_ADHD"])
print("Optimal Sex_F Threshold:", optimal_thresholds["Threshold_Sex"])
print("Maximized Final F1 Score:", optimal_thresholds["Final_F1_Score"])

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



#################################
# This uses 1 cutoff value (for each outcome) just for illustration

# # Get predicted probabilities for each target
# y_pred_probs = clf.predict_proba(X_test_fr)

# # Define custom thresholds
# thresholds = {"ADHD_Outcome": 0.50, "Sex_F": 0.30}

# # Convert to binary predictions using thresholds
# y_pred_adjusted = np.column_stack([
#     (y_pred_probs[0][:, 1] > thresholds["ADHD_Outcome"]).astype(int),  # ADHD
#     (y_pred_probs[1][:, 1] > thresholds["Sex_F"]).astype(int)          # Sex_F
# ])

# # Convert y_test to NumPy array for consistency
# y_true = y_test_fr.to_numpy()

# # Extract individual columns
# y_true_adhd = y_true[:, 0]  # True labels for ADHD
# y_true_sex = y_true[:, 1]   # True labels for Sex_F

# classification_adhd = y_pred_adjusted[:, 0]  # Predicted ADHD values
# classification_sex = y_pred_adjusted[:, 1]   # Predicted Sex_F values


# # Define sample weights (example: give extra weight to ADHD=1 & Sex_F=1 cases)
# sample_weight = np.ones(len(y_true))
# sample_weight[(y_true[:, 0] == 1) & (y_true[:, 1] == 1)] = 2  # Double weight for ADHD=1 & Sex_F=1

# # Compute weighted F1 score for ADHD
# f1_adhd = f1_score(y_true_adhd, classification_adhd, average="macro", sample_weight=sample_weight)
# print("F1 Score (ADHD):", f1_adhd)

# # Compute weighted F1 score for Sex_F
# f1_sex = f1_score(y_true_sex, classification_sex, average="macro", sample_weight=sample_weight)
# print("F1 Score (Sex_F):", f1_sex)

# # Final leaderboard score: Average of both
# final_f1_score = (f1_adhd + f1_sex) / 2
# print("Final Average Weighted F1 Score:", final_f1_score)



#################################
# this uses the same threshold for both adhd and sex

# y_true = y_test_fr.to_numpy()  # Convert y_test to NumPy array for consistency
# y_true_adhd = y_true[:, 0]  # True labels for ADHD
# y_true_sex = y_true[:, 1]   # True labels for Sex_F

# # Define range of thresholds to test
# thresholds = np.linspace(0, 1, 50)  # 50 thresholds from 0 to 1
# f1_scores = []

# # Loop over thresholds
# for threshold in thresholds:
#     # Convert probabilities to binary predictions using threshold
#     y_pred_adjusted = np.column_stack([
#         (y_pred_probs[0][:, 1] > threshold).astype(int),  # ADHD
#         (y_pred_probs[1][:, 1] > threshold).astype(int)   # Sex_F
#     ])

#     # Extract predictions
#     classification_adhd = y_pred_adjusted[:, 0]
#     classification_sex = y_pred_adjusted[:, 1]

#     # Define sample weights (double weight for ADHD=1 & Sex_F=1 cases)
#     sample_weight = np.ones(len(y_true))
#     sample_weight[(y_true[:, 0] == 1) & (y_true[:, 1] == 1)] = 2

#     # Compute F1 scores
#     f1_adhd = f1_score(y_true_adhd, classification_adhd, average="macro", sample_weight=sample_weight)
#     f1_sex = f1_score(y_true_sex, classification_sex, average="macro", sample_weight=sample_weight)

#     # Final leaderboard score (average of both F1 scores)
#     final_f1_score = (f1_adhd + f1_sex) / 2
#     f1_scores.append(final_f1_score)

# # Plot results
# plt.figure(figsize=(8, 5))
# plt.plot(thresholds, f1_scores, marker="o", linestyle="-")
# plt.xlabel("Threshold")
# plt.ylabel("Final Kaggle Leaderboard F1 Score")
# plt.title("Effect of Threshold on Final F1 Score")
# plt.grid(True)
# plt.show()





