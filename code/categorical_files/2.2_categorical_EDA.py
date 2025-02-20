import numpy as np
import pandas as pd
import seaborn as sns

import os
import matplotlib.pyplot as plt

import sklearn
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.stats import zscore, pearsonr, uniform, skew
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, train_test_split

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# categorical variable train dataframe

file_path_trainC = "/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx"
train_cat = pd.read_excel(file_path_trainC)

# Display the first few rows
train_cat.head()

file_path_trainS = "/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/TRAIN/TRAINING_SOLUTIONS.xlsx"
train_Solutions = pd.read_excel(file_path_trainS)
train_Solutions.head()

# Load test categorical dataset
file_path_testC = "/Users/noeliagarciaw/Desktop/IS5150/wids_project/data/TEST/TEST_CATEGORICAL.xlsx"
test_cat = pd.read_excel(file_path_testC)
test_cat.head()

# Plot individual distributions and print value counts for all categorical variables
categorical_cols = ['Basic_Demos_Enroll_Year', 'Basic_Demos_Study_Site', 'PreInt_Demos_Fam_Child_Ethnicity',
                    'PreInt_Demos_Fam_Child_Race', 'MRI_Track_Scan_Location', 'Barratt_Barratt_P1_Edu',
                    'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ']

for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, data=train_cat)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"Value counts for {col}:")
    print(train_cat[col].value_counts())
    print("\n" + "="*40 + "\n")

    for col in categorical_cols:
        print(f"Value counts for {col}:")
        print(train_cat[col].value_counts())
        print("\n" + "="*40 + "\n")

#Manual weights for all colums:
# Calculate total number of samples
total_samples = len(train_cat)

for col in categorical_cols:
    category_counts = train_cat[col].value_counts()
    weights = total_samples / category_counts

    print(f"Manual Weights for {col}:")
    print(weights)
    print("\n" + "="*40 + "\n")