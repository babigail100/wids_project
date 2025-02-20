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

train_Solutions.columns

train_cat.info()

test_cat.info()

train_cat.describe()

test_cat.describe()

# Print the count of nulls/NaNs for each column in train_cat
print("Missing values in train_cat:")
print(train_cat.isnull().sum())

# Print the count of nulls/NaNs for each column in test_cat
print("\nMissing values in test_cat:")
print(test_cat.isnull().sum())