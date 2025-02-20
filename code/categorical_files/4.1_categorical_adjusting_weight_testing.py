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

# Reassign categories for Parent 1 and Parent 2 Education
def reassign_education(edu_value):
    if edu_value in [3, 6, 9]:
        return 0  # Below High School
    elif edu_value == 12:
        return 1  # High School Graduate
    elif edu_value == 15:
        return 2  # Some College
    elif edu_value in [18, 21]:
        return 3  # College Degree
    else:
        return edu_value  # Leave other values unchanged if any

# Apply the reassignment to Parent 1 and Parent 2 Education columns
train_cat['Barratt_Barratt_P1_Edu'] = train_cat['Barratt_Barratt_P1_Edu'].apply(reassign_education)
train_cat['Barratt_Barratt_P2_Edu'] = train_cat['Barratt_Barratt_P2_Edu'].apply(reassign_education)

# Display updated value counts for verification
print("Updated Value Counts for Barratt_Barratt_P1_Edu:")
print(train_cat['Barratt_Barratt_P1_Edu'].value_counts())

print("\nUpdated Value Counts for Barratt_Barratt_P2_Edu:")
print(train_cat['Barratt_Barratt_P2_Edu'].value_counts())

# Reassign categories for Race
def reassign_race(race_value):
    if race_value == 0:
        return 0  # White/Caucasian
    elif race_value == 1:
        return 1  # Black/African American
    elif race_value == 2:
        return 2  # Hispanic
    elif race_value == 3:
        return 3  # Asian
    elif race_value in [4, 5, 6, 7]:
        return 4  # Indigenous/Native (Indian, Native American Indian, American Indian/Alaskan Native, Native Hawaiian/Pacific Islander)
    elif race_value == 8:
        return 5  # Two or more races
    elif race_value in [9, 10]:
        return 6  # Other/Unknown
    elif race_value == 11:
        return 7  # Choose not to specify
    else:
        return race_value  # Leave other values unchanged

# Apply the reassignment to Race column
train_cat['PreInt_Demos_Fam_Child_Race'] = train_cat['PreInt_Demos_Fam_Child_Race'].apply(reassign_race)

# Display updated value counts for verification
print("Updated Value Counts for PreInt_Demos_Fam_Child_Race:")
print(train_cat['PreInt_Demos_Fam_Child_Race'].value_counts())


# Reassign categories for Parent 1 and Parent 2 Occupation
def reassign_occupation(occ_value):
    if occ_value == 0:
        return 0  # Homemaker / Stay-at-home parent
    elif occ_value in [5, 10]:
        return 1  # Low-skilled labor (Janitor, Farm Worker, Food Preparation, Garbage Collector, etc.)
    elif occ_value in [15, 20]:
        return 2  # Skilled Trade & Clerical (Painter, Sales Clerk, Mechanic, Carpenter, Hairdresser)
    elif occ_value in [25, 30]:
        return 3  # Mid-level professions (Machinist, Secretary, Insurance Sales, Artist, Military Enlisted)
    elif occ_value in [35, 40]:
        return 4  # Technical & Management (Nurse, Engineer, Manager, Military Officer, Teacher)
    elif occ_value == 45:
        return 5  # High-Level Professions (Physician, Attorney, CEO, Judge, Professor)
    else:
        return occ_value  # Leave other values unchanged

# Apply the reassignment to Parent 1 and Parent 2 Occupation columns
train_cat['Barratt_Barratt_P1_Occ'] = train_cat['Barratt_Barratt_P1_Occ'].apply(reassign_occupation)
train_cat['Barratt_Barratt_P2_Occ'] = train_cat['Barratt_Barratt_P2_Occ'].apply(reassign_occupation)

# Display updated value counts for verification
print("Updated Value Counts for Barratt_Barratt_P1_Occ:")
print(train_cat['Barratt_Barratt_P1_Occ'].value_counts())

print("\nUpdated Value Counts for Barratt_Barratt_P2_Occ:")
print(train_cat['Barratt_Barratt_P2_Occ'].value_counts())

# Apply the reassignment functions
train_cat['PreInt_Demos_Fam_Child_Race'] = train_cat['PreInt_Demos_Fam_Child_Race'].apply(reassign_race)
train_cat['Barratt_Barratt_P1_Occ'] = train_cat['Barratt_Barratt_P1_Occ'].apply(reassign_occupation)
train_cat['Barratt_Barratt_P2_Occ'] = train_cat['Barratt_Barratt_P2_Occ'].apply(reassign_occupation)

test_cat['PreInt_Demos_Fam_Child_Race'] = test_cat['PreInt_Demos_Fam_Child_Race'].apply(reassign_race)
test_cat['Barratt_Barratt_P1_Occ'] = test_cat['Barratt_Barratt_P1_Occ'].apply(reassign_occupation)
test_cat['Barratt_Barratt_P2_Occ'] = test_cat['Barratt_Barratt_P2_Occ'].apply(reassign_occupation)
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

import importlib.util
import os

# Define the file path (adjust if needed)
module_path = os.path.join(os.getcwd(), "/Users/noeliagarciaw/Desktop/IS5150/wids_project/code/categorical_files/4_categorical_adjusting_weight.py")

# Load the module from the file
spec = importlib.util.spec_from_file_location("cat_adjust", module_path)
cat_adjust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cat_adjust)


print(cat_adjust.train_cat.head)
