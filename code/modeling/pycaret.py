from pycaret.classification import *
import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Load training data
imp_train = pd.read_excel(r".\data\imputed_data\train_out_path.xlsx")
fmri_train = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
s_train = pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx")
train_df = imp_train.merge(s_train, on='participant_id', how='left').merge(fmri_train, on='participant_id', how='left')

# Load testing data
imp_test = pd.read_excel(r".\data\imputed_data\test_out_path.xlsx")
fmri_test = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_df = imp_test.merge(fmri_test, on='participant_id', how='left')

# Define features and targets
X_train = train_df.drop(columns=['ADHD_Outcome', 'Sex_F', 'participant_id'])
y_train = train_df[['ADHD_Outcome', 'Sex_F']]
X_test = test_df.drop(columns=['participant_id'])

# Train-test split for threshold tuning
X_train_fr, X_test_fr, y_train_fr, y_test_fr = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
train_data = pd.concat([X_train_fr,y_train_fr],axis=1) # data to use in initial PyCaret exp

# PyCaret setup for multi-label classification
clf_setup = setup(
    data=train_data,  # Ensure targets are included
    target=y_train_fr,  # Specify both targets
    session_id=42,
    fold_strategy='stratifiedkfold',
    fold=5,
    use_gpu=True
)

# Compare models and select the best one
best_model = compare_models()

# Train the best model on the train split
final_model = create_model(best_model)

# Get probabilities for threshold adjustment
y_pred_probs = predict_model(final_model, data=X_test_fr, raw_score=True)

# Extract probability values
y_true = y_test_fr.to_numpy()
y_true_adhd = y_true[:, 0]
y_true_sex = y_true[:, 1]

thresholds_adhd = np.linspace(0, 1, 50)
thresholds_sex = np.linspace(0, 1, 50)
results = []

# Find optimal thresholds
for threshold_adhd, threshold_sex in product(thresholds_adhd, thresholds_sex):
    y_pred_adjusted = np.column_stack([
        (y_pred_probs['ADHD_Outcome'] > threshold_adhd).astype(int),
        (y_pred_probs['Sex_F'] > threshold_sex).astype(int)
    ])
    
    f1_adhd = f1_score(y_true_adhd, y_pred_adjusted[:, 0], average='macro')
    f1_sex = f1_score(y_true_sex, y_pred_adjusted[:, 1], average='macro')
    final_f1_score = (f1_adhd + f1_sex) / 2
    results.append((threshold_adhd, threshold_sex, final_f1_score))

# Convert results to DataFrame and get the best thresholds
results_df = pd.DataFrame(results, columns=['Threshold_ADHD', 'Threshold_Sex', 'Final_F1_Score'])
optimal_thresholds = results_df.loc[results_df['Final_F1_Score'].idxmax()]

# Retrain model on the full training set
clf_setup_final = setup(
    data=pd.concat([X_train, y_train], axis=1),
    target=['ADHD_Outcome', 'Sex_F'],
    session_id=42,
    multi_label=True,
    fold_strategy='stratifiedkfold',
    fold=5,
    use_gpu=True
)

final_model_full = create_model(best_model)

# Predict on the final test set
final_preds = predict_model(final_model_full, data=X_test, raw_score=True)

# Apply optimal thresholds
y_pred_final = np.column_stack([
    (final_preds['ADHD_Outcome'] > optimal_thresholds['Threshold_ADHD']).astype(int),
    (final_preds['Sex_F'] > optimal_thresholds['Threshold_Sex']).astype(int)
])

# Save final predictions
final_results_df = pd.DataFrame({
    'participant_id': test_df['participant_id'],
    'ADHD_Outcome': y_pred_final[:, 0],
    'Sex_F': y_pred_final[:, 1]
})
final_results_df.to_csv("Final_PyCaret_Results.csv", index=False)