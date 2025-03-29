from pycaret.classification import *
import pandas as pd
import numpy as np

# training data
imp_train = pd.read_excel(r".\data\imputed_data\train_out_path.xlsx")
fmri_train = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv") # this dataset cannot be stored in GitHub; found in Kaggle
s_train = pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx")
train_df = imp_train.merge(s_train, on='participant_id',how='left').merge(fmri_train, on='participant_id',how='left')

# testing data
imp_test = pd.read_excel(r".\data\imputed_data\test_out_path.xlsx")
fmri_test = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_df = imp_test.merge(fmri_test, on='participant_id',how='left')


sample_weight = np.ones(len(y))
sample_weight[(y["ADHD_Outcome"] == 1) & (y["Sex_F"] == 1)] = 2

# Convert y to a list of labels per sample (for PyCaret multi-label classification)
y_labels = y.apply(lambda row: [col for col in y.columns if row[col] == 1], axis=1)

# Add target labels to X for PyCaret
X["labels"] = y_labels

# PyCaret Setup
clf_setup = setup(
    data=X,
    target="labels",
    session_id=42,
    multi_label=True,
    fold_strategy='stratifiedkfold',
    fold=5,
    use_gpu=True
)

# Compare models
best_model = compare_models()

# Train selected model
final_model = create_model(best_model)

# Get probabilities for threshold adjustment
pred_probs = predict_model(final_model, data=X_test, raw_score=True)

# Adjust predictions using optimal thresholds
optimal_thresholds = {
    "Threshold_ADHD": 0.5,  # Adjust based on tuning
    "Threshold_Sex": 0.5
}

y_pred_adjusted = np.column_stack([
    (pred_probs["ADHD_Outcome"] > optimal_thresholds["Threshold_ADHD"]).astype(int),
    (pred_probs["Sex_F"] > optimal_thresholds["Threshold_Sex"]).astype(int)
])
