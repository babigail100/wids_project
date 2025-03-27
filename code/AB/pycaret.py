from pycaret.classification import *
import pandas as pd
import numpy as np

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
