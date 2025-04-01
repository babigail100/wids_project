# Code


Imputation: scripts used for missing quantitative and categorical data imputation
- knn_imputation.py: 
- mask_dataset.py: 
- mice_imputation.py: 
- nan_exploration.py: 
- percentage_weight.py: 
- random_forest_imputation.py: 

Modeling: scripts containing multi-outcome model pipelines, including neural networks, random forest, and support vector machines optimized to maximize weighted F1-score between ADHD diagnosis and sex classification and identify a sufficient cut-off value for accurate prediction.
- nn_script_1.py: Neural Network
- rf_script_1.py: Random Forest
- svm_script_1.py: Support Vector Machines

# Workflow
Setup
- Clone this repository on your local computer in order for file directories within scripts to be correct
- Use requirements.txt to ensure correct packages are installed

Data Preparation
- Create a masked version of the training data
- Use the masked version of the training data to choose the "best" imputation method
- Run both TRAIN and TEST quantitative and categorical datasets through the "best" imputation method
- Store the imputed train and test data in the data folder
  - data/imputed_data/train_out_path.xlsx
  - data/imputed_data/test_out_path.xlsx
- Save the fMRI data files from Kaggle: https://www.kaggle.com/competitions/widsdatathon2025/data
  - TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv
  - TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv

Modeling
- In each model file ({model}_script_1), ensure the directories to the train and test fMRI data are correct, as they are not initially stored within the repository due to their size
- Change the .csv results output file to a unique file name (generally by date)
- if using the neural network model, change the {model}.keras file name to a unique name to save the model (wids_project/model.keras)
- Run the script and locate the results file in the main directory (wids_project/output_file_csv)
  - nn_script_1.py: inputs original quantitative/categorical data and unaltered fMRI data for predictions
  - rf_script_1.py: inputs imputed quantitative/categorical data and unaltered fMRI data for predictions
  - svm_script_1.py: inputs imputed quantitative/categorical data and unaltered fMRI data for predictions
