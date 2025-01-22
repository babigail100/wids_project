import pandas as pd
import numpy as np

test_df = pd.read_excel(r".\data\TEST\TEST_CATEGORICAL.xlsx").merge(
    pd.read_excel(r".\data\TEST\TEST_QUANTITATIVE_METADATA.xlsx"), on='participant_id',how='left').merge(
    pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"), on='participant_id',how='left')

test_df.head()

train_df = pd.read_excel(r".\data\TRAIN\TRAIN_CATEGORICAL_METADATA.xlsx").merge(
    pd.read_excel(r".\data\TRAIN\TRAIN_QUANTITATIVE_METADATA.xlsx"), on='participant_id',how='left').merge(
    pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx"),on='participant_id',how='left').merge(
    pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"), on='participant_id',how='left')

train_df.head()

train_fmri = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")

'''
PREPROCESSING:
feature selection/dimensionality reduction necessary:
- PCA, ICA
normalization?
missing values?
one-hot encode categorical variables

MODEL DESIGN:
multi-outcome-> neural networks, multi-output random forest
regularization to prevent overfit (l1, l2)

OTHER EXPLORATION:
visualization
calculate correlation between non-fMRI features and ADHD/sex


fMRI: measures brain activity by detecting changes in blood flow via Blood Oxygen 
    Level Dependent (BOLD) as time-series measurements across brain regions
connectome matrices: represent connectivity between regions of the brain; each element
    quantifies a relationship (correlation) between two regions over time
matrix structure: rows/columns are brain regions or nodes; values are pairwise correlations 
    between BOLD signalls from two regions
- our data: vectorized form of symmetric connectome matrix
    a 200-region matrix would have (200 choose 2) = 19,900 unique connections
    A symmetric matrix means the correlation between Region A and Region B is the same as between Region B and Region A
    Reduce the dimensionality of the data by working only with the upper triangle

fMRI connectome matrices:
- reflect brain region correlations
- A "functional MRI connectome matrix" is a square matrix that represents the 
    strength of functional connections between different brain regions, derived from 
    data acquired using functional magnetic resonance imaging (fMRI), where each 
    cell in the matrix represents the correlation between the activity time series of 
    two brain regions, essentially showing how strongly they are functionally 
    connected to each other; essentially creating a "map" of the brain's functional network
- Structure:
    Each row and column of the matrix corresponds to a specific brain region (defined by 
    a brain atlas), and the value at the intersection of a row and column indicates the 
    strength of the functional connection between those two regions

Questions:
What atlas or parcellation was used to generate the connectome (e.g., AAL, Schaefer)?
Are the connectome values preprocessed (e.g., detrended, denoised, filtered)?
'''
### Check if original matrix structure is symmetric

# Compute the upper triangle indices including the diagonal
triu_indices_with_diag = np.triu_indices(199, k=0)  # Include diagonal (k=0)

# Extract one participant's flattened vector 
flattened_vector = train_fmri.iloc[0, 1:].values  # Skip participant ID

# Create an empty matrix
num_regions = 199
connectome_matrix = np.zeros((num_regions, num_regions))

# Fill the upper triangle including the diagonal
connectome_matrix[triu_indices_with_diag] = flattened_vector

# Mirror to complete the lower triangle
connectome_matrix += np.triu(connectome_matrix, k=1).T

# Check symmetry
symmetric = np.allclose(connectome_matrix, connectome_matrix.T)
print(f"Matrix is symmetric: {symmetric}") #true

'''
# for all participants:
connectome_matrices = []  # Store all matrices

for idx, row in train_fmri.iterrows():
    flattened_vector = row[1:].values  # Skip participant ID
    connectome_matrix = np.zeros((num_regions, num_regions))
    connectome_matrix[triu_indices_with_diag] = flattened_vector
    connectome_matrix += np.triu(connectome_matrix, k=1).T
    connectome_matrices.append(connectome_matrix)

connectome_matrices = np.array(connectome_matrices)  # Convert to NumPy array
'''

upper_triangle_indices = np.triu_indices(num_regions, k=1)  # Exclude diagonal
upper_triangle_features = connectome_matrix[upper_triangle_indices]
