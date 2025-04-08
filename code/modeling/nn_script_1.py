import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

### bring in data files
test_df = pd.read_excel(r".\data\TEST\TEST_CATEGORICAL.xlsx").merge(
    pd.read_excel(r".\data\TEST\TEST_QUANTITATIVE_METADATA.xlsx"), on='participant_id',how='left').merge(
    pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"), on='participant_id',how='left')

train_df = pd.read_excel(r".\data\TRAIN\TRAIN_CATEGORICAL_METADATA.xlsx").merge(
    pd.read_excel(r".\data\TRAIN\TRAIN_QUANTITATIVE_METADATA.xlsx"), on='participant_id',how='left').merge(
    pd.read_excel(r".\data\TRAIN\TRAINING_SOLUTIONS.xlsx"),on='participant_id',how='left').merge(
    pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"), on='participant_id',how='left')

train_fmri = pd.read_csv(r"\Users\babig\OneDrive\Documents\USU Sen\Data Competitions\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")

prep_train = train_df
prep_test = test_df

## identify data type for each column
nums = ['EHQ_EHQ_Total','ColorVision_CV_Score','APQ_P_APQ_P_CP','APQ_P_APQ_P_ID',
        'APQ_P_APQ_P_INV','APQ_P_APQ_P_OPD','APQ_P_APQ_P_PM','APQ_P_APQ_P_PP',
        'SDQ_SDQ_Conduct_Problems','SDQ_SDQ_Difficulties_Total','SDQ_SDQ_Emotional_Problems',
        'SDQ_SDQ_Externalizing','SDQ_SDQ_Generating_Impact','SDQ_SDQ_Hyperactivity',
        'SDQ_SDQ_Internalizing','SDQ_SDQ_Peer_Problems','SDQ_SDQ_Prosocial','MRI_Track_Age_at_Scan']

cats = ['Basic_Demos_Enroll_Year','Basic_Demos_Study_Site','PreInt_Demos_Fam_Child_Ethnicity',
        'PreInt_Demos_Fam_Child_Race','MRI_Track_Scan_Location','Barratt_Barratt_P1_Edu',
        'Barratt_Barratt_P1_Occ','Barratt_Barratt_P2_Edu','Barratt_Barratt_P2_Occ']

targs = ['ADHD_Outcome','Sex_F'] #both are categorical; 0 or 1

ids = ['participant_id']

fmri = prep_train.drop(columns=nums + cats + targs + ids).columns.tolist()

### data preprocessing
def preprocess_NN(df, nums=[],cats=[],fmri=[],targs=[],ids=[],cat_prep="",pca_type=""):
    
    for i in df[nums]:
        df[i] = df[i].astype('float32')
    for i in df[cats]:
        df[i] = df[i].astype('category')

    X = df.drop(columns=ids+targs)
    y = df[targs]

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
    
    ## preprocess quantitative data
    # imputation for now
    X_train.fillna(X_train.median(numeric_only=True), inplace=True)
    X_test.fillna(X_train.median(numeric_only=True), inplace=True)
    # normalize via standardscaler
    scaler = StandardScaler()
    X_train[nums] = scaler.fit_transform(X_train[nums])
    X_test[nums] = scaler.transform(X_test[nums])


    ## preprocess fMRI data
    # PCA
    scaler_f = StandardScaler()
    fmri_scaled = scaler_f.fit_transform(X_train[fmri])

    # 1. Explained Variance Method
    pca = PCA().fit(fmri_scaled)

    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()
    n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1 
    print(f"Number of components to retain 95% variance (EVM): {n_components_95}")

    # 2. Parallel Analysis
    random_data = np.random.normal(size=fmri_scaled.shape)
    pca_random = PCA().fit(random_data)

    real_eigvals = pca.explained_variance_
    random_eigvals = pca_random.explained_variance_

    n_components_parallel = np.sum(real_eigvals > random_eigvals)
    print("Optimal number of components (Parallel Analysis):", n_components_parallel)

    if pca_type == "parallel":
        n_components = n_components_parallel
    elif pca_type == "EVM":
        n_components = n_components_95
    else:
        n_components = (n_components_parallel + n_components_95) // 2
        print("Averaged EVM & PA Components:", n_components)

    pca = PCA(n_components=n_components)
    fmri_train_pca = pca.fit_transform(scaler_f.fit_transform(X_train[fmri]))
    print(fmri_train_pca)
    fmri_test_pca = pca.transform(scaler_f.transform(X_test[fmri]))
    top_fmri_features = [f"pca_{i+1}" for i in range(n_components)]
    fmri_train_pca_df = pd.DataFrame(fmri_train_pca, columns=top_fmri_features, index=X_train.index)
    fmri_test_pca_df = pd.DataFrame(fmri_test_pca, columns=top_fmri_features, index=X_test.index)
    X_train.drop(columns=fmri, inplace=True)
    X_test.drop(columns=fmri, inplace=True)
    X_train = pd.concat([X_train, fmri_train_pca_df], axis=1)
    X_test = pd.concat([X_test, fmri_test_pca_df], axis=1)
    

    ## preprocess categorical data
    # imputation tbd
    # option 1: one-hot encoding
    if cat_prep == "OHE":
        X_train = pd.get_dummies(X_train, columns=cats, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cats, drop_first=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        return X_train, X_test, y_train, y_test, scaler, scaler_f, pca, top_fmri_features

    # option 2: embedding
    elif cat_prep == "Embedding":
        cat_inps = []
        cat_embs = []

        for col in cats:
            X_train[col] = X_train[col].cat.codes
            X_test[col] = X_test[col].cat.codes  

            num_unique = df[col].nunique()
            emb_dim = min(50, (num_unique // 2 + 1))
            inp = Input(shape=(1,), name=col)
            emb = Embedding(input_dim=num_unique, output_dim=emb_dim, name=f"{col}_embedding")(inp)
            emb = Flatten()(emb)

            cat_inps.append(inp)
            cat_embs.append(emb)

            num_unique = df[col].nunique()
            emb_dim = min(50, (num_unique // 2 + 1))
            inp = Input(shape=(1,), name=col)  # Single-value input
            emb = Embedding(input_dim=num_unique, output_dim=emb_dim, name=f"{col}_embedding")(inp)
            emb = Flatten()(emb)
            cat_inps.append(inp)
            cat_embs.append(emb)

        return X_train, X_test, y_train, y_test, scaler, scaler_f, pca, top_fmri_features, cat_embs, cat_inps

    return X_train, X_test, y_train, y_test, scaler, scaler_f, pca, top_fmri_features

X_train, X_test, y_train, y_test, scaler, scaler_f, pca, top_fmri_features = preprocess_NN(prep_train, nums=nums, cats=cats, fmri=fmri, targs=targs, ids=ids, cat_prep="OHE")

def new_data_preprocess_NN(new_df, scaler=scaler, scaler_f=scaler_f, pca=pca, train_columns=prep_train.columns, fmri=fmri, top_fmri_features=top_fmri_features):
    # Apply the same preprocessing steps
    new_df = pd.get_dummies(new_df, columns=cats, drop_first=True)
    new_df = new_df.reindex(columns=train_columns, fill_value=0)  # Ensure same columns
    new_df_ids = new_df['participant_id']
    new_df.drop(columns=['participant_id'], inplace=True)
    # Apply trained scaler
    new_df[nums] = scaler.transform(new_df[nums])

    # Apply trained PCA
    fmri_unseen_pca = pca.transform(scaler_f.transform(new_df[fmri]))
    fmri_unseen_pca_df = pd.DataFrame(fmri_unseen_pca, columns=top_fmri_features, index=new_df.index)
    new_df = new_df.drop(columns=fmri, errors='ignore')
    new_df = pd.concat([new_df, fmri_unseen_pca_df], axis=1)
    new_df = new_df.reindex(columns=X_train.columns, fill_value=0)

    return new_df, new_df_ids

big_test, new_df_ids = new_data_preprocess_NN(prep_test)

def f1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)  # Convert probabilities to 0/1
    tp = tf.reduce_sum(y_true * y_pred_binary, axis=0)
    precision = tp / (tf.reduce_sum(y_pred_binary, axis=0) + 1e-7)
    recall = tp / (tf.reduce_sum(y_true, axis=0) + 1e-7)
    return 2 * (precision * recall) / (precision + recall + 1e-7)

def build_model_f1():
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(1024, activation='relu'),  
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),  
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'), 
        layers.BatchNormalization(),
        layers.Dense(2, activation='sigmoid', name="output_layer")  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])
    return model

early_stopping   = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(monitor='val_loss', save_best_only=True,save_weights_only=False, filepath="model1.keras")
callback_list    = [early_stopping,model_checkpoint]

model1 = build_model_f1()
model1.summary()

model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=callback_list)

def predict_on_new_data(model, new_X, ids=new_df_ids):
    preds = model.predict(new_X)
    preds_binary = (preds > 0.5).astype(int)
    preds_df = pd.DataFrame(preds_binary, columns=["ADHD_Outcome", "Sex_F"])
    preds_df.insert(0, 'participant_id', ids.values)
    return preds_df

results = predict_on_new_data(model1, big_test)
results
results.to_csv("NN1_results.csv", index=False)

y_preds = model1.predict(X_train)
y_preds
# use this to make a lambda threshold chart with threshold on the bottom and misclassification rate on the side

thresholds = np.linspace(0,1,50)
misclass_rates = []
for t in thresholds:
    pb = (y_preds[:,0] > t).astype(int)
    misclass_rate = np.mean(pb != y_train.values[:,0])
    misclass_rates.append(misclass_rate)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, misclass_rates, marker='o', linestyle='-')
plt.xlabel("Threshold")
plt.ylabel("Misclassification Rate")
plt.title("Threshold vs. Misclassification Rate - ADHD Outcome")
plt.grid()
plt.show()

thresholds2 = np.linspace(0,1,50)
misclass_rates2 = []
for t in thresholds2:
    pb = (y_preds[:,1] > t).astype(int)
    misclass_rate = np.mean(pb != y_train.values[:,1])
    misclass_rates2.append(misclass_rate)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, misclass_rates, marker='o', linestyle='-')
plt.xlabel("Threshold")
plt.ylabel("Misclassification Rate")
plt.title("Threshold vs. Misclassification Rate - Sex_F")
plt.grid()
plt.show()








'''
def NN_w_embs(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cat_inps=cat_inps, cat_embs=cat_embs):
    num_inputs = Input(shape=(X_train.shape[1],), name="numerical_inputs")
    all_inputs = [num_inputs]
    cat_inputs = cat_inps  # List of categorical inputs
    cat_embeddings = cat_embs  # List of embeddings
    all_inputs += cat_inputs  # Add categorical inputs to the model
    merged_inputs = Concatenate()(cat_embeddings + [num_inputs])
    x = Dense(256, activation='relu')(merged_inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # === Define Output Layers ===
    adhd_output = Dense(1, activation='sigmoid', name="adhd_output")(x)  # Binary classification for ADHD
    sex_output = Dense(1, activation='sigmoid', name="sex_output")(x)    # Binary classification for Sex

    # === Compile the Model ===
    model = Model(inputs=all_inputs, outputs=[adhd_output, sex_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"adhd_output": "binary_crossentropy", "sex_output": "binary_crossentropy"},
        metrics={"adhd_output": ["accuracy"], "sex_output": ["accuracy"]}
    )
    return model
'''
'''
emb_model = NN_w_embs()
emb_model.summary()
history = emb_model.fit(X_train,  # Numerical input + embeddings
    {"adhd_output": y_train["ADHD_Outcome"], "sex_output": y_train["Sex_F"]},  # Multi-output labels
    validation_data=(
        X_test,
        {"adhd_output": y_test["ADHD_Outcome"], "sex_output": y_test["Sex_F"]}
    ),
    batch_size=32,
    epochs=50,
    verbose=1
)
test_loss, adhd_acc, sex_acc = emb_model.evaluate(
    X_test, {"adhd_output": y_test["ADHD_Outcome"], "sex_output": y_test["Sex_F"]}
)
print(f"Test ADHD Accuracy: {adhd_acc:.4f}")
print(f"Test Sex Accuracy: {sex_acc:.4f}")

predictions = emb_model.predict(big_test)

adhd_preds = (predictions[0] > 0.5).astype(int)  # Convert to binary
sex_preds = (predictions[1] > 0.5).astype(int)

# Store results
results_df = pd.DataFrame({"ADHD_Outcome": adhd_preds.flatten(), "Sex_F": sex_preds.flatten()})
print(results_df.head())
'''


'''
# Input layers
quant_input = Input(shape=(X_quant_scaled.shape[1],), name="quant_input")
cat_input = Input(shape=(X_cat_encoded.shape[1],), name="cat_input")
fmri_input = Input(shape=(X_fmri_pca.shape[1],), name="fmri_input")

# Hidden layers for each input type
quant_branch = Dense(64, activation="relu")(quant_input)
cat_branch = Dense(64, activation="relu")(cat_input)
fmri_branch = Dense(128, activation="relu")(fmri_input)

# Merge branches
merged = Concatenate()([quant_branch, cat_branch, fmri_branch])
hidden = Dense(128, activation="relu")(merged)
hidden = Dense(64, activation="relu")(hidden)
output = Dense(2, activation="sigmoid")(hidden)  # 2 targets: ADHD, Sex_F

# Compile model
model = Model(inputs=[quant_input, cat_input, fmri_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit([X_quant_scaled, X_cat_encoded, X_fmri_pca], y, epochs=50, batch_size=32, validation_split=0.2)
'''
### USING EMBEDDINGS
'''
# Input layer for numerical + fMRI data
numerical_fmri_input = Input(shape=(num_numerical_fmri_features,), name="numerical_fmri")

# Concatenate embeddings and numerical/fMRI data
x = Concatenate()([numerical_fmri_input] + embeddings)

# Hidden layers
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
output = Dense(2, activation="sigmoid", name="output")  # Assuming binary classification

# Define and compile model
model = keras.Model(inputs=[numerical_fmri_input] + cat_inputs, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
'''
