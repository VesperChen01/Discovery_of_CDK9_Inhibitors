import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from tqdm import tqdm

# Set random seed
np.random.seed(30)

# Load the dataset
file_path = '/home/roufen/crf/ML_test/CDK9/dataset/cleaned_CDK9.csv'
data = pd.read_csv(file_path)

# Function to compute RDKit fingerprint
def compute_rdkit_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return RDKFingerprint(mol)
    else:
        return None

# Filter out invalid molecules and compute fingerprints
data = data[data['smiles'] != '0']  # Remove invalid SMILES
data['fingerprint'] = [compute_rdkit_fingerprint(smiles) for smiles in tqdm(data['smiles'])]

# Drop rows with None fingerprints
data = data.dropna(subset=['fingerprint'])

# Convert fingerprints to numpy array
X = np.array([fp.ToBitString() for fp in data['fingerprint']])
X = np.array([list(map(int, list(x))) for x in X])

# Target variable
y = data['pIC50'].astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Best hyperparameters from previous optimization
best_params = {
    'n_estimators': 380,
    'max_depth': 41,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.15625796048452906
}

# Train the final model with the best hyperparameters
rf = RandomForestRegressor(**best_params, random_state=42)
rf.fit(X_train, y_train)

# Save the model to a file
joblib.dump(rf, 'rf_train_model.pkl')

# Evaluate the model
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Calculate metrics
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
train_r2 = r2_score(y_train, y_pred_train)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training RMSE: {train_rmse}")
print(f"Training R2: {train_r2}")
print(f"Testing RMSE: {test_rmse}")
print(f"Testing R2: {test_r2}")


# # Predict on the test set
# y_pred = rf.predict(X_test)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)

# # Print the results
# print(f"RMSE: {rmse}")
# print(f"R2: {r2}")

# # Load the model from the file
# rf_loaded = joblib.load('rf_train_model.pkl')

# # Predict using the loaded model to ensure it works correctly
# y_pred_loaded = rf_loaded.predict(X_test)
# assert np.allclose(y_pred, y_pred_loaded), "Loaded model predictions do not match original model predictions"
