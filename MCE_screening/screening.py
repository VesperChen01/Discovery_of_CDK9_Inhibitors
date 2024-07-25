import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
import numpy as np
import joblib

# Function to compute RDKit fingerprint (reuse from previous code)
def compute_rdkit_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return RDKFingerprint(mol)
    else:
        return None

# Load the new dataset
new_file_path = '/home/roufen/crf/ML_test/CDK9/MCE_screening/MCE.csv'  # Replace with your new dataset file path
new_data = pd.read_csv(new_file_path)

# Filter out invalid molecules and compute fingerprints
new_data = new_data[new_data['smiles'] != '0']  # Remove invalid SMILES
new_data['fingerprint'] = new_data['smiles'].apply(compute_rdkit_fingerprint)
new_data = new_data.dropna(subset=['fingerprint'])

# Convert fingerprints to numpy array
X_new = np.array([fp.ToBitString() for fp in new_data['fingerprint']])
X_new = np.array([list(map(int, list(x))) for x in X_new])

# Load the trained model
rf_loaded = joblib.load('/home/roufen/crf/ML_test/CDK9/MCE_screening/rf_train_model.pkl')

# Predict on the new dataset
new_predictions = rf_loaded.predict(X_new)

# Add predictions to the new dataset
new_data['predicted_pIC50'] = new_predictions

# Save the results to a new CSV file
new_data.to_csv('/home/roufen/crf/ML_test/CDK9/MCE_screening/MCE_predictions.csv', index=False)

print("Predictions saved to MCE_predictions.csv")
