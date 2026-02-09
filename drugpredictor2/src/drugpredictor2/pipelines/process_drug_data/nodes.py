import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import the EXACT same featurization function used for lipinski model
import sys
from pathlib import Path
# Add process_data to path to import its featurize_molecule
sys.path.insert(0, str(Path(__file__).parent.parent / 'process_data'))
from nodes import featurize_molecule


# ===========================================
# Utility functions
# ===========================================

def mol_from_smiles(smiles: str):
    """Safely convert SMILES string to RDKit Mol."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def compute_fingerprints_from_smiles(smiles: str):
    """
    Wrapper to convert SMILES to mol and compute fingerprints using the same
    function as the lipinski model training (featurize_molecule from process_data).
    """
    if not isinstance(smiles, str):
        # Return zero array with correct size - will be determined by featurize_molecule
        mol = None
    else:
        mol = mol_from_smiles(smiles)
    
    # Use the EXACT same featurization as lipinski model
    return featurize_molecule(mol)

# ===========================================
# Node: process drug dataset
# ===========================================

def process_drug_dataset(drug_raw: pd.DataFrame):
    """Process the drug dataset to generate features and labels."""
    df = drug_raw.copy()

    # Compute fingerprints/descriptors using THE SAME function as lipinski model
    print("Computing fingerprints for all molecules (using lipinski fingerprint function)...")
    X = np.stack(df["IsomericSMILES"].map(compute_fingerprints_from_smiles))


    # Encode ATC classes
    le = LabelEncoder()
    y_atc_int = le.fit_transform(df["MATC_Code_Short"])
    y_atc = to_categorical(y_atc_int)

    # Prepare feature dataframe
    drug_y_drug = df["is_drug"].values.astype(int)

    # Save mapping for later use
    atc_mapping = pd.DataFrame({
        "ATC_Code": df["MATC_Code_Short"].unique(),
        "Encoded_Label": range(len(df["MATC_Code_Short"].unique()))
    })

    # Train/test split needed


    return X, drug_y_drug, y_atc, atc_mapping

def split_drug_data(X, y_drug, y_atc, test_size=0.2, random_state=42):
    """Split data into train and validation sets."""
    indices = np.arange(len(X))
    
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_drug  # stratify by drug/non-drug
    )
    
    return {
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "y_drug_train": y_drug[train_idx],
        "y_drug_val": y_drug[val_idx],
        "y_atc_train": y_atc[train_idx],
        "y_atc_val": y_atc[val_idx],
        "n_atc_classes": y_atc.shape[1]
    }

