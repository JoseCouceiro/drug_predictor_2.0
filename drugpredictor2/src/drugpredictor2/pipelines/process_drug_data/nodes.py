import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# ===========================================
# Utility functions
# ===========================================

def mol_from_smiles(smiles: str):
    """Safely convert SMILES string to RDKit Mol."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def compute_fingerprints(smiles: str,
                        morgan_bits=2048,
                        maccs_bits=167,
                        scalar_bits=1,
                        morgan_radius=2,
                        morgan_feat_bits=1024,
                        ap_bits=1024,
                        tt_bits=1024):
    """
    Returns concatenated numpy array of:
      - Morgan (ECFP)
      - Feature-based Morgan (FCFP)
      - MACCS keys
      - Hashed Atom Pair fingerprint
      - Hashed Topological Torsion fingerprint
      - TPSA (1 value)
    """
    total_len = morgan_bits + morgan_feat_bits + maccs_bits + ap_bits + tt_bits + scalar_bits
    out = np.zeros(total_len, dtype=float)

    if not isinstance(smiles, str):
        return out
    
    try:
        mol = mol_from_smiles(smiles)
        if mol is None:
            return out

        # Morgan ECFP
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits, useFeatures=False)
        morgan_arr = np.array(morgan_fp, dtype=float)

        # Feature-based Morgan (FCFP-like)
        morgan_feat_fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_feat_bits, useFeatures=True)
        morgan_feat_arr = np.array(morgan_feat_fp, dtype=float)

        # MACCS
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.array(maccs_fp, dtype=float)

        # Hashed Atom Pair
        ap_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=ap_bits)
        ap_arr = np.array(ap_fp, dtype=float)

        # Hashed Topological Torsion
        tt_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=tt_bits)
        tt_arr = np.array(tt_fp, dtype=float)

        # TPSA
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        tpsa_arr = np.array([tpsa], dtype=float)

        # Concatenate everything
        return np.concatenate([morgan_arr, morgan_feat_arr, maccs_arr, ap_arr, tt_arr, tpsa_arr])
    
    except Exception:
        return out

# ===========================================
# Node: process drug dataset
# ===========================================

def process_drug_dataset(drug_raw: pd.DataFrame):
    """Process the drug dataset to generate features and labels."""
    df = drug_raw.copy()

    # Compute fingerprints/descriptors
    print("Computing fingerprints for all molecules...")
    X = np.stack(df["IsomericSMILES"].map(compute_fingerprints))


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

