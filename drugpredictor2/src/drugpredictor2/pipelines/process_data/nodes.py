import gzip
import io
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools as pt, Descriptors
from typing import Optional, Tuple, List, Dict, Any

# --- Core functions ---
def compute_morgan_fp(mol: Chem.Mol, depth: int = 2, nBits: int = 2048) -> Optional[np.ndarray]:
    """Computes the Morgan fingerprint for a single RDKit molecule.

    This function generates a Morgan fingerprint (equivalent to ECFP/FCFP)
    as a bit vector for the given RDKit molecule. It handles potential
    issues during fingerprint generation by returning None upon failure.

    Args:
        mol: The RDKit molecule object.
        depth: The radius of the fingerprint, defining the maximum distance
               from the central atom to consider for atom environments.
               (Equivalent to morgan_radius or ECFP/FCFP diameter/2).
        nBits: The desired length of the bit vector, which determines the
               number of possible features.

    Returns:
        A NumPy array of booleans (for memory saving) representing the Morgan fingerprint as a bit vector
        (e.g., array([0, 1, 0, ..., 1])). Returns None if the fingerprint
        computation fails (e.g., for an invalid molecule).
    """
    try:
        mor_fp = AllChem.GetMorganFingerprintAsBitVect(mol, depth, nBits)
        return np.array(mor_fp, dtype=np.bool_) # IMPORTANT MEMORY OPTIMIZATION
    except Exception as e:
        return None

def is_lipinski(x: pd.DataFrame) -> pd.DataFrame:
    """Applies Lipinski's Rule of Five to a DataFrame of molecular properties.

    Calculates whether a molecule adheres to at least three of the four main
    Lipinski rules (MW < 500, LogP <= 5, H-Bond Donors <= 5,
    H-Bond Acceptors <= 10). Adds a 'RuleFive' column where 1 indicates
    compliance (passes >= 3 rules) and 0 indicates failure.

    Args:
        x: DataFrame containing molecular properties, including 'Molecular Weight',
           'LogP', 'H-Bond Donors', and 'H-Bond Acceptors'.

    Returns:
        The input DataFrame with an added 'RuleFive' integer column.
    """
    # Lipinski rules
    hdonor = x['H-Bond Donors'] <= 5
    haccept = x['H-Bond Acceptors'] <= 10
    mw = x['Molecular Weight'] < 500
    clogP = x['LogP'] <= 5
    # Apply rules to dataframe
    x['RuleFive'] = np.where(((hdonor & haccept & mw) | (hdonor & haccept & clogP) | (hdonor & mw & clogP) | (haccept & mw & clogP)), 1, 0)
    return x

def add_RDKit_mol(df: pd.DataFrame) -> pd.DataFrame:
    base_column = 'SMILES'
    calculated_column = 'RDKit_Molecule'
    
    pt.AddMoleculeColumnToFrame(
        frame=df,
        smilesCol=base_column,
        molCol=calculated_column
    )
    
    # Drop rows where molecule conversion failed
    initial_mol_len = len(df)
    df.dropna(subset=[calculated_column], inplace=True)
    if len(df) < initial_mol_len:
        print(f"Dropped {initial_mol_len - len(df)} rows in due to failed molecule conversion in file.")

def add_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    base_column = 'RDKit_Molecule'
    calculated_column = 'Morgan2FP'

    # Compute Morgan fingerprints
    df[calculated_column] = df[base_column].map(compute_morgan_fp)
    
    # Drop the intermediate RDKit_Molecule column
    final_df = df.drop(columns=[base_column])

    # Drop rows where fingerprint computation failed
    initial_fp_len = len(final_df)
    final_df.dropna(subset=[calculated_column], inplace=True)
    if len(final_df) < initial_fp_len:
        print(f"Dropped {initial_fp_len - len(final_df)} rows in due to failed fingerprinting in file.")
        
    return final_df

# --- Processes a single raw SDF file from start to finish ---

def process_single_raw_sdf_to_fingerprints(
    sdf_partition: dict
) -> pd.DataFrame:
    """
    Processes a single gzipped SDF file content into a DataFrame with fingerprints.
    
    Args:
        sdf_content: The gzipped SDF file content as bytes
        file_id: The identifier for this file (from partitioned dataset)
        
    Returns:
        Processed DataFrame with molecular properties and fingerprints
    """
    try:
        sdf_content = sdf_partition['data']
        with gzip.open(io.BytesIO(sdf_content), 'rb') as gz:
            supplier = Chem.ForwardSDMolSupplier(gz)
            data = []
            for i, mol in enumerate(supplier):
                if mol is None:
                    print(f"Warning: Skipping invalid molecule in file at index {i}")
                    continue
                try:
                    data.append({
                        "SMILES": Chem.MolToSmiles(mol),
                        "Molecular Weight": Descriptors.MolWt(mol),
                        "H-Bond Donors": Chem.Lipinski.NumHDonors(mol),
                        "H-Bond Acceptors": Chem.Lipinski.NumHAcceptors(mol),
                        "LogP": Descriptors.MolLogP(mol)
                    })
                except Exception as e:
                    print(f"Error processing molecule {i+1} in file: {e}")

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df_with_lip = is_lipinski(df)
            df_with_mol = add_RDKit_mol(df_with_lip)
            df_with_fp = add_fingerprints(df_with_mol)
            
            return df_with_fp

    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()

        
""" def process_partition_with_id(partition: Tuple[str, Any]) -> pd.DataFrame:
    Wrapper to extract key and value from partitioned dataset.
    partition_key, data = partition
    return process_single_raw_sdf_to_fingerprints(raw_sdf_file_path=data, file_id=partition_key)

 """

""" # --- Dummy initializations ---
def create_empty_fingerprint_dataframe() -> pd.DataFrame:
    Creates an empty DataFrame with the expected columns for final 
    fingerprints.
    # Define columns that will be in your final fingerprint DataFrame
    return pd.DataFrame(columns=[
        'SMILES', 'Molecular Weight', 'H-Bond Donors', 'H-Bond Acceptors',
        'LogP', 'RuleFive', 'OriginalFileID', 'Morgan2FP'
    ]) """
