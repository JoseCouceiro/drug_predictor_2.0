import traceback
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools as pt
from typing import Optional, Callable, Dict, Set, Any

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

""" def is_lipinski(x: pd.DataFrame) -> pd.DataFrame:
    Applies Lipinski's Rule of Five to a DataFrame of molecular properties.

    Calculates whether a molecule adheres to at least three of the four main
    Lipinski rules (MW < 500, LogP <= 5, H-Bond Donors <= 5,
    H-Bond Acceptors <= 10). Adds a 'RuleFive' column where 1 indicates
    compliance (passes >= 3 rules) and 0 indicates failure.

    Args:
        x: DataFrame containing molecular properties, including 'Molecular Weight',
           'LogP', 'H-Bond Donors', and 'H-Bond Acceptors'.

    Returns:
        The input DataFrame with an added 'RuleFive' integer column.
   
    # Lipinski rules
    hdonor = x['H-Bond Donors'] <= 5
    haccept = x['H-Bond Acceptors'] <= 10
    mw = x['Molecular Weight'] < 500
    clogP = x['LogP'] <= 5
    # Apply rules to dataframe
    x['RuleFive'] = np.where(((hdonor & haccept & mw) | (hdonor & haccept & clogP) | (hdonor & mw & clogP) | (haccept & mw & clogP)), 1, 0)
    return x
 """
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
    return df

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

def get_existing_fnames(directory_path: str) -> Set[str]:
    """
    Scans a directory for processed files and returns a set of their
    base filenames (stems), ignoring extensions.

    Args:
        directory_path: The path to the folder containing already
                        processed files (e.g., 'featurized_data').

    Returns:
        A set of base filenames found in the directory.
    """
    processed_dir = Path(directory_path)
    if not processed_dir.is_dir():
        print(f"Directory '{directory_path}' not found. Assuming no files exist.")
        return set()
    
    existing_fnames = {p.stem for p in processed_dir.iterdir() if p.is_file()}
    
    print(f"Found {len(existing_fnames)} already processed files in '{directory_path}'.")
    return existing_fnames

def process_new_partitions(
    partitions_dict: Dict[str, Callable[[], pd.DataFrame]],
    existing_fnames: Set[str]
) -> Dict[str, pd.DataFrame]:
    """
    Processes only the new partitions that do not have a corresponding output file.

    It compares the base name of the raw files (from partitions_dict keys)
    against the base names of existing processed files.

    Args:
        partitions_dict: A dictionary of all raw partitions.
                         Keys are full filenames (e.g., 'my_file.csv').
                         Values are the loading functions.
        existing_fnames: A set of base filenames that are already processed,
                         as returned by get_existing_fnames().

    Returns:
        A dictionary containing only the newly processed DataFrames, where keys
        are the base filenames.
    """
    processed_data = {}
    print(f"Received {len(partitions_dict)} total raw partitions. Checking against {len(existing_fnames)} existing processed files.")

    for raw_filename, partition_load_func in partitions_dict.items():
        # Get the base name of the raw file to compare with existing files.
        partition_stem = Path(raw_filename).stem

        # SKIP if the base name is in the existing set
        if partition_stem in existing_fnames:
            continue

        print(f"--- Processing NEW partition: {raw_filename} ---")
        try:
            partition_df = partition_load_func()

            # Apply the sequence of transformations
            #df_with_lip = is_lipinski(partition_df)
            df_with_mol = add_RDKit_mol(partition_df)
            final_df = add_fingerprints(df_with_mol)
            
            processed_data[partition_stem] = final_df
            print(f"Successfully processed new partition: {partition_stem}")

        except Exception as e:
            print(f"ERROR: Failed to process new partition {raw_filename}. Error: {e}")
            traceback.print_exc()

    if not processed_data:
        print("No new partitions to process.")
        
    return processed_data