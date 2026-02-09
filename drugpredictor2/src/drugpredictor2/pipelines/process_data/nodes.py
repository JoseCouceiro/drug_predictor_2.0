import traceback
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, PandasTools as pt
from typing import Optional, Callable, Dict, Set, Any

# --- Core functions ---

def featurize_molecule(mol,
                                morgan_radius=2,
                                morgan_bits=2048,
                                morgan_feat_bits=1024,
                                MAACS_bits=167,
                                ap_bits=1024,
                                tt_bits=1024,
                                tapsa_weight=1) -> np.ndarray:
    """    Returns concatenated numpy array of:º
      - Morgan (ECFP)
      - Feature-based Morgan (FCFP)
      - MACCS keys
      - Hashed Atom Pair fingerprint
      - Hashed Topological Torsion fingerprint
      - TPSA (1 value)
    """

    if mol is None:
        total_len = morgan_bits + morgan_feat_bits + MAACS_bits + ap_bits + tt_bits + tapsa_weight
        return np.zeros(total_len, dtype=float)

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
    calculated_column = 'FP'

    # Compute fingerprints
    df[calculated_column] = df[base_column].map(featurize_molecule)
    
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