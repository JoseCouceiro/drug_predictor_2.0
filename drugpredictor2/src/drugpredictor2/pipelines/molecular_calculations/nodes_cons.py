from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pt
import os # Added for path manipulation

# Function to obtain fingerprints
def compute_morgan_fp(mol: Chem.Mol,
                      depth: int = 2,
                      nBits: int = 2048
                      ) -> Optional[np.ndarray]: # Changed to np.ndarray as in your code
    """Computes the Morgan fingerprint for a single RDKit molecule."""
    try:
        # Use dtype=np.bool_ for significant memory savings if fingerprints are binary (0s and 1s)
        # Otherwise, np.int8 if you need integers and 0-255 range.
        # Your current np.array(mor_fp) will likely default to np.int64 or np.int32,
        # which is much larger than needed for binary fingerprints.
        mor_fp = AllChem.GetMorganFingerprintAsBitVect(mol, depth, nBits)
        return np.array(mor_fp, dtype=np.bool_) # <-- IMPORTANT MEMORY OPTIMIZATION
    except Exception as e:
        # print(f'Error computing Morgan fingerprints: {e}') # Suppress repeated printing for large datasets
        return None

# Getting RDKit molecules for a single DataFrame batch
def add_molecule_column_to_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    """Adds an 'RDKit_Molecule' column to a single DataFrame batch.

    This function is designed to work on a pre-selected batch of data (e.g.,
    one of the DataFrames from your `processed_lipinski_batches`).

    Args:
        df_batch: A DataFrame containing a 'SMILES' column.

    Returns:
        The input DataFrame with the 'RDKit_Molecule' column added.
    """
    base_column = 'SMILES'
    calculated_column = 'RDKit_Molecule'

    # PandasTools modifies the dataframe in-place
    pt.AddMoleculeColumnToFrame(
        frame=df_batch,
        smilesCol=base_column,
        molCol=calculated_column,
        # If your 'SMILES' column has missing values, you might want to specify
        # `removeSmi=False` and handle invalid SMILES later, or drop them now.
        # For now, let's assume `AddMoleculeColumnToFrame` handles this by putting None.
    )
    # print('MOLECULE COLUMN APPENDED TO BATCH') # Suppress for frequent calls
    return df_batch

# Get fingerprints for a single batch
def get_fingerprints_for_batch(df_batch_with_mols: pd.DataFrame) -> pd.DataFrame:
    """Computes Morgan fingerprints for molecules in a single DataFrame batch.

    Args:
        df_batch_with_mols: A DataFrame containing an 'RDKit_Molecule' column.

    Returns:
        The input DataFrame with the 'RDKit_Molecule' column replaced by
        a 'Morgan2FP' column containing NumPy arrays (or None if failed).
    """
    base_column = 'RDKit_Molecule'
    calculated_column = 'Morgan2FP'

    # Compute fingerprints for the subset
    df_batch_with_mols[calculated_column] = df_batch_with_mols[base_column].map(compute_morgan_fp)

    # Drop the intermediate RDKit_Molecule column
    dropped_subset_df = df_batch_with_mols.drop(columns=base_column)

    # Convert NumPy arrays to lists IF you intend to save to CSV/JSON later AND rely on exact string representation.
    # For Parquet, NumPy arrays can be stored directly, which is generally better.
    # If your downstream model expects NumPy arrays, DO NOT convert to list here.
    # We should save as NumPy arrays in Parquet for efficiency.
    # dropped_subset_df[calculated_column] = dropped_subset_df[calculated_column].apply(lambda x: x.tolist() if x is not None else None)
    
    # print('FINGERPRINTS OBTAINED FOR BATCH') # Suppress for frequent calls
    return dropped_subset_df


# --- New orchestrating node for fingerprinting individual processed SDF files ---
def process_single_lipinski_batch_to_fingerprints(
    lipinski_batch_df: pd.DataFrame,
    # The filename/key that identifies this specific batch from the PartitionedDataSet
    # Kedro passes this as the actual name of the dataset from the PartitionedDataSet load.
    # We need to explicitly capture it for naming the output file.
    batch_name: str # Kedro will inject the key of the dictionary
) -> Tuple[str, pd.DataFrame]:
    """
    Processes a single batch of Lipinski-filtered data to generate Morgan fingerprints.

    This node takes one DataFrame (representing one original SDF file's processed data),
    adds RDKit molecule objects, computes fingerprints, and returns the resulting
    DataFrame along with a key to save it individually.

    Args:
        lipinski_batch_df: A DataFrame containing 'SMILES' and 'RuleFive' columns
                           for a single batch of molecules.
        batch_name: The string key/name of this specific batch from the PartitionedDataSet.

    Returns:
        A tuple: (key_for_saving, DataFrame_with_fingerprints).
    """
    if lipinski_batch_df.empty:
        print(f"Skipping empty batch: {batch_name}")
        return batch_name, pd.DataFrame() # Return empty if no data

    # 1. Add RDKit Molecule objects to the batch
    df_with_mols = add_molecule_column_to_batch(lipinski_batch_df)

    # 2. Compute fingerprints for this batch
    df_with_fps = get_fingerprints_for_batch(df_with_mols)

    # Ensure consistent columns before returning if some columns might be missing
    # Example: if 'SMILES', 'RuleFive', 'Morgan2FP', 'Label' are expected
    # Make sure 'Label' (if present in original `lipinski_batch_df`) is carried over
    expected_cols = ['SMILES', 'RuleFive', 'Morgan2FP'] # Adjust based on your actual data
    if 'Label' in lipinski_batch_df.columns and 'Label' not in df_with_fps.columns:
        df_with_fps['Label'] = lipinski_batch_df['Label']
        expected_cols.append('Label')

    # Optionally, drop rows where fingerprint computation failed
    initial_len = len(df_with_fps)
    df_with_fps.dropna(subset=['Morgan2FP'], inplace=True)
    if len(df_with_fps) < initial_len:
        print(f"Dropped {initial_len - len(df_with_fps)} rows in batch {batch_name} due to failed fingerprinting.")
    
    # Return the batch name and the processed DataFrame
    return batch_name, df_with_fps


# --- Dummy initializations (if needed for empty datasets) ---
# Removed get_model_input, get_fingerprints, extract_validataion_dataset as they
# are replaced by the new batch-oriented functions or external split.

def create_empty_fingerprint_dataframe() -> pd.DataFrame:
    """Creates an empty DataFrame with the expected columns for fingerprints."""
    return pd.DataFrame(columns=['SMILES', 'Molecular Weight', 'H-Bond Donors',
                                 'H-Bond Acceptors', 'LogP', 'RuleFive', 'Morgan2FP'])
