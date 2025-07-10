from typing import Optional, Tuple
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pt

# Function to obtain fingerprints

def compute_morgan_fp(mol: Chem.Mol,
                      depth: int =2,
                      nBits: int =2048
                      ) -> Optional[np.array]:
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
        return np.array(mor_fp, dtype=np.bool)
    except Exception as e:
        print(f'Error computing Morgan fingerprints: {e}')
        return None

# Getting RDKit molecules

def add_molecule_column(df_batch: pd.DataFrame
                        ) -> pd.DataFrame:
    """Adds an 'RDKit_Molecule' column to a DataFrame for new rows.

    This function converts 'SMILES' strings from the 'SMILES' column into RDKit molecule objects,
    adding them to a new 'RDKit_Molecule'.

    Args:
        df_batch: The input DataFrame containing a 'SMILES' column with molecular structures.

    Returns:
        A new DataFrame subset containing the 'RDKit_Molecule' column added.
    """
    base_column = 'SMILES'
    calculated_column = 'RDKit_Molecule'
    
    # PandasTools modifies the dataframe in-place
    pt.AddMoleculeColumnToFrame(
        frame=df_batch,
        smilesCol=base_column,
        molCol=calculated_column
        )
    print('MOLECULE COLUMN APPENDED FOR BATCH')
    return df_batch

# Adding fingerprints to the dataframe

def get_fingerprints(df_batch_with_mols: pd.DataFrame
                    ) -> pd.DataFrame:
    """Computes Morgan fingerprints for molecules in a DataFrame.

    This function takes a DataFrame subset (typically containing new rows with RDKit molecule
    objects), computes Morgan fingerprints for each molecule, and drops the molecule column.

    Args:
        df_batch_with_mols: A DataFrame containing an 'RDKit_Molecule' for which fingerprints 
            need to be computed.

    Returns:
        The input DataFrame with the 'RDKit_Molecule' column replaced by
        a 'Morgan2FP' column containing NumPy arrays (or None if failed).
    """
    base_column = 'RDKit_Molecule'
    calculated_column = 'Morgan2FP'

    # Compute fingerprints for the subset
    df_batch_with_mols[calculated_column] = df_batch_with_mols[base_column].map(compute_morgan_fp)

    # Drop the intermediate RDKit_Molecule column from the subset
    dropped_df_batch_with_mols = df_batch_with_mols.drop(columns=base_column)

    print('FINGERPRINTS OBTAINED FOR BATCH')
    return dropped_df_batch_with_mols

# Training and validation split

""" def extract_validataion_dataset(
        combined_df: pd.DataFrame,
        validation_percentage: int
        ) -> Tuple[pd.DataFrame,pd.DataFrame]:
    Splits a DataFrame into training and validation sets.

    Randomly samples a specified percentage of the DataFrame to create a
    validation set, with the remainder forming the training set.

    Args:
        combined_df: The complete dataset (e.g., after fingerprint generation)
            to be split.
        validation_percentage: An integer (0-100) representing the
            percentage of the data to allocate to the validation set.

    Returns:
        A tuple containing two DataFrames: (validation_set, training_set).
   
    n_validation_samples = int((validation_percentage / 100) * len(combined_df))
    validation_set = combined_df.sample(n_validation_samples)
    training_set = combined_df.drop(index=validation_set.index)
    print('TRAIN/VALIDATION SPLIT')
    return validation_set, training_set """

# Combined functions

def process_batch_to_fingerprints(
        batch_df: pd.DataFrame,
        batch_name: str
        ) -> Tuple[str,pd.DataFrame]:
    """Processes a single batch of data containing SMILES to generate Morgan fingerprints.

    This node takes one DataFrame (representing one original SDF file's processed data),
    adds RDKit molecule objects, computes fingerprints, and returns the resulting
    DataFrame along with a key to save it individually.

    Args:
        batch_df: A DataFrame containing at least a 'SMILES' column. This
            represents the current full input data, from which new rows are processed.
        batch_name: The string key/name of this specific batch from the PartitionedDataSet.

    Returns:
        A tuple: (key_for_saving, DataFrame_with_fingerprints).
    """
    # 1. Add RDKit Molecule objects to new rows in 'combined_df'
    df_with_mols = add_molecule_column(batch_df)

    # 2. Get fingerprints for the new subset and update the main fingerprint DataFrame
    df_with_fps = get_fingerprints(df_with_mols)

    # 3. Drop rows where fingerprint computation failed
    initial_len = len(df_with_fps)
    df_with_fps.dropna(subset=['Morgan2FP'], inplace=True)
    if len(df_with_fps) < initial_len:
        print(f"Dropped {initial_len - len(df_with_fps)} rows in batch {batch_name} due to failed fingerprinting.")
    
    return batch_name, df_with_fps

def create_empty_fingerprint_dataframe() -> pd.DataFrame:
    """Creates an empty DataFrame with the expected columns for fingerprints."""
    # Define columns that will be in your final fingerprint DataFrame
    # Adjust based on your actual data (e.g., 'SMILES', 'RuleFive', 'Morgan2FP', 'Label')
    return pd.DataFrame(columns=['SMILES', 'Molecular Weight', 'H-Bond Donors',
                                 'H-Bond Acceptors', 'LogP', 'RuleFive', 'Morgan2FP'])
