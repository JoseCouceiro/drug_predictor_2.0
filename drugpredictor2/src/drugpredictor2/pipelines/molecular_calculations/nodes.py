from typing import Optional, Tuple
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pt

"""IMPROVE: REMOVE THE WARNING FOR HIDROGEN BONDS"""
"""CREAR UNA FUNCIÓN QUE ANOTE LA ÚLTIMA MOLÉCULA CON FP"""
"""SALVAR COMO PARQUET, NO PICKLE"""
# Function to obtain fingerprints

def compute_morgan_fp(mol: Chem.Mol,
                      depth: int =2,
                      nBits: int =2048
                      ) -> Optional[np.array]:
    """Computes the Morgan fingerprint for a single RDKit molecule.

    Args:
        mol: The RDKit molecule object.
        depth: The radius of the fingerprint (morgan_radius).
        nBits: The length of the bit vector.

    Returns:
        A NumPy array representing the Morgan fingerprint as a bit vector.
        Returns None if the computation fails for any reason.
    """
    try:
        mor_fp = AllChem.GetMorganFingerprintAsBitVect(mol,depth,nBits)
        return np.array(mor_fp)
    except:
        print('Something went wrong computing Morgan fingerprints')
        return None

# Getting RDKit molecules

def add_molecule_column(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Adds an 'RDKit_Molecule' column to a DataFrame from a 'SMILES' column.

    This function uses RDKit's PandasTools to convert SMILES strings into
    RDKit molecule objects, modifying the DataFrame in-place.

    Args:
        combined_df: DataFrame containing a 'SMILES' column with molecular
            structures.

    Returns:
        The DataFrame with the new 'RDKit_Molecule' column added.
    """
    base_column = 'SMILES'
    calculated_column = 'RDKit_Molecule'
    pt.AddMoleculeColumnToFrame(
        frame=combined_df,
        smilesCol=base_column,
        molCol=calculated_column
        )
    return combined_df

# Adding fingerprints to the dataframe

def get_fingerprints(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Morgan fingerprints for each molecule in the DataFrame.

    This function maps the `compute_morgan_fp` function over the
    'RDKit_Molecule' column and adds the results to a new 'Morgan2FP' column.

    Args:
        combined_df: DataFrame containing an 'RDKit_Molecule' column with
            RDKit molecule objects.

    Returns:
        The DataFrame with a new 'Morgan2FP' column containing fingerprints.
    """
    combined_df['Morgan2FP'] = combined_df['RDKit_Molecule'].map(compute_morgan_fp) 
    return combined_df

# Training and validation split

def extract_validataion_dataset(
        combined_df: pd.DataFrame, validation_percentage: int
        ) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Splits a DataFrame into training and validation sets.

    Randomly samples a specified percentage of the DataFrame to create a
    validation set, with the remainder forming the training set.

    Args:
        combined_df: The complete dataset to be split.
        validation_percentage: An integer (0-100) representing the
            percentage of the data to allocate to the validation set.

    Returns:
        A tuple containing two DataFrames: (validation_set, training_set).
    """
    validation_set = combined_df.sample(int((validation_percentage/100)*len(combined_df)))
    training_set = combined_df.drop(index=validation_set.index)
    return validation_set, training_set

# Combined functions

def get_model_input(
        combined_df: pd.DataFrame, validation_percentage: int
        ) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Prepares training and validation datasets for model input.

    This function orchestrates a sequence of data preparation steps:
    1. Generates RDKit molecule objects from SMILES strings.
    2. Computes Morgan fingerprints for each molecule.
    3. Splits the resulting dataset into training and validation sets.

    Args:
        combined_df: A DataFrame containing at least a 'SMILES' column.
        validation_size: An integer (0-100) representing the percentage
            of data to use for the validation set.

    Returns:
        A tuple containing the two final DataFrames:
        (validation_set, training_set).
    """
    combined_df_mols = add_molecule_column(combined_df)
    combined_df_fingerprints = get_fingerprints(combined_df_mols)
    validation_set, training_set = extract_validataion_dataset(
        combined_df_fingerprints, validation_percentage
        )
    return validation_set, training_set
