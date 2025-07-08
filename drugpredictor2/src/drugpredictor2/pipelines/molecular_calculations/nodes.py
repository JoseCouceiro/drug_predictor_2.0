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
        A NumPy array representing the Morgan fingerprint as a bit vector
        (e.g., array([0, 1, 0, ..., 1])). Returns None if the fingerprint
        computation fails (e.g., for an invalid molecule).
    """
    try:
        mor_fp = AllChem.GetMorganFingerprintAsBitVect(mol, depth, nBits)
        return np.array(mor_fp)
    except Exception as e:
        print(f'Error computing Morgan fingerprints: {e}')
        return None

# Getting RDKit molecules

def add_molecule_column(df: pd.DataFrame,
                        fp_tracker: str
                        ) -> pd.DataFrame:
    """Adds an 'RDKit_Molecule' column to a DataFrame for new rows.

    This function identifies new rows in the input DataFrame based on `fp_tracker`
    and converts their 'SMILES' strings into RDKit molecule objects,
    adding them to a new 'RDKit_Molecule' column within a *subset* of the DataFrame.

    Args:
        df: The input DataFrame containing a 'SMILES' column with molecular structures.
            This DataFrame is expected to contain both previously processed and new rows.
        fp_tracker: An integer (as a string or int) indicating the index
            from which new rows begin. Rows at or after this index will be
            processed. This allows for incremental processing of large files.

    Returns:
        A new DataFrame subset containing only the newly processed rows
        with the 'RDKit_Molecule' column added. The original `df` is
        not modified by this function.
    """
    base_column = 'SMILES'
    calculated_column = 'RDKit_Molecule'

    # Process only new rows starting from the old value of n
    subset_df = df.iloc[int(fp_tracker):].copy()
    
    # PandasTools modifies the dataframe in-place
    pt.AddMoleculeColumnToFrame(
        frame=subset_df,
        smilesCol=base_column,
        molCol=calculated_column
        )
    print('MOLECULE COLUMN APPENDED')
    return subset_df

# Adding fingerprints to the dataframe

def get_fingerprints(subset_df: pd.DataFrame,
                     df_w_fp: pd.DataFrame,
                    ) -> pd.DataFrame:
    """Computes Morgan fingerprints for molecules in a DataFrame subset and updates a main DataFrame.

    This function takes a DataFrame subset (typically containing new rows with RDKit molecule
    objects), computes Morgan fingerprints for each molecule, drops the molecule column,
    and then concatenates these new fingerprint rows with an existing main DataFrame that
    already contains fingerprints.

    Crucially, it also ensures that the computed molecular fingerprints (which are
    initially NumPy arrays) are converted into standard Python lists. This conversion
    is vital to prevent data truncation (e.g., '...') when the combined DataFrame is
    later saved to text-based formats like CSV, allowing for reliable parsing upon reloading.

    Args:
        subset_df: A DataFrame containing an 'RDKit_Molecule' column (usually
            representing newly added molecules) for which fingerprints need to be computed.
        df_w_fp: The main DataFrame that already contains fingerprints (a 'Morgan2FP' column)
            and will be updated with the results from `subset_df`.

    Returns:
        A tuple containing:
        - combined_df_w_fp (pd.DataFrame): The updated main DataFrame with the
          newly computed fingerprints appended. The 'Morgan2FP' column in this
          DataFrame will contain Python lists (not NumPy arrays) for each fingerprint.
        - updated_fp_tracker (str): The new length of `combined_df_w_fp` as a string,
          to be used as the starting index for the next batch of processing.

    Notes:
        - Molecules for which fingerprint computation fails will have `None` in
          the 'Morgan2FP' column.
        - The `RDKit_Molecule` column is dropped from `subset_df` before concatenation.
        - This function is designed for incremental processing, appending new
          fingerprints to an existing DataFrame.
    """
    base_column = 'RDKit_Molecule'
    calculated_column = 'Morgan2FP'

    # Compute fingerprints for the subset
    subset_df[calculated_column] = subset_df[base_column].map(compute_morgan_fp)

    # Drop the intermediate RDKit_Molecule column from the subset
    dropped_subset_df = subset_df.drop(columns=base_column)

    #combined_df_w_fp.loc[fp_tracker:, calculated_column] = dropped_subset_df[calculated_column]
    combined_df_w_fp = pd.concat([df_w_fp, dropped_subset_df], ignore_index=True)
    
    # Update the tracker with the new total length of the combined DataFrame
    updated_fp_tracker = str(len(combined_df_w_fp))
    combined_df_w_fp[calculated_column] = combined_df_w_fp[calculated_column].apply(lambda x: x.tolist())
    print('FINGERPRINTS OBTAINED')
    return combined_df_w_fp, updated_fp_tracker

# Training and validation split

def extract_validataion_dataset(
        combined_df: pd.DataFrame,
        validation_percentage: int
        ) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Splits a DataFrame into training and validation sets.

    Randomly samples a specified percentage of the DataFrame to create a
    validation set, with the remainder forming the training set.

    Args:
        combined_df: The complete dataset (e.g., after fingerprint generation)
            to be split.
        validation_percentage: An integer (0-100) representing the
            percentage of the data to allocate to the validation set.

    Returns:
        A tuple containing two DataFrames: (validation_set, training_set).
    """
    n_validation_samples = int((validation_percentage / 100) * len(combined_df))
    validation_set = combined_df.sample(n_validation_samples)
    training_set = combined_df.drop(index=validation_set.index)
    print('TRAIN/VALIDATION SPLIT')
    return validation_set, training_set

# Combined functions

def get_model_input(
        combined_df: pd.DataFrame,
        df_w_fp: pd.DataFrame,
        validation_percentage: int,
        fp_tracker: str
        ) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Prepares training and validation datasets for model input, including incremental fingerprinting.

    This function orchestrates a sequence of data preparation steps:
    1. Identifies new rows for processing based on `fp_tracker`.
    2. Generates RDKit molecule objects from SMILES strings for these new rows.
    3. Computes Morgan fingerprints for the newly generated molecules.
    4. Appends these new fingerprints to an existing DataFrame containing previous fingerprints.
    5. Splits the *entire* updated dataset (after new fingerprints are appended)
       into training and validation sets.

    This design is suitable for continuously processing new data chunks from a large source
    file and incrementally building a complete fingerprint DataFrame.

    Args:
        combined_df: A DataFrame containing at least a 'SMILES' column. This
            represents the current full input data, from which new rows are processed.
        df_w_fp: The existing DataFrame that stores previously computed fingerprints.
            New fingerprints will be appended to this DataFrame. This should
            typically be empty at the first call or contain previous results.
        validation_percentage: An integer (0-100) representing the percentage
            of the final, complete dataset to allocate to the validation set.
        fp_tracker: An integer (as a string or int) indicating the index
            from which to start processing new rows from `combined_df`.
            This value is typically the length of `df_w_fp` from the previous iteration.

    Returns:
        A tuple containing the four results:
        - final_combined_df_fingerprints (pd.DataFrame): The complete DataFrame
          with all SMILES converted to Morgan fingerprints,
          including newly processed rows.
        - validation_set (pd.DataFrame): The DataFrame containing the
          validation subset.
        - training_set (pd.DataFrame): The DataFrame containing the training subset.
        - updated_fp_tracker (str): The new tracker value (total length of the
          `final_combined_df_fingerprints`), indicating where to start processing
          the *next* batch of new data.
    """
    # 1. Add RDKit Molecule objects to new rows in 'combined_df'
    subset_df_mols = add_molecule_column(combined_df, fp_tracker)

    # 2. Get fingerprints for the new subset and update the main fingerprint DataFrame
    combined_df_fingerprints, updated_fp_tracker= get_fingerprints(subset_df_mols, df_w_fp)
    
    # 3. Split the *entire* updated fingerprint DataFrame into training and validation sets
    validation_set, training_set = extract_validataion_dataset(
        combined_df_fingerprints, validation_percentage
        )
    return combined_df_fingerprints, validation_set, training_set, updated_fp_tracker
