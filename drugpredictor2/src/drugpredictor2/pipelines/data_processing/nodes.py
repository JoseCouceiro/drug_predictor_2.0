import gzip
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors
from typing import List, Tuple, Dict, Any, Optional

def get_unprocessed_files(directory_path: str, tracker: str) -> list[tuple[str,str]]:
    """Identifies new files in a directory that have not been processed yet.

    Compares the list of files in `directory_path` against a tracker string
    containing filenames of previously processed files.

    Args:
        directory_path: Path to the directory containing input files.
        tracker: A string containing newline-separated filenames that have
            already been processed.

    Returns:
        A list of tuples, where each tuple contains the filename and the full
        filepath of an unprocessed file.
    """
    directory_path_list = directory_path.split('/')
    directory_path_os = os.path.join(*directory_path_list)
    all_files = [
        (file, os.path.join(directory_path_os, file))
        for file in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, file))
    ]
    # Filter out files that are already listed in the tracker string
    return [(file, filepath) for (file, filepath) in all_files if file not in tracker]

def update_tracker(directory_path: str, tracker: str) -> tuple[str,list[tuple[str,str]]]:
    """Updates the tracker with new filenames and returns the list of new files.

    This function first identifies unprocessed files and then appends their
    filenames to the tracker string, separated by newlines.

    Args:
        directory_path: Path to the directory containing input files.
        tracker: The current tracker string of processed filenames.

    Returns:
        A tuple containing:
            1. The updated tracker string.
            2. The list of new files to be processed (filename, filepath).
    """
    new_files = get_unprocessed_files(directory_path, tracker)
    for (filename, filepath) in new_files:
        # Appends with a newline to ensure clean separation
        tracker +='\n'+filename
    return tracker, new_files

def process_inputs(directory_path: str, tracker: str) -> Tuple[List[Tuple[str, pd.DataFrame]], str]:
    """Reads new gzipped SDF files, extracts molecular properties, and creates DataFrames.

    Returns:
        A tuple containing:
            1. A list of tuples, where each tuple is (filename, pandas DataFrame)
               for each newly processed SDF file.
            2. The updated tracker string with the new filenames.
    """
    tracker_updated, new_files = update_tracker(directory_path, tracker)
    df_with_filenames_list = [] # Store tuples of (filename, DataFrame)

    for filename, file_path in new_files:
        data = []
        try:
            with gzip.open(file_path, 'rb') as gz:
                supplier = Chem.ForwardSDMolSupplier(gz)
                for i, mol in enumerate(supplier):
                    if mol is None:
                        print(f"Warning: Skipping invalid molecule in {filename} at index {i}")
                        continue
                    try:
                        data.append({
                            "SMILES": Chem.MolToSmiles(mol),
                            "Molecular Weight": Descriptors.MolWt(mol),
                            "H-Bond Donors": Chem.Lipinski.NumHDonors(mol),
                            "H-Bond Acceptors": Chem.Lipinski.NumHAcceptors(mol),
                            "LogP": Descriptors.MolLogP(mol),
                            "OriginalFile": filename
                        })
                    except Exception as e:
                        print(f"Error processing molecule {i+1} in {filename}: {e}")
            if data:
                df = pd.DataFrame(data)
                df_with_filenames_list.append((filename, df))
            else:
                print(f"No valid molecules found in {filename}.")

        except Exception as e:
            print(f"Error reading the SDF file {file_path}: {e}")

    return df_with_filenames_list, tracker_updated

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

""" def get_lipinski_dataframes(sdf_folder: str, tracker: str) -> tuple[list[pd.DataFrame],str]:
    Processes new SDF files and applies Lipinski's Rule of Five to each.

    This function orchestrates the reading of new SDF files and the subsequent
    application of the Lipinski filter to the resulting DataFrames.

    Args:
        sdf_folder: Path to the directory containing input SDF files.
        tracker: A string record of already processed filenames.

    Returns:
        A tuple containing:
            1. A list of DataFrames, each with the 'RuleFive' column added.
            2. The updated tracker string.
   
    df_list, tracker_updated = process_inputs(sdf_folder, tracker)
    df_lip_list = [is_lipinski(df) for df in df_list]
    return df_lip_list, tracker_updated """

""" def update_dataframe(combined_csv: pd.DataFrame, sdf_folder: str, tracker: str) -> tuple[pd.DataFrame,str]:
    Appends newly processed and filtered molecular data to an existing DataFrame.

    This node takes an existing DataFrame, processes all new SDF files from a
    specified folder, filters them using Lipinski's rules, and concatenates
    the new data with the existing DataFrame.

    Args:
        combined_csv: The cumulative DataFrame of previously processed molecules.
        sdf_folder: Path to the directory containing new SDF files to process.
        tracker: A string record of already processed filenames.

    Returns:
        A tuple containing:
            1. The combined DataFrame with new data appended.
            2. The updated tracker string with new filenames added.
   
    df_lip_list, tracker_updated = get_lipinski_dataframes(sdf_folder, tracker)
    combined_csv_updated = pd.concat([combined_csv]+df_lip_list, ignore_index=True)
    return combined_csv_updated, tracker_updated """

def apply_lipinski_and_prepare_for_saving(
    processed_dfs_with_filenames: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, pd.DataFrame]:
    """Applies Lipinski's rule to each DataFrame and prepares them for individual saving.

    Args:
        processed_dfs_with_filenames: A list of tuples, each containing
                                      (filename, DataFrame) from process_inputs.

    Returns:
        A dictionary where keys are unique identifiers for saving (e.g., cleaned filenames)
        and values are the processed DataFrames with the 'RuleFive' column.
        This allows Kedro to save each DataFrame individually.
    """
    output_dataframes = {}
    for filename, df in processed_dfs_with_filenames:
        df_lip = is_lipinski(df)
        # Create a unique key for the catalog.
        # Replace non-alphanumeric chars for valid dataset names if needed,
        # or use a simple prefix + filename without extension.
        # For example, "file_processed_data_<original_filename_no_ext>"
        base_filename = os.path.splitext(os.path.splitext(filename)[0])[0] # Remove .sdf.gz
        output_key = f"processed_sdf_file_{base_filename.replace('.', '_').replace('-', '_')}"
        output_dataframes[output_key] = df_lip
    return output_dataframes
