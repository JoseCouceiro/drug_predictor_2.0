import gzip
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors

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

def process_inputs(directory_path: str, tracker: str) -> tuple[list[pd.DataFrame],str]:
    """Reads new gzipped SDF files, extracts molecular properties, and creates DataFrames.

    For each unprocessed file found, this function opens it, iterates through
    the molecules, calculates key chemical descriptors using RDKit, and
    compiles the data into a DataFrame. Each file results in one DataFrame.

    Args:
        directory_path: Path to the directory containing new .sdf.gz files.
        tracker: The current tracker string of processed filenames.

    Returns:
        A tuple containing:
            1. A list of pandas DataFrames, one for each newly processed SDF file.
            2. The updated tracker string with the new filenames.
    """
    tracker_updated, new_files = update_tracker(directory_path, tracker)
    df_list = []
    for filename, file_path in new_files:
        # Open the gzipped SDF file
        try:
            with gzip.open(file_path, 'rb') as gz:
                supplier = Chem.ForwardSDMolSupplier(gz)
                # Initialize a list to store data
                data = []
                # Iterate over each molecule in the file
                n = 1
                for mol in supplier:
                    n += 1
                    if mol is None:
                        continue
                    try:
                        # Access molecule properties
                        properties = mol.GetPropsAsDict()
                        # Add properties
                        data.append({
                            "SMILES": Chem.MolToSmiles(mol),
                            "Molecular Weight": Descriptors.MolWt(mol),
                            "H-Bond Donors": Chem.Lipinski.NumHDonors(mol),
                            "H-Bond Acceptors": Chem.Lipinski.NumHAcceptors(mol),
                            "LogP": Descriptors.MolLogP(mol),
                        })
                    except Exception as e:
                        print(f"Error processing molecule: {e}")
                # Create a DataFrame from the list of dictionaries
                df = pd.DataFrame(data)
                df_list.append(df)
        except Exception as e:
            print(f"Error reading the SDF file: {e}")
    return df_list, tracker_updated

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

def get_lipinski_dataframes(sdf_folder: str, tracker: str) -> tuple[list[pd.DataFrame],str]:
    """Processes new SDF files and applies Lipinski's Rule of Five to each.

    This function orchestrates the reading of new SDF files and the subsequent
    application of the Lipinski filter to the resulting DataFrames.

    Args:
        sdf_folder: Path to the directory containing input SDF files.
        tracker: A string record of already processed filenames.

    Returns:
        A tuple containing:
            1. A list of DataFrames, each with the 'RuleFive' column added.
            2. The updated tracker string.
    """
    df_list, tracker_updated = process_inputs(sdf_folder, tracker)
    df_lip_list = [is_lipinski(df) for df in df_list]
    return df_lip_list, tracker_updated

def update_dataframe(combined_csv: pd.DataFrame, sdf_folder: str, tracker: str) -> tuple[pd.DataFrame,str]:
    """Appends newly processed and filtered molecular data to an existing DataFrame.

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
    """
    df_lip_list, tracker_updated = get_lipinski_dataframes(sdf_folder, tracker)
    combined_csv_updated = pd.concat([combined_csv]+df_lip_list, ignore_index=True)
    return combined_csv_updated, tracker_updated