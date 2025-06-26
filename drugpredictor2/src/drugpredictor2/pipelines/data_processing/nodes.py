import gzip
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors

"CAMBIA LA LÓGICA, EN VEZ DE SALVAR CSV INDIVIDUALES, SALVA SOLO UN CSV QUE SE CONCATENA AL YA EXISTENTE"

def get_unprocessed_files(directory_path, tracker):
    #processed_files = set()
    directory_path_list = directory_path.split('/')
    #print('LIST: ', directory_path_list)
    directory_path_os = os.path.join(*directory_path_list)
    #print('DIR OS: ', directory_path_os)
    all_files = [
        (file, os.path.join(directory_path_os, file))
        for file in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, file))
    ]
    #print('TRACKER: ', tracker, 'TYPE: ', type(tracker))
    #tracker_list = tracker.split('\n')
    #print('TRACKER_LIST: ', tracker_list)
    #print('ALL_FILES', all_files)
    #print('UNPROCESSED FILES: ', [(file, filepath) for (file, filepath) in all_files if file not in tracker_list])
    #returns a list of the tuples: (filename, filepath) with all the files not in tracker
    #print([(file, filepath) for (file, filepath) in all_files if file not in tracker])
    return [(file, filepath) for (file, filepath) in all_files if file not in tracker]

def update_tracker(directory_path, tracker):
    new_files = get_unprocessed_files(directory_path, tracker)
    for (filename, filepath) in new_files:
        tracker +='\n'+filename
    return tracker, new_files

def process_inputs(directory_path, tracker):
    #print('DIR PATH: ', directory_path)
    #print('TRACKER: ', tracker)
    tracker_updated, new_files = update_tracker(directory_path, tracker)
    df_list = []
    for filename, file_path in new_files:
        print('FILENAME: ', f'reading {filename}')
        # Open the gzipped SDF file
        try:
            with gzip.open(file_path, 'rb') as gz:
                supplier = Chem.ForwardSDMolSupplier(gz)
                # Initialize a list to store data
                data = []
                # Iterate over each molecule in the file
                n = 1
                for mol in supplier:
                    #print(n)
                    n += 1
                    if mol is None:
                        continue
                    try:
                        # Access molecule properties
                        properties = mol.GetPropsAsDict()
                        # Example: Add a specific property, add more as needed
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
    print('OUTPUTS: ', df_list, tracker_updated)
    return df_list, tracker_updated


def is_lipinski(x: pd.DataFrame) -> pd.DataFrame:
    """
    Function that applies a set of rules (Lipinski rules) to several columns of a pandas dataframe and returns \
          a dataframe with a new column that states if said rules were passed or not.
    Input: pandas dataframe.
    Output: pandas dataframe.
    """
    # Lipinski rules
    hdonor = x['H-Bond Donors'] <= 5
    haccept = x['H-Bond Acceptors'] <= 10
    mw = x['Molecular Weight'] < 500
    clogP = x['LogP'] <= 5
    # Apply rules to dataframe
    x['RuleFive'] = np.where(((hdonor & haccept & mw) | (hdonor & haccept & clogP) | (hdonor & mw & clogP) | (haccept & mw & clogP)), 1, 0)
    return x

def get_lipinski_dataframes(sdf_folder, tracker):
    df_list, tracker_updated = process_inputs(sdf_folder, tracker)
    df_lip_list = []
    for df in df_list:
        df_lip = is_lipinski(df)
        df_lip_list.append(df_lip)
    #print('OUTPUTS GET LIP: ', df_lip_list, tracker_updated)
    return df_lip_list, tracker_updated
        
def sdf_to_csv(sdf_folder, csv_folder, tracker):
    df_list, tracker_updated = process_inputs(sdf_folder, tracker)
    print('DF_LIST INSIDE SDF_TO_CSV: ', df_list)
    df_lip_list = get_lipinski_dataframes(df_list, csv_folder, sdf_folder, tracker_updated)
    for df_lip in df_lip_list:
        return df_lip, tracker_updated

# concat csv files in a combined csv
"""Cambiar esto para concadenar el combined_output con los nuevos csv.
Para ello hay que usar el tracker"""
def concat_dataframes(csv_path):
    # List to store DataFrames
    dataframes = []
    # Iterate over files in the folder
    for file in os.listdir(csv_path):
        # Check if the file is a CSV
        if file.endswith('.csv'):
            file_path = os.path.join(csv_path, file)
            
            # Read CSV into DataFrame and append to list
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    return combined_df

    print("CSV files concatenated successfully.")

def update_dataframe(combined_csv, sdf_folder, tracker):
    print('LOADED COMBINED: ', combined_csv)
    df_lip_list, tracker_updated = get_lipinski_dataframes(sdf_folder, tracker)
    print('DF_LIP_LIST IN UPDATE: ', df_lip_list)
    print('TO CONCAT: ', [combined_csv]+df_lip_list)
    combined_csv_updated = pd.concat([combined_csv]+df_lip_list, ignore_index=True)
    print('COMBINED: ', combined_csv_updated)
    return combined_csv_updated, tracker_updated