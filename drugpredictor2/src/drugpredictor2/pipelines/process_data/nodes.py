import traceback
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, QED, rdMolDescriptors, PandasTools as pt
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
        # float32, not float16: PyArrow/Parquet can't serialize half-precision
        # floats ("Unhandled type for Arrow to Parquet schema conversion: halffloat").
        return np.zeros(total_len, dtype=np.float32)

    # Morgan ECFP
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits, useFeatures=False)
    morgan_arr = np.array(morgan_fp, dtype=np.float32)

    # Feature-based Morgan (FCFP-like)
    morgan_feat_fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_feat_bits, useFeatures=True)
    morgan_feat_arr = np.array(morgan_feat_fp, dtype=np.float32)

    # MACCS
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.array(maccs_fp, dtype=np.float32)

    # Hashed Atom Pair
    ap_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=ap_bits)
    ap_arr = np.array(ap_fp, dtype=np.float32)

    # Hashed Topological Torsion
    tt_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=tt_bits)
    tt_arr = np.array(tt_fp, dtype=np.float32)

    # TPSA
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    tpsa_arr = np.array([tpsa], dtype=np.float32)

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

def add_auxiliary_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute richer structure-derived targets for multi-task pretraining.

    QED (continuous drug-likeness), rotatable bond count, aromatic ring count,
    and sp3 carbon fraction give the backbone more supervisory signal than the
    single binary Lipinski rule, computed from the same RDKit mol already built
    for fingerprinting.
    """
    base_column = 'RDKit_Molecule'

    def _safe_qed(mol):
        try:
            return QED.qed(mol)
        except Exception:
            return np.nan

    df['QED'] = df[base_column].map(_safe_qed)
    df['NumRotatableBonds'] = df[base_column].map(rdMolDescriptors.CalcNumRotatableBonds)
    df['NumAromaticRings'] = df[base_column].map(rdMolDescriptors.CalcNumAromaticRings)
    df['FractionCSP3'] = df[base_column].map(rdMolDescriptors.CalcFractionCSP3)

    aux_cols = ['QED', 'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3']
    initial_len = len(df)
    df.dropna(subset=aux_cols, inplace=True)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows due to failed auxiliary descriptor computation.")

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
    existing_fnames: Set[str],
    output_dir: str,
) -> Dict[str, int]:
    """
    Processes only the new partitions that do not have a corresponding output file,
    writing each one to disk IMMEDIATELY after it's computed.

    Previously all newly-processed partitions were accumulated in memory and
    only persisted via Kedro's PartitionedDataSet when this function returned.
    Since fingerprinting hundreds of thousands of molecules can crash or get
    OOM-killed partway through, that meant a crash on partition 20/31 lost the
    first 19 too — nothing was written until the very end. Writing per-partition
    means a crash only loses the partition being processed at that moment; a
    rerun correctly resumes from wherever it left off.

    Args:
        partitions_dict: A dictionary of all raw partitions.
                         Keys are full filenames (e.g., 'my_file.csv').
                         Values are the loading functions.
        existing_fnames: A set of base filenames that are already processed,
                         as returned by get_existing_fnames().
        output_dir: Directory to write each processed partition to immediately
                    (same directory backing the 'featurized_data' catalog entry).

    Returns:
        A small summary dict {partition_stem: row_count} for logging/provenance —
        NOT the actual DataFrames, which are already persisted to disk directly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {}
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
            df_with_aux = add_auxiliary_descriptors(df_with_mol)
            final_df = add_fingerprints(df_with_aux)

            # Persist immediately — survives a crash on a LATER partition.
            final_df.to_parquet(output_path / f"{partition_stem}.parquet")
            summary[partition_stem] = len(final_df)
            print(f"Successfully processed and saved partition: {partition_stem} ({len(final_df)} rows)")

        except Exception as e:
            print(f"ERROR: Failed to process new partition {raw_filename}. Error: {e}")
            traceback.print_exc()

    if not summary:
        print("No new partitions to process.")
        
    return summary