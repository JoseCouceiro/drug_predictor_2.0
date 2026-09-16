import os
import sys
import pickle
import numpy as np
import pandas as pd
import pubchempy as pcp
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# Reuse the EXACT fingerprint function every model in this project was
# trained on, so app-time features always match training-time features.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from drugpredictor2.pipelines.process_data.nodes import featurize_molecule

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
RES_DIR = os.path.join(os.path.dirname(__file__), 'res')

# ATC level-1 code -> human-readable description, as this dataset actually
# uses each code (NOT always the standard WHO meaning: this dataset
# repurposes 'I' for "Antiinflammatory" and 'O' for "Lipid regulation",
# verified against drug_raw['MATC_Code_Explanation']; see process_drug_data's
# ACTION_BASED_CODES/ORGAN_BASED_CODES comment for the full breakdown).
ATC_DESCRIPTIONS = {
    'A': 'Alimentary tract and metabolism',
    'B': 'Blood and blood forming organs',
    'C': 'Cardiovascular system',
    'D': 'Dermatologicals',
    'G': 'Genito-urinary system and sex hormones',
    'H': 'Systemic hormonal preparations',
    'I': 'Antiinflammatory',
    'J': 'Antiinfectives for systemic use',
    'L': 'Antineoplastic and immunomodulating agents',
    'M': 'Musculo-skeletal system',
    'N': 'Nervous system',
    'O': 'Lipid regulation',
    'P': 'Antiparasitic products',
    'R': 'Respiratory system',
    'S': 'Sensory organs',
}


@st.cache_resource(show_spinner=False)
def _load_model_cached(filename):
    """Each model is ~1GB / 20s+ to unpickle (large Dense layer after the
    Conv1D flatten). st.cache_resource keeps it in memory across Streamlit
    reruns so this only happens once per server process, not per interaction.
    """
    path = os.path.join(DATA_DIR, '06_models', filename)
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def _load_mapping_cached(filename):
    """Returns {encoded_label: atc_code} to decode argmax predictions."""
    path = os.path.join(DATA_DIR, '08_reporting', filename)
    df = pd.read_csv(path)
    return dict(zip(df['Encoded_Label'], df['ATC_Code']))


# Platt scaling for drug_classifier_model's raw sigmoid output, which is real
# but badly compressed (val set range ~[0.15, 0.65] for BOTH classes, even
# though the two classes' means do differ — val AUC 0.705). Fit as
# sigmoid(A * raw_prob + B) via LogisticRegression on
# drug_classifier_val_predictions.csv (raw pred_prob -> true label). This is a
# monotonic transform, so it cannot change the ROC AUC/ranking or accuracy at
# the optimal threshold — it only stretches the displayed probability toward
# 0/1 as far as the model's actual discriminative power honestly supports
# (recalibrated val range ~[0.15, 0.78]). Don't push this further: with AUC
# 0.705, more aggressive stretching would misrepresent the model's real
# uncertainty as false confidence.
_DRUG_PROB_PLATT_A = 5.894070
_DRUG_PROB_PLATT_B = -2.597218


def _recalibrate_drug_prob(raw_prob):
    return 1.0 / (1.0 + np.exp(-(_DRUG_PROB_PLATT_A * raw_prob + _DRUG_PROB_PLATT_B)))


class Display:
    """Page layout for the Drug Predictor 2.0 Streamlit app."""

    def show_tab1(self):
        st.title('Drug Predictor 2.0')
        st.markdown(
            'Predicts drug-likeness and ATC class (action-based and '
            'organ-based taxonomies) from molecular structure alone.'
        )

    def show_tab2(self):
        st.title('Drug Predictor 2.0 — high throughput')
        st.markdown('Batch prediction from a CSV of CIDs or SMILES.')

    def show_sidebar(self):
        with st.sidebar:
            st.write("Get your molecule's CID [here](https://pubchem.ncbi.nlm.nih.gov/)")
            st.image(os.path.join(RES_DIR, 'images', 'pubchem.png'), width=200)
            st.write(
                "Draw your molecule and get its SMILES "
                "[here](https://web.chemdoodle.com/demos/smiles#customise-template)"
            )
            st.image(os.path.join(RES_DIR, 'images', 'chemdoodleweb.png'), width=200)


class Calcs:
    """Loads the three trained models + ATC mappings and runs predictions."""

    def __init__(self):
        self.drug_model = _load_model_cached('drug_classifier_model.pkl')
        self.action_atc_model = _load_model_cached('action_atc_classifier_model.pkl')
        self.organ_atc_model = _load_model_cached('organ_atc_classifier_model.pkl')

        self.action_mapping = _load_mapping_cached('action_atc_mapping_drugs_only.csv')
        self.organ_mapping = _load_mapping_cached('organ_atc_mapping_drugs_only.csv')

    # ---- Molecule resolution ----

    def is_cid_or_smiles(self, query):
        return 'CID' if query.strip().isdigit() else 'SMILES'

    def get_molecule_from_cid(self, cid):
        try:
            comp_mol = pcp.Compound.from_cid(cid)
        except Exception:
            st.error(f'Wrong CID format! {cid} is not a valid CID')
            return None
        smiles = self._extract_smiles(comp_mol)
        if not smiles:
            st.error(f'CID {cid} has no SMILES available on PubChem')
            return None
        return Chem.MolFromSmiles(smiles)

    def _extract_smiles(self, comp_mol):
        """pubchempy 1.0.4's canonical_smiles/isomeric_smiles properties look
        for PubChem URN labels/names that PubChem's API has since renamed, so
        they always return None. Read the SMILES directly from the raw record
        instead (label 'SMILES', name 'Absolute' for isomeric, 'Connectivity'
        for canonical/non-isomeric).
        """
        for prop in comp_mol.record.get('props', []):
            urn = prop.get('urn', {})
            if urn.get('label') == 'SMILES' and urn.get('name') in ('Absolute', 'Connectivity'):
                return prop.get('value', {}).get('sval')
        return None

    def get_molecule_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error(f'Wrong SMILES! {smiles} is not a valid SMILES')
        return mol

    # ---- Featurization ----

    def get_fp(self, mol):
        return featurize_molecule(mol)

    def reshape_fp(self, fp, entry_type='single'):
        if entry_type == 'list':
            return fp.reshape((fp.shape[0], fp.shape[1], 1))
        return fp.reshape(1, fp.shape[0], 1)

    # ---- Single-molecule prediction ----

    def _top_k(self, probs, mapping, k=3):
        """Real drugs often straddle multiple ATC classes, and each taxonomy
        classifier only ever saw a single label per molecule during training,
        so top-1 confidence can be low/misleading. Return the top-k (code,
        description, probability) tuples so that uncertainty is visible
        instead of hidden behind one argmax guess.
        """
        top_idxs = np.argsort(probs)[::-1][:k]
        return [
            (mapping[i], ATC_DESCRIPTIONS.get(mapping[i], mapping[i]), float(probs[i]))
            for i in top_idxs
        ]

    def predict_single(self, mol):
        fp = self.get_fp(mol)
        arr = self.reshape_fp(fp, 'single')

        drug_prob = _recalibrate_drug_prob(
            float(self.drug_model.predict(arr, verbose=0).ravel()[0])
        )
        action_probs = self.action_atc_model.predict(arr, verbose=0).ravel()
        organ_probs = self.organ_atc_model.predict(arr, verbose=0).ravel()

        return {
            'drug_prob': drug_prob,
            'action_top3': self._top_k(action_probs, self.action_mapping),
            'organ_top3': self._top_k(organ_probs, self.organ_mapping),
        }

    def return_output(self, mol, query):
        st.write("This is your compound's structure")
        st.image(Draw.MolToImage(mol))

        result = self.predict_single(mol)

        st.markdown(f"##### Probability of being a drug: **{result['drug_prob']:.2%}**")

        st.markdown("##### Predicted action-based ATC class (top 3):")
        for code, desc, prob in result['action_top3']:
            st.markdown(f"- **{code} — {desc}** ({prob:.2%})")

        st.markdown("##### Predicted organ-based ATC class (top 3):")
        for code, desc, prob in result['organ_top3']:
            st.markdown(f"- **{code} — {desc}** ({prob:.2%})")

    def pred_from_cid(self, query):
        mol = self.get_molecule_from_cid(query)
        if mol:
            self.return_output(mol, query)

    def pred_from_smiles(self, query):
        mol = self.get_molecule_from_smiles(query)
        if mol:
            self.return_output(mol, query)

    # ---- High-throughput (batch) prediction ----

    def return_predictions_dataframe(self, df):
        if 'smiles' in df.columns:
            df['molecule'] = df['smiles'].map(self.get_molecule_from_smiles)
        elif 'cid' in df.columns:
            df['molecule'] = df['cid'].map(self.get_molecule_from_cid)
        else:
            st.error(':red[Wrong format! Please make sure there is a header called either "cid" or "smiles"]')
            return None

        df['fingerprints'] = df['molecule'].map(self.get_fp)
        fingerprints = np.stack(df['fingerprints'].to_numpy())
        arr = self.reshape_fp(fingerprints, 'list')

        drug_probs = _recalibrate_drug_prob(self.drug_model.predict(arr, verbose=0).ravel())

        action_probs_arr = self.action_atc_model.predict(arr, verbose=0)
        action_idxs = np.argmax(action_probs_arr, axis=1)
        action_codes = [self.action_mapping[i] for i in action_idxs]
        action_confs = action_probs_arr[np.arange(len(action_idxs)), action_idxs]

        organ_probs_arr = self.organ_atc_model.predict(arr, verbose=0)
        organ_idxs = np.argmax(organ_probs_arr, axis=1)
        organ_codes = [self.organ_mapping[i] for i in organ_idxs]
        organ_confs = organ_probs_arr[np.arange(len(organ_idxs)), organ_idxs]

        return {
            'drug_probability': drug_probs,
            'action_atc_code': action_codes,
            'action_atc_probability': action_confs,
            'organ_atc_code': organ_codes,
            'organ_atc_probability': organ_confs,
        }

    def return_output_dataframe(self, results, df):
        try:
            df['drug_probability'] = results['drug_probability']
            df['action_atc_code'] = results['action_atc_code']
            df['action_atc_description'] = [ATC_DESCRIPTIONS.get(c, c) for c in results['action_atc_code']]
            df['action_atc_probability'] = results['action_atc_probability']
            df['organ_atc_code'] = results['organ_atc_code']
            df['organ_atc_description'] = [ATC_DESCRIPTIONS.get(c, c) for c in results['organ_atc_code']]
            df['organ_atc_probability'] = results['organ_atc_probability']

            output_cols = [c for c in df.columns if c not in ('molecule', 'fingerprints')]
            output_df = df[output_cols]
            st.dataframe(output_df)
            st.download_button(
                label='Download as a CSV file',
                data=output_df.to_csv(index=False),
                file_name='drug_predictor_2_predictions.csv',
            )
        except Exception:
            st.error('Something went wrong while mounting the output dataframe :(')
