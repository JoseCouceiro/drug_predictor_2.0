import streamlit as st
import pandas as pd
from calculations import Display, Calcs

displayer = Display()

with st.spinner('Loading models (first load takes ~1 minute, cached after that)...'):
    calculator = Calcs()

tab1, tab2 = st.tabs(['Drug predictor', 'Drug predictor high throughput'])
displayer.show_sidebar()

with tab1:
    displayer.show_tab1()

    query = st.text_input("Insert molecule's CID or SMILES: ")

    if query:
        query_type = calculator.is_cid_or_smiles(query)
        if query_type == 'SMILES':
            calculator.pred_from_smiles(query)
        else:
            calculator.pred_from_cid(query)

with tab2:
    displayer.show_tab2()

    uploaded = st.file_uploader(
        "Please upload a CSV file. One molecule CID or SMILES per line. "
        "'cid' or 'smiles' as header for molecules."
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f'Analysing {len(df)} molecules')
        results = calculator.return_predictions_dataframe(df)
        if results is not None:
            calculator.return_output_dataframe(results, df)
