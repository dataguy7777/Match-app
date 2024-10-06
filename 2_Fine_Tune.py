#2_Fine_Tune.py page

import streamlit as st
import pandas as pd

st.title("Fine-Tune Matches")

if 'matches_df' in st.session_state:
    matches_df = st.session_state['matches_df'].copy()

    st.write("You can edit the 'Best Match' column to fine-tune the matches.")

    edited_df = st.experimental_data_editor(matches_df)

    # Save the edited dataframe back to session state
    st.session_state['matches_df'] = edited_df

    st.success("Your changes have been saved.")

    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Fine-Tuned Matches as CSV",
        data=csv,
        file_name='fine_tuned_matches.csv',
        mime='text/csv',
    )
else:
    st.warning("No matches to fine-tune. Please generate matches on the main page.")