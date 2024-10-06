#2_Fine_Tune.py page

import streamlit as st
import pandas as pd

st.set_page_config(page_title="✏️ Fine-Tune Matches", layout="wide")

st.title("✏️ Fine-Tune Matches")

if 'matches_df' in st.session_state:
    matches_df = st.session_state['matches_df'].copy()

    st.subheader("📝 Edit Your Matches")
    st.write("You can edit the 'Best Match' and 'Score' columns to fine-tune the matches.")

    # Allow editing of the dataframe
    edited_df = st.experimental_data_editor(matches_df, num_rows="dynamic")

    # Save the edited dataframe back to session state
    st.session_state['matches_df'] = edited_df

    st.success("✅ Your changes have been saved.")

    # Download Fine-Tuned Matches as CSV
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Fine-Tuned Matches as CSV",
        data=csv,
        file_name='fine_tuned_matches.csv',
        mime='text/csv',
    )
else:
    st.warning("⚠️ No matches to fine-tune. Please generate matches on the main page.")