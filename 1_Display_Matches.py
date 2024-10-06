#1_Display_Matches.py page

import streamlit as st
import pandas as pd

st.title("Display Matches")

if 'matches_df' in st.session_state:
    matches_df = st.session_state['matches_df']
    st.write(matches_df)
    csv = matches_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Matches as CSV",
        data=csv,
        file_name='matches.csv',
        mime='text/csv',
    )
else:
    st.warning("No matches to display. Please generate matches on the main page.")