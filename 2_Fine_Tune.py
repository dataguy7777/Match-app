#2_Fine_Tune.py page

import streamlit as st
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fine_tune_matches_page():
    st.title("✏️ Fine-Tune Matches")

    # Instructions Section
    with st.expander("ℹ️ Instructions"):
        st.write("""
            - **Edit Matches:** Modify the 'Best Match' and 'Score' columns to refine the matches.
            - **Save Changes:** Your edits are automatically saved.
            - **Download Fine-Tuned Data:** Download the updated matches in CSV format.
        """)

    # Check if 'matches_df' exists in session state
    if 'matches_df' in st.session_state:
        matches_df = st.session_state['matches_df'].copy()
        logger.info("Accessing matches from session state for fine-tuning.")

        if matches_df.empty:
            st.warning("The matches dataframe is empty. Please generate matches on the main page.")
            logger.warning("Matches dataframe is empty.")
        else:
            st.subheader("📝 Edit Your Matches")
            st.write("You can edit the 'Best Match' and 'Score' columns to fine-tune the matches.")

            # Allow editing of the dataframe
            edited_df = st.experimental_data_editor(matches_df, num_rows="dynamic")

            # Save the edited dataframe back to session state
            st.session_state['matches_df'] = edited_df

            st.success("✅ Your changes have been saved.")
            logger.info("Fine-tuned matches saved to session state.")

            # Download Fine-Tuned Matches as CSV
            csv = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Fine-Tuned Matches as CSV",
                data=csv,
                file_name='fine_tuned_matches.csv',
                mime='text/csv',
            )
            logger.info("Fine-tuned matches available for download.")
    else:
        st.warning("⚠️ No matches to fine-tune. Please generate matches on the main page.")
        logger.warning("No matches found in session state.")