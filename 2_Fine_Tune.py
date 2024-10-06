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

def save_edited_matches(df):
    """
    Saves the edited matches dataframe to session state.

    Args:
        df (pd.DataFrame): Edited matches dataframe.

    Returns:
        None

    Example:
        save_edited_matches(edited_df)
    """
    try:
        st.session_state['matches_df'] = df
        logger.info("Edited matches saved to session state.")
    except Exception as e:
        logger.error(f"Failed to save edited matches: {e}")
        st.error("An error occurred while saving your changes.")

st.title("🔧 Fine-Tune Matches")

# Instructions Section
with st.expander("ℹ️ Instructions"):
    st.write("""
        - **Edit Matches:** Modify the 'Best Match' column to refine your match selections.
        - **Save Changes:** Your edits are automatically saved.
        - **Download Fine-Tuned Matches:** Export the refined matches as a CSV file.
    """)

# Check if 'matches_df' exists in session state
if 'matches_df' in st.session_state:
    matches_df = st.session_state['matches_df'].copy()
    logger.info("Loaded matches from session state for fine-tuning.")

    if matches_df.empty:
        st.warning("The matches dataframe is empty. Please generate matches on the main page.")
        logger.warning("Matches dataframe is empty.")
    else:
        st.subheader("Edit 'Best Match' Column")
        st.write("You can edit the 'Best Match' column to fine-tune the matches.")

        try:
            # Ensure 'Best Match' column exists
            if 'Best Match' not in matches_df.columns:
                st.error("The 'Best Match' column is missing from the data.")
                logger.error("'Best Match' column not found in matches dataframe.")
            else:
                # Allow editing only the 'Best Match' column
                edited_df = st.experimental_data_editor(
                    matches_df,
                    use_container_width=True,
                    column_config={
                        "Best Match": st.column_config.TextColumn(
                            "Best Match",
                            help="Edit this column to refine the best matches."
                        )
                    },
                    key="data_editor"
                )
                logger.info("User edited the 'Best Match' column.")

                # Save the edited dataframe back to session state
                save_edited_matches(edited_df)

                st.success("Your changes have been saved.")
                logger.info("Changes successfully saved to session state.")

                # Provide download option
                csv = edited_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Fine-Tuned Matches as CSV",
                    data=csv,
                    file_name='fine_tuned_matches.csv',
                    mime='text/csv',
                )
                logger.info("Fine-tuned matches available for download.")
        except Exception as e:
            st.error("An error occurred while editing the matches.")
            logger.error(f"Error during data editing: {e}")
else:
    st.warning("No matches to fine-tune. Please generate matches on the main page.")
    logger.warning("No matches found in session state for fine-tuning.")