#1_Display_Matches.py page

import streamlit as st
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def filter_matches(df, category=None, score_threshold=None):
    """
    Filters matches based on category and score threshold.

    Args:
        df (pd.DataFrame): DataFrame containing matches.
        category (str, optional): Category to filter by. Defaults to None.
        score_threshold (float, optional): Minimum score to filter by. Defaults to None.

    Returns:
        pd.DataFrame: Filtered DataFrame.

    Example:
        filtered_df = filter_matches(df, category='A', score_threshold=50)
    """
    logger.info("Applying filters to matches.")
    try:
        if category:
            df = df[df['Category'] == category]
            logger.debug(f"Filtered by category: {category}")
        if score_threshold is not None:
            df = df[df['Score'] >= score_threshold]
            logger.debug(f"Filtered by score threshold: {score_threshold}")
        logger.info(f"Filtering completed. Number of matches after filtering: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error during filtering matches: {e}")
        st.error("An error occurred while filtering matches.")
        return pd.DataFrame()

st.title("📊 Display Matches")

# Instructions Section
with st.expander("ℹ️ Instructions"):
    st.write("""
        - **View Matches:** Browse through the generated matches.
        - **Search & Filter:** Use the sidebar to filter matches based on criteria.
        - **Sort Data:** Click on column headers to sort the data.
        - **Download Data:** Download the matches in your preferred format.
    """)

# Check if 'matches_df' exists in session state
if 'matches_df' in st.session_state:
    matches_df = st.session_state['matches_df'].copy()
    logger.info("Displaying matches from session state.")

    if matches_df.empty:
        st.warning("The matches dataframe is empty. Please generate matches on the main page.")
        logger.warning("Matches dataframe is empty.")
    else:
        # Sidebar for Filters
        st.sidebar.header("🔍 Filter Matches")
        
        # Example filters (modify based on actual dataframe columns)
        if 'Category' in matches_df.columns:
            categories = matches_df['Category'].dropna().unique().tolist()
            selected_categories = st.sidebar.multiselect(
                "Select Categories",
                options=categories,
                default=categories
            )
            matches_df = matches_df[matches_df['Category'].isin(selected_categories)]
            logger.debug(f"Selected categories: {selected_categories}")
        
        if 'Score' in matches_df.columns:
            min_score, max_score = float(matches_df['Score'].min()), float(matches_df['Score'].max())
            score_threshold = st.sidebar.slider(
                "Minimum Score",
                min_value=min_score,
                max_value=max_score,
                value=min_score
            )
            matches_df = matches_df[matches_df['Score'] >= score_threshold]
            logger.debug(f"Applied score threshold: {score_threshold}")
        
        st.subheader("Filtered Matches")
        st.dataframe(matches_df)

        # Download Options
        csv = matches_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Matches as CSV",
            data=csv,
            file_name='filtered_matches.csv',
            mime='text/csv',
        )
        logger.info("Filtered matches available for download.")
else:
    st.warning("No matches to display. Please generate matches on the main page.")
    logger.warning("No matches found in session state.")