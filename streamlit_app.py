#streamlit_app.py

import streamlit as st
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_matches(df):
    """
    Generates matches based on the input dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing data to generate matches.

    Returns:
        pd.DataFrame: DataFrame containing generated matches.

    Example:
        input_df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Score': [85, 90]})
        matches = generate_matches(input_df)
    """
    logger.info("Starting match generation.")
    try:
        # Placeholder for actual match generation logic
        df['Best Match'] = df['Name']  # Dummy implementation
        logger.debug(f"Generated matches:\n{df.head()}")
        logger.info("Match generation completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during match generation: {e}")
        st.error("An error occurred while generating matches.")
        return pd.DataFrame()

def read_uploaded_file(uploaded_file):
    """
    Reads the uploaded file into a pandas DataFrame.

    Args:
        uploaded_file (UploadedFile): File uploaded via Streamlit's file uploader.

    Returns:
        pd.DataFrame: DataFrame containing the uploaded data.

    Example:
        df = read_uploaded_file(uploaded_file)
    """
    logger.info(f"Reading uploaded file: {uploaded_file.name}")
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            logger.info("CSV file read successfully.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            logger.info("Excel file read successfully.")
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            logger.warning(f"Unsupported file type: {file_extension}")
            return pd.DataFrame()
        logger.debug(f"Dataframe shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading file {uploaded_file.name}: {e}")
        st.error("There was an error processing your file.")
        return pd.DataFrame()

st.title("🔍 Match Generator")

# Instructions Section
with st.expander("ℹ️ Instructions"):
    st.write("""
        - **Upload Data:** Provide a CSV or Excel file to generate matches.
        - **Generate Matches:** Click the button to create matches based on your data.
        - **Fine-Tune Matches:** Navigate to the Fine-Tune page to refine your matches.
    """)

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if not df.empty:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write("Data Preview:", df.head())
        logger.info(f"File '{uploaded_file.name}' uploaded and previewed.")

        if st.button("Generate Matches"):
            matches_df = generate_matches(df)
            if not matches_df.empty:
                st.session_state['matches_df'] = matches_df
                st.success("Matches generated and saved to session state.")
                logger.info("Matches generated and saved to session state.")
            else:
                st.warning("No matches were generated.")
                logger.warning("Generated matches dataframe is empty.")
    else:
        st.warning("Failed to process the uploaded file. Please check the file format and content.")
        logger.warning("Uploaded file resulted in an empty dataframe.")
else:
    st.warning("Please upload a CSV or Excel file to generate matches.")
    logger.warning("No file uploaded by the user.")