# streamlit_app.py

import streamlit as st
import pandas as pd
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if openai_api_key:
    import openai
    openai.api_key = openai_api_key
    logger.info("OpenAI API key loaded from environment.")
else:
    # Import sentence-transformers for open-source LLM
    from sentence_transformers import SentenceTransformer, util
    import torch
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another model
    logger.info("Using Hugging Face's SentenceTransformer for match generation.")

def generate_matches(df_source, df_target, source_col, target_col, top_x=5):
    """
    Generates matches based on the input dataframes.

    Args:
        df_source (pd.DataFrame): Source dataframe containing data to generate matches.
        df_target (pd.DataFrame): Target dataframe containing data to find matches.
        source_col (str): Name of the source column.
        target_col (str): Name of the target column.
        top_x (int): Number of top matches to find.

    Returns:
        pd.DataFrame: DataFrame containing generated matches.
    """
    logger.info("Starting match generation.")
    try:
        matches = []

        source_sentences = df_source[source_col].dropna().tolist()
        target_sentences = df_target[target_col].dropna().tolist()

        if openai_api_key:
            logger.info("Generating matches using OpenAI's API.")
            for source in source_sentences:
                # Prepare the prompt for the LLM
                prompt = (
                    f"Find the top {top_x} best matches for the following sentence from the target sentences:\n\n"
                    f"Source Sentence: {source}\n\nTarget Sentences:\n"
                )
                for idx, target in enumerate(target_sentences):
                    prompt += f"{idx + 1}. {target}\n"
                prompt += f"\nProvide the numbers of the top {top_x} matching target sentences, separated by commas."

                # Call the OpenAI API
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0
                )

                # Extract the match indices
                response_text = response.choices[0].text.strip()
                try:
                    match_indices = [int(idx.strip()) - 1 for idx in response_text.split(',') if idx.strip().isdigit()]
                except ValueError:
                    logger.error(f"Invalid response from OpenAI: {response_text}")
                    st.error(f"Invalid response from OpenAI for source sentence: {source}")
                    continue

                best_matches = [target_sentences[idx] for idx in match_indices if 0 <= idx < len(target_sentences)]

                for match in best_matches:
                    matches.append({
                        'Source Sentence': source,
                        'Best Match': match
                    })
        else:
            logger.info("Generating matches using Hugging Face's SentenceTransformer.")
            # Compute embeddings
            source_embeddings = model.encode(source_sentences, convert_to_tensor=True)
            target_embeddings = model.encode(target_sentences, convert_to_tensor=True)

            # Compute cosine similarities
            cosine_scores = util.cos_sim(source_embeddings, target_embeddings)

            for idx, source in enumerate(source_sentences):
                # Get the top X matches
                scores, top_indices = torch.topk(cosine_scores[idx], k=top_x)
                for score, target_idx in zip(scores, top_indices):
                    matches.append({
                        'Source Sentence': source,
                        'Best Match': target_sentences[target_idx],
                        'Score': score.item()
                    })

        logger.info("Match generation completed successfully.")
        return pd.DataFrame(matches)
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
    """
    logger.info(f"Reading uploaded file: {uploaded_file.name}")
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            logger.info("CSV file read successfully.")
        elif file_extension in ['xlsx', 'xls']:
            # Get sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            st.session_state['sheet_names'] = sheet_names  # Store sheet names in session state
            df = None  # Initialize as None; will be loaded per sheet
            logger.info("Excel file detected with sheets.")
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            logger.warning(f"Unsupported file type: {file_extension}")
            return pd.DataFrame()
        logger.debug(f"Dataframe shape: {df.shape if df is not None else 'Multiple sheets'}")
        return df
    except Exception as e:
        logger.error(f"Error reading file {uploaded_file.name}: {e}")
        st.error("There was an error processing your file.")
        return pd.DataFrame()

# Configure Streamlit Page
st.set_page_config(page_title="🔍 Match Generator", layout="wide")

# Sidebar Navigation
st.sidebar.title("🗂️ Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Display Matches", "Fine-Tune Matches"])

if app_mode == "Home":
    st.title("🔍 Match Generator")

    # Instructions Section
    with st.expander("ℹ️ Instructions"):
        st.write("""
            - **Upload Data:** Provide a CSV or Excel file to generate matches.
            - **Select Source and Target:** Choose the sheets and columns for source and target data.
            - **Set Parameters:** Define how many top matches to generate.
            - **Generate Matches:** Click the button to create matches based on your data.
            - **View & Fine-Tune:** Navigate to the 'Display Matches' and 'Fine-Tune Matches' pages to review and adjust your matches.
        """)

    # File Uploader
    uploaded_file = st.file_uploader("📂 Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        df_initial = read_uploaded_file(uploaded_file)
        if not df_initial.empty or ('sheet_names' in st.session_state):
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            if 'sheet_names' in st.session_state:
                sheet_names = st.session_state['sheet_names']
                # Tabs for Source and Target
                tabs = st.tabs(["📁 Source", "📁 Target"])

                with tabs[0]:
                    st.subheader("📁 Select Source Sheet and Column")
                    source_sheet = st.selectbox("🔹 Select Source Sheet", options=sheet_names, key='source_sheet')
                    source_df = pd.read_excel(uploaded_file, sheet_name=source_sheet)
                    source_columns = source_df.columns.tolist()
                    source_col = st.selectbox("🔹 Select Source Column", options=source_columns, key='source_col')

                with tabs[1]:
                    st.subheader("📁 Select Target Sheet and Column")
                    target_sheet = st.selectbox("🔹 Select Target Sheet", options=sheet_names, key='target_sheet')
                    target_df = pd.read_excel(uploaded_file, sheet_name=target_sheet)
                    target_columns = target_df.columns.tolist()
                    target_col = st.selectbox("🔹 Select Target Column", options=target_columns, key='target_col')

                # Parameter for top X matches
                st.subheader("⚙️ Set Parameters")
                top_x = st.number_input("🔢 Enter the number of top matches to find (X)", min_value=1, max_value=20, value=5)

                # Generate Matches Button
                if st.button("🚀 Generate Matches"):
                    with st.spinner("Generating matches, please wait..."):
                        matches_df = generate_matches(source_df, target_df, source_col, target_col, top_x=top_x)
                        if not matches_df.empty:
                            st.session_state['matches_df'] = matches_df
                            st.success("✅ Matches generated and saved to session state.")
                            logger.info("Matches generated and saved to session state.")
                        else:
                            st.warning("⚠️ No matches were generated.")
                            logger.warning("Generated matches dataframe is empty.")

            else:
                # If the file is CSV
                st.subheader("🔍 Data Preview")
                st.dataframe(df_initial.head())

                # Column Selection
                st.subheader("📊 Select Columns for Matching")
                columns = df_initial.columns.tolist()
                source_col = st.selectbox("🔹 Select the Source Column", options=columns, key='source_col_csv')
                target_col = st.selectbox("🔹 Select the Target Column", options=columns, key='target_col_csv')

                # Parameter for top X matches
                st.subheader("⚙️ Set Parameters")
                top_x = st.number_input("🔢 Enter the number of top matches to find (X)", min_value=1, max_value=20, value=5)

                # Generate Matches Button
                if st.button("🚀 Generate Matches"):
                    with st.spinner("Generating matches, please wait..."):
                        matches_df = generate_matches(df_initial, df_initial, source_col, target_col, top_x=top_x)
                        if not matches_df.empty:
                            st.session_state['matches_df'] = matches_df
                            st.success("✅ Matches generated and saved to session state.")
                            logger.info("Matches generated and saved to session state.")
                        else:
                            st.warning("⚠️ No matches were generated.")
                            logger.warning("Generated matches dataframe is empty.")
        else:
            st.warning("⚠️ Failed to process the uploaded file. Please check the file format and content.")
            logger.warning("Uploaded file resulted in an empty dataframe.")
    else:
        st.info("📌 Please upload a CSV or Excel file to begin.")
        logger.info("No file uploaded by the user.")

elif app_mode == "Display Matches":
    import pages.page_Display_Matches
    pages.page_Display_Matches.display_matches_page()

elif app_mode == "Fine-Tune Matches":
    import pages.page_Fine_Tune
    pages.page_Fine_Tune.fine_tune_matches_page()