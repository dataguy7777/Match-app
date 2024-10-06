#streamlit_app.py

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

def generate_matches(df, source_col, target_col, top_x=5):
    """
    Generates matches based on the input dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing data to generate matches.
        source_col (str): Name of the source column.
        target_col (str): Name of the target column.
        top_x (int): Number of top matches to find.

    Returns:
        pd.DataFrame: DataFrame containing generated matches.
    """
    logger.info("Starting match generation.")
    try:
        matches = []

        source_sentences = df[source_col].dropna().tolist()
        target_sentences = df[target_col].dropna().tolist()

        if openai_api_key:
            logger.info("Generating matches using OpenAI's API.")
            for source in source_sentences:
                # Prepare the prompt for the LLM
                prompt = f"Find the top {top_x} best matches for the following sentence from the target sentences:\n\nSource Sentence: {source}\n\nTarget Sentences:\n"
                for idx, target in enumerate(target_sentences):
                    prompt += f"{idx+1}. {target}\n"
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

st.set_page_config(page_title="🔍 Match Generator", layout="wide")

st.title("🔍 Match Generator")

# Instructions Section
with st.expander("ℹ️ Instructions"):
    st.write("""
        - **Upload Data:** Provide a CSV or Excel file to generate matches.
        - **Select Columns:** Choose the source and target columns from your data.
        - **Set Parameters:** Define how many top matches to generate.
        - **Generate Matches:** Click the button to create matches based on your data.
        - **View & Fine-Tune:** Navigate to the 'Display Matches' and 'Fine-Tune' pages to review and adjust your matches.
    """)

# File Uploader
uploaded_file = st.file_uploader("📂 Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if not df.empty:
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
        logger.info(f"File '{uploaded_file.name}' uploaded and previewed.")

        # Column Selection
        st.subheader("📊 Select Columns for Matching")
        columns = df.columns.tolist()
        source_col = st.selectbox("🔹 Select the Source Column", options=columns)
        target_col = st.selectbox("🔹 Select the Target Column", options=columns)

        # Parameter for top X matches
        st.subheader("⚙️ Set Parameters")
        top_x = st.number_input("🔢 Enter the number of top matches to find (X)", min_value=1, max_value=20, value=5)

        # Generate Matches Button
        if st.button("🚀 Generate Matches"):
            with st.spinner("Generating matches, please wait..."):
                matches_df = generate_matches(df, source_col, target_col, top_x=top_x)
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