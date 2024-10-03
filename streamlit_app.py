import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    import openai
    openai.api_key = openai_api_key
else:
    # Import sentence-transformers for open-source LLM
    from sentence_transformers import SentenceTransformer, util
    import torch
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Or choose another model

st.title("Sentence Matching App")

st.header("1. Upload Your File")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read the file into a dataframe
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("File Uploaded Successfully!")

    st.header("2. Select Source and Target Columns")

    # Column selection
    columns = df.columns.tolist()
    source_col = st.selectbox("Select the Source Column", columns)
    target_col = st.selectbox("Select the Target Column", columns)

    st.header("3. Set Parameters")

    # Parameter for top X matches
    top_x = st.number_input("Enter the number of top matches to find (X)", min_value=1, value=5)

    st.header("4. Generate Matches")

    if st.button("Generate Matches"):
        # Get the sentences
        source_sentences = df[source_col].dropna().tolist()
        target_sentences = df[target_col].dropna().tolist()

        st.info("Generating matches, please wait...")

        matches = []

        if openai_api_key:
            # Generate matches using OpenAI LLM
            for source in source_sentences:
                # Prepare the prompt for the LLM
                prompt = f"Find the best match for the following sentence from the target sentences:\n\nSource Sentence: {source}\n\nTarget Sentences:\n"
                for idx, target in enumerate(target_sentences):
                    prompt