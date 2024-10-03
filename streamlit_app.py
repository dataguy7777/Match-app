import streamlit as st
import pandas as pd
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

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

        # Generate matches using LLM
        st.info("Generating matches, please wait...")
        matches = []

        for source in source_sentences:
            # Prepare the prompt for the LLM
            prompt = f"Find the best match for the following sentence from the target sentences:\n\nSource Sentence: {source}\n\nTarget Sentences:\n"
            for idx, target in enumerate(target_sentences):
                prompt += f"{idx+1}. {target}\n"
            prompt += "\nProvide the number of the best matching target sentence."

            # Call the OpenAI API
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=5,
                temperature=0
            )

            # Extract the match index
            match_idx = int(response.choices[0].text.strip()) - 1
            best_match = target_sentences[match_idx]

            matches.append({
                'Source Sentence': source,
                'Best Match': best_match
            })

        # Save matches to session state
        st.session_state['matches_df'] = pd.DataFrame(matches)

        st.success("Matches generated successfully! Navigate to 'Display Matches' page to view them.")