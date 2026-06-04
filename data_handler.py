import pandas as pd
import streamlit as st

@st.cache_data
def load_imdb_sample():
    """
    Loads a sample of 10,000 IMDB reviews from a public raw URL on GitHub.
    """
    try:
        url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
        # Download and read only the first 10,000 rows
        df = pd.read_csv(url, nrows=10000)
        return df
    except Exception as e:
        st.error(f"Error loading IMDB dataset from GitHub: {e}")
        return None

def process_upload(uploaded_file) -> pd.DataFrame:
    """
    Processes the uploaded CSV or Excel file into a pandas DataFrame.
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    return df

