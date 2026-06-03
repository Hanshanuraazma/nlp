import pandas as pd
import tensorflow_datasets as tfds
import streamlit as st

@st.cache_data
def load_imdb_sample():
    """
    Loads a sample of 10,000 IMDB reviews using tensorflow_datasets.
    """
    try:
        # Load the imdb_reviews dataset
        ds = tfds.load('imdb_reviews', split='train[:10000]', as_supervised=True)
        
        # Convert to a pandas DataFrame
        texts = []
        labels = []
        for text, label in tfds.as_numpy(ds):
            texts.append(text.decode('utf-8'))
            labels.append('positive' if label == 1 else 'negative')
            
        df = pd.DataFrame({'review': texts, 'sentiment': labels})
        return df
    except Exception as e:
        st.error(f"Error loading IMDB dataset from TFDS: {e}")
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
