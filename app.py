import streamlit as st
import pandas as pd
import os
import wandb
from data_handler import load_imdb_sample, process_upload
from nlp_pipeline import SentimentPipeline
from visuals import plot_sentiment_distribution, plot_confusion_matrix, display_metrics
import io

st.set_page_config(page_title="Universal Sentiment Analysis", layout="wide")

# Initialize session state for Pipeline and Data
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = SentimentPipeline()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'text_col' not in st.session_state:
    st.session_state.text_col = None
if 'label_col' not in st.session_state:
    st.session_state.label_col = None

st.title("Universal Sentiment Analysis Dashboard")

# Define Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Discovery", 
    "Training & Evaluation", 
    "Error Analysis", 
    "Playground"
])

# --- TAB 1: Data Discovery ---
with tab1:
    st.header("Upload or Load Data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Custom Dataset")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])
        if uploaded_file is not None:
            try:
                st.session_state.df = process_upload(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                
    with col2:
        st.subheader("Fallback Dataset")
        if st.button("Load IMDB Sample (10k)"):
            with st.spinner("Fetching data from tensorflow_datasets..."):
                df_imdb = load_imdb_sample()
                if df_imdb is not None:
                    st.session_state.df = df_imdb
                    st.success("IMDB sample loaded successfully!")

    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("Data Preview & Column Mapping")
        
        cols = st.session_state.df.columns.tolist()
        
        col_text, col_label = st.columns(2)
        with col_text:
            text_col = st.selectbox("Select Text Column", options=cols, 
                                    index=cols.index('review') if 'review' in cols else 0)
        with col_label:
            label_col = st.selectbox("Select Label/Target Column", options=cols,
                                     index=cols.index('sentiment') if 'sentiment' in cols else 0)
            
        st.session_state.text_col = text_col
        st.session_state.label_col = label_col
        
        st.dataframe(st.session_state.df.head())
        
        st.markdown("---")
        st.subheader("Sentiment Distribution")
        fig_pie, fig_bar = plot_sentiment_distribution(st.session_state.df, label_col)
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with chart_col2:
            st.plotly_chart(fig_bar, use_container_width=True)


# --- TAB 2: Training & Evaluation ---
with tab2:
    st.header("Model Training Center")
    
    if st.session_state.df is None:
        st.warning("Please upload or load data in the 'Data Discovery' tab first.")
    else:
        st.info("Uses Logistic Regression & TF-IDF (10k features, ngram 1,2) with NLTK Cleaning.")
        if st.button("Train Model", type='primary'):
            with st.spinner("Cleaning text and training model... This may take a moment."):
                # W&B Integration
                wandb_api_key = os.environ.get("WANDB_API_KEY") or (st.secrets.get("WANDB_API_KEY") if hasattr(st, 'secrets') else None)
                use_wandb = bool(wandb_api_key)
                
                if use_wandb:
                    try:
                        wandb.login(key=wandb_api_key)
                        wandb.init(project="universal-sentiment-dashboard", 
                                   config={"model": "Logistic Regression", "vectorizer": "TF-IDF"})
                    except Exception as e:
                        st.warning(f"Failed to initialize W&B: {e}")
                        use_wandb = False
                else:
                    st.info("WANDB_API_KEY not found in environment or secrets. Logging only to Streamlit.")
                
                # Train
                acc, f1, cm, evaluated_df, classes = st.session_state.pipeline.train(
                    st.session_state.df, 
                    st.session_state.text_col, 
                    st.session_state.label_col
                )
                
                # Update dataframe with predictions
                st.session_state.df = evaluated_df
                
                # Log metrics
                if use_wandb:
                    wandb.log({"accuracy": acc, "f1_score": f1})
                    wandb.finish()
                    
                st.success("Model trained successfully!")
                
                # Display Results
                display_metrics(acc, f1)
                
                st.markdown("---")
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(cm, classes)
                st.plotly_chart(fig_cm, use_container_width=True)
                
        elif st.session_state.pipeline.is_trained:
            st.success("Model is currently trained and ready for predictions.")
            

# --- TAB 3: Error Analysis ---
with tab3:
    st.header("Error Deep-Dive")
    if not st.session_state.pipeline.is_trained or 'is_error' not in st.session_state.df.columns:
        st.warning("Please train the model first to see error analysis.")
    else:
        st.write("Analyze the texts that were misclassified by the model.")
        
        df_errors = st.session_state.df[st.session_state.df['is_error'] == True]
        st.metric("Total Misclassified Samples", len(df_errors))
        
        if len(df_errors) > 0:
            # Filterable table
            true_label_filter = st.multiselect("Filter by True Label", options=df_errors[st.session_state.label_col].unique())
            pred_label_filter = st.multiselect("Filter by Predicted Label", options=df_errors['prediction'].unique())
            
            filtered_errors = df_errors.copy()
            if true_label_filter:
                filtered_errors = filtered_errors[filtered_errors[st.session_state.label_col].isin(true_label_filter)]
            if pred_label_filter:
                filtered_errors = filtered_errors[filtered_errors['prediction'].isin(pred_label_filter)]
                
            display_cols = [st.session_state.text_col, st.session_state.label_col, 'prediction', 'cleaned_text']
            st.dataframe(filtered_errors[display_cols], use_container_width=True)
        else:
            st.success("No errors found in the current evaluation set!")

        st.markdown("---")
        st.subheader("Export Predictions")
        
        # Convert df to csv for download
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Results as CSV",
            data=csv,
            file_name='sentiment_predictions.csv',
            mime='text/csv',
        )


# --- TAB 4: Playground ---
with tab4:
    st.header("Live Prediction Tool")
    
    if not st.session_state.pipeline.is_trained:
        st.warning("Please train the model first before using the playground.")
    else:
        st.write("Test the model in real-time with custom text.")
        
        user_input = st.text_area("Enter text here:", height=150)
        
        if st.button("Predict Sentiment"):
            if user_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    prediction = st.session_state.pipeline.predict_single(user_input)
                    st.success(f"**Predicted Sentiment:** {prediction}")
