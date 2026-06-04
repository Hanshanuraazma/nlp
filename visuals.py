import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import streamlit as st
import numpy as np

def plot_sentiment_distribution(df: pd.DataFrame, label_col: str):
    """
    Plots a pie chart and bar chart for sentiment distribution.
    Uses custom Harmonious HSL colors (Indigo for positive, Rose for negative).
    """
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    
    # Custom color palette matching the UI
    color_map = {
        'positive': '#6366f1',  # Indigo
        'pos': '#6366f1',
        '1': '#6366f1',
        'negative': '#f43f5e',  # Rose
        'neg': '#f43f5e',
        '0': '#f43f5e'
    }
    
    fig_pie = px.pie(counts, values='Count', names='Sentiment', 
                     title="Distribusi Sentimen (Pie Chart)",
                     color='Sentiment',
                     color_discrete_map=color_map)
                     
    fig_bar = px.bar(counts, x='Sentiment', y='Count', 
                     title="Distribusi Sentimen (Bar Chart)",
                     color='Sentiment',
                     color_discrete_map=color_map)
                     
    fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    
    return fig_pie, fig_bar

def plot_confusion_matrix(cm, class_names):
    """
    Plots a heatmap for the confusion matrix.
    """
    # Inverse cm for plotly heatmap standard format (y axis ascending from bottom)
    cm_display = cm[::-1]
    y_labels = class_names[::-1]
    
    fig = ff.create_annotated_heatmap(
        z=cm_display,
        x=list(class_names),
        y=list(y_labels),
        colorscale='Purples',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix (Evaluasi Model)',
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label'),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def display_metrics(accuracy: float, f1: float):
    """
    Displays accuracy and F1 score using Streamlit metric cards.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Akurasi Model", value=f"{accuracy:.4%}")
    with col2:
        st.metric(label="F1-Score (Weighted)", value=f"{f1:.4f}")

