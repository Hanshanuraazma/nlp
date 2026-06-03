import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import streamlit as st
import numpy as np

def plot_sentiment_distribution(df: pd.DataFrame, label_col: str):
    """
    Plots a pie chart and bar chart for sentiment distribution.
    """
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    
    fig_pie = px.pie(counts, values='Count', names='Sentiment', 
                     title="Sentiment Distribution (Pie)",
                     color_discrete_sequence=px.colors.sequential.RdBu)
                     
    fig_bar = px.bar(counts, x='Sentiment', y='Count', 
                     title="Sentiment Distribution (Bar)",
                     color='Sentiment',
                     color_discrete_sequence=px.colors.sequential.RdBu)
    
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
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label')
    )
    
    return fig

def display_metrics(accuracy: float, f1: float):
    """
    Displays accuracy and F1 score using Streamlit metric cards.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Accuracy", value=f"{accuracy:.4f}")
    with col2:
        st.metric(label="F1-Score (Weighted)", value=f"{f1:.4f}")
