import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import plotly.express as px
from flan_t5_prompting import one_shot_generate  # Import the function

# Load RoBERTa model for transcript classification
@st.cache_resource
def load_model():
    MODEL_NAME = "Ash00win/roberta-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit Layout
st.title("Interview Transcript Classification and Question Generation")

# 1. Transcript Classification Tool
st.header("Transcript Classification")
user_input = st.text_area("Enter Interview Transcript:")
if st.button("Classify Transcript"):
    try:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).item()
        st.write(f"Predicted Category: {predictions}")  # Just show the numeric label
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")

# 2. Question and Answer Generation Tool (unchanged from your working version)
st.header("Question and Answer Generation")
category = st.selectbox("Select Interview Category:", 
                       ["post_game_reaction", "in-game_analysis", "injury_report", 
                        "match_preview", "player_commentary"])
question_input = st.text_input("Enter a Question:")
if st.button("Generate Answer"):
    try:
        generated_answer = one_shot_generate(category, question_input)
        st.write(f"Generated Answer: {generated_answer}")
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")

# 3. Visualize Data Clusters using UMAP
st.header("Visualize Data Clusters")

# Load UMAP data from GitHub
UMAP_DATA_URL = "umap_embeddings.csv"

try:
    umap_data = pd.read_csv(UMAP_DATA_URL)
    
    # Create interactive plot
    fig = px.scatter(
        umap_data, 
        x='x', 
        y='y', 
        color='label',
        hover_data=['text'],
        title='Transcript Embeddings Visualization'
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate="<b>Label:</b> %{marker.color}<br><b>Text:</b> %{customdata[0]}<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error loading or visualizing UMAP data: {str(e)}")
    st.info(f"Could not load data from: {UMAP_DATA_URL}")
