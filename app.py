import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import pandas as pd
import numpy as np

# Loading the model from Hugging Face Hub
MODEL_NAME = "Ash00win/roberta-finetuned"  # Replace with your Hugging Face model repo name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Streamlit Layout
st.title("Interview Transcript Classification and Question Generation")

# 1. Transcript Classification Tool
st.header("Transcript Classification")
user_input = st.text_area("Enter Interview Transcript:")
if st.button("Classify Transcript"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=1).item()  # Get the class prediction
    st.write(f"Predicted Category: {predictions}")

# 2. Question and Answer Generation Tool (Using FLAN-T5)
st.header("Question and Answer Generation")
category = st.selectbox("Select Interview Category:", ["post_game_reaction", "in-game_analysis", "injury_report", "match_preview", "player_commentary"])
question_input = st.text_input("Enter a Question:")
if st.button("Generate Answer"):
    # FLAN-T5 question generation (code already added)
    generated_answer = one_shot_generate(category, question_input)
    st.write(f"Generated Answer: {generated_answer}")

# 3. Visualize Data Clusters using UMAP
st.header("Visualize Data Clusters")
umap_data = pd.read_csv("umap_embeddings.csv")  # Make sure you generate and upload this file
st.write(umap_data)
st.write("UMAP Visualization will go here.")
