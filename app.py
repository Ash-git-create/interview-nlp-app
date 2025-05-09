import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import umap
from transformers import AutoModelForSeq2SeqLM

# Load RoBERTa model for transcript classification
roberta_model_name = "Ash00win/roberta-finetuned"  # Your Hugging Face model repo
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)

# Load FLAN-T5 model for Q&A generation
flan_model_id = "google/flan-t5-large"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_id)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_id)
generator = pipeline("text2text-generation", model=flan_model, tokenizer=flan_tokenizer)

# Streamlit Layout
st.title("Interview Transcript Classification and Question Generation")

# === Transcript Classification ===
st.header("Transcript Classification")
user_input = st.text_area("Enter Interview Transcript:")
if st.button("Classify Transcript"):
    # Tokenize and classify using RoBERTa model
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = roberta_model(**inputs).logits
    label_id = torch.argmax(logits, dim=1).item()  # Get the label ID prediction
    st.write(f"Predicted Label ID: {label_id}")  # You can map this label ID to a category if needed

# === Question and Answer Generation ===
st.header("Question and Answer Generation")
category = st.selectbox("Select Interview Category:", ["post_game_reaction", "in-game_analysis", "injury_report", "match_preview", "player_commentary"])
question_input = st.text_input("Enter a Question:")
if st.button("Generate Answer"):
    # FLAN-T5 question generation (code already added)
    generated_answer = one_shot_generate(category, question_input)
    st.write(f"Generated Answer: {generated_answer}")

# === UMAP Visualization of Transcript Clusters ===
st.header("Visualize Data Clusters")

# Load UMAP embeddings CSV (make sure it's available and uploaded in your repo)
umap_data = pd.read_csv("umap_embeddings.csv")  # Make sure you generate and upload this file
st.write("Data Cluster visualization using UMAP embeddings")

# Plot the UMAP embeddings as a scatter plot
fig = px.scatter(umap_data, x="x", y="y", color="label", hover_data=["text"])
st.plotly_chart(fig)

# === Helper Function for FLAN-T5 One-shot Generation ===
def one_shot_generate(category, new_question):
    category_prompts = {
        "post_game_reaction": {
            "example_q": "How do you feel about today's win?",
            "example_a": "It was a great team effort. We stuck to our game plan and executed well.",
        },
        "in-game_analysis": {
            "example_q": "What changed in the second quarter?",
            "example_a": "We switched to a zone defense and started controlling the pace better.",
        },
        "injury_report": {
            "example_q": "What’s the update on Alex’s condition?",
            "example_a": "He’s undergoing further tests, but we hope it’s just a minor sprain.",
        },
        "match_preview": {
            "example_q": "What are you expecting from tomorrow's game?",
            "example_a": "They're a strong team, so we’ll focus on tightening our defense and staying disciplined.",
        },
        "player_commentary": {
            "example_q": "How would you describe your opponent's performance?",
            "example_a": "He was aggressive from the start and kept pressure on us throughout the game.",
        }
    }

    shot = category_prompts[category]
    prompt = (
        f"Category: {category}\n"
        f"Q: {shot['example_q']}\nA: {shot['example_a']}\n\n"
        f"Q: {new_question}\nA:"
    )
    output = generator(prompt, max_length=100, do_sample=True, temperature=0.8)[0]["generated_text"]
    return output
