import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM, pipeline  # Import the missing class here
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load model and tokenizer from Hugging Face
MODEL_NAME = "Ash00win/roberta-finetuned"  # Replace with your Hugging Face model repo name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load FLAN-T5 model for question generation
flan_model_id = "google/flan-t5-large"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_id)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_id)  # Fixing this import
generator = pipeline("text2text-generation", model=flan_model, tokenizer=flan_tokenizer)

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

# Function to generate answers using FLAN-T5
def one_shot_generate(category, question_input):
    shot = category_prompts[category]
    prompt = f"Category: {category}\nQ: {shot['example_q']}\nA: {shot['example_a']}\n\nQ: {question_input}\nA:"
    output = generator(prompt, max_length=100, do_sample=True, temperature=0.8)[0]["generated_text"]
    return output

# Streamlit Layout
st.title("Interview Transcript Classification and Question Generation")

# 1. Transcript Classification Tool
st.header("Transcript Classification")
user_input = st.text_area("Enter Interview Transcript:")
if st.button("Classify Transcript"):
    # Tokenize the user input text
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        # Get model logits for classification
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class index
    st.write(f"Predicted Category: {predicted_class}")  # Display the predicted category

# 2. Question and Answer Generation Tool (Using FLAN-T5)
st.header("Question and Answer Generation")
category = st.selectbox("Select Interview Category:", ["post_game_reaction", "in-game_analysis", "injury_report", "match_preview", "player_commentary"])
question_input = st.text_input("Enter a Question:")
if st.button("Generate Answer"):
    # Generate answer using the one-shot generation function
    generated_answer = one_shot_generate(category, question_input)
    st.write(f"Generated Answer: {generated_answer}")

# 3. Visualize Data Clusters using UMAP
st.header("Visualize Data Clusters")
umap_data = pd.read_csv("umap_embeddings.csv")  # Ensure that this file is available and uploaded

# Plot the UMAP clusters
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(umap_data["x"], umap_data["y"], c=umap_data["label"], cmap='viridis', s=50)
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_title("UMAP Visualization of Data Clusters")

# Display the plot in Streamlit
st.pyplot(fig)
