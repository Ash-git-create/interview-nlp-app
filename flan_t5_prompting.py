from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load FLAN-T5
model_id = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Category prompts for one-shot generation
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

def one_shot_generate(category, new_question):
    shot = category_prompts[category]
    prompt = (
        f"Category: {category}\n"
        f"Q: {shot['example_q']}\nA: {shot['example_a']}\n\n"
        f"Q: {new_question}\nA:"
    )
    output = generator(prompt, max_length=100, do_sample=True, temperature=0.8)[0]["generated_text"]
    return output
