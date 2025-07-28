import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_path = "./results"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(torch.device("cpu"))
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- UI Section ---
st.title("🧠 Emotion Classifier App")
st.info("⚠️ This app uses a fine-tuned DistilBERT model. Accuracy may vary depending on the training data quality.")
st.caption("AI PM Portfolio Demo · Created by Jovita Amos")

# ✅ Define input BEFORE using it
text = st.text_input("Enter a sentence to analyze:", "")

# --- Inference ---
if text:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    predicted_label = label_map[prediction]
    
    st.success(f"Predicted Emotion: {predicted_label}")

# Ask for user feedback
st.markdown("### 🧐 Was this prediction accurate?")
feedback = st.radio("Your feedback helps improve this model:", ["Yes", "No"], horizontal=True)

if feedback == "Yes":
    st.success("👍 Great! Thanks for confirming.")
elif feedback == "No":
    corrected = st.selectbox("What should the correct emotion be?", list(label_map.values()))
    st.warning(f"⚠️ Got it! You said it should be **{corrected}**.")
    # Optional: Append to feedback log for retraining
