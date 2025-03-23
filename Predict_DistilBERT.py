import torch
import requests
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# API Key for NewsAPI (Replace with your key)
NEWS_API_KEY = "28013c60e5004c928b8325e13d72179d"

#  Load Tokenizer & Model
MODEL_PATH = "fake_news_model.pt"
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Move Model to GPU (If Available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to Predict Fake or Real News with Confidence Score
def predict_fake_news(text):
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    #  Compute confidence score
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence = probs.max().item() * 100  
    predicted_class = torch.argmax(outputs.logits).item()

    if predicted_class == 1:
        return f" **Fake News** ({confidence:.2f}% confidence)"
    else:
        return f" **Real News** ({confidence:.2f}% confidence)"

# Fetch Live News from API
def fetch_live_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        st.error(f" Error fetching news: {response.json().get('message', 'Unknown error')}")
        return []

# Streamlit UI
st.title("Real-Time Fake News Detector")

# 🔹 User Input Section
st.header("Check Your Own News")
user_input = st.text_area("Enter News Headline or Article:")

if st.button("Check Authenticity"):
    if user_input.strip():
        result = predict_fake_news(user_input)
        st.write(f"### 🔍 Prediction: {result}")
    else:
        st.warning(" Please enter some text.")

# 🔹 Live News Analysis
st.header("Live News Analysis")
api_key = st.text_input("Enter Your NewsAPI Key:", type="password")

if st.button("Fetch & Analyze Live News"):
    if api_key.strip():
        articles = fetch_live_news(api_key)
        if articles:
            for article in articles:
                title = article.get("title", "No Title")
                description = article.get("description", "")
                news_text = title + " " + description
                result = predict_fake_news(news_text)

                st.subheader(f"🔹 {title}")
                st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                st.write(f" **Prediction:** {result}")
                st.write("---")
        else:
            st.write("No news articles found.")
    else:
        st.warning(" Please enter a valid API key.")
