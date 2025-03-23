import torch
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st

MODEL_PATH = "fake_news_model.pt"
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

#Load Trained Model
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Predict Fake or Real News
def predict_fake_news(text):
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    return "Fake News" if prediction == 1 else "Real News"

# Fetch Live News from API
def fetch_latest_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        st.error(f"Error fetching news: {response.json().get('message', 'Unknown error')}")
        return []

# Streamlit UI
st.title("Real-Time Fake News Detection System")

st.sidebar.header("Enter News Headline or Article")
news_input = st.sidebar.text_area("Paste news text here:")

if st.sidebar.button("Check Authenticity"):
    if news_input.strip():
        result = predict_fake_news(news_input)
        st.write(f"### Prediction: {result}")
    else:
        st.warning("Please enter some text.")

st.sidebar.header("Fetch Live News")
api_key = st.sidebar.text_input("Enter NewsAPI Key:", type="password")

if st.sidebar.button("Get Latest News"):
    if api_key.strip():
        articles = fetch_latest_news(api_key)
        if articles:
            for article in articles:
                title = article.get('title', 'No Title')
                source = article.get('source', {}).get('name', 'Unknown Source')
                description = article.get('description', '')

                news_text = (title or "No Title") + " " + (description or "")

                result = predict_fake_news(news_text)

                st.write(f"🔹 **{title}**")
                st.write(f"**Source:** {source}")
                st.write(f" **Prediction:** {result}")
                st.write("---")
        else:
            st.write("No news articles found.")
    else:
        st.warning("Please enter a valid API key.")
