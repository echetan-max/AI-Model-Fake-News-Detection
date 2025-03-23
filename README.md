First make sure you have installed all the required libraries :
torch,  
transformers,  
datasets,  
evaluate,  
streamlit,  
requests,  
numpy,  
pandas,  
tqdm,  
scikit-learn.

File Descriptions:

-train_fake_news_model.py (Train the Model)

Loads and preprocesses the LIAR dataset

Fine-tunes a DistilBERT model for fake news detection

Implements early stopping & model checkpointing

Saves the trained model as fake_news_model.pt

-predict_fake_news.py (Detect Fake News)

Loads the trained model (fake_news_model.pt)

Accepts manual user input or fetches live news from NewsAPI

Classifies news as Real or Fake with a confidence score

Displays results in an interactive Streamlit UI

-To run the model 
python -m streamlit run predict_fake_news.py
