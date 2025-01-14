import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the TF-IDF and model from pickle
def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

# Predict toxicity of the input text
def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

# Set up the page title and layout
st.set_page_config(page_title="Toxicity Detection App", layout="centered")

st.markdown(
    """
    <style>
    .css-1v3fvcr {
        background-color: #f1f1f1;
        text-align: center;
    }
    .stTextInput input {
        font-size: 20px;
        padding: 10px;
    }
    .stButton button {
        background-color: #ff6347;
        color: black;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #ff4500;
    }
    .stInfo {
        font-size: 18px;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# Page Title
st.title("Toxicity Detection")

# Main interaction flow
if st.button("Start"):
    st.subheader("Enter your text for toxicity analysis:")
    text_input = st.text_input("Text Input")

    if text_input:
        if st.button("Analyze"):
            result = toxicity_prediction(text_input)
            st.subheader("Result:")
            st.info(f"The result is: {result}.")
    else:
        st.write("Please input some text for analysis.")
