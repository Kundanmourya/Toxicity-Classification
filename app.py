import streamlit as st
import pickle
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

tfidf = load_tfidf()
nb_model = load_model()

def toxicity_prediction(text):    
    text_tfidf = tfidf.transform([text]).toarray()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name
<<<<<<< Updated upstream
    
#st.title("Health Check Page")
#st.write("App is running smoothly!"
             
# Set up the page title and layout
=======

>>>>>>> Stashed changes
st.set_page_config(page_title="Toxicity Detection App", layout="centered")

if "start_clicked" not in st.session_state:
    st.session_state.start_clicked = False

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

st.title("Toxicity Detection")

if not st.session_state.start_clicked:
    if st.button("Start"):
        st.session_state.start_clicked = True
        st.rerun()  

else:
    st.subheader("Enter your text for toxicity analysis:")
    text_input = st.text_input("Text Input")

    if text_input:
        if st.button("Analyze"):
            result = toxicity_prediction(text_input)
            st.subheader("Result:")
            st.info(f"The result is: {result}.")
    else:
        st.write("Please input some text for analysis.")

