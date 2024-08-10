import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense


word_index=imdb.get_word_index()
reverse_word_index={v:k for k,v in word_index.items()}

model=load_model("simple_rnn_imdb.h5")

def preprocessing_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocessing_text(review)
    predict=model.predict(preprocessed_input)

    sentiment="Positive" if predict[0][0] >0.5 else 'Negative'
    return sentiment,predict[0][0]

#streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input=st.text_input("Movie Review")
if st.button("classify"):
    sentiment,score=predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {score}")
else:
    st.write("Enter a moview review.")