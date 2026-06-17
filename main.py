import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ======================
# Load Model
# ======================

model = load_model("next_word_lstm.keras")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl","rb") as f:
    max_sequence_len = pickle.load(f)


# ======================
# Prediction Function
# ======================

def predict_next_word(text):

    token_list = tokenizer .texts_to_sequences([text])[0]
    token_list =pad_sequences(
        [token_list],
        maxlen = max_sequence_len-1,
        padding ="pre"
    )
    predicted = np.argmax(
        model.predict(token_list,verbose=0),
        axis = -1
    )[0]

    for word,index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return "no Prediction"

    return "No Prediction"

# ======================
# Streamlit UI
# ======================

st.set_page_config(
    page_title = "next word Predictor",

)

st.title("Next Word Predict using LSTM")

st.write(
    "Enter a sentence and predict the next word."
)
user_input = st.text_input(
    "Enter Text",
    placeholder="Artifical intelligence"
)


if st.button("Predict"):

    if user_input.strip():
        next_word = predict_next_word(
            user_input.lower()
        )

        st.success(
            f"Predicted Next Word:{next_word}"
        )
    else:
        st.warning("Please enter some text.")