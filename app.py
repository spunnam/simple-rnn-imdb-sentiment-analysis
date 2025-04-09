# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# -------------------------------------------
# Data Loading and Preparation
# -------------------------------------------

# Load the IMDB dataset word index that maps words to integers
word_index = imdb.get_word_index()
# Create a reversed mapping to decode integers back into words
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained sentiment analysis model
model = load_model("simple_rnn_imdb.h5")

# -------------------------------------------
# Helper Functions
# -------------------------------------------


def decode_review(encoded_review):
    """
    Decodes an encoded movie review back into human-readable text.

    Parameters:
    encoded_review (list): A list of integers representing the encoded review.

    Returns:
    str: The decoded review as a string.
    """
    # The index values are offset by 3 for special tokens, thus subtract 3.
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


def preprocess_review(text):
    """
    Preprocesses the user input review text so that it can be fed into the model.

    This function:
      1. Converts the text to lowercase and splits it into words.
      2. Encodes each word using the IMDB word index. Unknown words are encoded as the integer 2.
      3. Pads the encoded review to a fixed length of 500 words.

    Parameters:
    text (str): The raw review text provided by the user.

    Returns:
    numpy.ndarray: The padded encoded review ready for model prediction.
    """
    words = text.lower().split()
    # Encode words by mapping each to its integer index in the dictionary; add offset of 3.
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Pad sequences to ensure all inputs have the same length (500 words)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    """
    Predicts the sentiment of a given movie review using the pre-trained model.

    Parameters:
    review (str): The movie review text.

    Returns:
    tuple: A tuple with the sentiment label ('Positive' or 'Negative')
           and the prediction probability (float).
    """
    preprocessed_review = preprocess_review(review)
    # Model prediction returns a probability between 0 and 1
    prediction = model.predict(preprocessed_review)
    # Assign sentiment based on a threshold of 0.5
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]


# -------------------------------------------
# Streamlit Application: User Interface
# -------------------------------------------

# Configure the Streamlit page settings
st.set_page_config(
    page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered"
)

# Title and introduction of the app
st.title("IMDB Movie Review Sentiment Analysis")
st.markdown(
    """
This application uses a pre-trained recurrent neural network (RNN) model to predict whether a movie review expresses a **positive** or **negative** sentiment.

Enter your review in the text area below and click on **Analyze Sentiment**.
"""
)

# Input area for the user's movie review
user_review = st.text_area("Enter your movie review here:", height=200)

# Button that triggers the sentiment prediction
if st.button("Analyze Sentiment"):
    # Check if the review input is not empty
    if user_review.strip() == "":
        st.error("Please enter a movie review to analyze!")
    else:
        # Get sentiment prediction and its associated probability
        sentiment, probability = predict_sentiment(user_review)
        # Display the sentiment result in a visually distinct format
        if probability > 0.5:
            st.success(f"The review sentiment is: **{sentiment}**")
        else:
            st.error(f"The review sentiment is: **{sentiment}**")
        st.write(f"Prediction probability: **{probability:.4f}**")
        st.markdown("Thank you for using the IMDB Sentiment Analysis tool!")
