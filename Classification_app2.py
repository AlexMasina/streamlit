
# Import necessary libraries
import streamlit as st
import joblib, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




predictor  = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

train_data = pd.read_csv("train.csv")

# Creates a main title and subheader on your page -
# these are static across all pages
st.title("News Classifier")
st.subheader("Classifying news articles into categories")

# Creating sidebar with selection box -
# you can create multiple pages this way
options = ["Prediction", "Information"]
selection = st.sidebar.selectbox("Choose Option", options)

# Building out the "Information" page
if selection == "Information":
    st.info("General Information")
    st.markdown("This app classifies news articles into predefined categories like Business, Technology, Sports, Education, and Entertainment.")

# Building out the prediction page
if selection == "Prediction":
    st.info("Prediction with ML Models")
    # Creating a text box for user input
    news_text = st.text_area("Enter News Content", "Type here...")

    if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = vectorizer.transform([news_text])
        prediction = predictor.predict(vect_text)[0]
        predicted_category = train_data['category'].unique()[prediction]
        st.success(f"Text Categorized as: {predicted_category}")




