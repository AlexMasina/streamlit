
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import streamlit as st
import joblib, os


predictor  = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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




