
# Import necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import streamlit as st
import joblib, os

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv") 

train_data.dropna(subset=['content', 'category'], inplace=True)
test_data.dropna(subset=['content'], inplace=True)

# Encode categories
y_train = train_data['category'].astype('category').cat.codes
y_test = test_data['category'].astype('category').cat.codes

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['content'])
X_test = vectorizer.transform(test_data['content'])

# Define models 
models = { 
    "Logistic Regression": LogisticRegression(max_iter=200), 
    "Random Forest": RandomForestClassifier(), 
    "Support Vector Machine": SVC() } 



# Train and evaluate models with cross-validation 
results = {} 
for model_name, model in models.items(): 
    model.fit(X_train, y_train) 
    predictions = model.predict(X_test) 
    accuracy = accuracy_score(y_test, predictions) 
    results[model_name] = { 
                           "Accuracy": accuracy, 
                           "Classification Report": classification_report(y_test, predictions, target_names=train_data['category'].unique()) } 
    
# Save the best model
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = models[best_model_name]



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
        input_vectorized = vectorizer.transform([input_text])
        prediction = best_model.predict(input_vectorized)[0]
        predicted_category = data['category'].unique()[prediction]
        st.write(f"Predicted Category: {predicted_category}")



