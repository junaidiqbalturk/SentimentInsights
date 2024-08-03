from flask import Blueprint, render_template, request, jsonify
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

main = Blueprint('main', __name__)

# Load the model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Preprocess function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
    return ' '.join(words)

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    prediction = model.predict(vectorized_review)[0]
    sentiment = 'positive' if prediction == 1 else 'negative'
    return jsonify({'sentiment': sentiment})
