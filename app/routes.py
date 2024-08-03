from flask import Blueprint, request, jsonify, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

main = Blueprint('main', __name__)

model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@main.route('/')
def index():
    return render_template('dashboard.html')

@main.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    processed_review = preprocess_text(review)
    transformed_review = vectorizer.transform([processed_review])
    prediction = model.predict(transformed_review)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)
