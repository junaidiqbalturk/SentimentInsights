from flask import Blueprint, render_template, request, jsonify
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from .utils import preprocess_text


main = Blueprint('main',__name__)

model = joblib.load('models/sentiment_model_pkl')
vectorizer = joblib.load('models/sentiment_vectorizer_pkl')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.form['text']
    cleaned_text = preprocess_text(text)
    features = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(features)
    sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
    return jsonify({'sentiment': sentiment})

@main.route('/dashboard')
def dashboard():
    df = pd.read_csv('dataset\imdb_reviews.csv')
    df['cleaned_text'] = df['review'].apply(preprocess_text)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

    sns.countplot(df['sentiment'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('dashboard.html', plot_url=plot_url)

