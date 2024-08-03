import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

nltk.download('stopwords')
nltk.download('wordset')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # removing punctuation and special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # converting text to lower case
    text = text.lower()
    #spiliting the texts in tokens
    words = text.split()
    #removing stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def train_mode():
    df = pd.read_csv('dataset/imdb_review.csv')
    df['cleaned_text'] = df['review'].apply(preprocess_text)
    X = df['cleaned_text']
    Y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #printing on the console to see the accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification_report",report(y_test,y_pred))

    joblib.dump(model, 'models/sentiment_model.pkl')

