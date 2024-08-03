import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import nltk

# Ensure required NLTK packages are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# Preprocess text function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
    return ' '.join(words)


def train_model():
    df = pd.read_csv('dataset/imdb_review.csv')
    df['cleaned_text'] = df['review'].apply(preprocess_text)
    X = df['cleaned_text']
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')


if __name__ == "__main__":
    train_model()
