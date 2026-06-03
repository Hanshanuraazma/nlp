import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_clean(text: str) -> str:
    """
    Cleans text by removing punctuation, numbers, special characters,
    stopwords, and applies lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation, numbers, special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(cleaned_words)

class SentimentPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False
        
    def train(self, df: pd.DataFrame, text_col: str, label_col: str):
        """
        Trains the TF-IDF vectorizer and Logistic Regression model.
        Returns evaluation metrics and the confusion matrix.
        """
        # Clean text
        df['cleaned_text'] = df[text_col].apply(advanced_clean)
        
        # Split data (for evaluation). Here we just evaluate on training data for simplicity in dashboard, 
        # or we could do a train/test split. Let's do a simple 80/20 split.
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df[label_col], test_size=0.2, random_state=42
        )
        
        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        # Full dataset prediction for error analysis and export
        df['prediction'] = self.model.predict(self.vectorizer.transform(df['cleaned_text']))
        df['is_error'] = df[label_col] != df['prediction']
        
        return acc, f1, cm, df, self.model.classes_

    def predict_single(self, text: str) -> str:
        """
        Predicts sentiment for a single text input.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        cleaned = advanced_clean(text)
        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)
        return pred[0]
