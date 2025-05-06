import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

fake_df = pd.read_csv('fake.csv')
true_df = pd.read_csv('true.csv')
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_news(news_text):
    cleaned = clean_text(news_text)
    print("Cleaned Text:", cleaned)  # Debugging: See cleaned text
    vect_text = vectorizer.transform([cleaned])
    return model.predict(vect_text)[0]
#testing
sample = "The government has announced a new healthcare policy effective next week."
print(f"Prediction for sample news: {predict_news(sample)}")
print(f"Prediction for 'NASA confirms alien life on Mars.': {predict_news('NASA confirms alien life on Mars.')}")
print(f"Prediction for 'Stock market rises after tech earnings beat expectations.': {predict_news('Stock market rises after tech earnings beat expectations.')}")
print(predict_news("Scientists discover a new planet in a neighboring galaxy."))
print(predict_news("Breaking: Tech company stocks plummet after cybersecurity breach."))
