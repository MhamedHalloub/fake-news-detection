# Fake News Detection using Machine Learning

This project implements a machine learning model to detect fake news articles. Using a dataset of real and fake news, the model applies text preprocessing, TF-IDF vectorization, and a Multinomial Naive Bayes classifier to predict whether a given article is **FAKE** or **REAL**.

## Key Features:
- **Data Preprocessing**: Cleans and prepares text data by removing stopwords and punctuation.
- **TF-IDF Vectorization**: Converts text data into numerical features for model training.
- **Naive Bayes Classifier**: Uses a Multinomial Naive Bayes classifier to classify news as fake or real.
- **Model Evaluation**: Provides accuracy and confusion matrix metrics for performance analysis.

## Installation:
1. Clone the repository:
   ```bash
   git clone https://MhamedHalloub/fake-news-detection.git
Install dependencies:
  pip install -r requirements.txt
Run the model:
  python fake_news_detector.py
How it Works:
      The dataset is loaded and labeled as FAKE or REAL.
      Text data is cleaned and transformed into numerical features using TF-IDF.
      A Multinomial Naive Bayes model is trained and evaluated using accuracy and confusion matrix.
