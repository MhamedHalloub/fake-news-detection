# Fake News Detection Model

This project contains a Python script for detecting fake news using machine learning. The script uses Natural Language Processing (NLP) and machine learning algorithms to classify news articles as either "REAL" or "FAKE."

## Dataset

The dataset used for this project contains labeled news articles (real or fake). You can download the dataset from [this link](<insert_download_link_here>), or you can use your own dataset in CSV format with the same structure.

### Dataset Structure

The dataset must contain two CSV files:

- **`fake.csv`**: A CSV file containing fake news articles. The structure should have two columns:
  - `text`: The content of the news article
  - `label`: The label indicating that the news is "FAKE"
  
- **`true.csv`**: A CSV file containing real news articles. The structure should have two columns:
  - `text`: The content of the news article
  - `label`: The label indicating that the news is "REAL"

After downloading the dataset, save the `fake.csv` and `true.csv` files in the same directory as the `fake_news_detector.py` file.

## Requirements

You need to install the following Python libraries to run this project:
            pip install pandas scikit-learn nltk

Running the Script
            Download the dataset files (fake.csv and true.csv) and place them in the same directory as the Python script.
            Run the script using the following command:
               python fake_news_detector.py
The model will be trained on the dataset and will predict whether the news articles are "REAL" or "FAKE."

Code Explanation
            The script first loads the fake.csv and true.csv files using pandas.
            It combines both datasets, assigns labels ("REAL" and "FAKE"), and cleans the text data.
            Then, the text data is vectorized using the TfidfVectorizer from scikit-learn.
            A machine learning model (PassiveAggressiveClassifier) is trained to classify the news articles.
            The script then evaluates the model's accuracy and prints the confusion matrix.
            Finally, the script makes predictions on sample news articles to demonstrate how it works.
