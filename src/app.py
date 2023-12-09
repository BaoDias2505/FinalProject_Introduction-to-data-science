from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
import re
from typing import List
import os

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
# nltk.download('punkt')

# ------------------------------ Prepare ------------------------------

# Load the saved KMeans model
with open(f"./data/models/my_best_model.pkl", "rb") as model_file:
    my_best_model = pickle.load(model_file)

# Preprocess the data: convert to lowercase, remove special characters, and stopwords
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [token for token in text.split() if token not in stop_words]
    return ' '.join(tokens)


# Read the data
clustered_data = pd.read_csv("./data/processed/video_data_clustered.csv")
clustered_data['clean_title'] = clustered_data['title'].apply(preprocess_text)

# Create a TF-IDF vectorizer to vectorize the text
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',
                             min_df=2,  # !ADD THIS LINE TO REDUCE THE VOCABULARY SIZE
                             dtype=np.float32)
vectorizer.fit(clustered_data['clean_title'])


# Return top `n` video titles in the same cluster


def _recommend_top_n_videos(cluster_id, top_n: int = 10) -> List[str]:
    # Define a dataframe to store the recommended videos
    recommended_videos = pd.DataFrame(columns=['channelTitle', 'title'])
    # !USING THE BELOW CODE FOR TESTING
    # recommended_videos = pd.DataFrame(columns=['channelTitle', 'title', 'viewCount'])

    # Define a set to store the channels that have been recommended
    set_of_channels = set()

    # Get all videos in the same cluster
    #   Then sort them by their views
    video_in_cluster = clustered_data.query(f"cluster_id == {cluster_id}")\
        .sort_values(by='viewCount', ascending=False)

    # Iterate through each video in the cluster
    for i, video in video_in_cluster.iterrows():
        channelTitle = video['channelTitle']
        videoTitle = video['title']

        # If the channel has already been recommended, skip it
        if channelTitle in set_of_channels:
            continue

        # Otherwise, add it to the list of recommended channels
        if len(recommended_videos.columns) == 2:
            recommended_videos.loc[len(recommended_videos)] = [
                channelTitle, videoTitle]
        else:  # !USING THE BELOW CODE FOR TESTING
            recommended_videos.loc[len(recommended_videos)] = [
                channelTitle, videoTitle, video['viewCount']]

        # Add the channel to the set
        set_of_channels.add(channelTitle)

        # If we have enough channels, stop
        if len(recommended_videos) >= top_n:
            break

    return recommended_videos


# ------------------------------ APP ------------------------------
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        # Preprocess the input text
        text_data = pd.Series(text)
        text_data = text_data.apply(preprocess_text)
        text_transformed = vectorizer.transform(text_data)

        # Use the trained model to predict cluster for the input text
        cluster = my_best_model.predict(text_transformed)

        # Additional logic to recommend videos based on the cluster
        recommended_videos = _recommend_top_n_videos(cluster, top_n=5)

        return render_template('result.html',
                               tables=[recommended_videos.to_html(
                                   classes='data')],
                               titles=recommended_videos.columns.values)


if __name__ == '__main__':
    app.run(debug=False)
