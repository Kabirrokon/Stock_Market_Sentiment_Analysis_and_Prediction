# Stock Market Sentiment Analysis and Prediction

import numpy as np
import pandas as pd
import yfinance as yf
import tweepy
import re
import nltk
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Twitter API credentials (replace with your own)
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

# Twitter API authentication
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Fetching stock price data
def get_stock_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2025-01-01')
    data = data[['Close']]
    return data

# Fetching real-time tweets
def get_tweets(keyword):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang='en', tweet_mode='extended').items(100)
    tweet_list = [tweet.full_text for tweet in tweets]
    return tweet_list

# Preprocess tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    tweet = tweet.lower()
    return tweet

# Sentiment analysis
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

# Predicting stock prices using LSTM
def predict_stock(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    predictions = model.predict(X_train)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Visualization
def plot_predictions(data, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Example execution
stock_data = get_stock_data('AAPL')
tweets = get_tweets('AAPL')
sentiments = [analyze_sentiment(preprocess_tweet(tweet)) for tweet in tweets]
predictions = predict_stock(stock_data)
plot_predictions(stock_data.values, predictions)
