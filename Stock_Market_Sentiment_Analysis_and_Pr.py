# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import datetime

# Step 1: Download Historical Stock Data
stock_symbol = 'AAPL'  # Change this to any other symbol if needed
data = yf.download(stock_symbol, start='2020-01-01', end=datetime.date.today())

# Step 2: Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare training data
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 3: Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# Step 5: Predicting the Stock Prices
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Step 6: Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.plot(data['Close'].values, label='Actual Price', color='blue')
plt.plot(np.arange(60, len(predictions) + 60), predictions, label='Predicted Price', color='red')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
