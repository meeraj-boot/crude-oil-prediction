#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st


# In[2]:


import pandas as pd

# Read the CSV file and parse the 'Date' column as dates with the correct date format
data = pd.read_csv('Crude Oil Prices Daily.csv', parse_dates=['Date'], dayfirst=True)

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Now your 'Date' column should be interpreted correctly as dates
print(data.head())


# In[3]:


# Assuming your DataFrame is named 'data', change the column name from 'Closing Value' to 'Close'
data.rename(columns={'Closing Value': 'Close'}, inplace=True)

# Now the column name 'Closing Value' has been changed to 'Close'
print(data.head())

prices = data[['Close']]


# In[4]:


# Create a Streamlit web app
st.title('Crude Oil Price Prediction using LSTM')

st.subheader('Raw data')
st.write(data.tail())

# Display historical price chart
st.write("### Historical Crude Oil Price Chart")
st.line_chart(prices)

# Display distribution plots
st.write("### Data Distribution")
plt.figure(figsize=(8, 3))
sns.distplot(data['Close'], color='crimson', hist_kws={"edgecolor": 'white'}, norm_hist=False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Data Scaling
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# Create sequences for input and target
sequence_length = 10
X, y = [], []

for i in range(len(scaled_prices) - sequence_length):
    X.append(scaled_prices[i:i + sequence_length])
    y.append(scaled_prices[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split the data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=120, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test)

# Display prediction results
st.write("### Prediction Results")
actual_prices_df = pd.DataFrame(actual_prices, columns=['actual_prices'])
predicted_prices_df = pd.DataFrame(predicted_prices, columns=['predicted_prices'])
st.write(f"Actual Close Prices: {float(actual_prices_df.iloc[-1])}")
st.write(f"Predicted Prices: {float(predicted_prices_df.iloc[-1])}")

# Plot the results
plt.plot(actual_prices, color='red', label='Real Crude Oil Prices')
plt.plot(predicted_prices, color='blue', label=f"Predicted Crude Oil Prices is {round(float(predicted_prices_df.iloc[-1]), 2)}")
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Crude Oil Price')
plt.title('Crude Oil Price Prediction using LSTM')
plt.axhline(round(float(predicted_prices_df.iloc[-1]), 2), color='red', linestyle=':')
st.pyplot()


# In[ ]:




