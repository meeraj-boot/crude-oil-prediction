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
import pickle
from datetime import datetime, timedelta


# In[6]:


import pandas as pd

# Read the CSV file and parse the 'Date' column as dates with the correct date format
data = pd.read_csv('Crude Oil Prices Daily.csv', parse_dates=['Date'], dayfirst=True)

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Now your 'Date' column should be interpreted correctly as dates
print(data.head())


# In[12]:


# Assuming your DataFrame is named 'data', change the column name from 'Closing Value' to 'Close'
data.rename(columns={'Closing Value': 'Close'}, inplace=True)

# Now the column name 'Closing Value' has been changed to 'Close'
print(data.head())

prices = data[['Close']]


# In[8]:


# Create a Streamlit web app
st.title('Crude Oil Price Prediction using LSTM')


# In[9]:


st.subheader('Raw data')
st.write(data.tail())


# In[10]:


st.subheader('Raw data')
st.write(data.tail())


# In[13]:


# Display historical price chart
st.write("### Historical Crude Oil Price Chart")
st.line_chart(prices)


# In[14]:


# Display distribution plots
st.write("### Data Distribution")
plt.figure(figsize=(8, 3))
sns.distplot(data['Close'], color='crimson', hist_kws={"edgecolor": 'white'}, norm_hist=False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


# In[15]:


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


# In[16]:


# Load the trained model
with open('oil_price_lstm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
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


#future_days = 7 
#st.sidebar.header('Select Date Range')
start_date = st.date_input('Start Date')
future_days = st.number_input('days')
button_clicked = st.button("Predict")
if button_clicked:
    end_date = start_date + pd.DateOffset(days=future_days)

    st.text(f"Forecasting from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Forecast future oil prices
    forecast_dates = pd.date_range(start=datetime.now().strftime('%Y-%m-%d'), end=end_date)
    days_difference = (forecast_dates[-1] - forecast_dates[0]).days
    #st.write(forecast_dates)
    #st.write(days_difference)
    forecast_scaled = []
    last_batch = X[-1]

    for _ in range(int(days_difference+1)):
        last_batch = last_batch.reshape(1, sequence_length, 1)
        forecast = model.predict(last_batch)
        forecast_scaled.append(forecast[0])
        last_batch = np.roll(last_batch, -1, axis=1)
        last_batch[:, -1, 0] = forecast[0]

    forecast_scaled = np.array(forecast_scaled)
    forecasted_prices = scaler.inverse_transform(forecast_scaled)
    #st.write(forecasted_prices)
    #Display forecasted prices
    st.write('### Forecasted Oil Prices')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data.values, label='Historical Prices')
    ax.plot(forecast_dates, forecasted_prices, label='Forecasted Prices', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Oil Price')
    ax.legend()
    st.pyplot(fig)
    new_dates =[]
    for dates in forecast_dates:
        new_dates.append(datetime.date(dates))

    #st.write(new_dates)
    new_prices =[]
    #st.write(forecasted_prices)
    #forecasted_prices = forecasted_prices[-future_days:]
    for price in forecasted_prices:
        new_prices.append(price)

    #st.write(new_prices)



    # Create a DataFrame
    df = pd.DataFrame({'date': new_dates, 'price': new_prices})
    df = df.set_index('date')

    st.write(df.tail(int(future_days+1)))


# In[ ]:




