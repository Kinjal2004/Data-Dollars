import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime as dt
import streamlit as st


user_input = st.text_input('Enter Stock Ticker','HAL')

st.title(user_input+' Stock Price Predictor For Tomorrow')

# Retrieve historical data for Apple stock
data = yf.download(user_input, start="2016-01-01", end=dt.datetime.now())

# Prepare the data by selecting the closing prices and shifting them by one day
data = pd.DataFrame(data["Close"])
data["Prediction"] = data["Close"].shift(-1)

# Split the data into training and testing sets
train_data = data[:-1]
test_data = data[-1:]

# Train a linear regression model on the training data
X_train = train_data.drop("Prediction", axis=1)
y_train = train_data["Prediction"]
model = LinearRegression()
model.fit(X_train, y_train)

# Use the model to make a prediction for the next day's closing price
X_test = test_data.drop("Prediction", axis=1)
prediction = model.predict(X_test)

st.subheader(prediction[0])
