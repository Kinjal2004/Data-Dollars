# importing all important libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas_datareader.data as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
import mplfinance as mpf
from sklearn.linear_model import LinearRegression
import datetime as dt

# defining pages

def app():
    st.title('Stock Trend Predictor')
    user_input = st.text_input('Enter Stock Ticker', 'HAL')
    yf.pdr_override()
    start = '2022-01-01'
    end = '2022-12-31'
    df = data.get_data_yahoo(user_input, start=start, end=end)
    st.subheader('Data from 1st January 2022 to 31st December 2022')
    st.write(df.describe())
    # Visualization
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training_array. shape[0]):
        x_train.append(data_training_array[i-100: i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = load_model('keras_model.h5')
    past_100_days = data_training.tail(100)

    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    st.subheader('Predictions vs Original')

    fig2 = plt.figure(figsize=(12, 6))

    plt.plot(y_test, 'b', label='Original Price', color="red")

    plt.plot(y_predicted,
             'p', label='Predicted Price', color="green")
    plt.xlabel('Time')
    plt.ylabel('Price')

    plt. legend()

    st.pyplot(fig2)
def candlestick():
    st.title('HAL Real-Time Stock Data')

    # Define the stock ticker
    ticker = 'HAL'

    # Fetch the real-time stock data
    stock_data = yf.download(ticker, period='1d', interval='1m')

    # Check if there is any data for the entered ticker
    if stock_data.empty:
        st.warning("No data found. Please enter a valid stock ticker.")
    else:
        # Plot the data in a candlestick chart with volume bars
        fig, ax = mpf.plot(stock_data, type='candle', volume=True, figratio=(2,1), figscale=1.5, style='yahoo', ylabel='Price', volume_panel=1, ylabel_lower='Volume', show_nontrading=True, returnfig=True)
        st.pyplot(fig)
def EMA():
    # Set the plotting style
    style.use("ggplot")

    # Define the stock symbol
    symbol = "HAL"

    # Define the timeframe (default is 1 day)
    timeframe = "1d"

    # Get the live stock data from Yahoo Finance
    data = yf.download(symbol, period="1d", interval="1m")

    # Calculate the 20-day exponential moving average (EMA)
    ema = data["Close"].ewm(span=20, adjust=False).mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["Close"], label="Price")
    ax.plot(ema, label="EMA")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol} Live Stock Data")
    ax.legend()

    # Convert the plot to a Streamlit-friendly format
    st.pyplot(fig)
def rsi():
    st.title("RSI Predictor")
    # Download the HAL stock data from Yahoo Finance
    hal = yf.download('HAL', start='2022-01-01', end='2022-12-31')

    # Calculate the price difference between each day
    hal['Diff'] = hal['Close'].diff()

    # Calculate the positive and negative price differences
    hal['PosDiff'] = hal['Diff'].apply(lambda x: x if x > 0 else 0)
    hal['NegDiff'] = hal['Diff'].apply(lambda x: -x if x < 0 else 0)

    # Calculate the rolling average of positive and negative price differences
    hal['PosAvg'] = hal['PosDiff'].rolling(window=14).mean()
    hal['NegAvg'] = hal['NegDiff'].rolling(window=14).mean()

    # Calculate the relative strength
    hal['RS'] = hal['PosAvg'] / hal['NegAvg']

    # Calculate the RSI
    hal['RSI'] = 100 - (100 / (1 + hal['RS']))

    # Create a new column for the buy and sell signals
    hal['Signal'] = 0
    hal.loc[hal['RSI'] < 30, 'Signal'] = 1
    hal.loc[hal['RSI'] > 70, 'Signal'] = -1

    # Plot the data and RSI with buy and sell signals
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
    ax1.plot(hal['Close'])
    ax1.set_ylabel('Price')
    ax1.set_title('HAL Stock Price in 2022')

    # Plot the buy and sell signals
    ax1.plot(hal.loc[hal['Signal'] == 1].index, hal['Close'][hal['Signal'] == 1], '^', markersize=10, color='g')
    ax1.plot(hal.loc[hal['Signal'] == -1].index, hal['Close'][hal['Signal'] == -1], 'v', markersize=10, color='r')

    ax2.plot(hal['RSI'])
    ax2.set_ylabel('RSI')
    ax2.set_title('RSI of HAL Stock in 2022')

    # Plot the buy and sell signals
    ax2.axhline(y=30, color='b', linestyle='-')
    ax2.axhline(y=70, color='b', linestyle='-')
    ax2.fill_between(hal.index, y1=30, y2=70, color='#adccff', alpha=0.3)

    # Display the plot in Streamlit
    st.pyplot(fig)
def new():
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

#dictionary 
pages= {
    "Home": app,
    "Candlestick Model": candlestick,
    "EMA Indicator": EMA,
    "RSI Indicator": rsi,
    "Stock Value":new,
}

# Create a menu with the different page options
page = st.sidebar.selectbox("Select a page", tuple(pages.keys()))

# Call the function associated with the selected page
pages[page]()

