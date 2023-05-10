import streamlit as st
import yfinance as yf
import mplfinance as mpf

# Set the app title
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
