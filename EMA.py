import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.style as style
import streamlit as st

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
