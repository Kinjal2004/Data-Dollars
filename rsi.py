import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
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
