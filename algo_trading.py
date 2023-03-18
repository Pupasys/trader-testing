import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

ticker = 'AMZN'  # You can replace this with any stock symbol you prefer
start_date = '2020-01-01'
end_date = '2021-01-01'

stock_data = fetch_stock_data(ticker, start_date, end_date)

def calculate_moving_averages(data, short_window, long_window):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    return data

def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    signals['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1.0, 0.0)
    signals['Positions'] = signals['Signal'].diff()

    return signals

short_window = 20
long_window = 50

stock_data = calculate_moving_averages(stock_data, short_window, long_window)
signals = generate_signals(stock_data)

def plot_strategy(data, signals):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot stock closing price and moving averages
    ax1.plot(data.index, data['Close'], label='Close', linewidth=1)
    ax1.plot(data.index, data['Short_MA'], label='Short MA', linewidth=1)
    ax1.plot(data.index, data['Long_MA'], label='Long MA', linewidth=1)

    # Plot buy signals
    ax1.plot(signals.loc[signals['Positions'] == 1].index, data['Close'][signals['Positions'] == 1], '^', markersize=10, color='g', label='Buy signal')

    # Plot sell signals
    ax1.plot(signals.loc[signals['Positions'] == -1].index, data['Close'][signals['Positions'] == -1], 'v', markersize=10, color='r', label='Sell signal')

    ax1.set_ylabel('Price')
    ax1.set_title(ticker +' Stock Price, Moving Averages & Buy/Sell Signals')
    ax1.legend(loc='best')
    plt.xlabel('Date')
    plt.show()

plot_strategy(stock_data, signals)
