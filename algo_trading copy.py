import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

def analyze_ticker(ticker, start_date, end_date, short_window, long_window):
    stock_data = fletch_stock_data(ticker, start_date, end_date)
    stock_data = calculate_moving_averages(stock_data, short_window, long_window)
    signals = generate_signals(stock_data)
    success_rate = calculate_success_rate(stock_data, signals)
    return success_rate

def fletch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

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

def calculate_success_rate(data, signals):
    # Add shifted closing price data to the signals DataFrame
    signals['Shifted_Close'] = data['Close'].shift(-1)
    # Calculate whether the stock price increased or decreased
    signals['Actual_Change'] = np.where(signals['Shifted_Close'] > data['Close'], 1, 0)
    # Calculate the number of correct predictions
    correct_predictions = (signals['Signal'] == signals['Actual_Change']).sum()
    # Calculate the success rate
    success_rate = correct_predictions / len(signals)
    return success_rate

def plot_strategy(tickers, success_rates):
    fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the figsize for better visibility
    bars = ax.bar(tickers, success_rates)

    ax.set_ylabel('Success Rate')
    ax.set_title('Prediction Success Rates for Multiple Stocks')
    ax.set_xticks(tickers)

    # Add percentage labels above
    for bar, success_rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.annotate('{:.2%}'.format(success_rate),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()
def fletch_tickers(top):
    url1 = "https://ca.finance.yahoo.com/most-active?count=25&offset=0"
    url2 = "https://ca.finance.yahoo.com/most-active?count=25&offset=25"
    
    tables1 = pd.read_html(url1)
    tables2 = pd.read_html(url2)
    
    most_active_stocks1 = tables1[0]
    most_active_stocks2 = tables2[0]
    
    most_active_stocks = pd.concat([most_active_stocks1, most_active_stocks2], ignore_index=True)
    
    tickers = most_active_stocks['Symbol'].tolist()
    tickers = [ticker + '.TO' if '.TO' not in ticker else ticker for ticker in tickers]
    top_tickers = tickers[:top]
    
    return top_tickers
top = 50
tickers = fletch_tickers(top)  # Replace with the top tickers in tsx
start_date = datetime(1900, 1, 1)
end_date = datetime.now()
short_window = 20
long_window = 50
success_rates = [analyze_ticker(ticker, start_date, end_date, short_window, long_window) for ticker in tickers]
ticker_success_pairs = [(ticker, analyze_ticker(ticker, start_date, end_date, short_window, long_window)) for ticker in tickers]
sorted_pairs = sorted(ticker_success_pairs, key=lambda x: x[1], reverse=True)
average_success_rate = sum([success_rate for _, success_rate in sorted_pairs]) / len(sorted_pairs)

print("Prediction Success Rates (sorted from highest to lowest):")
for ticker, success_rate in sorted_pairs:
    print("{}: {:.2%}".format(ticker, success_rate))

print("\nAverage Success Rate: {:.2%}".format(average_success_rate))
sorted_tickers, sorted_success_rates = zip(*sorted_pairs)
plot_strategy(sorted_tickers, sorted_success_rates)