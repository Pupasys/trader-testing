import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def prepare_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# Replace 'Close' with the feature you want to predict
data = stock_data['Close'].values.astype('float32')
data = data.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Split data into train and test sets (e.g., 80% for training and 20% for testing)
train_size = int(len(data) * 0.8)
train, test = data[0:train_size, :], data[train_size:len(data), :]

# Prepare the data for LSTM (using a sliding window approach)
window_size = 20  # Define your desired window size
X_train, y_train = prepare_data(train, window_size)
X_test, y_test = prepare_data(test, window_size)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, window_size)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert the predictions back to the original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

train_score = np.sqrt(mean_squared_error(y_train[0], train_predict[:, 0]))
test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
print('Train Score: {:.2f} RMSE'.format(train_score))
print('Test Score: {:.2f} RMSE'.format(test_score))
