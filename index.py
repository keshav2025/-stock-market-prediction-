import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/keshavkumar/Downloads/NIFTY 50-22-06-2023-to-22-06-2024 (1).csv'
data = pd.read_csv(file_path)

# Display the first few rows to verify the data is loaded correctly
print("Original DataFrame:")
print(data.head(10))

# Ensure that column names are correctly stripped of spaces
data.columns = data.columns.str.strip()

# Display the DataFrame with stripped column names
print("\nDataFrame with stripped column names:")
print(data.head(10))

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y', errors='coerce')

# Drop rows with invalid dates
data = data.dropna(subset=['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Check if 'Close' column exists and is not empty
if 'Close' not in data.columns or data['Close'].isnull().all():
    raise ValueError("The 'Close' column is missing or contains all null values.")

# Select the 'Close' column as the target variable
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Define a function to create training and test datasets
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Set the time step (number of previous days to use for prediction)
time_step = 30  # Adjusted time step to 30

# Check the shape of the scaled data
print(f"Scaled data shape: {scaled_data.shape}")

# Adjust the time step if necessary
if len(scaled_data) <= time_step:
    raise ValueError("Time step is too large for the available data. Reduce the time step.")

# Split the data into training and test sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

# Create the training and test datasets
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Check if test data is empty
if X_test.size == 0 or y_test.size == 0:
    raise ValueError("Test data is empty. Adjust the time step or check the data range.")
else:
    # Reshape the input data to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform the predictions and actual values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Debugging information for shapes
    print(f"Shape of train_predict: {train_predict.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of test_predict: {test_predict.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, close_prices, label='Original Data')

    train_predict_plot = np.empty_like(scaled_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step] = train_predict

    test_predict_plot = np.empty_like(scaled_data)
    test_predict_plot[:, :] = np.nan
    test_predict_start_idx = len(train_predict) + (time_step * 2)
    test_predict_plot[test_predict_start_idx:test_predict_start_idx + len(test_predict)] = test_predict

    plt.plot(data.index, train_predict_plot, label='Training Predictions')
    plt.plot(data.index, test_predict_plot, label='Test Predictions')
    
    plt.title('Nifty 50 Closing Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
