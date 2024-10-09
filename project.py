# Import necessary libraries
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Load the dataset
data = pd.read_csv('Microsoft_Stock.csv".csv')  # Replace 'your_stock_data.csv' with your CSV file

# Display the first few rows of the dataset
print(data.head())

# Check the closing price
plt.figure(figsize=(10,6))
plt.plot(data['Close'], label='Closing Price')
plt.title('Stock Price History')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Preprocessing: Use only the 'Close' price for prediction
data = data[['Close']]

# Convert the dataframe to a numpy array
dataset = data.values

# Scale the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Define the time step to create features
time_step = 60

# Create training data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]

X_train = []
y_train = []

for i in range(time_step, len(train_data)):
    X_train.append(train_data[i-time_step:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy arrays and reshape
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=20)

# Test data
test_data = scaled_data[train_size - time_step:]
X_test = []
y_test = dataset[train_size:]

for i in range(time_step, len(test_data)):
    X_test.append(test_data[i-time_step:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Visualize the results
train = data[:train_size]
valid = data[train_size:]
valid['Predictions'] = predictions

# Plot the data
plt.figure(figsize=(10,6))
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.plot(train['Close'], label='Training Data')
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Training', 'Actual Price', 'Predicted Price'])
plt.show()
