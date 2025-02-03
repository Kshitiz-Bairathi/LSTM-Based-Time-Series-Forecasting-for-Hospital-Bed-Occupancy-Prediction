import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all but error messages from TensorFlow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from tensorflow.keras.layers import Input
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import joblib

# Load data from the Excel file
file_path = 'Aiims Jodhpur_11.xlsx'
df = pd.read_excel(file_path)

# Ensure the date column is in datetime format and sort the data by date
df = df[pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').notna()]
df = df.sort_values('Date')

# Drop any rows with missing values
df = df.dropna()

# Set date as index
df.set_index('Date', inplace=True)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Define the function to create sequences
def create_sequences(data, sequence_length=30):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Define sequence length and split data into sequences
sequence_length = 30  # Use the last 30 days to predict the next day
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(y_train.shape[1])  # Output layer for each ward
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# Evaluate the model on the test set
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Calculate the evaluation metrics
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
rmse = sqrt(mse)

# Print the evaluation results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the trained model and scaler for later use
model.save('hospital_bed_model_11.keras')
joblib.dump(scaler, 'scaler.pkl')

