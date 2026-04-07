"""
Deep Learning Stock Price Predictor (LSTM)
------------------------------------------
Uses Long Short-Term Memory (LSTM) Recurrent Neural Networks to predict stock prices.

Designed to run beautifully in Google Colab or your local machine!
Dependencies: pip install yfinance matplotlib scikit-learn tensorflow pandas numpy
"""

import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# For Colab compatibility or local environments
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
except ImportError:
    print("TensorFlow/Keras not found! Please run 'pip install tensorflow' or run this in Google Colab.")
    sys.exit()

# ==========================================
# 🎨 Console Visuals
# ==========================================
def typing_print(text, delay=0.015):
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Loading", dots=3, speed=0.3):
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(dots):
        time.sleep(speed)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()

# ==========================================
# 🚀 The Application
# ==========================================
def run_prediction():
    typing_print("=== 📈 Deep Learning LSTM Stock Predictor ===", delay=0.03)
    
    # 1. Fetch Data
    stock_symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    loading_animation(f"📥 Fetching historic stock data for {stock_symbol} ({start_date} to {end_date})", dots=3, speed=0.2)
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    
    typing_print(f"✅ Data fetched successfully! Shape: {df.shape}")
    
    # Extract only the 'Close' prices
    data = df.filter(['Close']).values
    
    # 2. Preprocess the Data (Scaling is critical for Neural Networks)
    loading_animation("🧹 Scaling data using MinMaxScaler", dots=3, speed=0.1)
    dataset_length = len(data)
    train_data_len = int(np.ceil(dataset_length * .8))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create the training dataset
    train_data = scaled_data[0:int(train_data_len), :]
    x_train, y_train = [], []
    
    # Look back 60 days to predict the next day
    look_back = 60
    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i-look_back:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape the data for LSTM [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # 3. Build the LSTM Model
    loading_animation("🧠 Architecting the Long Short-Term Memory (LSTM) Network", dots=4, speed=0.2)
    
    model = Sequential()
    # First LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Prevent overfitting
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Single predicted stock price
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 4. Train the Model
    typing_print("\n🔥 Commencing Neural Network Training (Epochs=10)...")
    time.sleep(1)
    
    # We use a small batch size and epochs for quick demonstration
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)
    
    typing_print("✅ Model Training Complete!\n")
    loading_animation("🔮 Generating Predictions against Test Data", dots=3, speed=0.2)
    
    # 5. Create the testing dataset
    test_data = scaled_data[train_data_len - look_back:, :]
    x_test = []
    y_test = data[train_data_len:, :]
    
    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i-look_back:i, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Get the models predicted price values 
    predictions = model.predict(x_test)
    # Unscale the data back to real $ values
    predictions = scaler.inverse_transform(predictions)
    
    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    typing_print(f"📊 Accuracy Metric - Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # 6. Plotting the visualization
    typing_print("📈 Rendering plot... (Check your windows or Colab output)")
    
    train = df[:train_data_len]
    valid = df[train_data_len:]
    valid['Predictions'] = predictions
    
    plt.figure(figsize=(16,6))
    plt.title(f'LSTM Model Prediction for {stock_symbol}')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price USD ($)', fontsize=12)
    plt.plot(train['Close'], color='blue', label='Historical Training Data')
    plt.plot(valid['Close'], color='orange', label='Actual Price')
    plt.plot(valid['Predictions'], color='green', label='LSTM Prediction')
    plt.legend(loc='lower right')
    plt.show()
    
    typing_print("\n👋 Prediction Complete!")

if __name__ == "__main__":
    run_prediction()
