"""
Script to train and save LSTM models for stock price prediction.
Run this script before using the prediction API.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Directory where trained models will be stored
MODELS_DIR = 'models'

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Created directory: {MODELS_DIR}")

def prepare_data(data, time_steps):
    """Prepare data for LSTM training."""
    X, y = [], []
    # Ensure data is 2D with shape (n, 1)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    # Ensure X has shape (samples, time_steps, features)
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], time_steps, 1))
    elif len(X.shape) == 3 and X.shape[2] != 1:
        X = X.reshape((X.shape[0], time_steps, 1))
    
    # Ensure y has shape (samples, features)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    return X, y

def train_and_save_model(ticker, start_date='2010-01-01', end_date='2023-05-31', time_steps=30):
    """
    Train an LSTM model for a given ticker and save it to disk.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for training data
        end_date: End date for training data
        time_steps: Number of time steps for LSTM input
    """
    print(f"\n{'='*60}")
    print(f"Training model for {ticker}")
    print(f"{'='*60}")
    
    # Sanitize ticker for filename
    safe_ticker = ticker.replace('.', '_')
    
    # Download data
    try:
        print(f"Downloading data from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return False
    
    # Check if data is empty
    if data.empty:
        print(f"No data found for ticker {ticker}")
        return False
    
    # Check if Close column exists
    if 'Close' not in data.columns:
        print(f"No 'Close' price data found for {ticker}")
        return False
    
    prices = data['Close'].values.reshape(-1, 1)
    
    # Check for sufficient data
    if len(prices) < 100:
        print(f"Insufficient data for {ticker}. Got {len(prices)} data points, need at least 100.")
        return False
    
    print(f"Downloaded {len(prices)} data points")
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Use all data for training (no split for production models)
    train_data = scaled_prices
    
    # Prepare training data
    print("Preparing training sequences...")
    X_train, y_train = prepare_data(train_data, time_steps)
    
    # Check if we have enough training data
    if len(X_train) == 0:
        print(f"Not enough training data after preparing sequences. Need more than {time_steps} data points.")
        return False
    
    print(f"Prepared {len(X_train)} training sequences")
    
    # Build model
    print("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    print("Training model (this may take a while)...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f'{safe_ticker}_model.h5')
    scaler_path = os.path.join(MODELS_DIR, f'{safe_ticker}_scaler.pkl')
    metadata_path = os.path.join(MODELS_DIR, f'{safe_ticker}_metadata.pkl')
    
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'ticker': ticker,
        'time_steps': time_steps,
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'training_samples': len(X_train),
        'trained_date': datetime.now().isoformat()
    }
    
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Successfully trained and saved model for {ticker}")
    return True

if __name__ == '__main__':
    # List of common tickers to train
    # You can modify this list to include the tickers you want to support
    tickers = [
        'AAPL',      # Apple
        'MSFT',      # Microsoft
        'GOOGL',     # Google
        'AMZN',      # Amazon
        'TSLA',      # Tesla
        'TATAMOTORS.NS',  # Tata Motors (Indian stock)
        'RELIANCE.NS',    # Reliance (Indian stock)
        'TCS.NS',         # TCS (Indian stock)
    ]
    
    print("="*60)
    print("Stock Price Prediction Model Training")
    print("="*60)
    print(f"\nThis script will train models for {len(tickers)} tickers.")
    print("This may take a while depending on your system...")
    
    successful = 0
    failed = 0
    
    for ticker in tickers:
        try:
            if train_and_save_model(ticker):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error training model for {ticker}: {str(e)}")
            failed += 1
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Successfully trained: {successful} models")
    print(f"Failed: {failed} models")
    print(f"Models saved in: {MODELS_DIR}/")
    print("="*60)

