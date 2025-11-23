"""
Training script for stock price prediction models.
Trains models on all available historical data and saves them to files.
"""
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Popular tickers to train
POPULAR_TICKERS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
    'JPM', 'BAC', 'GS', 'WMT', 'PG', 'KO',
    'TATAMOTORS.NS', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS',
    'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS',
    'JNJ', 'V', 'MA', 'DIS', 'VZ'
]

# Directory to store trained models
MODELS_DIR = 'models'

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

def train_model(ticker, start_date=None, end_date=None):
    """
    Train a model for a specific ticker using all available data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Optional start date (if None, uses all available data)
        end_date: Optional end date (if None, uses current date)
    
    Returns:
        tuple: (model, scaler, metadata)
    """
    print(f"\nTraining model for {ticker}...")
    
    # Download data - use Ticker object for cleaner data access
    try:
        ticker_obj = yf.Ticker(ticker)
        
        if start_date and end_date:
            # Use history method for date range
            data = ticker_obj.history(start=start_date, end=end_date)
        else:
            # Download maximum available historical data
            from datetime import datetime, timedelta
            end_date_obj = datetime.now()
            start_date_obj = end_date_obj - timedelta(days=3650)  # 10 years
            
            # Try to get maximum available data using history method
            # First try 10-year period
            data = ticker_obj.history(period="10y")
            
            # If that didn't work well, try explicit date range
            if data.empty or len(data) < 100:
                data = ticker_obj.history(start=start_date_obj.strftime('%Y-%m-%d'), 
                                        end=end_date_obj.strftime('%Y-%m-%d'))
            
            # If still not enough, try max period
            if data.empty or len(data) < 100:
                data = ticker_obj.history(period="max")
    except Exception as e:
        raise Exception(f"Failed to download data for {ticker}: {str(e)}")
    
    # Check if data is empty
    if data.empty:
        raise Exception(f"No data found for ticker {ticker}")
    
    # Ticker.history() returns a DataFrame with standard columns: Open, High, Low, Close, Volume, etc.
    # Check if Close column exists
    if 'Close' not in data.columns:
        raise Exception(f"No 'Close' price data found for {ticker}. Available columns: {list(data.columns)}")
    
    # Get Close prices
    prices = data['Close'].values.reshape(-1, 1)
    
    # Check for sufficient data
    if len(prices) < 100:
        raise Exception(f"Insufficient data for {ticker}. Got {len(prices)} data points, need at least 100. "
                       f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    print(f"  Downloaded {len(prices)} data points")
    print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Normalize - use ALL data for fitting (no train/test split for training)
    scaled_prices = scaler.fit_transform(prices)
    
    # Prepare training data - use all available data
    time_steps = 30
    X_train, y_train = prepare_data(scaled_prices, time_steps)
    
    # Check if we have enough training data
    if len(X_train) == 0:
        raise Exception(f"Not enough training data after preparing sequences. Need more than {time_steps} data points.")
    
    print(f"  Prepared {len(X_train)} training sequences")
    
    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    print(f"  Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Create metadata
    metadata = {
        'ticker': ticker,
        'data_points': len(prices),
        'training_sequences': len(X_train),
        'date_range': {
            'start': data.index[0].strftime('%Y-%m-%d'),
            'end': data.index[-1].strftime('%Y-%m-%d')
        },
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'time_steps': time_steps
    }
    
    print(f"  ✓ Model trained successfully")
    
    return model, scaler, metadata

def save_model(ticker, model, scaler, metadata):
    """Save model, scaler, and metadata to files."""
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Sanitize ticker for filename (replace . with _)
    safe_ticker = ticker.replace('.', '_')
    
    # Save Keras model
    model_path = os.path.join(MODELS_DIR, f'{safe_ticker}_model.h5')
    model.save(model_path)
    
    # Save scaler using pickle
    scaler_path = os.path.join(MODELS_DIR, f'{safe_ticker}_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata using pickle
    metadata_path = os.path.join(MODELS_DIR, f'{safe_ticker}_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"  ✓ Saved model files to {MODELS_DIR}/")

def train_all_models():
    """Train models for all popular tickers."""
    print("=" * 60)
    print("Starting model training for all tickers")
    print("=" * 60)
    
    successful = 0
    failed = 0
    failed_tickers = []
    
    for ticker in POPULAR_TICKERS:
        try:
            model, scaler, metadata = train_model(ticker)
            save_model(ticker, model, scaler, metadata)
            successful += 1
        except Exception as e:
            failed += 1
            failed_tickers.append((ticker, str(e)))
            print(f"  ✗ Failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_tickers:
        print("\nFailed tickers:")
        for ticker, error in failed_tickers:
            print(f"  - {ticker}: {error}")
    
    print(f"\nModels saved to: {os.path.abspath(MODELS_DIR)}/")
    print("=" * 60)

if __name__ == '__main__':
    train_all_models()

