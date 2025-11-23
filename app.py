from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Allowed origins including deployed frontend
allowed_origins = [
    os.getenv("FRONTEND_URL"),
    "http://localhost:3000",
    "https://stock-price-prediction-beryl.vercel.app",
]

# Filter out None values (in case FRONTEND_URL is not set)
allowed_origins = [origin for origin in allowed_origins if origin]

# Enable CORS for all routes with proper configuration
# Using both global and per-resource configuration for maximum compatibility
CORS(app, 
     resources={
         r"/*": {
             "origins": allowed_origins,
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
             "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept"],
             "expose_headers": ["Content-Type"],
             "supports_credentials": True,  # Can use True with specific origins (not wildcard)
             "max_age": 3600
         }
     },
     # Global CORS settings as fallback
     origins=allowed_origins,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
     supports_credentials=True
)

# Directory where trained models are stored
MODELS_DIR = 'models'

# Cache for loaded models (to avoid reloading on every request)
models_cache = {}

def prepare_data(data, time_steps):
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

def train_model_on_demand(ticker):
    """
    Train a model for a specific ticker on-demand.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        tuple: (model, scaler, metadata) or None if training fails
    """
    print(f"Training model on-demand for {ticker}...")
    
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Try to get maximum available data
        data = ticker_obj.history(period="10y")
        
        # If that didn't work well, try explicit date range
        if data.empty or len(data) < 100:
            end_date_obj = datetime.now()
            start_date_obj = end_date_obj - timedelta(days=3650)  # 10 years
            data = ticker_obj.history(start=start_date_obj.strftime('%Y-%m-%d'), 
                                    end=end_date_obj.strftime('%Y-%m-%d'))
        
        # If still not enough, try max period
        if data.empty or len(data) < 100:
            data = ticker_obj.history(period="max")
        
        # Check if data is empty
        if data.empty:
            raise Exception(f"No data found for ticker {ticker}")
        
        # Check if Close column exists
        if 'Close' not in data.columns:
            raise Exception(f"No 'Close' price data found for {ticker}")
        
        # Get Close prices
        prices = data['Close'].values.reshape(-1, 1)
        
        # Check for sufficient data
        if len(prices) < 100:
            raise Exception(f"Insufficient data for {ticker}. Got {len(prices)} data points, need at least 100.")
        
        print(f"  Downloaded {len(prices)} data points for {ticker}")
        
        # Create scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Normalize - use ALL data for fitting
        scaled_prices = scaler.fit_transform(prices)
        
        # Prepare training data - use all available data
        time_steps = 30
        X_train, y_train = prepare_data(scaled_prices, time_steps)
        
        # Check if we have enough training data
        if len(X_train) == 0:
            raise Exception(f"Not enough training data after preparing sequences. Need more than {time_steps} data points.")
        
        print(f"  Prepared {len(X_train)} training sequences for {ticker}")
        
        # Build model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train
        print(f"  Training model for {ticker}...")
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
        
        # Save the model for future use
        save_trained_model(ticker, model, scaler, metadata)
        
        print(f"  ✓ Model trained and saved for {ticker}")
        
        return model, scaler, metadata
    except Exception as e:
        print(f"Error training model for {ticker}: {str(e)}")
        return None

def save_trained_model(ticker, model, scaler, metadata):
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

def load_trained_model(ticker):
    """
    Load a trained model from disk, or train it on-demand if not found.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        tuple: (model, scaler, metadata) or None if not found and training fails
    """
    # Check cache first
    if ticker in models_cache:
        return models_cache[ticker]
    
    # Sanitize ticker for filename
    safe_ticker = ticker.replace('.', '_')
    
    # Construct file paths
    model_path = os.path.join(MODELS_DIR, f'{safe_ticker}_model.h5')
    scaler_path = os.path.join(MODELS_DIR, f'{safe_ticker}_scaler.pkl')
    metadata_path = os.path.join(MODELS_DIR, f'{safe_ticker}_metadata.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Model doesn't exist, train it on-demand
        print(f"Model not found for {ticker}, training on-demand...")
        model_data = train_model_on_demand(ticker)
        if model_data is None:
            return None
        model, scaler, metadata = model_data
        # Cache the newly trained model
        models_cache[ticker] = (model, scaler, metadata)
        return model, scaler, metadata
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata if available
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
        # Cache the loaded model
        models_cache[ticker] = (model, scaler, metadata)
        
        return model, scaler, metadata
    except Exception as e:
        print(f"Error loading model for {ticker}: {str(e)}")
        # If loading fails, try training on-demand
        print(f"Attempting to train model on-demand for {ticker}...")
        model_data = train_model_on_demand(ticker)
        if model_data is None:
            return None
        model, scaler, metadata = model_data
        # Cache the newly trained model
        models_cache[ticker] = (model, scaler, metadata)
        return model, scaler, metadata

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response
    try:
        data = request.json
        ticker = data.get('ticker', 'TATAMOTORS.NS')
        train_start_date = data.get('train_start_date', None)
        train_end_date = data.get('train_end_date', None)
        predict_start_date = data.get('predict_start_date', None)
        predict_end_date = data.get('predict_end_date', None)
        
        # Fallback for old API format
        if not train_start_date:
            train_start_date = data.get('start_date', '2010-01-01')
            train_end_date = data.get('end_date', '2023-05-31')
            predict_start_date = train_end_date
            predict_end_date = train_end_date
            is_future_prediction = False
        else:
            is_future_prediction = True
        
        # Load trained model from file (or train on-demand if not found)
        model_data = load_trained_model(ticker)
        if model_data is None:
            return jsonify({
                'error': f'Failed to load or train model for ticker {ticker}. Please check if the ticker symbol is valid and try again.'
            }), 500
        
        model, scaler, metadata = model_data
        
        # Download stock data for predictions
        # Use the date range from metadata if available, otherwise use requested dates
        if metadata and 'date_range' in metadata:
            data_start = metadata['date_range']['start']
            data_end = metadata['date_range']['end']
        else:
            data_start = train_start_date
            data_end = train_end_date
        
        stock_data = yf.download(ticker, start=data_start, end=data_end, progress=False)
        
        # Check if data is empty
        if stock_data.empty:
            return jsonify({
                'error': f'No data found for ticker {ticker} in the specified date range.'
            }), 400
        
        prices = stock_data['Close'].values.reshape(-1, 1)
        scaled_prices = scaler.transform(prices)
        
        # Get time_steps from metadata or use default
        time_steps = metadata.get('time_steps', 30) if metadata else 30
        
        # For predictions, use all available data (model was trained on all data)
        all_data = scaled_prices
        
        # Get predictions on available data for comparison
        # Use a portion of the data for evaluation (last 20% for test, rest for training visualization)
        eval_size = int(len(all_data) * 0.2)
        train_data = all_data[:-eval_size] if eval_size > 0 else all_data
        
        X_train_pred, y_train_actual = prepare_data(train_data, time_steps)
        if len(X_train_pred) > 0:
            train_predictions = model.predict(X_train_pred, verbose=0)
            train_predictions = scaler.inverse_transform(train_predictions)
            train_actual = scaler.inverse_transform(y_train_actual)
            train_dates = stock_data.index[time_steps:len(train_data) + time_steps]
            
            # Calculate training RMSE
            train_mse = np.mean((train_predictions - train_actual) ** 2)
            train_rmse = np.sqrt(train_mse)
        else:
            train_predictions = []
            train_actual = []
            train_dates = []
            train_rmse = None
        
        if is_future_prediction:
            # For future predictions: use last part of available data to predict future
            # Use the last 'time_steps' days to predict future
            last_sequence = all_data[-time_steps:].reshape(1, time_steps, 1)
            
            # Generate predictions for future dates
            predict_dates = pd.date_range(start=predict_start_date, end=predict_end_date, freq='B')  # Business days
            predicted_future = []
            
            # Start with last sequence from training data
            current_sequence = last_sequence.copy()
            
            for i in range(len(predict_dates)):
                # Predict next price
                next_pred = model.predict(current_sequence, verbose=0)
                predicted_future.append(next_pred[0, 0])
                
                # Update sequence: remove first, add prediction
                current_sequence = np.append(current_sequence[0, 1:, :], next_pred).reshape(1, time_steps, 1)
            
            # Inverse transform predictions
            predicted = scaler.inverse_transform(np.array(predicted_future).reshape(-1, 1))
            
            # For future predictions, we don't have actual values yet
            actual = None
            rmse = None
            
            response = {
                'predicted': predicted.flatten().tolist(),
                'dates': predict_dates.strftime('%Y-%m-%d').tolist(),
                'is_future': True,
                'ticker': ticker,
                'training_data': {
                    'predicted': train_predictions.flatten().tolist() if len(train_predictions) > 0 else [],
                    'actual': train_actual.flatten().tolist() if len(train_actual) > 0 else [],
                    'dates': train_dates.strftime('%Y-%m-%d').tolist() if len(train_dates) > 0 else [],
                    'rmse': float(train_rmse) if train_rmse is not None else None
                }
            }
            
        else:
            # Original behavior: predict on test set from historical data
            # Use the last portion of data as test set
            eval_size = int(len(all_data) * 0.2)
            if eval_size == 0:
                eval_size = min(50, len(all_data) - time_steps - 1)
            
            test_data = all_data[-eval_size:] if eval_size > 0 else all_data
            
            # Ensure we have enough data points
            if len(test_data) < time_steps + 1:
                return jsonify({'error': f'Not enough test data. Need at least {time_steps + 1} data points, got {len(test_data)}'}), 400
            
            X_test, y_test = prepare_data(test_data, time_steps)
            
            # Make predictions (X_test should now have correct shape from prepare_data)
            predicted = model.predict(X_test, verbose=0)
            predicted = scaler.inverse_transform(predicted)
            actual = scaler.inverse_transform(y_test)
            
            # Calculate metrics
            mse = np.mean((predicted - actual) ** 2)
            rmse = np.sqrt(mse)
            
            # Prepare response - get dates corresponding to test predictions
            test_start_idx = len(all_data) - len(test_data) + time_steps
            test_dates = stock_data.index[test_start_idx:]
            
            response = {
                'actual': actual.flatten().tolist(),
                'predicted': predicted.flatten().tolist(),
                'rmse': float(rmse),
                'dates': test_dates.strftime('%Y-%m-%d').tolist(),
                'is_future': False,
                'training_data': {
                    'predicted': train_predictions.flatten().tolist() if len(train_predictions) > 0 else [],
                    'actual': train_actual.flatten().tolist() if len(train_actual) > 0 else [],
                    'dates': train_dates.strftime('%Y-%m-%d').tolist() if len(train_dates) > 0 else [],
                    'rmse': float(train_rmse) if train_rmse is not None else None
                }
            }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # Log full error to console for debugging
        print("=" * 50)
        print("ERROR OCCURRED:")
        print(str(e))
        print("=" * 50)
        print("TRACEBACK:")
        print(error_details)
        print("=" * 50)
        # Return user-friendly error message
        return jsonify({
            'error': str(e),
            'details': error_details.split('\n')[-5:] if len(error_details) > 100 else error_details
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    # Count available models
    model_count = 0
    if os.path.exists(MODELS_DIR):
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.h5')]
        model_count = len(model_files)
    
    return jsonify({
        'status': 'ok', 
        'message': 'Backend is running',
        'cached_models': len(models_cache),
        'available_models': model_count
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
