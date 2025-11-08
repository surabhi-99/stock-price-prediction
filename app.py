from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
# Allow all origins for CORS with proper headers - more permissive for local development
CORS(app, 
     resources={r"/*": {"origins": "*"}},  # Allow all routes, not just /api/*
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "HEAD"],
     expose_headers=["Content-Type", "Authorization"])

# Global variables
model = None
scaler = MinMaxScaler(feature_range=(0, 1))

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE, HEAD')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

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

def train_model(ticker, start_date, end_date):
    global model, scaler
    
    # Download data
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        raise Exception(f"Failed to download data for {ticker}: {str(e)}")
    
    # Check if data is empty
    if data.empty:
        raise Exception(f"No data found for ticker {ticker}. Please check the ticker symbol and date range.")
    
    # Check if Close column exists
    if 'Close' not in data.columns:
        raise Exception(f"No 'Close' price data found for {ticker}")
    
    prices = data['Close'].values.reshape(-1, 1)
    
    # Check for sufficient data
    if len(prices) < 100:
        raise Exception(f"Insufficient data for {ticker}. Got {len(prices)} data points, need at least 100.")
    
    # Normalize
    scaled_prices = scaler.fit_transform(prices)
    
    # Split
    train_size = int(len(scaled_prices) * 0.8)
    train_data = scaled_prices[:train_size]
    
    # Prepare training data
    time_steps = 30
    X_train, y_train = prepare_data(train_data, time_steps)
    
    # Check if we have enough training data
    if len(X_train) == 0:
        raise Exception(f"Not enough training data after preparing sequences. Need more than {time_steps} data points.")
    
    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    return model, scaler

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
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
        
        # Train model on historical data
        model, scaler = train_model(ticker, train_start_date, train_end_date)
        
        # Always calculate training data predictions for comparison
        stock_data = yf.download(ticker, start=train_start_date, end=train_end_date, progress=False)
        prices = stock_data['Close'].values.reshape(-1, 1)
        scaled_prices = scaler.transform(prices)
        
        # Prepare training data for predictions
        train_size = int(len(scaled_prices) * 0.8)
        train_data = scaled_prices[:train_size]
        time_steps = 30
        
        # Get predictions on training data
        X_train_pred, y_train_actual = prepare_data(train_data, time_steps)
        if len(X_train_pred) > 0:
            train_predictions = model.predict(X_train_pred, verbose=0)
            train_predictions = scaler.inverse_transform(train_predictions)
            train_actual = scaler.inverse_transform(y_train_actual)
            train_dates = stock_data.index[time_steps:train_size + time_steps]
            
            # Calculate training RMSE
            train_mse = np.mean((train_predictions - train_actual) ** 2)
            train_rmse = np.sqrt(train_mse)
        else:
            train_predictions = []
            train_actual = []
            train_dates = []
            train_rmse = None
        
        if is_future_prediction:
            # For future predictions: use last part of training data to predict future
            # Use the last 'time_steps' days to predict future
            last_sequence = scaled_prices[-time_steps:].reshape(1, time_steps, 1)
            
            # Generate predictions for future dates
            from datetime import datetime, timedelta
            import pandas as pd
            
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
            # Prepare test data - use the same scaler and split as training
            test_data = scaled_prices[train_size:]
            time_steps = 30
            
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
            test_dates = stock_data.index[train_size + time_steps:]
            
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

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

