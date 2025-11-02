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
CORS(app)

# Global variables
model = None
scaler = MinMaxScaler(feature_range=(0, 1))

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

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker', 'TATAMOTORS.NS')
        start_date = data.get('start_date', '2010-01-01')
        end_date = data.get('end_date', '2023-05-31')
        
        # Train model
        model, scaler = train_model(ticker, start_date, end_date)
        
        # Download data for prediction
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        prices = stock_data['Close'].values.reshape(-1, 1)
        scaled_prices = scaler.transform(prices)
        
        # Prepare test data - use the same scaler and split as training
        train_size = int(len(scaled_prices) * 0.8)
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
            'dates': test_dates.strftime('%Y-%m-%d').tolist()
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
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

