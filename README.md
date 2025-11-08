# 📈 Stock Price Prediction App

A full-stack web application for predicting stock prices using LSTM (Long Short-Term Memory) neural networks.

## 🌐 Live Demo

| Service | Link |
|----------|------|
| **Backend (API)** | [https://stock-price-prediction-api.onrender.com](https://stock-price-prediction-api.onrender.com) |
| **Frontend (Web App)** | [https://stock-price-prediction-beryl.vercel.app](https://stock-price-prediction-beryl.vercel.app) |

---

## 🚀 Features

- **Real-time Predictions**: Train and predict stock prices using LSTM models
- **Interactive Charts**: Visualize actual vs predicted prices with Recharts
- **Multiple Stocks**: Support for any stock ticker from Yahoo Finance
- **RMSE Metrics**: Evaluate model performance with Root Mean Squared Error
- **Beautiful UI**: Modern, responsive React interface
- **Training vs Actual Chart**: See how well the model fits historical data with interactive comparison between actual prices and model predictions over the training period.
- **Future Prediction Chart**: Visualize the model's forecast for unseen future prices, based on learned data patterns.


## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## 🛠️ Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install React Dependencies

```bash
cd stock-prediction-frontend
npm install
```

## 🎯 Running the Application

### Start Backend (Terminal 1)

```bash
python app.py
```

Backend runs on: http://localhost:5000

### Start Frontend (Terminal 2)

```bash
cd stock-prediction-frontend
npm start
```

Frontend runs on: http://localhost:3000

The browser will automatically open at http://localhost:3000

## 📖 How to Use

1. **Enter Stock Ticker**: 
   - Examples: `AAPL`, `GOOGL`, `MSFT`, `TATAMOTORS.NS` (for NSE stocks)
   - Format: Standard ticker symbols from Yahoo Finance

2. **Set Date Range**:
   - Start Date: Beginning of historical data
   - End Date: End of historical data period
   - Recommended: At least 3-4 years of data for better results

3. **Click "Predict Stock Price"**:
   - Model will train (takes 1-2 minutes)
   - Results will show RMSE metric and chart

4. **Switch Chart View**:
   - Use the "View Switcher" to toggle between "Training vs Actual" and "Future Prediction" charts.
   - RMSE accuracy metric is shown alongside the charts for comparison.
  
## 📊 Visualizations

- **Training vs Actual Chart**: Interactive line chart comparing real historical prices and model predictions for training data. Lets users evaluate model fit and accuracy.
- **Future Prediction Chart**: Displays the model's forecast for future prices beyond the training period.

## 🧪 Testing

Test the API directly:

```bash
python test_api.py
```

Or use curl:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "start_date": "2020-01-01", "end_date": "2023-12-31"}'
```

## 🏗️ Project Structure

```
Stock price prediction/
├── app.py                          # Flask backend API
├── requirements.txt                # Python dependencies
├── test_api.py                    # API testing script
├── Stock_Price_Prediction.ipynb   # Jupyter Notebook with LSTM model implementation
└── stock-prediction-frontend/      # React frontend
    ├── src/
    │   ├── App.js                 # Main app component
    │   ├── components/
    │   │   ├── PredictionForm.js  # Input form
    │   │   ├── PredictionChart.js # Chart visualization
    │   │   └── MetricsDisplay.js   # RMSE display
    └── package.json
```

## 🔧 API Endpoints

### POST `/api/predict`

Predict stock prices for a given ticker and date range.

**Request:**
```json
{
  "ticker": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31"
}
```

**Response:**
```json
{
  "actual": [100.5, 101.2, ...],
  "predicted": [99.8, 101.5, ...],
  "rmse": 15.23,
  "dates": ["2023-01-01", "2023-01-02", ...]
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## 🎨 Technologies Used

### Backend
- **Flask**: Web framework
- **TensorFlow/Keras**: LSTM neural network
- **yfinance**: Stock data fetching
- **scikit-learn**: Data preprocessing
- **NumPy/Pandas**: Data manipulation

### Frontend
- **React**: UI framework
- **Recharts**: Chart visualization
- **Axios**: HTTP client
- **CSS3**: Styling

## 🔮 Future Enhancements

- [ ] Model caching to avoid retraining
- [ ] Support for multiple stocks comparison
- [ ] Future price forecasting
- [ ] Model performance history
- [ ] Export predictions to CSV
- [ ] Real-time stock data updates
- [ ] User authentication
- [ ] Save favorite stocks

## 🐛 Troubleshooting

### Backend Issues
- **Port 5000 in use**: Change port in `app.py` (line 105)
- **Import errors**: Run `pip install -r requirements.txt`
- **Model shape errors**: Ensure enough data points (minimum 31 for 30 time steps)

### Frontend Issues
- **CORS errors**: Backend has CORS enabled, ensure backend is running
- **Connection refused**: Check if backend is running on port 5000
- **npm errors**: Run `npm install` in frontend directory

### Data Issues
- **Invalid ticker**: Use correct Yahoo Finance ticker format
- **Insufficient data**: Ensure date range has enough data points
- **Download errors**: Check internet connection

## ❓ FAQ

**Q: What does the RMSE metric mean?**  
A: RMSE (Root Mean Squared Error) shows the average difference between your model's predictions and actual prices. Lower is better.

**Q: Can I customize the date range for analysis?**  
A: Yes! Use the date selector to set your training or prediction window.

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

