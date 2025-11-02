# Stock Price Prediction App - Setup Guide

## ✅ Setup Complete!

All files have been created successfully. Here's how to run the application:

## 📁 Project Structure

```
Stock price prediction/
├── app.py                          # Flask backend API
├── requirements.txt                # Python dependencies
├── Stock_Price_Prediction.ipynb   # Original notebook
└── stock-prediction-frontend/      # React frontend
    ├── src/
    │   ├── App.js                 # Main app component
    │   ├── App.css                # Main styles
    │   ├── index.js               # React entry point
    │   └── components/
    │       ├── PredictionForm.js  # Input form
    │       ├── PredictionChart.js # Chart visualization
    │       └── MetricsDisplay.js   # RMSE display
    ├── package.json
    └── README.md
```

## 🚀 How to Run

### Step 1: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### Step 2: Start the Flask Backend

```powershell
python app.py
```

The backend will run on **http://localhost:5000**

### Step 3: Start the React Frontend

Open a **new terminal window** and run:

```powershell
cd stock-prediction-frontend
npm start
```

The frontend will automatically open in your browser at **http://localhost:3000**

## 🎯 Usage

1. **Enter Stock Ticker**: Enter a stock symbol (e.g., `TATAMOTORS.NS`, `AAPL`, `GOOGL`)
2. **Set Date Range**: Choose start and end dates for historical data
3. **Click "Predict Stock Price"**: The model will train and make predictions
4. **View Results**: 
   - See RMSE (Root Mean Squared Error) metric
   - View interactive chart comparing actual vs predicted prices

## 🔧 Troubleshooting

### Backend Issues
- Make sure port 5000 is not in use
- Check that all Python packages are installed
- Ensure you have internet connection (for downloading stock data)

### Frontend Issues
- Make sure Node.js and npm are installed
- Run `npm install` again if you see dependency errors
- Check that the backend is running before making predictions

### CORS Errors
- The Flask backend has CORS enabled, but if you see errors, make sure the backend is running on port 5000

## 📝 Notes

- The model trains each time you make a prediction (this may take 1-2 minutes)
- For faster predictions, you could modify the backend to save/load trained models
- The app uses Yahoo Finance API (via yfinance) to fetch stock data

## 🎨 Features

- ✨ Beautiful, responsive UI
- 📊 Interactive charts using Recharts
- 🔄 Real-time model training
- 📈 Stock price predictions using LSTM neural networks
- 📉 RMSE metrics for model evaluation

