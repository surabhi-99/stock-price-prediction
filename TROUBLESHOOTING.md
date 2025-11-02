# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### ❌ Error: "Cannot connect to backend" or "Connection refused"

**Problem:** The Flask backend server is not running.

**Solution:**
1. Open a terminal/PowerShell window
2. Navigate to the project directory:
   ```bash
   cd "C:\Users\user\Downloads\Stock price prediction"
   ```
3. Start the backend:
   ```bash
   python app.py
   ```
4. You should see: `Running on http://127.0.0.1:5000`
5. Keep this terminal window open while using the app

---

### ❌ Error: "No data found for ticker"

**Problem:** Invalid ticker symbol or no data available for the date range.

**Solutions:**
- **For US stocks:** Use standard symbols like `AAPL`, `GOOGL`, `MSFT`, `TSLA`
- **For Indian stocks (NSE):** Add `.NS` suffix, e.g., `TATAMOTORS.NS`, `RELIANCE.NS`
- **For other exchanges:** Add appropriate suffix (`.L` for London, `.TO` for Toronto, etc.)
- Check Yahoo Finance to verify the ticker exists
- Ensure date range has historical data available

---

### ❌ Error: "Insufficient data" or "Not enough data points"

**Problem:** The date range doesn't have enough historical data.

**Solutions:**
- Use a longer date range (at least 1-2 years)
- Check if the stock has enough trading days
- Try a different stock with more history
- Minimum required: ~100+ data points

---

### ❌ Error: Shape mismatch errors

**Problem:** Data shape incompatibility (should be fixed in current version).

**Solutions:**
- Restart the Flask backend
- Try again with a valid ticker
- If persists, check backend terminal for detailed error

---

### ❌ Model takes too long to train

**Problem:** Training takes 1-3 minutes per request (normal behavior).

**Solutions:**
- Wait for training to complete (progress shown in button)
- For faster results, consider:
  - Using shorter date ranges (but ensure minimum data)
  - Implementing model caching (save/load trained models)
  - Reducing epochs or model complexity

---

### ❌ CORS errors in browser console

**Problem:** Frontend can't connect to backend due to CORS policy.

**Solutions:**
- Ensure backend has `CORS(app)` enabled (already in code)
- Make sure backend is running on `http://localhost:5000`
- Check browser console for specific CORS error

---

### ❌ Frontend doesn't load

**Problem:** React development server not running.

**Solutions:**
1. Open a new terminal window
2. Navigate to frontend directory:
   ```bash
   cd "C:\Users\user\Downloads\Stock price prediction\stock-prediction-frontend"
   ```
3. Start React app:
   ```bash
   npm start
   ```
4. Browser should open at http://localhost:3000

---

### ❌ Python import errors

**Problem:** Missing Python packages.

**Solutions:**
```bash
pip install -r requirements.txt
```

Make sure all packages are installed:
- flask
- flask-cors
- yfinance
- tensorflow
- numpy
- pandas
- scikit-learn

---

### ❌ npm/node errors

**Problem:** Missing Node.js packages or wrong Node version.

**Solutions:**
1. Make sure Node.js is installed: `node --version`
2. Install dependencies:
   ```bash
   cd stock-prediction-frontend
   npm install
   ```

---

## 🔍 How to Check What's Running

### Check if Flask backend is running:
```bash
curl http://localhost:5000/api/health
```
Or open in browser: http://localhost:5000/api/health

Should return: `{"status":"ok"}`

### Check if React frontend is running:
Open in browser: http://localhost:3000

### Check what's using port 5000 (if needed):
```powershell
netstat -ano | findstr :5000
```

---

## 📝 Debugging Steps

1. **Check Backend Logs:**
   - Look at the terminal where Flask is running
   - Errors will be printed there

2. **Check Browser Console:**
   - Press F12 in browser
   - Go to Console tab
   - Look for errors in red

3. **Check Network Tab:**
   - Press F12 in browser
   - Go to Network tab
   - Click "Predict" button
   - Check the `/api/predict` request
   - See response/error details

4. **Test API Directly:**
   ```bash
   python test_api.py
   ```

---

## ✅ Quick Health Check

Run this to verify everything is set up:

```bash
# Terminal 1: Start backend
python app.py

# Terminal 2: Start frontend
cd stock-prediction-frontend
npm start

# Terminal 3: Test API
python test_api.py
```

---

## 🆘 Still Having Issues?

1. **Check error messages** - They now show detailed information
2. **Check backend terminal** - Flask logs all errors there
3. **Check browser console** - Frontend errors appear there
4. **Verify ports** - Make sure 5000 and 3000 are not in use by other apps
5. **Restart both servers** - Sometimes a fresh start fixes issues

---

## 📞 Common Error Messages

| Error | Solution |
|-------|----------|
| Connection refused | Start Flask backend |
| Network Error | Check backend is running |
| No data found | Check ticker symbol |
| Insufficient data | Use longer date range |
| Timeout | Model training takes time (normal) |

