import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import PredictionForm from "./components/PredictionForm";
import PredictionChart from "./components/PredictionChart";
import MetricsDisplay from "./components/MetricsDisplay";
import InvestmentSuggestion from "./components/InvestmentSuggestion";
import TopStocks from "./components/TopStocks";
import LearningResources from "./components/LearningResources";

function App() {
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [activeView, setActiveView] = useState("future"); // 'future' or 'training'
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState(null);

  // Test backend connection
  const testBackendConnection = async () => {
    setTestingConnection(true);
    setConnectionStatus(null);
    setError(null);

    const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";
    
    try {
      console.log("Testing connection to:", `${API_BASE_URL}/api/health`);
      const response = await axios.get(`${API_BASE_URL}/api/health`, {
        timeout: 20000,
      });
      setConnectionStatus({
        success: true,
        message: `✅ Backend is reachable! Status: ${response.status}`,
        data: response.data,
      });
      console.log("Connection test successful:", response.data);
    } catch (err) {
      const errorMsg = err.response 
        ? `Backend responded with status ${err.response.status}: ${err.response.statusText}`
        : err.code === "ECONNABORTED" || err.message.includes("timeout")
        ? "Request timed out - server might be starting up (wait 30-60 seconds)"
        : err.code === "ERR_NETWORK" || err.message.includes("Network Error")
        ? "Network error - cannot reach backend server"
        : `Error: ${err.message || err.code || "Unknown error"}`;
      
      setConnectionStatus({
        success: false,
        message: `❌ Connection failed: ${errorMsg}`,
        error: err,
      });
      console.error("Connection test failed:", err);
    } finally {
      setTestingConnection(false);
    }
  };

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);

    // Use environment variable if available, otherwise use default
    const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

    console.log("Making request to:", `${API_BASE_URL}/api/predict`);
    console.log("Request data:", formData);

    try {
      // First, check if backend is awake by calling health endpoint
      try {
        console.log("Checking backend health at:", `${API_BASE_URL}/api/health`);
        const healthResponse = await axios.get(`${API_BASE_URL}/api/health`, {
          timeout: 15000, // 15 seconds for health check (increased for cold start)
        });
        console.log("Health check successful:", healthResponse.data);
      } catch (healthErr) {
        console.warn("Health check failed:", {
          code: healthErr.code,
          message: healthErr.message,
          response: healthErr.response?.data,
          status: healthErr.response?.status,
        });
        // Continue anyway - the server might still respond to the prediction request
      }

      console.log("Sending prediction request...");
      const response = await axios.post(
        `${API_BASE_URL}/api/predict`,
        formData,
        {
          timeout: 300000, // 5 minutes timeout for model training
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      
      console.log("Response received:", response.status, response.data);
      
      // Only check for error if status is not 2xx
      if (response.status >= 200 && response.status < 300) {
        // Success - check if response has error field
        if (response.data?.error) {
          setError(response.data.error);
          setLoading(false);
          return; // Don't set predictionData if there's an error
        }
        
        // Validate response has required data
        if (!response.data || (!response.data.predicted && !response.data.training_data)) {
          setError("Invalid response from server: missing prediction data");
          setLoading(false);
          return;
        }
        
        // Success - set the data
        console.log("Setting prediction data:", response.data);
        console.log("Prediction data set:", {
          hasData: !!response.data,
          hasPredicted: !!response.data.predicted,
          hasDates: !!response.data.dates,
          hasTrainingData: !!response.data.training_data,
          isFuture: response.data.is_future,
        });
        setPredictionData(response.data);
        setActiveView("future");
        setError(null); // Clear any previous errors
      } else {
        // Non-2xx status
        throw new Error(`Server returned status ${response.status}: ${response.data?.error || response.statusText}`);
      }
    } catch (err) {
      let errorMessage = "An error occurred";
      const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

      // Check if it's an axios error
      const isAxiosError = err.isAxiosError !== undefined;
      const errorCode = err.code || (isAxiosError ? err.code : null);
      const errMsg = err.message || "";
      
      // Check for specific error types
      if (errorCode === "ECONNREFUSED" || errMsg.includes("ECONNREFUSED")) {
        errorMessage =
          "❌ Cannot connect to backend server.\n\n" +
          "Possible reasons:\n" +
          "• The Render server is starting up (free tier takes ~30-60 seconds)\n" +
          "• The server might be sleeping (free tier auto-sleeps after inactivity)\n\n" +
          "💡 Solution: Wait 30-60 seconds and try again. The first request will wake up the server.";
      } else if (err.code === "ECONNABORTED" || err.message.includes("timeout")) {
        if (err.config?.timeout === 10000) {
          // Health check timeout
          errorMessage =
            "⏱️ Backend server is slow to respond (likely waking up from sleep).\n\n" +
            "The server is starting but taking longer than expected. Please wait a moment and try again.";
        } else {
          errorMessage =
            "⏱️ Request timed out after 5 minutes.\n\n" +
            "The model training is taking longer than expected. This might happen if:\n" +
            "• The Render server is waking up from sleep\n" +
            "• The dataset is very large\n" +
            "• Network connection is slow\n\n" +
            "💡 Solution: Try again with a shorter date range or wait a bit longer.";
        }
      } else if (
        err.message.includes("Network Error") || 
        err.message.includes("ERR_NETWORK") || 
        err.code === "ERR_NETWORK" ||
        err.message.includes("Failed to fetch") ||
        err.message.includes("CORS") ||
        (err.request && !err.response)
      ) {
        // Enhanced network error diagnostics
        const errorDetails = [];
        errorDetails.push(`Error Code: ${err.code || 'Unknown'}`);
        errorDetails.push(`Error Message: ${err.message || 'No message'}`);
        if (err.config?.url) {
          errorDetails.push(`Request URL: ${err.config.url}`);
        }
        if (err.response?.status) {
          errorDetails.push(`Response Status: ${err.response.status}`);
        }
        
        // Check for CORS specifically
        const isCorsError = err.message.includes("CORS") || 
                           (err.response?.status === 0) ||
                           (err.code === "ERR_NETWORK" && !err.response);
        
        errorMessage =
          "🌐 Network error occurred.\n\n" +
          (isCorsError 
            ? "⚠️ This appears to be a CORS (Cross-Origin Resource Sharing) error.\n\n"
            : "Possible reasons:\n") +
          (isCorsError
            ? "The backend server is not allowing requests from this domain.\n\n"
            : "• Backend server is down or unreachable\n" +
              "• CORS policy blocking the request\n" +
              "• Network connectivity issues\n" +
              "• Backend URL might be incorrect\n\n") +
          `🔍 Debug Info:\n${errorDetails.join('\n')}\n\n` +
          "💡 Solutions:\n" +
          "1. Click 'Test Backend Connection' button above to verify connectivity\n" +
          "2. Verify the backend is running at: " + API_BASE_URL + "\n" +
          "3. Check browser console (F12) for detailed error messages\n" +
          "4. Try accessing the health endpoint directly in browser: " + API_BASE_URL + "/api/health\n" +
          (isCorsError
            ? "5. Backend needs to allow CORS from your Vercel domain\n"
            : "5. Wait 30-60 seconds if using Render free tier (cold start)\n") +
          "6. Check your internet connection";
      } else if (err.response) {
        // Server responded with error
        const status = err.response.status;
        if (status === 500) {
          errorMessage =
            "🔴 Server Error (500)\n\n" +
            (err.response.data?.error || "An internal server error occurred.") +
            "\n\n" +
            (err.response.data?.details 
              ? `Details: ${Array.isArray(err.response.data.details) 
                  ? err.response.data.details.join('\n') 
                  : err.response.data.details}`
              : "");
        } else if (status === 400) {
          errorMessage =
            "⚠️ Bad Request (400)\n\n" +
            (err.response.data?.error || err.response.data?.message || "Invalid request parameters.");
        } else {
          errorMessage =
            `Server error (${status}):\n\n` +
            (err.response.data?.error || err.response.data?.message || "An error occurred on the server.");
        }
      } else if (err.request) {
        errorMessage =
          "📡 No response from server.\n\n" +
          "The request was sent but no response was received. This usually means:\n" +
          "• The backend server is sleeping (Render free tier)\n" +
          "• The server is starting up\n" +
          "• Network timeout\n\n" +
          "💡 Solution: Wait 30-60 seconds and try again. The first request will wake up the server.";
      } else {
        errorMessage = err.message || "An unexpected error occurred";
      }

      setError(errorMessage);
      
      // Enhanced error logging
      console.error("=".repeat(50));
      console.error("PREDICTION ERROR DETAILS:");
      console.error("Error Code:", err.code);
      console.error("Error Message:", err.message);
      console.error("Error Name:", err.name);
      console.error("Request URL:", err.config?.url || "Unknown");
      console.error("Request Method:", err.config?.method || "Unknown");
      console.error("Response Status:", err.response?.status);
      console.error("Response Data:", err.response?.data);
      console.error("Response Headers:", err.response?.headers);
      console.error("Full Error Object:", err);
      console.error("=".repeat(50));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Price Prediction</h1>
        <p>Predict stock prices using LSTM Neural Networks</p>
      </header>

      <main className="App-main">
        <div className="connection-test-section" style={{ marginBottom: '20px', textAlign: 'center' }}>
          <button 
            onClick={testBackendConnection} 
            disabled={testingConnection}
            className="test-connection-btn"
            style={{
              padding: '10px 20px',
              backgroundColor: '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: testingConnection ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '600',
              opacity: testingConnection ? 0.6 : 1,
            }}
          >
            {testingConnection ? '⏳ Testing Connection...' : '🔍 Test Backend Connection'}
          </button>
          {connectionStatus && (
            <div 
              className="connection-status"
              style={{
                padding: '15px',
                marginTop: '15px',
                borderRadius: '8px',
                backgroundColor: connectionStatus.success ? '#d1fae5' : '#fee2e2',
                color: connectionStatus.success ? '#065f46' : '#991b1b',
                border: `2px solid ${connectionStatus.success ? '#10b981' : '#ef4444'}`,
                textAlign: 'left',
              }}
            >
              <p style={{ margin: 0, fontWeight: '600' }}>{connectionStatus.message}</p>
              {connectionStatus.data && (
                <pre style={{ marginTop: '10px', fontSize: '12px', overflow: 'auto', backgroundColor: 'rgba(0,0,0,0.05)', padding: '10px', borderRadius: '4px' }}>
                  {JSON.stringify(connectionStatus.data, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
        
        <TopStocks onStockSelect={(stock) => setSelectedTicker(stock.symbol)} />
        <PredictionForm
          onSubmit={handlePredict}
          loading={loading}
          initialTicker={selectedTicker}
        />

        {error && (
          <div className="error-message">
            <p>
              <strong>❌ Error:</strong>
            </p>
            <pre style={{ whiteSpace: "pre-wrap", textAlign: "left" }}>
              {error}
            </pre>
          </div>
        )}

        {predictionData && (
          <>
            <div className="view-switcher">
              <button
                className={`view-button ${
                  activeView === "training" ? "active" : ""
                }`}
                onClick={() => setActiveView("training")}
              >
                📊 Model vs Actual on Training Data
              </button>
              <button
                className={`view-button ${
                  activeView === "future" ? "active" : ""
                }`}
                onClick={() => setActiveView("future")}
              >
                🔮 Future Price Prediction
              </button>
            </div>

            {activeView === "training" && (
              <>
                {predictionData.training_data &&
                predictionData.training_data.dates &&
                predictionData.training_data.dates.length > 0 ? (
                  <>
                    <PredictionChart 
                      actual={predictionData.training_data.actual}
                      predicted={predictionData.training_data.predicted}
                      dates={predictionData.training_data.dates}
                      isFuture={false}
                      title="Model vs Actual on Training Data"
                      showDateRange={true}
                    />
                    {predictionData.training_data.rmse && (
                      <MetricsDisplay
                        rmse={predictionData.training_data.rmse}
                        label="Training RMSE"
                      />
                    )}
                  </>
                ) : (
                  <div className="error-message">
                    <p>
                      <strong>⚠️ Training Data Not Available:</strong>
                    </p>
                    <p>
                      Training data predictions are not available. Please try
                      running the prediction again.
                    </p>
                  </div>
                )}
              </>
            )}

            {activeView === "future" && (
              <>
                <PredictionChart
                  actual={predictionData.actual}
                  predicted={predictionData.predicted}
                  dates={predictionData.dates}
                  isFuture={predictionData.is_future}
                  title="Future Price Prediction"
                  showDateRange={false}
                />
                {!predictionData.is_future && predictionData.rmse && (
                  <MetricsDisplay rmse={predictionData.rmse} />
                )}
                <InvestmentSuggestion
                  rmse={predictionData.rmse}
                  predicted={predictionData.predicted}
                  actual={predictionData.actual}
                  isFuture={predictionData.is_future}
                  ticker={predictionData.ticker || selectedTicker}
                />
                {predictionData.is_future && (
                  <div className="future-note">
                    <p>
                      <strong>📈 Future Prediction:</strong> These are
                      forecasted prices based on the model's learning from
                      historical data. Actual future prices may vary.
                    </p>
                  </div>
                )}
              </>
            )}
          </>
        )}

        <LearningResources />
      </main>
    </div>
  );
}

export default App;
