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

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        "https://stock-price-prediction-api.onrender.com/api/predict",
        formData,
        {
          timeout: 300000, // 5 minutes timeout for model training
        }
      );
      setPredictionData(response.data);
      // Reset to future view by default when new data arrives
      setActiveView("future");
    } catch (err) {
      let errorMessage = "An error occurred";

      if (
        err.code === "ECONNREFUSED" ||
        err.message.includes("Network Error")
      ) {
        errorMessage =
          "Cannot connect to backend. Please make sure Flask server is running on https://stock-price-prediction-api.onrender.com/api/predict";
      } else if (err.response) {
        // Server responded with error
        errorMessage =
          err.response.data?.error ||
          err.response.data?.message ||
          `Server error: ${err.response.status}`;
        if (err.response.data?.details) {
          errorMessage += `\n\nDetails: ${err.response.data.details}`;
        }
      } else if (err.request) {
        errorMessage =
          "No response from server. Please check if the backend is running.";
      } else {
        errorMessage = err.message || "An unexpected error occurred";
      }

      setError(errorMessage);
      console.error("Prediction error:", err);
      console.error("Error details:", err.response?.data);
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
