import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import PredictionForm from './components/PredictionForm';
import PredictionChart from './components/PredictionChart';
import MetricsDisplay from './components/MetricsDisplay';

function App() {
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:5000/api/predict', formData, {
        timeout: 300000, // 5 minutes timeout for model training
      });
      setPredictionData(response.data);
    } catch (err) {
      let errorMessage = 'An error occurred';
      
      if (err.code === 'ECONNREFUSED' || err.message.includes('Network Error')) {
        errorMessage = 'Cannot connect to backend. Please make sure Flask server is running on http://localhost:5000';
      } else if (err.response) {
        // Server responded with error
        errorMessage = err.response.data?.error || err.response.data?.message || `Server error: ${err.response.status}`;
        if (err.response.data?.details) {
          errorMessage += `\n\nDetails: ${err.response.data.details}`;
        }
      } else if (err.request) {
        errorMessage = 'No response from server. Please check if the backend is running.';
      } else {
        errorMessage = err.message || 'An unexpected error occurred';
      }
      
      setError(errorMessage);
      console.error('Prediction error:', err);
      console.error('Error details:', err.response?.data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>📈 Stock Price Prediction</h1>
        <p>Predict stock prices using LSTM Neural Networks</p>
      </header>

      <main className="App-main">
        <PredictionForm onSubmit={handlePredict} loading={loading} />
        
        {error && (
          <div className="error-message">
            <p><strong>❌ Error:</strong></p>
            <pre style={{ whiteSpace: 'pre-wrap', textAlign: 'left' }}>{error}</pre>
          </div>
        )}

        {predictionData && (
          <>
            <MetricsDisplay rmse={predictionData.rmse} />
            <PredictionChart 
              actual={predictionData.actual}
              predicted={predictionData.predicted}
              dates={predictionData.dates}
            />
          </>
        )}
      </main>
    </div>
  );
}

export default App;

