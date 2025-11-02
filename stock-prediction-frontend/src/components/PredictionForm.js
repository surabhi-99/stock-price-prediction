import React, { useState } from 'react';
import './PredictionForm.css';

function PredictionForm({ onSubmit, loading }) {
  const [formData, setFormData] = useState({
    ticker: 'TATAMOTORS.NS',
    start_date: '2010-01-01',
    end_date: '2023-05-31'
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="ticker">Stock Ticker:</label>
        <input
          type="text"
          id="ticker"
          name="ticker"
          value={formData.ticker}
          onChange={handleChange}
          placeholder="e.g., TATAMOTORS.NS, AAPL, GOOGL"
          required
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="start_date">Start Date:</label>
          <input
            type="date"
            id="start_date"
            name="start_date"
            value={formData.start_date}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="end_date">End Date:</label>
          <input
            type="date"
            id="end_date"
            name="end_date"
            value={formData.end_date}
            onChange={handleChange}
            required
          />
        </div>
      </div>

      <button type="submit" disabled={loading} className="predict-button">
        {loading ? '⏳ Training Model...' : '🚀 Predict Stock Price'}
      </button>
    </form>
  );
}

export default PredictionForm;

