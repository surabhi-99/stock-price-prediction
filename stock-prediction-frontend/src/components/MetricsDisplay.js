import React from 'react';
import './MetricsDisplay.css';

function MetricsDisplay({ rmse }) {
  return (
    <div className="metrics-container">
      <div className="metric-card">
        <div className="metric-icon">📉</div>
        <h3>RMSE</h3>
        <p className="metric-value">{rmse.toFixed(2)}</p>
        <p className="metric-label">Root Mean Squared Error</p>
      </div>
    </div>
  );
}

export default MetricsDisplay;

