import React from 'react';
import './MetricsDisplay.css';

function MetricsDisplay({ rmse, label }) {
  // Calculate percentage accuracy (rough estimate)
  const getAccuracyLevel = () => {
    if (rmse < 5) return { level: 'Excellent', color: '#10b981' };
    if (rmse < 10) return { level: 'Good', color: '#3b82f6' };
    if (rmse < 20) return { level: 'Fair', color: '#f59e0b' };
    return { level: 'Needs Improvement', color: '#ef4444' };
  };

  const accuracy = getAccuracyLevel();
  const displayLabel = label || 'Model Accuracy';

  return (
    <div className="metrics-container">
      <div className="metric-card">
        <div className="metric-header">
          <div className="metric-icon">📊</div>
          <h3>{displayLabel}</h3>
        </div>
        <div className="metric-content">
          <p className="metric-value">{rmse.toFixed(2)}</p>
          <p className="metric-label">RMSE (Root Mean Squared Error)</p>
          <div className="accuracy-badge" style={{ backgroundColor: `${accuracy.color}15`, color: accuracy.color }}>
            {accuracy.level}
          </div>
          <div className="metric-info">
            <p>
              <strong>What is RMSE?</strong><br />
              Lower values indicate better predictions. This measures how close our predictions are to actual stock prices.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MetricsDisplay;

