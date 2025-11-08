import React from 'react';
import './InvestmentSuggestion.css';

function InvestmentSuggestion({ rmse, predicted, actual, isFuture, ticker }) {
  // Calculate investment suggestion
  const getInvestmentAdvice = () => {
    if (!isFuture && actual && predicted && actual.length > 0 && predicted.length > 0) {
      const recentActual = actual.slice(-10); // Last 10 actual values
      const recentPredicted = predicted.slice(-10); // Last 10 predicted values
      
      // Calculate trend (rising/falling)
      const actualTrend = recentActual[recentActual.length - 1] - recentActual[0];
      const predictedTrend = recentPredicted[recentPredicted.length - 1] - recentPredicted[0];
      
      // Calculate accuracy
      const avgAccuracy = 100 - (rmse / (recentActual.reduce((a, b) => a + b, 0) / recentActual.length)) * 100;
      
      // Prediction for next period if future
      let futureOutlook = null;
      if (isFuture && predicted.length > 0) {
        const futureTrend = predicted[predicted.length - 1] - predicted[0];
        futureOutlook = futureTrend > 0 ? 'bullish' : 'bearish';
      }
      
      // Determine suggestion
      if (!isFuture) {
        // For historical validation
        if (avgAccuracy > 85 && actualTrend > 0) {
          return {
            recommendation: 'BUY',
            confidence: 'High',
            color: '#10b981',
            reason: 'Model shows high accuracy with upward trend. Strong buy signal.',
            icon: '📈'
          };
        } else if (avgAccuracy > 85 && actualTrend < 0) {
          return {
            recommendation: 'SELL',
            confidence: 'High',
            color: '#ef4444',
            reason: 'Model shows high accuracy with downward trend. Consider selling.',
            icon: '📉'
          };
        } else if (avgAccuracy > 70) {
          return {
            recommendation: 'HOLD',
            confidence: 'Medium',
            color: '#f59e0b',
            reason: 'Model accuracy is moderate. Monitor closely before making decisions.',
            icon: '⚠️'
          };
        } else {
          return {
            recommendation: 'CAUTION',
            confidence: 'Low',
            color: '#6366f1',
            reason: 'Model accuracy is low. Seek additional analysis before investing.',
            icon: '🔍'
          };
        }
      } else {
        // For future predictions
        if (futureOutlook === 'bullish' && rmse < 20) {
          return {
            recommendation: 'CONSIDER BUYING',
            confidence: 'Medium-High',
            color: '#10b981',
            reason: 'Model predicts upward trend. Consider investment but do your own research.',
            icon: '📈'
          };
        } else if (futureOutlook === 'bearish') {
          return {
            recommendation: 'BE CAUTIOUS',
            confidence: 'Medium',
            color: '#f59e0b',
            reason: 'Model predicts downward trend. Wait for better entry point.',
            icon: '⚠️'
          };
        } else {
          return {
            recommendation: 'MONITOR',
            confidence: 'Medium',
            color: '#6366f1',
            reason: 'Monitor price movements. Consider consulting a financial advisor.',
            icon: '👁️'
          };
        }
      }
    }
    
    // Default for future predictions without historical validation
    if (isFuture && predicted && predicted.length > 0) {
      const futureTrend = predicted[predicted.length - 1] - predicted[0];
      if (futureTrend > 0) {
        return {
          recommendation: 'CONSIDER BUYING',
          confidence: 'Medium',
          color: '#10b981',
          reason: 'Model predicts upward trend. Always do your own research before investing.',
          icon: '📈'
        };
      } else {
        return {
          recommendation: 'BE CAUTIOUS',
          confidence: 'Medium',
          color: '#f59e0b',
          reason: 'Model predicts downward trend. Consider waiting for better entry point.',
          icon: '⚠️'
        };
      }
    }
    
    return null;
  };

  const advice = getInvestmentAdvice();

  if (!advice) return null;

  return (
    <div className="investment-suggestion">
      <div className="suggestion-header">
        <h3>💡 Investment Suggestion</h3>
        <span className="disclaimer">Not Financial Advice</span>
      </div>
      <div className="suggestion-card" style={{ borderColor: advice.color }}>
        <div className="recommendation-badge" style={{ backgroundColor: `${advice.color}15`, color: advice.color }}>
          <span className="badge-icon">{advice.icon}</span>
          <div>
            <div className="badge-text">{advice.recommendation}</div>
            <div className="badge-confidence">Confidence: {advice.confidence}</div>
          </div>
        </div>
        <p className="suggestion-reason">{advice.reason}</p>
        <div className="suggestion-footer">
          <p><strong>⚠️ Important:</strong> This is an AI-generated prediction, not professional financial advice. Always conduct your own research and consult with financial advisors before making investment decisions.</p>
        </div>
      </div>
    </div>
  );
}

export default InvestmentSuggestion;



