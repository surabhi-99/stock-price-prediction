import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './PredictionChart.css';

function PredictionChart({ actual, predicted, dates }) {
  const data = dates.map((date, index) => ({
    date: date,
    actual: actual[index]?.toFixed(2) || 0,
    predicted: predicted[index]?.toFixed(2) || 0
  }));

  return (
    <div className="chart-container">
      <h2>📊 Prediction vs Actual Prices</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            angle={-45}
            textAnchor="end"
            height={100}
            interval="preserveStartEnd"
            tick={{ fontSize: 12 }}
          />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="#8884d8" 
            name="Actual Price"
            strokeWidth={2}
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#82ca9d" 
            name="Predicted Price"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PredictionChart;

