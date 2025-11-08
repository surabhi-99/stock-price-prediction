import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { parseISO } from 'date-fns';
import './PredictionChart.css';

function PredictionChart({ actual, predicted, dates, isFuture, title, showDateRange, ticker }) {
  const [dateRange, setDateRange] = useState({ start: null, end: null });
  const [filteredData, setFilteredData] = useState([]);

  // Initialize date range to full range when component mounts or data changes
  useEffect(() => {
    if (dates && dates.length > 0) {
      const startDate = parseISO(dates[0]);
      const endDate = parseISO(dates[dates.length - 1]);
      setDateRange({ start: startDate, end: endDate });
    }
  }, [dates]);

  // Filter data based on date range
  useEffect(() => {
    if (!dates || dates.length === 0) {
      setFilteredData([]);
      return;
    }

    const buildDataPoint = (date, index) => {
      const dataPoint = {
        date: date,
        predicted: predicted && predicted[index] ? parseFloat(predicted[index]).toFixed(2) : 0
      };
      
      if (!isFuture && actual && actual[index] !== undefined && actual[index] !== null) {
        dataPoint.actual = parseFloat(actual[index]).toFixed(2);
      }
      
      return dataPoint;
    };

    if (!dateRange.start || !dateRange.end) {
      // Show all data if no range selected
      const data = dates.map((date, index) => buildDataPoint(date, index));
      setFilteredData(data);
      return;
    }

    const filtered = dates
      .map((date, index) => {
        const dateObj = parseISO(date);
        if (dateObj >= dateRange.start && dateObj <= dateRange.end) {
          return buildDataPoint(date, index);
        }
        return null;
      })
      .filter(item => item !== null);

    setFilteredData(filtered);
  }, [dates, predicted, actual, isFuture, dateRange]);

  // Use filtered data for the chart
  const data = filteredData;

  const chartTitle = title || "📊 Stock Price Prediction Chart";

  return (
    <div className="chart-container">
      <h2>{chartTitle}</h2>
      
      {showDateRange && dates && dates.length > 0 && (
        <div className="date-range-selector">
          <div className="date-range-group">
            <label>Start Date:</label>
            <DatePicker
              selected={dateRange.start}
              onChange={(date) => setDateRange(prev => ({ ...prev, start: date }))}
              selectsStart
              startDate={dateRange.start}
              endDate={dateRange.end}
              maxDate={dateRange.end || parseISO(dates[dates.length - 1])}
              minDate={parseISO(dates[0])}
              dateFormat="MMM d, yyyy"
              className="date-picker-input"
              showYearDropdown
              showMonthDropdown
              dropdownMode="select"
            />
          </div>
          <div className="date-range-group">
            <label>End Date:</label>
            <DatePicker
              selected={dateRange.end}
              onChange={(date) => setDateRange(prev => ({ ...prev, end: date }))}
              selectsEnd
              startDate={dateRange.start}
              endDate={dateRange.end}
              minDate={dateRange.start || parseISO(dates[0])}
              maxDate={parseISO(dates[dates.length - 1])}
              dateFormat="MMM d, yyyy"
              className="date-picker-input"
              showYearDropdown
              showMonthDropdown
              dropdownMode="select"
            />
          </div>
          <button 
            className="reset-date-range"
            onClick={() => {
              if (dates && dates.length > 0) {
                setDateRange({ 
                  start: parseISO(dates[0]), 
                  end: parseISO(dates[dates.length - 1]) 
                });
              }
            }}
          >
            Reset to Full Range
          </button>
        </div>
      )}
      
      <div className="chart-info">
        <p>
          <strong>Chart Overview:</strong> {
            isFuture 
              ? "This visualization shows predicted stock prices for future dates (green line). These are forecasts based on historical patterns learned by the AI model."
              : "This visualization compares the actual stock prices (blue line) with our AI model's predictions (green line). The closer the lines are, the more accurate the predictions."
          }
        </p>
      </div>
      <ResponsiveContainer width="100%" height={450}>
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
          {!isFuture && actual && (
            <Line 
              type="monotone" 
              dataKey="actual" 
              stroke="#8884d8" 
              name="Actual Price"
              strokeWidth={2}
              dot={false}
            />
          )}
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke={isFuture ? "#667eea" : "#82ca9d"}
            name={isFuture ? "Predicted Future Price" : "Predicted Price"}
            strokeWidth={3}
            dot={isFuture}
            strokeDasharray={isFuture ? "5 5" : "0"}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PredictionChart;

