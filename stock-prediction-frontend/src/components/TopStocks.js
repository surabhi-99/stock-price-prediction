import React, { useState } from 'react';
import './TopStocks.css';
import { popularStocks } from '../data/popularStocks';

function TopStocks({ onStockSelect }) {
  const [selectedCategory, setSelectedCategory] = useState('All');
  
  const categories = ['All', 'Technology', 'Finance', 'Automotive (India)', 'IT Services (India)', 'Banking (India)'];
  
  const filteredStocks = selectedCategory === 'All' 
    ? popularStocks 
    : popularStocks.filter(stock => stock.category === selectedCategory);

  const topStocksByCategory = {
    'Technology': ['AAPL', 'GOOGL', 'MSFT', 'NVDA'],
    'Finance': ['JPM', 'BAC', 'V', 'MA'],
    'Automotive (India)': ['TATAMOTORS.NS'],
    'IT Services (India)': ['TCS.NS', 'INFY.NS'],
    'Banking (India)': ['HDFCBANK.NS', 'ICICIBANK.NS']
  };

  return (
    <div className="top-stocks-section">
      <h2>⭐ Popular Stocks to Watch</h2>
      <p className="section-subtitle">Click on any stock to quickly select it for prediction</p>
      
      <div className="category-filters">
        {categories.map(category => (
          <button
            key={category}
            className={`filter-btn ${selectedCategory === category ? 'active' : ''}`}
            onClick={() => setSelectedCategory(category)}
          >
            {category}
          </button>
        ))}
      </div>

      <div className="stocks-grid">
        {filteredStocks.map((stock) => (
          <div
            key={stock.symbol}
            className="stock-card"
            onClick={() => onStockSelect(stock)}
          >
            <div className="stock-symbol-large">{stock.symbol}</div>
            <div className="stock-name-large">{stock.name}</div>
            <div className="stock-category-small">{stock.category}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default TopStocks;



