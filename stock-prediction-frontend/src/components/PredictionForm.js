import React, { useState, useRef, useEffect } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { format } from 'date-fns';
import { popularStocks, searchStocks } from '../data/popularStocks';
import './PredictionForm.css';

function PredictionForm({ onSubmit, loading, initialTicker }) {
  const [formData, setFormData] = useState({
    ticker: initialTicker || '',
    train_start_date: (() => {
      const date = new Date();
      date.setFullYear(date.getFullYear() - 10); // 10 years ago
      return date;
    })(),
    train_end_date: new Date(), // Today - end of training data
    predict_start_date: (() => {
      const date = new Date();
      date.setDate(date.getDate() + 1); // Tomorrow
      return date;
    })(),
    predict_end_date: (() => {
      const date = new Date();
      date.setDate(date.getDate() + 30); // 30 days from now
      return date;
    })()
  });

  const [tickerSearch, setTickerSearch] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [filteredStocks, setFilteredStocks] = useState(popularStocks);
  const dropdownRef = useRef(null);
  const inputRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target) && 
          inputRef.current && !inputRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Filter stocks when search changes
  useEffect(() => {
    if (tickerSearch) {
      setFilteredStocks(searchStocks(tickerSearch));
    } else {
      setFilteredStocks(popularStocks);
    }
  }, [tickerSearch]);

  // Update ticker when initialTicker changes
  useEffect(() => {
    if (initialTicker) {
      setFormData(prev => ({ ...prev, ticker: initialTicker }));
      const stock = popularStocks.find(s => s.symbol === initialTicker);
      if (stock) {
        setTickerSearch(stock.name);
      } else {
        setTickerSearch(initialTicker);
      }
    }
  }, [initialTicker]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.ticker) {
      alert('Please select or enter a stock ticker');
      return;
    }
    if (formData.predict_start_date <= formData.train_end_date) {
      alert('Prediction start date must be after training end date');
      return;
    }
    if (formData.predict_end_date <= formData.predict_start_date) {
      alert('Prediction end date must be after prediction start date');
      return;
    }
    onSubmit({
      ticker: formData.ticker,
      train_start_date: format(formData.train_start_date, 'yyyy-MM-dd'),
      train_end_date: format(formData.train_end_date, 'yyyy-MM-dd'),
      predict_start_date: format(formData.predict_start_date, 'yyyy-MM-dd'),
      predict_end_date: format(formData.predict_end_date, 'yyyy-MM-dd')
    });
  };

  const handleTickerChange = (e) => {
    const value = e.target.value;
    setTickerSearch(value);
    setFormData({ ...formData, ticker: value });
    setShowDropdown(true);
  };

  const handleStockSelect = (stock) => {
    setFormData({ ...formData, ticker: stock.symbol });
    setTickerSearch(stock.name);
    setShowDropdown(false);
  };

  const handleDateChange = (date, field) => {
    const updatedFormData = { ...formData, [field]: date };
    
    // Auto-adjust predict_start_date if train_end_date changes
    if (field === 'train_end_date' && date >= updatedFormData.predict_start_date) {
      const tomorrow = new Date(date);
      tomorrow.setDate(tomorrow.getDate() + 1);
      updatedFormData.predict_start_date = tomorrow;
    }
    
    // Auto-adjust predict_end_date if predict_start_date changes
    if (field === 'predict_start_date' && date >= updatedFormData.predict_end_date) {
      const futureDate = new Date(date);
      futureDate.setDate(futureDate.getDate() + 30);
      updatedFormData.predict_end_date = futureDate;
    }
    
    setFormData(updatedFormData);
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="ticker">
          Stock Ticker
          <span className="info-tooltip" title="Search for a company or enter a ticker symbol manually">
            ℹ️
          </span>
        </label>
        <div className="ticker-input-wrapper">
          <input
            ref={inputRef}
            type="text"
            id="ticker"
            name="ticker"
            value={tickerSearch || formData.ticker}
            onChange={handleTickerChange}
            onFocus={() => setShowDropdown(true)}
            placeholder="Search company (e.g., Apple, Google) or enter ticker (AAPL, GOOGL)"
            required
            autoComplete="off"
          />
          {showDropdown && filteredStocks.length > 0 && (
            <div className="ticker-dropdown" ref={dropdownRef}>
              <div className="dropdown-header">Popular Stocks</div>
              {filteredStocks.slice(0, 10).map((stock) => (
                <div
                  key={stock.symbol}
                  className="dropdown-item"
                  onClick={() => handleStockSelect(stock)}
                >
                  <div className="stock-symbol">{stock.symbol}</div>
                  <div className="stock-name">{stock.name}</div>
                  <div className="stock-category">{stock.category}</div>
                </div>
              ))}
              {filteredStocks.length > 10 && (
                <div className="dropdown-footer">
                  Showing 10 of {filteredStocks.length} results
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="date-section">
        <h3 className="section-title">📊 Training Data Range</h3>
        <p className="section-description">Historical data used to train the model</p>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="train_start_date">Training Start Date</label>
            <DatePicker
              id="train_start_date"
              selected={formData.train_start_date}
              onChange={(date) => handleDateChange(date, 'train_start_date')}
              selectsStart
              startDate={formData.train_start_date}
              endDate={formData.train_end_date}
              maxDate={formData.train_end_date}
              dateFormat="MMMM d, yyyy"
              className="date-picker-input"
              showYearDropdown
              showMonthDropdown
              dropdownMode="select"
              placeholderText="Select training start date"
            />
          </div>

          <div className="form-group">
            <label htmlFor="train_end_date">Training End Date</label>
            <DatePicker
              id="train_end_date"
              selected={formData.train_end_date}
              onChange={(date) => handleDateChange(date, 'train_end_date')}
              selectsEnd
              startDate={formData.train_start_date}
              endDate={formData.train_end_date}
              minDate={formData.train_start_date}
              maxDate={new Date()}
              dateFormat="MMMM d, yyyy"
              className="date-picker-input"
              showYearDropdown
              showMonthDropdown
              dropdownMode="select"
              placeholderText="Select training end date"
            />
          </div>
        </div>
      </div>

      <div className="date-section">
        <h3 className="section-title">🔮 Future Prediction Range</h3>
        <p className="section-description">Future dates to predict stock prices</p>
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="predict_start_date">Prediction Start Date</label>
            <DatePicker
              id="predict_start_date"
              selected={formData.predict_start_date}
              onChange={(date) => handleDateChange(date, 'predict_start_date')}
              selectsStart
              startDate={formData.predict_start_date}
              endDate={formData.predict_end_date}
              minDate={new Date(new Date(formData.train_end_date).getTime() + 86400000)} // Day after training end
              maxDate={formData.predict_end_date}
              dateFormat="MMMM d, yyyy"
              className="date-picker-input"
              showYearDropdown
              showMonthDropdown
              dropdownMode="select"
              placeholderText="Select prediction start date"
            />
          </div>

          <div className="form-group">
            <label htmlFor="predict_end_date">Prediction End Date</label>
            <DatePicker
              id="predict_end_date"
              selected={formData.predict_end_date}
              onChange={(date) => handleDateChange(date, 'predict_end_date')}
              selectsEnd
              startDate={formData.predict_start_date}
              endDate={formData.predict_end_date}
              minDate={formData.predict_start_date}
              dateFormat="MMMM d, yyyy"
              className="date-picker-input"
              showYearDropdown
              showMonthDropdown
              dropdownMode="select"
              placeholderText="Select prediction end date"
            />
          </div>
        </div>
      </div>

      <button type="submit" disabled={loading} className="predict-button">
        {loading ? (
          <>
            <span className="spinner"></span>
            Training Model... This may take 1-2 minutes
            {loading && (
              <div style={{ fontSize: '12px', marginTop: '8px', opacity: 0.8 }}>
                (First request may take longer if server is waking up)
              </div>
            )}
          </>
        ) : (
          '🚀 Predict Stock Price'
        )}
      </button>
    </form>
  );
}

export default PredictionForm;
