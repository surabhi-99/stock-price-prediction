// Popular stocks organized by market/region
export const popularStocks = [
  // US Tech Stocks
  { symbol: 'AAPL', name: 'Apple Inc.', category: 'Technology' },
  { symbol: 'GOOGL', name: 'Alphabet Inc. (Google)', category: 'Technology' },
  { symbol: 'MSFT', name: 'Microsoft Corporation', category: 'Technology' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', category: 'E-commerce' },
  { symbol: 'META', name: 'Meta Platforms (Facebook)', category: 'Technology' },
  { symbol: 'TSLA', name: 'Tesla Inc.', category: 'Automotive' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation', category: 'Technology' },
  { symbol: 'NFLX', name: 'Netflix Inc.', category: 'Entertainment' },
  
  // US Finance
  { symbol: 'JPM', name: 'JPMorgan Chase & Co.', category: 'Finance' },
  { symbol: 'BAC', name: 'Bank of America', category: 'Finance' },
  { symbol: 'GS', name: 'Goldman Sachs', category: 'Finance' },
  
  // US Consumer
  { symbol: 'WMT', name: 'Walmart Inc.', category: 'Retail' },
  { symbol: 'PG', name: 'Procter & Gamble', category: 'Consumer Goods' },
  { symbol: 'KO', name: 'The Coca-Cola Company', category: 'Beverages' },
  
  // Indian Stocks (NSE)
  { symbol: 'TATAMOTORS.NS', name: 'Tata Motors Ltd', category: 'Automotive (India)' },
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries', category: 'Conglomerate (India)' },
  { symbol: 'TCS.NS', name: 'Tata Consultancy Services', category: 'IT Services (India)' },
  { symbol: 'INFY.NS', name: 'Infosys Limited', category: 'IT Services (India)' },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank', category: 'Banking (India)' },
  { symbol: 'ICICIBANK.NS', name: 'ICICI Bank', category: 'Banking (India)' },
  { symbol: 'SBIN.NS', name: 'State Bank of India', category: 'Banking (India)' },
  { symbol: 'BHARTIARTL.NS', name: 'Bharti Airtel', category: 'Telecom (India)' },
  
  // Other Popular
  { symbol: 'JNJ', name: 'Johnson & Johnson', category: 'Healthcare' },
  { symbol: 'V', name: 'Visa Inc.', category: 'Finance' },
  { symbol: 'MA', name: 'Mastercard Inc.', category: 'Finance' },
  { symbol: 'DIS', name: 'The Walt Disney Company', category: 'Entertainment' },
  { symbol: 'VZ', name: 'Verizon Communications', category: 'Telecom' },
];

// Get all symbols for search
export const getAllSymbols = () => popularStocks.map(stock => stock.symbol);

// Search stocks by symbol or name
export const searchStocks = (query) => {
  if (!query) return popularStocks;
  const lowerQuery = query.toLowerCase();
  return popularStocks.filter(
    stock => 
      stock.symbol.toLowerCase().includes(lowerQuery) ||
      stock.name.toLowerCase().includes(lowerQuery) ||
      stock.category.toLowerCase().includes(lowerQuery)
  );
};




