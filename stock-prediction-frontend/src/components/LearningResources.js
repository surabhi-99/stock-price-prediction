import React, { useState } from 'react';
import './LearningResources.css';

function LearningResources() {
  const [activeTab, setActiveTab] = useState('resources');

  const resources = [
    {
      title: 'Beginner\'s Guide to Stock Investing',
      type: 'Article',
      link: 'https://www.investopedia.com/articles/basics/06/invest1000.asp',
      description: 'Learn the fundamentals of stock market investing'
    },
    {
      title: 'Understanding Stock Market Basics',
      type: 'Video',
      link: 'https://www.youtube.com/results?search_query=stock+market+basics',
      description: 'Visual guides to understand how stock markets work'
    },
    {
      title: 'Financial Modeling & Analysis',
      type: 'Course',
      link: 'https://www.coursera.org/browse/business/finance',
      description: 'Learn financial analysis and valuation techniques'
    },
    {
      title: 'Yahoo Finance',
      type: 'Platform',
      link: 'https://finance.yahoo.com/',
      description: 'Real-time stock data, news, and analysis'
    },
    {
      title: 'Investopedia Academy',
      type: 'Education',
      link: 'https://www.investopedia.com/academy/',
      description: 'Comprehensive investment education resources'
    }
  ];

  const precautions = [
    {
      icon: '💰',
      title: 'Never Invest More Than You Can Afford to Lose',
      description: 'Only invest money you can afford to lose without affecting your essential expenses.'
    },
    {
      icon: '📊',
      title: 'Diversify Your Portfolio',
      description: 'Don\'t put all your money in one stock. Spread investments across different sectors and assets.'
    },
    {
      icon: '⏰',
      title: 'Invest for the Long Term',
      description: 'Stock markets are volatile in the short term. Long-term investing reduces risk.'
    },
    {
      icon: '🔍',
      title: 'Do Your Own Research',
      description: 'Never rely solely on predictions. Research company fundamentals, financials, and market conditions.'
    },
    {
      icon: '📰',
      title: 'Stay Informed',
      description: 'Keep up with market news, company announcements, and economic indicators.'
    },
    {
      icon: '👨‍💼',
      title: 'Consult Financial Advisors',
      description: 'For significant investments, seek advice from certified financial advisors.'
    },
    {
      icon: '🎯',
      title: 'Set Clear Goals',
      description: 'Define your investment objectives, timeline, and risk tolerance before investing.'
    },
    {
      icon: '🚨',
      title: 'Beware of Market Manipulation',
      description: 'Be cautious of "get rich quick" schemes and unverified investment advice.'
    }
  ];

  return (
    <div className="learning-resources-section">
      <h2>📚 Investment Education & Resources</h2>
      
      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'resources' ? 'active' : ''}`}
          onClick={() => setActiveTab('resources')}
        >
          Learning Resources
        </button>
        <button 
          className={`tab ${activeTab === 'precautions' ? 'active' : ''}`}
          onClick={() => setActiveTab('precautions')}
        >
          Important Precautions
        </button>
      </div>

      {activeTab === 'resources' && (
        <div className="resources-content">
          <p className="content-intro">
            Before investing, it's crucial to understand how stock markets work. Here are valuable resources to get started:
          </p>
          <div className="resources-list">
            {resources.map((resource, index) => (
              <div key={index} className="resource-item">
                <div className="resource-header">
                  <span className="resource-type">{resource.type}</span>
                  <h4>{resource.title}</h4>
                </div>
                <p className="resource-description">{resource.description}</p>
                <a 
                  href={resource.link} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="resource-link"
                >
                  Visit Resource →
                </a>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'precautions' && (
        <div className="precautions-content">
          <p className="content-intro">
            Important precautions to take before investing in stocks:
          </p>
          <div className="precautions-grid">
            {precautions.map((precaution, index) => (
              <div key={index} className="precaution-card">
                <div className="precaution-icon">{precaution.icon}</div>
                <h4>{precaution.title}</h4>
                <p>{precaution.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default LearningResources;



