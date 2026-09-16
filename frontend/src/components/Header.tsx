import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header animate-fade-in-up">
      <div className="header-icon">
        <svg 
          width="60" 
          height="60" 
          viewBox="0 0 60 60" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
          className="arrow-icon"
        >
          <path 
            d="M10 50 L30 10 L50 50 L45 50 L30 20 L15 50 Z" 
            stroke="url(#gradient)" 
            strokeWidth="3" 
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#00ff88" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      
      <h1 className="header-title gradient-text">
        Financial News–Driven Stock Index Movement Prediction
      </h1>
      
      <p className="header-subtitle">
        AI-powered sentiment, topic, and explainable prediction system
      </p>
    </header>
  );
};

export default Header;
