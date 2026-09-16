import React from 'react';
import './ShapExplainability.css';

interface ShapFeature {
    feature: string;
    value: number;
    impact: 'positive' | 'negative';
    description: string;
}

const ShapExplainability: React.FC = () => {
    // Demo SHAP values - replace with actual data from your model
    const shapFeatures: ShapFeature[] = [
        {
            feature: 'Sentiment Score',
            value: 0.78,
            impact: 'positive',
            description: 'Overall positive sentiment in financial news articles'
        },
        {
            feature: 'Market Volatility Index',
            value: -0.62,
            impact: 'negative',
            description: 'High volatility indicates market uncertainty'
        },
        {
            feature: 'Economic Growth Keywords',
            value: 0.54,
            impact: 'positive',
            description: 'Frequency of growth-related terms in news'
        },
        {
            feature: 'Interest Rate Mentions',
            value: -0.48,
            impact: 'negative',
            description: 'Concerns about rising interest rates'
        },
        {
            feature: 'Corporate Earnings Reports',
            value: 0.41,
            impact: 'positive',
            description: 'Positive earnings announcements from major companies'
        },
        {
            feature: 'Geopolitical Risk Score',
            value: -0.35,
            impact: 'negative',
            description: 'International tensions affecting market confidence'
        }
    ];

    return (
        <div className="shap-section animate-fade-in-up">
            <div className="shap-card glass-strong">
                <div className="shap-header">
                    <h2 className="shap-title">Model Explainability (SHAP Values)</h2>
                    <p className="shap-subtitle">
                        Understanding which features most influenced today's prediction
                    </p>
                </div>

                <div className="shap-content">
                    <div className="shap-legend">
                        <div className="legend-item">
                            <span className="legend-indicator positive"></span>
                            <span>Positive Impact (↑)</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-indicator negative"></span>
                            <span>Negative Impact (↓)</span>
                        </div>
                    </div>

                    <div className="features-list">
                        {shapFeatures.map((feature, index) => (
                            <div
                                key={index}
                                className="feature-item transition-smooth"
                                style={{ animationDelay: `${index * 0.1}s` }}
                            >
                                <div className="feature-header">
                                    <div className="feature-name-container">
                                        <span className={`impact-icon ${feature.impact}`}>
                                            {feature.impact === 'positive' ? '↑' : '↓'}
                                        </span>
                                        <h3 className="feature-name">{feature.feature}</h3>
                                    </div>
                                    <span className={`feature-value ${feature.impact}`}>
                                        {feature.value > 0 ? '+' : ''}{feature.value.toFixed(2)}
                                    </span>
                                </div>

                                <div className="feature-bar-container">
                                    <div
                                        className={`feature-bar ${feature.impact}`}
                                        style={{ width: `${Math.abs(feature.value) * 100}%` }}
                                    ></div>
                                </div>

                                <p className="feature-description">{feature.description}</p>
                            </div>
                        ))}
                    </div>

                    <div className="shap-summary">
                        <div className="summary-card glass">
                            <h4>Prediction Confidence</h4>
                            <p className="confidence-value">87.3%</p>
                        </div>
                        <div className="summary-card glass">
                            <h4>Top Influencing Factor</h4>
                            <p className="top-factor">Sentiment Score</p>
                        </div>
                        <div className="summary-card glass">
                            <h4>Model Type</h4>
                            <p className="model-type">XGBoost + SHAP</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ShapExplainability;
