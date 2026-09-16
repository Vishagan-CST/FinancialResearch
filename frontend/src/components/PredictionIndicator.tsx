import React from "react";
import "./PredictionIndicator.css";

type PredictionType = "up" | "down" | null;

export interface FeatureImpact {
  feature: string;
  display_name: string;
  category: string;
  source_type: string;
  shap_value: number;
  feature_value: number;
  impact_direction: string;
  impact_effect: string;
}

export interface PredictionResult {
  prediction: PredictionType;
  probability?: number;
  confidence?: number;
  probabilities?: {
    up: number;
    down: number;
  };
  is_relevant?: boolean;
  explanation?: {
    top_topic?: string;
    top_tfidf_group?: string;
    top_keywords?: string[];
    explanation_text?: string;
    dominant_feature?: FeatureImpact;
    top_features?: FeatureImpact[];
    category_summary?: Record<string, number>;
    shap_values?: Record<string, number>;
    base_value?: number;
  };
}

interface PredictionIndicatorProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

const PredictionIndicator: React.FC<PredictionIndicatorProps> = ({
  prediction,
  isLoading,
}) => {
  const predictionType = prediction?.prediction || null;
  const explanation = prediction?.explanation;
  const topKeywords = explanation?.top_keywords || [];
  const shapValues = explanation?.shap_values || {};
  const explanationText = explanation?.explanation_text;
  const dominantFeature = explanation?.dominant_feature;

  return (
    <div className="prediction-section animate-fade-in-up">
      <h2 className="prediction-title">Market Movement Prediction</h2>

      {isLoading && (
        <div className="loading-indicator">
          <div className="loading-spinner"></div>
          <p>Analyzing article with AI models...</p>
        </div>
      )}

      {(!prediction || prediction.is_relevant !== false) && (
        <div className="bulbs-container">
          <div
            className={`bulb-wrapper ${predictionType === "up" ? "active" : "inactive"}`}
          >
            <div
              className={`bulb bulb-green ${predictionType === "up" ? "glow-green" : ""}`}
            >
              <div className="bulb-inner">
                <svg
                  width="80"
                  height="80"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polyline points="18 15 12 9 6 15" />
                </svg>
              </div>
            </div>
            <p className="bulb-label">Index Movement: UP</p>
            <div className="bulb-status">
              {predictionType === "up" && (
                <span className="status-badge status-active">ACTIVE</span>
              )}
            </div>
          </div>

          <div
            className={`bulb-wrapper ${predictionType === "down" ? "active" : "inactive"}`}
          >
            <div
              className={`bulb bulb-red ${predictionType === "down" ? "glow-red" : ""}`}
            >
              <div className="bulb-inner">
                <svg
                  width="80"
                  height="80"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </div>
            </div>
            <p className="bulb-label">Index Movement: DOWN</p>
            <div className="bulb-status">
              {predictionType === "down" && (
                <span className="status-badge status-active">ACTIVE</span>
              )}
            </div>
          </div>
        </div>
      )}

      {!isLoading &&
        prediction &&
        prediction.is_relevant !== false &&
        explanation && (
          <div className="explanation-section animate-fade-in">
            <h3 className="explanation-subtitle">Thematic Impact Analysis</h3>

            {dominantFeature && (
              <div className="dominant-driver-card">
                <span className="dominant-badge">👑 Dominant Driver</span>
                <span className="dominant-title">{dominantFeature.display_name}</span>
                <span className="dominant-category-tag">{dominantFeature.category}</span>
              </div>
            )}

            <div className="shap-chart">
              {Object.entries(shapValues)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .map(([topic, value]) => {
                  const isTopTopic =
                    topic === dominantFeature?.feature ||
                    topic === explanation?.top_topic ||
                    topic.replace(/_/g, " ").toLowerCase() === (explanation?.top_topic || "").toLowerCase();

                  const labelDisplay = topic.startsWith("tf_")
                    ? `TF-IDF: ${topic.slice(3).replace(/_/g, " ").toUpperCase()}`
                    : topic.replace(/_/g, " ").toUpperCase();

                  return (
                    <div
                      key={topic}
                      className={`shap-row ${isTopTopic ? "top-row" : ""}`}
                    >
                      <div className="shap-label">{labelDisplay}</div>
                      <div className="shap-bar-container">
                        <div
                          className={`shap-bar ${isTopTopic ? "importance-top" : "importance"}`}
                          style={{
                            width: `${Math.min(Math.abs(value) * 100, 100)}%`,
                            marginLeft: "0",
                          }}
                        ></div>
                        <div className="shap-value">
                          {value > 0 ? "+" : ""}{value.toFixed(4)}
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>

            <div className="explanation-card">
              <div className="explanation-icon">
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="16" x2="12" y2="12" />
                  <line x1="12" y1="8" x2="12.01" y2="8" />
                </svg>
              </div>
              <p className="explanation-text">
                {explanationText ? (
                  explanationText
                ) : (
                  <>
                    This prediction was influenced by{" "}
                    <span className="highlight-topic">
                      {explanation.top_topic || "Market Themes"}
                    </span>
                    {topKeywords.length > 0 && (
                      <>
                        , specifically because of the features:{" "}
                        {topKeywords.map((kw, i) => (
                          <React.Fragment key={kw}>
                            <span className="highlight-keyword">'{kw}'</span>
                            {i < topKeywords.length - 1 ? ", " : ""}
                          </React.Fragment>
                        ))}
                      </>
                    )}
                  </>
                )}
              </p>
            </div>
          </div>
        )}
    </div>
  );
};

export default PredictionIndicator;
