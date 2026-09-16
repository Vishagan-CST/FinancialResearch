import "./App.css";
import { useState } from "react";
import Header from "./components/Header";
import ArticleUpload from "./components/ArticleUpload";
import PredictionIndicator from "./components/PredictionIndicator";
import ShapExplainability from "./components/ShapExplainability";
import Footer from "./components/Footer";

interface PredictionResult {
  prediction: "up" | "down";
  probability: number;
  confidence: number;
  probabilities: {
    up: number;
    down: number;
  };
  is_relevant?: boolean;
  features?: any;
  explanation?: {
    top_topic: string;
    top_keywords: string[];
    shap_values: Record<string, number>;
  };
}

function App() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredictionComplete = (result: PredictionResult) => {
    setPrediction(result);
    setIsLoading(false);
  };

  const handleAnalysisStart = () => {
    setIsLoading(true);
    setPrediction(null);
  };

  return (
    <div className="app">
      <Header />
      <ArticleUpload
        onAnalysisStart={handleAnalysisStart}
        onPredictionComplete={handlePredictionComplete}
      />
      <PredictionIndicator prediction={prediction} isLoading={isLoading} />
      {/* <ShapExplainability /> */}
      <Footer />
    </div>
  );
}

export default App;
