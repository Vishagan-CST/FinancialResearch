import React, { useState } from "react";
import "./ArticleUpload.css";

const API_BASE_URL = "http://localhost:5001";

interface ArticleUploadProps {
  onAnalysisStart: () => void;
  onPredictionComplete: (result: any) => void;
}

const ArticleUpload: React.FC<ArticleUploadProps> = ({
  onAnalysisStart,
  onPredictionComplete,
}) => {
  const [uploadMode, setUploadMode] = useState<"pdf" | "text">("pdf");
  const [articleText, setArticleText] = useState("");
  const [fileName, setFileName] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      setSelectedFile(file);
      setError("");
    }
  };

  const handleAnalyze = async () => {
    setError("");

    // Validation
    if (
      uploadMode === "text" &&
      (!articleText || articleText.trim().length < 50)
    ) {
      setError("Please enter at least 50 characters of article text");
      return;
    }

    if (uploadMode === "pdf" && !selectedFile) {
      setError("Please select a PDF file to upload");
      return;
    }

    setIsAnalyzing(true);
    onAnalysisStart();

    try {
      let response;

      if (uploadMode === "text") {
        // Send text to backend
        response = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: articleText,
          }),
        });
      } else {
        // Send PDF file to backend
        const formData = new FormData();
        formData.append("file", selectedFile as File);

        response = await fetch(`${API_BASE_URL}/predict/pdf`, {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Analysis failed");
      }

      const result = await response.json();

      if (result.is_relevant === false) {
        setError("This content is not related to business or financial news. Please try again with relevant articles.");
        onPredictionComplete({ ...result, prediction: null }); // Clear previous prediction
        return;
      }

      onPredictionComplete(result);
    } catch (err: any) {
      setError(
        err.message ||
        "Failed to connect to backend. Make sure the API server is running.",
      );
      console.error("Analysis error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="upload-section animate-fade-in-up">
      <div className="upload-card glass-strong">
        <h2 className="upload-title">Validate Financial News (PDF or Text)</h2>

        <div className="upload-mode-toggle">
          <button
            className={`mode-btn transition-smooth ${uploadMode === "pdf" ? "active" : ""}`}
            onClick={() => setUploadMode("pdf")}
            disabled={isAnalyzing}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            Upload PDF
          </button>
          <button
            className={`mode-btn transition-smooth ${uploadMode === "text" ? "active" : ""}`}
            onClick={() => setUploadMode("text")}
            disabled={isAnalyzing}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            Paste Text
          </button>
        </div>

        {uploadMode === "pdf" ? (
          <div className="upload-area">
            <input
              type="file"
              id="file-upload"
              accept=".pdf"
              onChange={handleFileUpload}
              className="file-input"
              disabled={isAnalyzing}
            />
            <label
              htmlFor="file-upload"
              className="file-label glass transition-smooth hover-lift"
            >
              <svg
                width="48"
                height="48"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <span className="upload-text">
                {fileName || "Click to upload PDF or drag and drop"}
              </span>
              <span className="upload-hint">PDF files only</span>
            </label>
          </div>
        ) : (
          <div className="text-area-container">
            <textarea
              className="article-textarea glass transition-smooth"
              placeholder="Paste your financial news article here..."
              value={articleText}
              onChange={(e) => setArticleText(e.target.value)}
              rows={8}
              disabled={isAnalyzing}
            />
          </div>
        )}

        {error && (
          <div className="error-message">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {error}
          </div>
        )}

        <button
          className="analyze-btn transition-smooth hover-scale"
          onClick={handleAnalyze}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <svg
                className="spinner"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="12" cy="12" r="10" />
              </svg>
              Analyzing...
            </>
          ) : (
            <>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
              Analyze Article
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default ArticleUpload;
