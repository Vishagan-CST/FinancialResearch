from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference_pipeline import InferencePipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize inference pipeline
pipeline = InferencePipeline()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Backend API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        article_text = data['text']
        date = data.get('date', None)
        
        if not article_text or len(article_text.strip()) < 50:
            return jsonify({"error": "Article text too short (minimum 50 characters)"}), 400
        
        # Run inference pipeline
        result = pipeline.predict(article_text, date)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/pdf', methods=['POST'])
def predict_from_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        # Extract text from PDF
        article_text = pipeline.extract_text_from_pdf(file)
        
        if not article_text or len(article_text.strip()) < 50:
            return jsonify({"error": "Could not extract sufficient text from PDF"}), 400
        
        # Run inference pipeline
        result = pipeline.predict(article_text)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Starting S&P 500 Prediction API Server (New Model)")
    print("=" * 50)
    print(f"Model loaded: {pipeline.model_loaded}")
    print("Available endpoints:")
    print("  - GET  /health")
    print("  - POST /predict")
    print("  - POST /predict/pdf")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
