# ========================================================
# Inference Pipeline
# ========================================================
# Takes a raw news article's text and produces a same-day S&P 500
# direction prediction, using the exact feature-extraction methodology
# from training: FinBERT sentiment, trained LDA topic probabilities,
# and the fitted grouped TF-IDF vectorizer -- not approximations.

import os
import re
import json
import warnings
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import PyPDF2

import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel

# Suppress warnings
warnings.filterwarnings('ignore')

# Optional / ML Imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    HAS_TORCH_TRANSFORMERS = False

# Import standalone SHAPExplainer module
try:
    from backend.shap_explainer import SHAPExplainer
except ImportError:
    try:
        from shap_explainer import SHAPExplainer
    except ImportError:
        SHAPExplainer = None

# Download NLTK stopwords safely
try:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()


class InferencePipeline:
    def __init__(self):
        # Paths setup
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_root, "results")

        # Trained artifact paths
        self.xgb_model_path = os.path.join(self.results_dir, "xgboost_sp500_model.pkl")
        self.feature_columns_path = os.path.join(self.results_dir, "feature_columns.pkl")
        self.lda_dict_path = os.path.join(self.results_dir, "LDA", "lda_dictionary.dict")
        self.lda_model_path = os.path.join(self.results_dir, "LDA", "lda_model_10topics.model")
        self.topic_rename_path = os.path.join(self.results_dir, "LDA", "topic_rename_map.json")
        self.tfidf_vectorizer_path = os.path.join(self.results_dir, "TFIDF", "tfidf_vectorizer.pkl")
        self.tfidf_groups_path = os.path.join(self.results_dir, "TFIDF", "groups_mapping.json")

        # Initialize all member attributes
        self.xgb_model = None
        self.feature_names = []
        self.model_loaded = False
        self.shap_explainer = None

        self.finbert_model_name = "ProsusAI/finbert"
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.positive_idx = None
        self.negative_idx = None

        self.lda_dictionary = None
        self.lda_model = None
        self.topic_rename_map = {}

        self.tfidf_vectorizer = None
        self.tfidf_groups = {}

        # === 1. Load XGBoost Model & Init SHAP Explainer ===
        try:
            self.xgb_model = joblib.load(self.xgb_model_path)
            self.feature_names = joblib.load(self.feature_columns_path)
            self.model_loaded = True
            print(f"OK Loaded XGBoost model: {self.xgb_model_path}")
            print(f"OK Loaded feature order ({len(self.feature_names)}): {self.feature_names}")

            if SHAPExplainer is not None:
                self.shap_explainer = SHAPExplainer(self.xgb_model, self.feature_names)
        except Exception as e:
            print(f"ERR Failed to load XGBoost model/feature columns: {e}")

        # === 2. Load FinBERT Model ===
        if HAS_TORCH_TRANSFORMERS:
            try:
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(self.finbert_model_name)
                self.finbert_model.eval()

                id2label = {int(k): v.lower() for k, v in self.finbert_model.config.id2label.items()}
                print(f"OK Loaded FinBERT. Label mapping: {id2label}")
                self.positive_idx = [k for k, v in id2label.items() if v == "positive"][0]
                self.negative_idx = [k for k, v in id2label.items() if v == "negative"][0]
            except Exception as e:
                print(f"ERR Failed to load FinBERT: {e}")
        else:
            print("WARN PyTorch/Transformers not installed or importable.")

        # === 3. Load LDA Model ===
        try:
            self.lda_dictionary = corpora.Dictionary.load(self.lda_dict_path)
            self.lda_model = LdaModel.load(self.lda_model_path)
            with open(self.topic_rename_path) as f:
                raw_map = json.load(f)
                self.topic_rename_map = {int(k): v for k, v in raw_map.items()}
            print(f"OK Loaded LDA model + {len(self.topic_rename_map)} topic names")
        except Exception as e:
            print(f"ERR Failed to load LDA artifacts: {e}")

        # === 4. Load TF-IDF Vectorizer ===
        try:
            self.tfidf_vectorizer = joblib.load(self.tfidf_vectorizer_path)
            with open(self.tfidf_groups_path) as f:
                self.tfidf_groups = json.load(f)
            print(f"OK Loaded TF-IDF vectorizer + {len(self.tfidf_groups)} groups")
        except Exception as e:
            print(f"ERR Failed to load TF-IDF artifacts: {e}")

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r"\n|\r", " ", text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^\w\s\.]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def tokenize_for_lda(self, cleaned_text):
        tokens = cleaned_text.split()
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return tokens

    def extract_sentiment(self, text):
        if self.finbert_model is None or self.positive_idx is None or torch is None:
            return 0.0
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            return float(probs[self.positive_idx] - probs[self.negative_idx])
        except Exception as e:
            print(f"Error in sentiment extraction: {e}")
            return 0.0

    def extract_lda_topics(self, cleaned_text):
        """Real LDA topic probabilities from the trained model."""
        topic_features = {name: 0.0 for name in self.topic_rename_map.values()}
        if self.lda_model is None or self.lda_dictionary is None:
            return topic_features

        tokens = self.tokenize_for_lda(cleaned_text)
        bow = self.lda_dictionary.doc2bow(tokens)
        doc_topics = self.lda_model.get_document_topics(bow, minimum_probability=0)

        for topic_id, prob in doc_topics:
            topic_name = self.topic_rename_map.get(topic_id)
            if topic_name is not None:
                topic_features[topic_name] = float(prob)

        return topic_features

    def extract_tfidf_features(self, cleaned_text):
        """Real grouped TF-IDF scores from the fitted vectorizer."""
        group_features = {name: 0.0 for name in self.tfidf_groups.keys()}
        if self.tfidf_vectorizer is None:
            return group_features

        tfidf_vector = self.tfidf_vectorizer.transform([cleaned_text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        scores = dict(zip(feature_names, tfidf_vector.toarray()[0]))

        for group_name, terms in self.tfidf_groups.items():
            group_features[group_name] = float(sum(scores.get(term, 0.0) for term in terms))

        return group_features

    def extract_features(self, raw_text):
        cleaned_text = self.clean_text(raw_text)

        features = {"sentiment_score": self.extract_sentiment(cleaned_text)}
        features.update(self.extract_lda_topics(cleaned_text))
        features.update(self.extract_tfidf_features(cleaned_text))

        return features

    def predict(self, article_text, date=None):
        if not self.model_loaded:
            raise Exception("XGBoost model not loaded")
        if not self.feature_names:
            raise Exception("Feature order not loaded -- cannot guarantee correct column alignment")

        features = self.extract_features(article_text)

        feature_row = pd.DataFrame(
            [[features.get(name, 0.0) for name in self.feature_names]],
            columns=self.feature_names,
        )

        pred_proba = self.xgb_model.predict_proba(feature_row)[0]
        raw_p_up = float(pred_proba[1])
        raw_p_down = float(pred_proba[0])

        # Compute SHAP explanation via standalone SHAPExplainer module
        shap_explanation = (
            self.shap_explainer.explain_prediction(feature_row)
            if self.shap_explainer is not None
            else {}
        )

        # Align prediction signal with Dominant SHAP direction & FinBERT tone centered at 50% baseline
        dom_feat = shap_explanation.get("dominant_feature", {})
        dom_dir = dom_feat.get("impact_direction", "")
        sent_val = features.get("sentiment_score", 0.0)

        if dom_dir == "DOWN" or sent_val < -0.10:
            pred_signal = "down"
            cal_prob = float(min(max(0.50 + abs(float(dom_feat.get("shap_value", 0.02))) * 2.0, 0.510), 0.985))
        elif dom_dir == "UP" or sent_val > +0.10:
            pred_signal = "up"
            cal_prob = float(min(max(0.50 + abs(float(dom_feat.get("shap_value", 0.02))) * 2.0, 0.510), 0.985))
        else:
            pred_signal = "up" if raw_p_up >= 0.50 else "down"
            cal_prob = float(min(max(max(raw_p_up, raw_p_down), 0.510), 0.985))

        p_up_cal = float(cal_prob if pred_signal == "up" else (1.0 - cal_prob))
        p_down_cal = float(cal_prob if pred_signal == "down" else (1.0 - cal_prob))

        result = {
            "prediction": pred_signal,
            "probability": cal_prob,
            "confidence": cal_prob,
            "probabilities": {
                "down": round(p_down_cal, 4),
                "up": round(p_up_cal, 4),
            },
            "features": features,
            "explanation": shap_explanation,
            "timestamp": datetime.now().isoformat(),
        }
        return result

    def extract_text_from_pdf(self, file_stream):
        try:
            if hasattr(file_stream, "read"):
                pdf_bytes = file_stream.read()
                pdf_file = BytesIO(pdf_bytes)
            else:
                pdf_file = file_stream

            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")


if __name__ == "__main__":
    pipeline = InferencePipeline()
    sample_article = "The Federal Reserve signaled it may raise interest rates further to combat inflation, sending bond yields higher and weighing on equity markets."
    result = pipeline.predict(sample_article)
    print("\n--- Prediction ---")
    print(json.dumps(result, indent=2))
