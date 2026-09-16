# ========================================================
# SHAP Category Multiplier Calibration
# ========================================================
# Derives the LDA/TF-IDF multipliers used in shap_explainer.py from data,
# instead of guessing them. sentiment_score is one undivided feature, while
# LDA topics and TF-IDF groups each split their true impact across ~10
# correlated sub-features, so their raw per-feature SHAP values run much
# smaller on average -- not because they matter less, but because SHAP
# divides credit among correlated features.
#
# Method: run shap.TreeExplainer over the full dataset with the trained
# model, take the mean |SHAP| per feature, then set each type's multiplier
# so its average magnitude matches sentiment_score's average magnitude.
#
# Re-run this whenever the model is retrained and copy the printed
# LDA_MULTIPLIER / TFIDF_MULTIPLIER values into shap_explainer.py.

import os
import numpy as np
import pandas as pd
import joblib
import shap

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_PATH, "results")
DATASET_FILE = os.path.join(PROJECT_PATH, "Dataset", "xgboost_ready_dataset.csv")

model = joblib.load(os.path.join(RESULTS_DIR, "xgboost_sp500_model.pkl"))
feature_cols = joblib.load(os.path.join(RESULTS_DIR, "feature_columns.pkl"))

df = pd.read_csv(DATASET_FILE)
X = df[feature_cols]

explainer = shap.TreeExplainer(model)
raw = explainer.shap_values(X)
if isinstance(raw, list):
    raw = raw[1] if len(raw) > 1 else raw[0]
if hasattr(raw, "ndim") and raw.ndim == 3:
    raw = raw[:, :, -1]

mean_abs = np.abs(raw).mean(axis=0)
per_feat = dict(zip(feature_cols, mean_abs))

sentiment_mean = per_feat["sentiment_score"]
lda_feats = [f for f in feature_cols if not f.startswith("tf_") and f != "sentiment_score"]
tfidf_feats = [f for f in feature_cols if f.startswith("tf_")]

lda_mean = float(np.mean([per_feat[f] for f in lda_feats]))
tfidf_mean = float(np.mean([per_feat[f] for f in tfidf_feats]))

lda_multiplier = sentiment_mean / lda_mean
tfidf_multiplier = sentiment_mean / tfidf_mean

print(f"sentiment mean|SHAP|          = {sentiment_mean:.6f}")
print(f"LDA per-topic mean|SHAP|      = {lda_mean:.6f}  -> LDA_MULTIPLIER   = {lda_multiplier:.3f}")
print(f"TF-IDF per-group mean|SHAP|   = {tfidf_mean:.6f}  -> TFIDF_MULTIPLIER = {tfidf_multiplier:.3f}")

print("\nPer-feature mean |SHAP| (for reference):")
for f in feature_cols:
    print(f"  {f:<35} {per_feat[f]:.6f}")
