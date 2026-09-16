# ========================================================
# XGBoost Training with Hyperparameter Comparison
# ========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, log_loss
)

import xgboost as xgb
import joblib


# -----------------------------
# Paths relative to this script so the training works from any working directory.
# -----------------------------
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_PATH, "Dataset", "xgboost_ready_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date").reset_index(drop=True)

y = df["index_change"].astype(int)
X = df.drop(columns=["date", "index_change"])

print("[OK] Loaded dataset:", df.shape)
print("[OK] Features:", X.shape, " Target:", y.shape)
print("[OK] Target distribution:\n", y.value_counts())


# -----------------------------
# Split (time-series safe — chronological, no shuffle)
# -----------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, shuffle=False
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# -----------------------------
# Dynamic scale_pos_weight
# -----------------------------
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

train_baseline = max(y_train.mean(), 1 - y_train.mean())

print(f"\n[OK] Train class counts - Negative: {neg_count}, Positive: {pos_count}")
print(f"[OK] Computed scale_pos_weight: {scale_pos_weight:.4f}")
print(f"[OK] Train majority-class baseline accuracy: {train_baseline:.4f}")


# -----------------------------
# Train XGBoost Model
# -----------------------------
best_params = dict(
    learning_rate=0.005, max_depth=2, min_child_weight=10,
    gamma=0.5, reg_alpha=0.5, reg_lambda=5.0,
    subsample=0.5, colsample_bytree=0.5
)

print("\n" + "=" * 60)
print("TRAINING XGBOOST MODEL")
print("=" * 60)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=2000,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric=["logloss", "error"],
    early_stopping_rounds=50,
    **best_params
)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

best_config_name = "Final Model"


# -----------------------------
# Full evaluation on the selected best model
# -----------------------------
evals = model.evals_result()

train_logloss = evals["validation_0"]["logloss"]
val_logloss   = evals["validation_1"]["logloss"]
train_error   = evals["validation_0"]["error"]
val_error     = evals["validation_1"]["error"]

train_acc_curve = [1 - e for e in train_error]
val_acc_curve   = [1 - e for e in val_error]

best_epoch = int(np.argmin(val_logloss)) + 1
print(f"\n[OK] Best epoch: {best_epoch}")
print(f"[OK] Best val logloss: {min(val_logloss):.4f}")
print(f"[OK] Val accuracy at best epoch: {val_acc_curve[best_epoch-1]:.4f}")

train_proba = model.predict_proba(X_train)[:, 1]
val_proba   = model.predict_proba(X_val)[:, 1]
test_proba  = model.predict_proba(X_test)[:, 1]

train_pred = (train_proba >= 0.5).astype(int)
val_pred   = (val_proba >= 0.5).astype(int)
test_pred  = (test_proba >= 0.5).astype(int)


def print_metrics(name, y_true, pred, proba):
    acc = accuracy_score(y_true, pred)
    ll = log_loss(y_true, proba)
    print(f"\n{name} Accuracy: {acc:.4f} | Loss: {ll:.4f}")
    print(classification_report(y_true, pred, digits=4))
    return acc, ll

print("\n" + "=" * 60)
print(f"FINAL MODEL EVALUATION — Config: {best_config_name}")
print("=" * 60)

train_acc_final, train_ll_final = print_metrics("TRAIN", y_train, train_pred, train_proba)
val_acc_final, val_ll_final     = print_metrics("VAL", y_val, val_pred, val_proba)
test_acc_final, test_ll_final   = print_metrics("TEST", y_test, test_pred, test_proba)

majority_class = y_train.mode()[0]
baseline_test_acc = accuracy_score(y_test, np.full(len(y_test), majority_class))
print(f"\n[OK] Baseline accuracy (always predict majority class '{majority_class}'): {baseline_test_acc:.4f}")
print(f"[OK] Test accuracy vs baseline: {test_acc_final:.4f} vs {baseline_test_acc:.4f} "
      f"({'beats baseline' if test_acc_final > baseline_test_acc else 'DOES NOT beat baseline - investigate'})")


# -----------------------------
# Graphs: Accuracy
# -----------------------------
epochs = list(range(1, len(train_logloss) + 1))

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_logloss, label="Train loss")
plt.plot(epochs, val_logloss, label="Val loss")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
# plt.show()

# Accuracy Curve
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc_curve, label="Train accuracy")
plt.plot(epochs, val_acc_curve, label="Val accuracy")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
# plt.show()


# -----------------------------
# Confusion Matrices (Val + Test)
# -----------------------------
def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1],
        yticklabels=[0, 1]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    # plt.show()

plot_cm(y_val,  val_pred,  "Validation Confusion Matrix", "cm_val.png")
plot_cm(y_test, test_pred, "Test Confusion Matrix",       "cm_test.png")



# -----------------------------
# Save model + summary
# -----------------------------
MODEL_FILE = os.path.join(OUTPUT_DIR, "xgboost_sp500_model.pkl")
joblib.dump(model, MODEL_FILE)
joblib.dump(list(X.columns), os.path.join(OUTPUT_DIR, "feature_columns.pkl"))

summary = pd.DataFrame([{
    "best_config": best_config_name,
    "best_epoch": best_epoch,
    "train_accuracy": train_acc_final,
    "val_accuracy": val_acc_final,
    "test_accuracy": test_acc_final,
    "baseline_test_accuracy": baseline_test_acc,
    "train_logloss": train_ll_final,
    "val_logloss": val_ll_final,
    "test_logloss": test_ll_final,
    **best_params
}])

summary.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"), index=False)

print(f"\n[OK] Saved model: {MODEL_FILE}")
print(f"[OK] Saved results in: {OUTPUT_DIR}")
