# ========================================================
# Statistical Validation for the S&P 500 Direction Model
# ========================================================
# The single chronological train/val/test split in XGboostModel_Training.py
# cannot support a claim of "the model beats baseline" on its own: one split
# is one draw from history, with no significance test and no evidence the
# result holds across different time windows. This script adds the three
# things a top-tier reviewer asks for first:
#
#   1. Expanding-window walk-forward validation (N_FOLDS folds spanning
#      different chronological periods, model retrained from scratch each
#      fold on only the data available up to that point).
#   2. McNemar's test: model vs. majority-class baseline, on hard
#      predictions (paired test on the same instances -- appropriate here
#      because both classifiers are evaluated on identical test points).
#   3. Diebold-Mariano test: model vs. a constant "climatological"
#      probability baseline, on per-sample log-loss (appropriate for
#      probabilistic forecast comparison; HLN small-sample correction
#      applied since T is a few hundred, not thousands).
#   4. Circular block-bootstrap 95% CI on pooled walk-forward accuracy
#      (block bootstrap, not iid bootstrap, because daily market data is
#      autocorrelated).
#
# Hyperparameters are reused as-is from the tuned final model in
# XGboostModel_Training.py -- re-running hyperparameter search inside every
# fold is out of scope here; this script isolates the *validation scheme*
# question from the *hyperparameter search* question, and that reuse is
# stated explicitly so it can be cited as a disclosed limitation.

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import xgboost as xgb


# -----------------------------
# Paths
# -----------------------------
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_PATH, "Dataset", "xgboost_ready_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reused from the tuned "Final Model" / "ultra conservative" config in
# XGboostModel_Training.py (see results/model_summary.csv).
BEST_PARAMS = dict(
    learning_rate=0.005, max_depth=2, min_child_weight=10,
    gamma=0.5, reg_alpha=0.5, reg_lambda=5.0,
    subsample=0.5, colsample_bytree=0.5,
)

N_FOLDS = 5
INITIAL_TRAIN_FRAC = 0.5          # first fold trains on the oldest 50% of history
VAL_FRAC_WITHIN_TRAIN = 0.15      # tail slice of each fold's train window, held out for early stopping
N_BOOT = 10000
BLOCK_SIZE = 20                   # ~1 trading month, for the circular block bootstrap
RANDOM_SEED = 42


# -----------------------------
# Data loading (identical preprocessing to XGboostModel_Training.py)
# -----------------------------
def load_data():
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    y = df["index_change"].astype(int)
    X = df.drop(columns=["date", "index_change"])
    return X, y, df["date"]


# -----------------------------
# Expanding-window fold boundaries
# -----------------------------
def make_folds(n, n_folds=N_FOLDS, initial_train_frac=INITIAL_TRAIN_FRAC):
    initial_train_end = int(n * initial_train_frac)
    remaining = n - initial_train_end
    fold_size = remaining // n_folds
    folds = []
    train_end = initial_train_end
    for i in range(n_folds):
        test_start = train_end
        test_end = test_start + fold_size if i < n_folds - 1 else n
        folds.append((train_end, test_start, test_end))
        train_end = test_end
    return folds


# -----------------------------
# Train + evaluate a single walk-forward fold
# -----------------------------
def train_fold(X, y, train_end, test_start, test_end):
    X_train_full = X.iloc[:train_end]
    y_train_full = y.iloc[:train_end]
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    val_size = max(1, int(len(X_train_full) * VAL_FRAC_WITHIN_TRAIN))
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric=["logloss", "error"],
        early_stopping_rounds=50,
        **BEST_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    # Baselines are fit ONLY on this fold's training history, so they see
    # no information the model itself did not also have.
    majority_class = int(y_train_full.mode()[0])
    baseline_pred = np.full(len(y_test), majority_class)
    baseline_proba = np.full(len(y_test), float(y_train_full.mean()))

    return {
        "y_test": y_test.to_numpy(),
        "model_pred": test_pred,
        "model_proba": test_proba,
        "baseline_pred": baseline_pred,
        "baseline_proba": baseline_proba,
    }


# -----------------------------
# Circular block bootstrap CI (Politis & Romano, 1994)
# -----------------------------
def block_bootstrap_ci(correct, block_size=BLOCK_SIZE, n_boot=N_BOOT, alpha=0.05, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(correct)
    n_blocks = int(np.ceil(n / block_size))
    boot_accs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
        idx = idx.ravel()[:n]
        boot_accs[b] = correct[idx].mean()
    lo, hi = np.percentile(boot_accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# -----------------------------
# McNemar's test (model vs. baseline, paired on hard predictions)
# -----------------------------
def mcnemar_test(y_true, pred_a, pred_b):
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int(np.sum(a_correct & ~b_correct))   # a right, b wrong
    n10 = int(np.sum(~a_correct & b_correct))   # a wrong, b right
    n = n01 + n10

    if n == 0:
        return {"n01": n01, "n10": n10, "statistic": None, "p_value": 1.0,
                "method": "degenerate (no discordant pairs)"}
    if n < 25:
        k = min(n01, n10)
        p = stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
        return {"n01": n01, "n10": n10, "statistic": None, "p_value": float(p),
                "method": "exact binomial (n<25)"}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    p = float(1 - stats.chi2.cdf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p,
            "method": "chi-square, continuity-corrected"}


# -----------------------------
# Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction
# -----------------------------
def diebold_mariano(loss_a, loss_b, h=1):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    d_bar = float(d.mean())

    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if T > lag else 0.0
        var_d += 2 * (1 - lag / h) * cov
    var_d_bar = var_d / T

    if var_d_bar <= 0:
        return {"dm_statistic": 0.0, "p_value": 1.0, "mean_loss_diff": d_bar, "n": T, "lag_h": h}

    dm_stat = d_bar / np.sqrt(var_d_bar)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_corrected = dm_stat * hln_factor
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_stat_corrected), df=T - 1)))
    return {"dm_statistic": float(dm_stat_corrected), "p_value": p_value,
            "mean_loss_diff": d_bar, "n": T, "lag_h": h}


def per_sample_log_loss(y_true, proba):
    eps = 1e-15
    p = np.clip(proba, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


# -----------------------------
# Run walk-forward validation
# -----------------------------
def main():
    X, y, dates = load_data()
    folds = make_folds(len(X))

    print("=" * 70)
    print(f"WALK-FORWARD VALIDATION -- {N_FOLDS} expanding-window folds")
    print("=" * 70)

    fold_rows = []
    raw_results = []
    for i, (train_end, test_start, test_end) in enumerate(folds, start=1):
        r = train_fold(X, y, train_end, test_start, test_end)
        acc = float((r["model_pred"] == r["y_test"]).mean())
        baseline_acc = float((r["baseline_pred"] == r["y_test"]).mean())

        fold_rows.append({
            "fold": i,
            "train_n": train_end,
            "test_n": len(r["y_test"]),
            "test_start_date": str(dates.iloc[test_start].date()),
            "test_end_date": str(dates.iloc[test_end - 1].date()),
            "model_accuracy": acc,
            "baseline_accuracy": baseline_acc,
            "edge_vs_baseline": acc - baseline_acc,
        })
        raw_results.append(r)

        print(f"Fold {i}: train_n={train_end:4d}  test_n={len(r['y_test']):3d}  "
              f"[{fold_rows[-1]['test_start_date']} -> {fold_rows[-1]['test_end_date']}]  "
              f"model_acc={acc:.4f}  baseline_acc={baseline_acc:.4f}  edge={acc - baseline_acc:+.4f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_results.csv"), index=False)

    # ---- Pool all held-out folds (chronological order preserved) ----
    pooled_y = np.concatenate([r["y_test"] for r in raw_results])
    pooled_model_pred = np.concatenate([r["model_pred"] for r in raw_results])
    pooled_model_proba = np.concatenate([r["model_proba"] for r in raw_results])
    pooled_baseline_pred = np.concatenate([r["baseline_pred"] for r in raw_results])
    pooled_baseline_proba = np.concatenate([r["baseline_proba"] for r in raw_results])

    pooled_acc = float((pooled_model_pred == pooled_y).mean())
    pooled_baseline_acc = float((pooled_baseline_pred == pooled_y).mean())

    model_correct = (pooled_model_pred == pooled_y).astype(float)
    ci_lo, ci_hi = block_bootstrap_ci(model_correct)

    baseline_correct = (pooled_baseline_pred == pooled_y).astype(float)
    edge = model_correct - baseline_correct  # paired per-sample edge, for a CI on the improvement itself
    edge_lo, edge_hi = block_bootstrap_ci(edge, block_size=BLOCK_SIZE)  # note: mean of `edge` samples equals accuracy difference

    mcnemar_result = mcnemar_test(pooled_y, pooled_model_pred, pooled_baseline_pred)

    model_ll = per_sample_log_loss(pooled_y, pooled_model_proba)
    baseline_ll = per_sample_log_loss(pooled_y, pooled_baseline_proba)
    dm_h = max(1, int(round(len(model_ll) ** (1 / 3))))
    dm_result = diebold_mariano(model_ll, baseline_ll, h=dm_h)

    report = {
        "n_folds": N_FOLDS,
        "pooled_n_test": int(len(pooled_y)),
        "pooled_model_accuracy": pooled_acc,
        "pooled_baseline_accuracy": pooled_baseline_acc,
        "pooled_edge": pooled_acc - pooled_baseline_acc,
        "fold_accuracy_mean": float(fold_df["model_accuracy"].mean()),
        "fold_accuracy_std": float(fold_df["model_accuracy"].std(ddof=1)),
        "block_bootstrap_ci_95_accuracy": {"lower": ci_lo, "upper": ci_hi},
        "block_bootstrap_ci_95_edge_over_baseline": {"lower": edge_lo, "upper": edge_hi},
        "mcnemar_test": mcnemar_result,
        "diebold_mariano_test": dm_result,
    }

    print("\n" + "=" * 70)
    print("POOLED RESULTS ACROSS ALL WALK-FORWARD FOLDS")
    print("=" * 70)
    print(f"Pooled test n:                 {report['pooled_n_test']}")
    print(f"Pooled model accuracy:         {pooled_acc:.4f}")
    print(f"Pooled baseline accuracy:      {pooled_baseline_acc:.4f}")
    print(f"Pooled edge over baseline:     {report['pooled_edge']:+.4f}")
    print(f"Per-fold accuracy: mean={report['fold_accuracy_mean']:.4f}  std={report['fold_accuracy_std']:.4f}")
    print(f"95% block-bootstrap CI on accuracy:            [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"95% block-bootstrap CI on edge over baseline:  [{edge_lo:+.4f}, {edge_hi:+.4f}]")
    print(f"\nMcNemar's test (model vs. majority-class baseline):")
    print(f"  discordant pairs: model-only-correct={mcnemar_result['n01']}, "
          f"baseline-only-correct={mcnemar_result['n10']}")
    print(f"  method={mcnemar_result['method']}  p-value={mcnemar_result['p_value']:.4f}")
    print(f"\nDiebold-Mariano test (model log-loss vs. constant-probability baseline log-loss):")
    print(f"  lag h={dm_result['lag_h']}  DM statistic={dm_result['dm_statistic']:.4f}  "
          f"p-value={dm_result['p_value']:.4f}  mean loss diff={dm_result['mean_loss_diff']:+.5f}")

    verdict_mcnemar = "SIGNIFICANT (p<0.05)" if mcnemar_result["p_value"] < 0.05 else "NOT significant (p>=0.05)"
    verdict_dm = "SIGNIFICANT (p<0.05)" if dm_result["p_value"] < 0.05 else "NOT significant (p>=0.05)"
    print(f"\n[VERDICT] McNemar's test: {verdict_mcnemar}")
    print(f"[VERDICT] Diebold-Mariano test: {verdict_dm}")

    report["mcnemar_verdict"] = verdict_mcnemar
    report["diebold_mariano_verdict"] = verdict_dm

    with open(os.path.join(OUTPUT_DIR, "statistical_validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- Plot: per-fold accuracy vs baseline, with pooled CI ----
    plt.figure(figsize=(10, 5))
    x = fold_df["fold"]
    width = 0.35
    plt.bar(x - width / 2, fold_df["model_accuracy"], width, label="Model")
    plt.bar(x + width / 2, fold_df["baseline_accuracy"], width, label="Majority-class baseline")
    plt.axhline(pooled_acc, color="tab:blue", linestyle="--", alpha=0.6,
                label=f"Pooled model accuracy ({pooled_acc:.3f})")
    plt.fill_between([0.4, N_FOLDS + 0.6], ci_lo, ci_hi, color="tab:blue", alpha=0.12,
                      label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    plt.xticks(x)
    plt.xlabel("Walk-forward fold (chronological)")
    plt.ylabel("Accuracy")
    plt.title("Walk-Forward Validation: Model vs. Baseline Accuracy by Fold")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.savefig(os.path.join(OUTPUT_DIR, "walk_forward_accuracy.png"), dpi=300, bbox_inches="tight")

    print(f"\n[OK] Saved: results/walk_forward_results.csv")
    print(f"[OK] Saved: results/statistical_validation_report.json")
    print(f"[OK] Saved: results/walk_forward_accuracy.png")


if __name__ == "__main__":
    main()
