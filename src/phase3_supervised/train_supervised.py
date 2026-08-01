"""
Phase 3, Task 3.3 - supervised classifier.
Scales on the training split only, compares 3 models with 5-fold CV, tunes the
decision threshold for high recall on a validation split, then scores the winner
on a held-out test set. Saves the model, the preprocessor (scaler+medians+
threshold), metrics and plots.
Run: python train_supervised.py
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import SAMPLES
RESULTS = Path(__file__).resolve().parents[2] / "results"
ATTACKS = ["DDoS-HTTP_Flood", "DoS-HTTP_Flood", "DNS_Spoofing", "XSS", "Brute_Force"]

# Phase 3 is the precision stage (the counterpart to Phase 2's recall-first cutoff),
# so we pick the F1-optimal decision threshold - a balanced operating point.

if __name__ == "__main__":
    d = np.load(SAMPLES / "flow_preprocessed.npz", allow_pickle=True)
    X, y, attack_type = d["X"], d["y"].astype(int), d["attack_type"].astype(str)
    names = d["feature_names"].astype(str).tolist()
    medians = pd.Series(d["medians"], index=names)

    # Flows belonging to Phase 2's held-out packets are forced out of training, so
    # Phase 4 never asks this model about a flow it learned from.
    from recheck_phase2_alerts import flow_key
    book = pd.read_csv(SAMPLES / "flow_bookkeeping.csv")
    all_keys = flow_key(book["Src IP"], book["Dst IP"],
                        book["Src Port"], book["Dst Port"])

    idx = np.arange(len(X))
    held = np.zeros(len(X), dtype=bool)
    ep = RESULTS / "heldout_endpoints.csv"
    if ep.exists():
        e = pd.read_csv(ep)
        wanted = set(flow_key(e["src_ip"], e["dst_ip"], e["src_port"], e["dst_port"]))
        held = all_keys.isin(wanted).to_numpy()
        print(f"flows excluded from training (Phase 2 held-out): {held.sum()} "
              f"({held.mean():.1%})")
    else:
        print("heldout_endpoints.csv not found - run phase2 first for a leak-free split")

    free = idx[~held]
    dev_i, te_i = train_test_split(free, test_size=0.2, random_state=42,
                                   stratify=y[free])
    te_i = np.concatenate([te_i, idx[held]])
    Xdev, Xte, ydev, yte, tte = X[dev_i], X[te_i], y[dev_i], y[te_i], attack_type[te_i]
    print(f"flow split: dev={len(dev_i)} test={len(te_i)}")

    RESULTS.mkdir(exist_ok=True)
    pd.Series(sorted(set(all_keys.iloc[dev_i]))).to_frame("key").to_csv(
        RESULTS / "flow_train_keys.csv", index=False)

    scaler = StandardScaler().fit(Xdev)
    Xdev, Xte = scaler.transform(Xdev), scaler.transform(Xte)

    # class_weight="balanced" makes each model take the rare attacks seriously
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced",
                                               random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }

    # 5-fold cross-validation on dev; pick the best mean F1
    print("k-fold model comparison (dev set):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = {}
    for name, m in models.items():
        cv_f1[name] = float(cross_val_score(m, Xdev, ydev, cv=cv, scoring="f1", n_jobs=-1).mean())
        print(f"  {name:22s} CV F1={cv_f1[name]:.3f}")
    best = max(cv_f1, key=cv_f1.get)
    print("best model:", best)

    # tune the decision threshold on a validation slice of dev (for high recall)
    Xtr, Xval, ytr, yval = train_test_split(Xdev, ydev, test_size=0.2,
                                            random_state=42, stratify=ydev)
    tuned = models[best].fit(Xtr, ytr)
    prec, rec, thr = precision_recall_curve(yval, tuned.predict_proba(Xval)[:, 1])
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[f1s[:-1].argmax()])            # F1-optimal cutoff (balanced)

    # refit the winner on all of dev, then score the untouched test set at that threshold
    model = models[best].fit(Xdev, ydev)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

    metrics = {
        "model": best, "cv_f1": cv_f1, "threshold": threshold,
        "AUC": float(roc_auc_score(yte, proba)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred)),
        "f1": float(f1_score(yte, pred)),
        "FPR": float(fp / (fp + tn)), "FNR": float(fn / (fn + tp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "per_attack_detection": {a: {"n": int((tte == a).sum()),
                                     "detection_rate": float(pred[tte == a].mean())}
                                 for a in ATTACKS if (tte == a).sum()},
    }
    print(f"[test] threshold={threshold:.3f} AUC={metrics['AUC']:.3f} "
          f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    joblib.dump(model, SAMPLES / "flow_model.joblib")
    joblib.dump({"scaler": scaler, "medians": medians, "features": names, "threshold": threshold},
                SAMPLES / "flow_preprocessor.joblib")
    (RESULTS / "phase3_metrics.json").write_text(json.dumps(metrics, indent=2))

    plt.figure(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Greens",
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Phase 3 ({best})")
    plt.tight_layout(); plt.savefig(RESULTS / "phase3_confusion_matrix.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(yte, proba)
    plt.figure(figsize=(4, 3.5))
    plt.plot(fpr, tpr, label=f"{best} (AUC={metrics['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Phase 3 ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "phase3_roc.png", dpi=150); plt.close()

    print("saved flow_model.joblib, flow_preprocessor.joblib, phase3_metrics.json, plots")
