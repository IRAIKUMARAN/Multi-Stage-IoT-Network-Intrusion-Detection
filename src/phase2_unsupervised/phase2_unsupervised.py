"""
PHASE 2 - Unsupervised anomaly detection on PACKET-level data

Task 2.1: train unsupervised models that separate normal vs anomalous packets
          (autoencoder, k-means, isolation forest). NO labels in training.
Task 2.2: pick alert thresholds, analyze false positives vs negatives.
Metrics : precision/recall/F1, per-attack detection rate, FPR/FNR, AUC-ROC,
          confusion matrix.
"""

import numpy as np, pandas as pd
from pathlib import Path

# ---- TUNABLE PARAMETERS ----
# Phase 2 is stage 1 of a cascade. Stage 2 (Phase 3) can only REMOVE false
# positives, never recover a missed attack, so we bias Phase 2 toward RECALL.
# Lower this (e.g. 0.80) if Phase 3 gets overwhelmed by too many false alarms.
RECALL_TARGET = 0.95

# SECTION 1 - LOAD DATA
# Load the preprocessed feature matrix (already scaled). Labels are read too, but ONLY for evaluation / threshold tuning - the models never train on them.

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import SAMPLES as DATA
d          = np.load(DATA / "packet_preprocessed.npz", allow_pickle=True)
X          = d["X"]
feat_names = d["feature_names"].astype(str)

book   = pd.read_csv(DATA / "packet_bookkeeping.csv")
y_type = book["attack_type"].to_numpy()
y      = (y_type != "Benign").astype(int)          

# drop the leaky columns the verifier flagged. Match port_class_dst exactly, and any
# http_content_type=application/octet-stream* one-hot (covers the ;charset=UTF-8 suffix),
# so the drop actually happens instead of silently matching nothing.
mask = ~np.array([f == "port_class_dst"
                  or f.startswith("http_content_type=application/octet-stream")
                  for f in feat_names])
X, feat_names = X[:, mask], feat_names[mask]
print("dropped leaky cols:", int((~mask).sum()))
print("X:", X.shape, "| attack rate:", round(y.mean(), 4))

# SECTION 2 - TRAIN / VAL / TEST SPLIT
from sklearn.model_selection import train_test_split
idx = np.arange(len(X))
itr, ite = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
itr, iva = train_test_split(itr, test_size=0.20, random_state=42, stratify=y[itr])

Xtr, Xva, Xte = X[itr], X[iva], X[ite]
ytr, yva, yte = y[itr], y[iva], y[ite]
tte = y_type[ite]
print("train/val/test:", len(itr), len(iva), len(ite))

# SECTION 3 - AUTOENCODER (Task 2.1)
# Learns to reconstruct NORMAL traffic; attacks reconstruct poorly, so the per-row reconstruction error becomes the anomaly score.
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score

def train_ae(Xtrain, hidden, max_iter=80):
    """Train an autoencoder (input==target) with a given architecture. No labels."""
    ae = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                      max_iter=max_iter, early_stopping=True, random_state=42)
    ae.fit(Xtrain, Xtrain)
    return ae

def recon_error(model, A):
    """Per-row reconstruction error (MSE). Higher = more abnormal."""
    return np.mean((A - model.predict(A)) ** 2, axis=1)

# SECTION 4 - AE HYPERPARAMETER SWEEP (Task 2.1: "tune systematically")
# Try several architectures; keep the one with the best VALIDATION AUC. Using val AUC for model selection is standard - the AE itself never sees y.
configs = [(64, 16, 64), (128, 32, 128), (64, 8, 64), (32,)]
sweep = []
for h in configs:
    ae_h  = train_ae(Xtr, h, max_iter=40)        
    auc_h = roc_auc_score(yva, recon_error(ae_h, Xva))
    sweep.append((auc_h, h))
    print(f"  AE {str(h):16s} val AUC={auc_h:.3f}")

best_auc, best_h = max(sweep)
print(f"best AE architecture: {best_h}  (val AUC={best_auc:.3f})")

ae = train_ae(Xtr, best_h, max_iter=80)          
err_tr, err_va, err_te = recon_error(ae, Xtr), recon_error(ae, Xva), recon_error(ae, Xte)

# SECTION 5 - ISOLATION FOREST (second unsupervised method, for comparison)
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=200, contamination=0.02,
                      random_state=42, n_jobs=-1).fit(Xtr)
iso_score_te = -iso.score_samples(Xte)             # higher = more anomalous
iso_pred_te  = (iso.predict(Xte) == -1).astype(int)

# SECTION 6 - AE + K-MEANS (the instructor-suggested combo, label-free cutoff)
# Cluster the (log-scaled) error into 2 groups; the higher-error group = alerts.
from sklearn.cluster import KMeans
km   = KMeans(n_clusters=2, n_init=10, random_state=42).fit(np.log1p(err_tr).reshape(-1, 1))
anom = int(np.argmax([err_tr[km.labels_ == c].mean() for c in (0, 1)]))
km_pred_te = (km.predict(np.log1p(err_te).reshape(-1, 1)) == anom).astype(int)

# SECTION 7 - THRESHOLD SELECTION ON VALIDATION (Task 2.2)
# Pick the AE cutoff on VAL (not test). Two defensible operating points:
#   thr_f1     - balanced (max F1)
#   thr_recall - recall-first, the right choice for a 2-stage IDS (Phase 3 cleans the extra false positives later).

from sklearn.metrics import precision_recall_curve
prec, rec, thrs = precision_recall_curve(yva, err_va)
f1s  = 2 * prec * rec / (prec + rec + 1e-12)
best = f1s[:-1].argmax()
thr_f1 = thrs[best]
ok = rec[:-1] >= RECALL_TARGET
thr_recall = thrs[ok].max() if ok.any() else thr_f1
# Two-stage design: Phase 2 should favour RECALL (catch nearly everything), because
# Phase 3 can only REMOVE false positives later — it can never recover a missed attack.
# So we use the recall-first cutoff, not max-F1.
THR         = thr_recall
ae_pred_te  = (err_te > THR).astype(int)
print(f"\nchosen AE threshold={THR:.4g}  (recall-first; F1-optimal was {f1s[best]:.3f})")

# SECTION 7b - PHASE 2 SWEET-SPOT CURVE: recall vs false-positive-rate across cutoffs.
# The "sweet spot" is where recall is high but the false-alarm rate is still tolerable
# for Phase 3. Computed on validation so it does not peek at the test set.
import json as _json
from sklearn.metrics import confusion_matrix as _cm
_curve = []
for q in np.linspace(0.50, 0.999, 25):
    t = float(np.quantile(err_va, q))
    p = (err_va > t).astype(int)
    tn, fp, fn, tp = _cm(yva, p).ravel()
    _curve.append({"threshold": round(t, 5),
                   "recall": round(tp / (tp + fn), 3),
                   "FPR": round(fp / (fp + tn), 3),
                   "precision": round(tp / (tp + fp + 1e-9), 3)})
_P2 = Path(__file__).resolve().parents[2] / "results"
_P2.mkdir(exist_ok=True)
(_P2 / "phase2_operating_curve.json").write_text(_json.dumps(_curve, indent=2))
print("saved results/phase2_operating_curve.json (recall vs false-alarm trade-off)")

# SECTION 8 - EVALUATE ALL METHODS ON TEST
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix)
def evaluate(name, y_true, pred, cont_score):
    """Print the full metric line for one method AND return it as a dict."""
    auc = roc_auc_score(y_true, cont_score)
    p   = precision_score(y_true, pred, zero_division=0)
    r   = recall_score(y_true, pred)
    f1  = f1_score(y_true, pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    fpr, fnr = fp / (fp + tn), fn / (fn + tp)
    print(f"[{name:>14}] AUC={auc:.3f} P={p:.3f} R={r:.3f} "
          f"F1={f1:.3f} FPR={fpr:.3f} FNR={fnr:.3f}")
    return {"AUC": float(auc), "precision": float(p), "recall": float(r), "f1": float(f1),
            "FPR": float(fpr), "FNR": float(fnr),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

print()
metrics = {}
metrics["AE_threshold"]    = evaluate("AE-threshold", yte, ae_pred_te,  err_te)
metrics["AE_kmeans"]       = evaluate("AE-kmeans",    yte, km_pred_te,  err_te)
metrics["IsolationForest"] = evaluate("IsolationFor", yte, iso_pred_te, iso_score_te)

pred = ae_pred_te

# SECTION 9 - PER-ATTACK DETECTION RATE (primary method)

print("\nPer-attack detection rate (AE-threshold):")
per_attack = {}
for a in ["DDoS-HTTP_Flood", "DoS-HTTP_Flood", "DNS_Spoofing", "XSS", "Brute_Force"]:
    m = (tte == a)
    if m.sum():
        rate = float(pred[m].mean())
        per_attack[a] = {"n": int(m.sum()), "detection_rate": rate}
        print(f"  {a:16s} n={m.sum():4d}  caught={rate:.3f}")
metrics["per_attack_detection"] = per_attack
metrics["threshold"] = float(THR)

# SECTION 10 - PLOTS (confusion matrix + ROC) -> results/
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve

RESULTS = Path(__file__).resolve().parents[2] / "results"
RESULTS.mkdir(exist_ok=True)

cm = confusion_matrix(yte, pred)
plt.figure(figsize=(4, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Phase 2 Confusion Matrix")
plt.tight_layout(); plt.savefig(RESULTS / "phase2_confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(4, 3.5))
for nm, sc in [("Autoencoder", err_te), ("IsolationForest", iso_score_te)]:
    fpr_c, tpr_c, _ = roc_curve(yte, sc)
    plt.plot(fpr_c, tpr_c, label=f"{nm} (AUC={roc_auc_score(yte, sc):.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Phase 2 ROC"); plt.legend(); plt.tight_layout()
plt.savefig(RESULTS / "phase2_roc.png", dpi=150); plt.close()
print("\nplots saved to", RESULTS)

# SECTION 11 - SAVE FLAGGED ALERTS (handoff to Phase 3) + METRICS
flagged = book.iloc[ite][pred.astype(bool)]
flagged.to_csv(RESULTS / "flagged_packet_ids.csv", index=False)
print("flagged alerts saved:", len(flagged))

import json
(RESULTS / "phase2_metrics.json").write_text(json.dumps(metrics, indent=2))
print("metrics saved to", RESULTS / "phase2_metrics.json")
