"""
Phase 3, Task 3.1 (optional) + novelty experiment.
1) Run an unsupervised detector (Isolation Forest) on the FLOW features and
   compare its AUC to the packet-level detector from Phase 2 (flow vs packet).
2) Novelty: add that anomaly score as an extra feature to the supervised model
   and compare F1 with vs without it.
No labels are used to train the unsupervised model.
Run: python flow_anomaly.py   (run preprocess_flow.py first)
"""
import sys, json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import SAMPLES
RESULTS = Path(__file__).resolve().parents[2] / "results"


def evaluate_with(features, y):
    """Train RandomForest on the given features and return test scores."""
    Xtr, Xte, ytr, yte = train_test_split(features, y, test_size=0.2,
                                          random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pred = m.predict(Xte)
    proba = m.predict_proba(Xte)[:, 1]
    return {"f1": round(f1_score(yte, pred), 4),
            "precision": round(precision_score(yte, pred, zero_division=0), 4),
            "recall": round(recall_score(yte, pred), 4),
            "AUC": round(roc_auc_score(yte, proba), 4)}


if __name__ == "__main__":
    d = np.load(SAMPLES / "flow_preprocessed.npz", allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)

    # --- Task 3.1: unsupervised on flows (no labels in training) ---
    iso = IsolationForest(n_estimators=200, contamination=0.02,
                          random_state=42, n_jobs=-1).fit(X)
    anomaly = -iso.score_samples(X)                     # higher = more anomalous
    flow_auc = float(roc_auc_score(y, anomaly))

    # fair comparison: SAME method (Isolation Forest) on packets, read from Phase 2's metrics
    packet_if = packet_ae = None
    p2 = RESULTS / "phase2_metrics.json"
    if p2.exists():
        m2 = json.loads(p2.read_text())
        packet_if = m2.get("IsolationForest", {}).get("AUC")
        packet_ae = m2.get("AE_threshold", {}).get("AUC")
    print(f"Task 3.1  flow IsolationForest AUC = {flow_auc:.3f}")
    if packet_if:
        print(f"          packet IsolationForest AUC (same method) = {packet_if:.3f}")
    if packet_ae:
        print(f"          packet autoencoder AUC (best packet detector) = {packet_ae:.3f}")

    # --- Novelty: supervised with vs without the anomaly score as a feature ---
    without = evaluate_with(X, y)
    withscore = evaluate_with(np.column_stack([X, anomaly]), y)
    print("supervised without anomaly feature:", without)
    print("supervised WITH    anomaly feature:", withscore)

    result = {"flow_IsolationForest_AUC": round(flow_auc, 4),
              "packet_IsolationForest_AUC": packet_if,   # fair, same-method comparison
              "packet_autoencoder_AUC": packet_ae,       # best packet detector overall
              "supervised_without_anomaly": without,
              "supervised_with_anomaly": withscore}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase3_anomaly.json").write_text(json.dumps(result, indent=2))
    print("saved results/phase3_anomaly.json")
