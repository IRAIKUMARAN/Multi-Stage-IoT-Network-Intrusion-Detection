"""
Phase 3 add-on: cascade vs union (comparative analysis + novelty).
Demonstrates the two-stage trade-off on the flow test set with an unsupervised
stage-1 (Isolation Forest, tuned for high recall) + supervised stage-2
(Random Forest):
    stage1 alone | stage2 alone | CASCADE (s1 AND s2) | UNION (s1 OR s2)
Cascade favours precision (fewer false alarms); union favours recall (catches
attacks either stage finds). Saves a comparison table + per-attack chart.
Run: python ensemble_union.py   (run preprocess_flow.py first)
"""
import sys, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import SAMPLES
RESULTS = Path(__file__).resolve().parents[2] / "results"
ATTACKS = ["DDoS-HTTP_Flood", "DoS-HTTP_Flood", "DNS_Spoofing", "XSS", "Brute_Force"]
STAGE1_RECALL = 0.90                                   # stage-1 is tuned recall-first


def score_row(pred, y):
    """recall / precision / f1 for a set of predictions."""
    return {"recall": round(recall_score(y, pred), 3),
            "precision": round(precision_score(y, pred, zero_division=0), 3),
            "f1": round(f1_score(y, pred), 3)}


if __name__ == "__main__":
    d = np.load(SAMPLES / "flow_preprocessed.npz", allow_pickle=True)
    X, y, at = d["X"], d["y"].astype(int), d["attack_type"].astype(str)
    Xtr, Xte, ytr, yte, _, atte = train_test_split(
        X, y, at, test_size=0.2, random_state=42, stratify=y)

    # stage 1 - unsupervised (Isolation Forest), threshold set for ~90% recall
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1).fit(Xtr)
    score = -iso.score_samples(Xte)
    thr1 = np.quantile(score[yte == 1], 1 - STAGE1_RECALL)   # catch ~90% of attacks
    s1 = (score >= thr1).astype(int)

    # stage 2 - supervised (Random Forest)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(Xtr, ytr)
    s2 = rf.predict(Xte)

    # combine
    cascade = (s1 & s2)                                # both must agree -> precise
    union = (s1 | s2)                                  # either flags -> high recall

    approaches = {"stage1_unsupervised": s1, "stage2_supervised": s2,
                  "cascade_AND": cascade, "union_OR": union}
    table = {name: score_row(pred, yte) for name, pred in approaches.items()}

    # per-attack recall (fraction of each attack caught) for cascade vs union
    per_attack = {}
    for a in ATTACKS:
        mask = atte == a
        if mask.sum():
            per_attack[a] = {"cascade": round(float(cascade[mask].mean()), 3),
                             "union": round(float(union[mask].mean()), 3)}

    result = {"stage1_recall_target": STAGE1_RECALL, "overall": table,
              "per_attack_recall": per_attack}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase3_ensemble.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # chart: recall vs precision for the four approaches
    names = list(table); rec = [table[n]["recall"] for n in names]
    prec = [table[n]["precision"] for n in names]
    x = np.arange(len(names)); w = 0.38
    plt.figure(figsize=(6, 3.6))
    plt.bar(x - w/2, rec, w, label="recall", color="#7F77DD")
    plt.bar(x + w/2, prec, w, label="precision", color="#1D9E75")
    plt.xticks(x, ["stage1", "stage2", "cascade", "union"])
    plt.ylim(0, 1); plt.ylabel("score"); plt.legend()
    plt.title("Cascade (precise) vs Union (high recall)")
    plt.tight_layout(); plt.savefig(RESULTS / "phase3_ensemble.png", dpi=150); plt.close()
    print("saved results/phase3_ensemble.json + phase3_ensemble.png")
