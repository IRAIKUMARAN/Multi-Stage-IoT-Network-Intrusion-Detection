"""
PHASE 2 - Unsupervised anomaly detection on packet-level data.

Compares six unsupervised detectors, then reports Phase 2 at two operating points:
  STANDALONE - F1-optimal, how good Phase 2 is as an IDS on its own.
  CASCADE    - recall-first, the alert set handed to Phase 3 (which can only
               remove false positives, never recover an attack Phase 2 missed).

Labels are never used to train. They are used only to score and to pick thresholds
on the validation split.

Run: python phase2_unsupervised.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display in this environment, just save figures to disk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# reuse the sampling config from phase 1 instead of duplicating paths here
sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import SAMPLES as DATA

import detectors as det
import scoring as sc
import thresholds as th

CASCADE_RECALL_TARGET = 0.85
ARCHS = [(64, 16, 64), (128, 32, 128), (96, 24, 96)] # autoencoder shapes to sweep
ATTACKS = ["DDoS-HTTP_Flood", "DoS-HTTP_Flood", "DNS_Spoofing", "XSS", "Brute_Force"]
RESULTS = Path(__file__).resolve().parents[2] / "results"
SEED = 42


def load_data():
  """Load the preprocessed packet features + bookkeeping table and drop leaky cols."""
    d = np.load(DATA / "packet_preprocessed.npz", allow_pickle=True)
    X, feat = d["X"], d["feature_names"].astype(str)
    book = pd.read_csv(DATA / "packet_bookkeeping.csv")
    y_type = book["attack_type"].to_numpy()

    # these two features basically leak the label (found during EDA), so they get
    # dropped before anything is trained on them
    leaky = np.array([f == "port_class_dst" or
                      f.startswith("http_content_type=application/octet-stream")
                      for f in feat])
    X, feat = X[:, ~leaky], feat[~leaky]
    print(f"dropped leaky cols: {int(leaky.sum())}")
    return X, (y_type != "Benign").astype(int), y_type, book


def split(X, y):
    """Stratified train/val/test split, 70/(0.2*70)/30 roughly, all keyed off SEED."""
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    itr, iva = train_test_split(itr, test_size=0.20, random_state=SEED, stratify=y[itr])
    return itr, iva, ite


def build_detectors(Xtr, Xva, Xte, yva):
    """Return {name: (val_score, test_score)} for every detector. No labels in training."""
    scorer = lambda s: roc_auc_score(yva, s)

    # pick the AE architecture using val AUC only, then train it for real
    arch, arch_auc, sweep = det.sweep_autoencoder(Xtr, Xva, scorer, ARCHS)
    for auc, h in sweep:
        print(f"  AE {str(h):16s} val AUC={auc:.3f}")
    print(f"best architecture: {arch} (val AUC={arch_auc:.3f})")

    ae = det.train_autoencoder(Xtr, arch, max_iter=100)
    scale = sc.feature_error_scale(ae, Xtr)

    # robust variant is trained separately (different loss), gets its own scale
    rae = det.train_robust_autoencoder(Xtr, arch, max_iter=100)
    rscale = sc.feature_error_scale(rae, Xtr)

    pca_va, pca_te = det.pca_scores(Xtr, [Xva, Xte])
    if_va, if_te = det.iforest_scores(Xtr, [Xva, Xte])

    out = {
        "AE_plain":      (sc.recon_error(ae, Xva), sc.recon_error(ae, Xte)),
        "AE_normalised": (sc.norm_recon_error(ae, Xva, scale),
                          sc.norm_recon_error(ae, Xte, scale)),
        "AE_robust":     (sc.norm_recon_error(rae, Xva, rscale),
                          sc.norm_recon_error(rae, Xte, rscale)),
        "PCA":           (pca_va, pca_te),
        "IsolationForest": (if_va, if_te),
    }

    # fuse the three strongest, independent detectors; weight each by how much
    # better than random it actually is on validation (clip negatives to 0 so a
    # detector that's worse than chance doesn't drag the fusion down)
    parts = ["AE_robust", "PCA", "IsolationForest"]
    w = [max(roc_auc_score(yva, out[p][0]) - 0.5, 0) for p in parts]
    print("fusion weights: " + ", ".join(f"{p}={x:.3f}" for p, x in zip(parts, w)))
    out["Fusion"] = (sc.fuse([out[p][0] for p in parts], w),
                     sc.fuse([out[p][1] for p in parts], w))
    return out, arch


def compare(scores, yva, yte):
    """Score every detector at its own F1-optimal threshold (chosen on validation).

    Returns the test table AND the validation F1 of each detector, so the winning
    detector can be selected without ever looking at the test set.
    """
    table, val_f1 = {}, {}
    for name, (s_va, s_te) in scores.items():
        thr, f1_va = th.f1_optimal(yva, s_va)   # threshold picked on val only
        m = sc.evaluate(yte, (s_te > thr).astype(int), s_te)  # test metrics reported after
        m["threshold"] = thr
        m["val_f1"] = round(f1_va, 4)
        table[name] = m
        val_f1[name] = f1_va
    return table, val_f1


def plots(yte, pred, scores):
    """Save the confusion matrix (standalone pick) and ROC comparison across detectors."""
    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Phase 2 Confusion Matrix")
    plt.tight_layout(); plt.savefig(RESULTS / "phase2_confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    for name, (_, st) in scores.items():
        fpr, tpr, _ = roc_curve(yte, st)
        plt.plot(fpr, tpr, lw=1.6, label=f"{name} ({roc_auc_score(yte, st):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)  # random-guess reference line
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Phase 2 ROC - detector comparison")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(RESULTS / "phase2_roc.png", dpi=150); plt.close()


if __name__ == "__main__":
    # lets rerun the whole pipeline with a different seed for a quick
    # stability check without touching the code
    if "--seed" in sys.argv:
        SEED = int(sys.argv[sys.argv.index("--seed") + 1])
        det.SEED = SEED
        print(f"seed override: {SEED}")

    X, y, y_type, book = load_data()
    print(f"X: {X.shape} | attack rate: {y.mean():.4f}")

    itr, iva, ite = split(X, y)
    Xtr, Xva, Xte = X[itr], X[iva], X[ite]
    yva, yte, tte = y[iva], y[ite], y_type[ite]
    print(f"train/val/test: {len(itr)} {len(iva)} {len(ite)}")

    scores, arch = build_detectors(Xtr, Xva, Xte, yva)
    table, val_f1 = compare(scores, yva, yte)

    print("\ndetector comparison (F1-optimal operating point):")
    print(f"  {'method':18s} {'AUC':>6} {'prec':>7} {'recall':>7} {'testF1':>7} {'valF1':>7}")
    for name, m in sorted(table.items(), key=lambda kv: -kv[1]["val_f1"]):
        print(f"  {name:18s} {m['AUC']:>6.3f} {m['precision']:>7.3f} "
              f"{m['recall']:>7.3f} {m['f1']:>7.3f} {m['val_f1']:>7.3f}")

    # selection happens on val F1 - test set is only for reporting, never for choosing
    best = max(val_f1, key=val_f1.get)          # selected on validation, never on test
    s_va, s_te = scores[best]
    print(f"\nbest standalone detector: {best} "
          f"(val F1={val_f1[best]:.3f} -> test F1={table[best]['f1']:.3f})")

    thr_standalone = table[best]["threshold"]
    pred_standalone = (s_te > thr_standalone).astype(int)

    # cascade threshold: use the tuned value from the cascade sweep if it exists,
    # otherwise fall back to a plain recall-target threshold on validation
    tuned = RESULTS / "operating_point.json"
    if tuned.exists():
        thr_cascade = json.loads(tuned.read_text())["phase2_threshold"]
        cascade_rule = "tuned on validation by the cascade sweep"
    else:
        thr_cascade = th.recall_target(yva, s_va, CASCADE_RECALL_TARGET)
        cascade_rule = f"recall>={CASCADE_RECALL_TARGET} (untuned fallback)"
    pred_cascade = (s_te > thr_cascade).astype(int)
    m_cascade = sc.evaluate(yte, pred_cascade, s_te)
    print(f"cascade operating point: {cascade_rule}")

    print(f"\nSTANDALONE  thr={thr_standalone:.4g}  "
          f"P={table[best]['precision']:.3f} R={table[best]['recall']:.3f} "
          f"F1={table[best]['f1']:.3f}")
    print(f"CASCADE     thr={thr_cascade:.4g}  "
          f"P={m_cascade['precision']:.3f} R={m_cascade['recall']:.3f} "
          f"F1={m_cascade['f1']:.3f}  alerts={int(pred_cascade.sum())}")

    metrics = {
        "best_detector": best,
        "architecture": list(arch),
        "comparison": table,
        "standalone": {**table[best], "operating_point": "F1-optimal"},
        "cascade": {**m_cascade, "threshold": thr_cascade,
                    "operating_point": cascade_rule,
                    "alerts": int(pred_cascade.sum())},
        "per_attack_detection": sc.per_attack_rate(pred_standalone, tte, ATTACKS),
        "per_attack_detection_cascade": sc.per_attack_rate(pred_cascade, tte, ATTACKS),
        "AE_threshold": {**m_cascade, "threshold": thr_cascade},
    }

    metrics["seed"] = SEED
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase2_metrics.json").write_text(json.dumps(metrics, indent=2))
    # keep a per-seed copy too, so re-running with --seed doesn't clobber the main run
    if SEED != 42:
        (RESULTS / f"phase2_metrics_seed{SEED}.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / "phase2_operating_curve.json").write_text(
        json.dumps(th.operating_curve(yva, s_va), indent=2))

    # only the packets flagged at the cascade threshold get passed forward -
    # this is literally the input Phase 3 will work with  
    book.iloc[ite][pred_cascade.astype(bool)].to_csv(
        RESULTS / "flagged_packet_ids.csv", index=False)
    print(f"\nflagged alerts handed to Phase 3: {int(pred_cascade.sum())}")

    np.savez_compressed(RESULTS / "phase2_test_scores.npz",
                        test_index=ite, y=yte, score=s_te,
                        attack_type=tte.astype(str), detector=best,
                        val_index=iva, y_val=yva, score_val=s_va)

    # Endpoints of every held-out packet. Phase 3 excludes the matching flows from
    # its training set, so the flow model never scores a flow it learned from.
    cols = ["src_ip", "dst_ip", "src_port", "dst_port"]
    book.iloc[np.concatenate([iva, ite])][cols].drop_duplicates().to_csv(
        RESULTS / "heldout_endpoints.csv", index=False)
    print("saved heldout_endpoints.csv (flows Phase 3 must not train on)")

    plots(yte, pred_standalone, scores)
    print("saved metrics, operating curve, plots to", RESULTS)
