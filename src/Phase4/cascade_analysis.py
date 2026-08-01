"""
PHASE 4 - Does the cascade actually beat either stage alone?

Scores every validation and test packet with BOTH stages, then compares three
systems on identical data:

  Phase 2 alone   - packet detector only
  Phase 3 alone   - flow classifier applied to all traffic, no Phase 2 gate
  Cascade         - Phase 2 flags, then Phase 3 confirms

Every threshold is chosen on VALIDATION and reported on TEST, so no system is
tuned on the data it is scored on.

Run: python cascade_analysis.py   (needs phase2 + phase3 to have run first)
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "phase1_sampling"))
sys.path.append(str(HERE.parent / "phase3_supervised"))
from config import SAMPLES, RAW_FLOW
from build_flow_dataset import FLOW_FILES, ID_COLS, agg_rule
from recheck_phase2_alerts import flow_key

RESULTS = HERE.parents[1] / "results"
EPS = 1e-12


def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    p = tp / (tp + fp + EPS)
    r = tp / (tp + fn + EPS)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r + EPS), 4),
            "accuracy": round((tp + tn) / len(y), 4),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "alerts": tp + fp}


def phase3_proba_for(book):
    """Match each packet to its flow and return the flow model's attack probability."""
    keys = flow_key(book["src_ip"], book["dst_ip"], book["src_port"], book["dst_port"])
    wanted = set(keys)

    prep = joblib.load(SAMPLES / "flow_preprocessor.joblib")
    model = joblib.load(SAMPLES / "flow_model.joblib")

    hits = []
    for files in FLOW_FILES.values():
        for name in files:
            path = list(RAW_FLOW.rglob(name))[0]
            for chunk in pd.read_csv(path, low_memory=False, chunksize=200_000):
                k = flow_key(chunk["Src IP"], chunk["Dst IP"],
                             chunk["Src Port"], chunk["Dst Port"])
                sub = chunk[k.isin(wanted)]
                if len(sub):
                    hits.append(sub)
    if not hits:
        sys.exit("No packets matched the flow files.")

    matched = pd.concat(hits, ignore_index=True)
    keep = {c: "first" for c in ID_COLS if c in matched and c != "Flow ID"}
    agg = matched.groupby("Flow ID", as_index=False).agg(
        {**keep, **agg_rule(matched.columns)})

    Xdf = agg.reindex(columns=prep["features"]).apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(prep["medians"])
    agg["proba"] = model.predict_proba(prep["scaler"].transform(Xdf.values))[:, 1]
    agg["key"] = flow_key(agg["Src IP"], agg["Dst IP"], agg["Src Port"], agg["Dst Port"])

    lookup = dict(zip(agg["key"], agg["proba"]))
    return keys.map(lookup).to_numpy(dtype=float)


def pred_p2(s2, t2):
    return (s2 > t2).astype(int)


def pred_p3(p3, matched, t3):
    return (matched & (np.nan_to_num(p3) >= t3)).astype(int)


def pred_cascade(s2, p3, matched, t2, t3):
    """Phase 2 gates; Phase 3 confirms. Unmatched alerts keep Phase 2's verdict."""
    return ((s2 > t2) & np.where(matched, np.nan_to_num(p3) >= t3, True)).astype(int)


if __name__ == "__main__":
    d = np.load(RESULTS / "phase2_test_scores.npz", allow_pickle=True)
    if "val_index" not in d:
        sys.exit("Re-run phase2_unsupervised.py first (validation scores missing).")

    iva, yva, s2va = d["val_index"], d["y_val"].astype(int), d["score_val"]
    ite, yte, s2te = d["test_index"], d["y"].astype(int), d["score"]
    print(f"Phase 2 detector: {d['detector']}  |  val {len(yva)}  test {len(yte)}")

    book = pd.read_csv(SAMPLES / "packet_bookkeeping.csv")
    both = book.iloc[np.concatenate([iva, ite])].reset_index(drop=True)
    p3_all = phase3_proba_for(both)
    p3va, p3te = p3_all[:len(iva)], p3_all[len(iva):]
    mva, mte = ~np.isnan(p3va), ~np.isnan(p3te)
    print(f"matched to a flow: val {mva.mean():.1%}  test {mte.mean():.1%}")

    t2_grid = np.quantile(s2va, np.linspace(0.50, 0.995, 24))
    t3_grid = np.round(np.linspace(0.05, 0.95, 19), 2)

    # --- tune every system on VALIDATION ---
    t2_p2 = max(t2_grid, key=lambda t: metrics(yva, pred_p2(s2va, t))["f1"])
    t3_p3 = max(t3_grid, key=lambda t: metrics(yva, pred_p3(p3va, mva, t))["f1"])
    grid = [(t2, t3, metrics(yva, pred_cascade(s2va, p3va, mva, t2, t3))["f1"])
            for t2 in t2_grid for t3 in t3_grid]
    t2_c, t3_c, _ = max(grid, key=lambda g: g[2])

    print(f"\nthresholds chosen on validation:")
    print(f"  Phase 2 alone   t2={t2_p2:.4g}")
    print(f"  Phase 3 alone   t3={t3_p3:.2f}")
    print(f"  Cascade         t2={t2_c:.4g}  t3={t3_c:.2f}")

    # --- report every system on TEST ---
    rows = [
        ("Phase 2 alone", metrics(yte, pred_p2(s2te, t2_p2)) | {"t2": float(t2_p2)}),
        ("Phase 3 alone", metrics(yte, pred_p3(p3te, mte, t3_p3)) | {"t3": float(t3_p3)}),
        ("Cascade (2 then 3)",
         metrics(yte, pred_cascade(s2te, p3te, mte, t2_c, t3_c))
         | {"t2": float(t2_c), "t3": float(t3_c)}),
    ]

    print(f"\ntest-set results (thresholds fixed from validation)")
    print(f"{'system':22} {'prec':>7} {'recall':>7} {'F1':>7} {'acc':>7} {'alerts':>8}")
    for name, m in rows:
        print(f"{name:22} {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} {m['accuracy']:>7.3f} {m['alerts']:>8}")

    winner = max(rows, key=lambda r: r[1]["f1"])[0]
    print(f"\nbest by F1 on test: {winner}")

    out = {"selection": "thresholds tuned on validation, reported on test",
           "phase2_alone": rows[0][1], "phase3_alone": rows[1][1],
           "cascade_best": rows[2][1], "winner": winner,
           "match_rate_test": round(float(mte.mean()), 4)}
    (RESULTS / "phase4_cascade_analysis.json").write_text(json.dumps(out, indent=2))

    (RESULTS / "operating_point.json").write_text(json.dumps({
        "phase2_threshold": float(t2_c),
        "phase3_threshold": float(t3_c),
        "source": "validation cascade sweep (cascade_analysis.py)",
    }, indent=2))
    print(f"wrote operating_point.json  t2={t2_c:.4g}  t3={t3_c:.2f}")

    f1s = np.array([g[2] for g in grid]).reshape(len(t2_grid), len(t3_grid))
    plt.figure(figsize=(7, 5))
    plt.imshow(f1s, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="validation F1")
    plt.xticks(range(0, len(t3_grid), 3), [f"{t:.2f}" for t in t3_grid[::3]])
    plt.yticks(range(0, len(t2_grid), 3), [f"{t:.3g}" for t in t2_grid[::3]])
    plt.xlabel("Phase 3 threshold"); plt.ylabel("Phase 2 threshold")
    plt.title("Cascade F1 across both thresholds (validation)")
    plt.tight_layout(); plt.savefig(RESULTS / "phase4_cascade_grid.png", dpi=150)
    plt.close()
    print("saved phase4_cascade_analysis.json + phase4_cascade_grid.png")
