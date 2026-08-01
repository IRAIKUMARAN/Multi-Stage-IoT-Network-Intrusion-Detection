"""
Phase 3 - re-check Phase 2's alerts (the headline two-stage result) + novelty.
Take the packets Phase 2 flagged, find each one's flow, run the trained flow
model on it, and measure how many of Phase 2's false positives get dismissed
while real attacks are kept.
Novelty: sweep the Phase 3 threshold to map the false-positive-reduction vs
attack-retention trade-off, then pick the operating point that removes the most
false positives while keeping >= 90% of the attacks.
Run: python recheck_phase2_alerts.py   (run train_supervised.py first)
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import SAMPLES, RAW_FLOW
from build_flow_dataset import FLOW_FILES, ID_COLS, agg_rule

RESULTS = Path(__file__).resolve().parents[2] / "results"


def flow_key(ip_s, ip_d, p_s, p_d):
    """Direction-independent key: sort the two endpoints so A->B and B->A match the same flow."""
    port = lambda s: pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    a = ip_s.astype(str) + ":" + port(p_s)
    b = ip_d.astype(str) + ":" + port(p_d)
    return a.where(a <= b, b) + "|" + a.where(a > b, b)


if __name__ == "__main__":
    # 1. load Phase 2's flagged packets; mark each as a real attack or a false positive
    flagged = pd.read_csv(RESULTS / "flagged_packet_ids.csv")
    flagged["key"] = flow_key(flagged["src_ip"], flagged["dst_ip"],
                              flagged["src_port"], flagged["dst_port"])
    flagged["is_attack"] = (flagged["attack_type"] != "Benign").astype(int)
    wanted = set(flagged["key"])
    print(f"flagged alerts: {len(flagged)}  ({int((flagged['is_attack']==0).sum())} benign = FPs)")

    prep = joblib.load(SAMPLES / "flow_preprocessor.joblib")
    model = joblib.load(SAMPLES / "flow_model.joblib")

    # 2. scan the flow files and keep only the flagged flows
    print("scanning flow files for the flagged flows ...")
    hits = []
    for files in FLOW_FILES.values():
        for name in files:
            path = list(RAW_FLOW.rglob(name))[0]
            for chunk in pd.read_csv(path, low_memory=False, chunksize=200_000):
                k = flow_key(chunk["Src IP"], chunk["Dst IP"], chunk["Src Port"], chunk["Dst Port"])
                sub = chunk[k.isin(wanted)]
                if len(sub):
                    hits.append(sub)
    if not hits:
        sys.exit("No flagged flows matched the flow files - check IP/port columns.")
    matched = pd.concat(hits, ignore_index=True)

    # 3. merge segments the SAME way as training, build features, predict probabilities
    matched["Label"] = "unknown"
    keep_ids = {c: "first" for c in ID_COLS if c in matched and c != "Flow ID"}
    agg = matched.groupby("Flow ID", as_index=False).agg({**keep_ids, **agg_rule(matched.columns)})

    Xdf = agg.reindex(columns=prep["features"]).apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(prep["medians"])
    agg["proba"] = model.predict_proba(prep["scaler"].transform(Xdf.values))[:, 1]
    agg["key"] = flow_key(agg["Src IP"], agg["Dst IP"], agg["Src Port"], agg["Dst Port"])
    proba_by_key = dict(zip(agg["key"], agg["proba"]))

    m = flagged[flagged["key"].isin(proba_by_key)].copy()
    m["proba"] = m["key"].map(proba_by_key)
    fp_before = int((m["is_attack"] == 0).sum())
    tp_before = int((m["is_attack"] == 1).sum())

    # 4. NOVELTY: sweep the threshold to map the two-stage trade-off curve
    curve = []
    for t in np.round(np.linspace(0.05, 0.95, 19), 2):
        k = m["proba"] >= t
        fp_a = int(((m["is_attack"] == 0) & k).sum())
        tp_a = int(((m["is_attack"] == 1) & k).sum())
        curve.append({"threshold": float(t),
                      "fp_reduction_pct": round(100 * (1 - fp_a / fp_before), 1) if fp_before else None,
                      "attack_retention_pct": round(100 * tp_a / tp_before, 1) if tp_before else None})
    (RESULTS / "phase3_operating_curve.json").write_text(json.dumps(curve, indent=2))

    # 5. pick the operating point: most false-positive reduction while keeping >= 90% of attacks
    RETENTION_FLOOR = 90.0
    viable = [c for c in curve if c["attack_retention_pct"] and c["attack_retention_pct"] >= RETENTION_FLOOR]
    chosen = max(viable, key=lambda c: c["fp_reduction_pct"]) if viable \
        else max(curve, key=lambda c: c["attack_retention_pct"])

    op = m["proba"] >= chosen["threshold"]
    result = {
        "operating_threshold": chosen["threshold"],
        "retention_floor_pct": RETENTION_FLOOR,
        "flagged_total": int(len(flagged)),
        "matched_to_flow": int(len(m)),
        "match_rate": round(len(m) / len(flagged), 3),
        "false_positives_before": fp_before,
        "false_positives_after": int(((m["is_attack"] == 0) & op).sum()),
        "false_positive_reduction_pct": chosen["fp_reduction_pct"],
        "attacks_reaching_phase3": tp_before,
        "attacks_kept": int(((m["is_attack"] == 1) & op).sum()),
        "attack_retention_pct": chosen["attack_retention_pct"],
    }
    (RESULTS / "phase3_recheck.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    red = [c["fp_reduction_pct"] for c in curve]
    ret = [c["attack_retention_pct"] for c in curve]
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(red, ret, marker="o")
    plt.xlabel("False-positive reduction (%)"); plt.ylabel("Attack retention (%)")
    plt.title("Two-stage operating curve"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "phase3_operating_curve.png", dpi=150); plt.close()

    print("saved results/phase3_recheck.json, phase3_operating_curve.json + .png")
