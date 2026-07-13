"""
Phase 3, Task 3.2 - build the flow dataset.
Label each flow by its file, merge its 2-minute segments into one row per Flow ID,
then sample ~200k benign + ~5k attack flows (same proportions as Task 1.1).
Run: python build_flow_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import RAW_FLOW, SAMPLES

# Each class -> its flow file(s).
FLOW_FILES = {
    "Benign":          ["BenignTraffic.pcap_Flow.csv", "BenignTraffic1.pcap_Flow.csv",
                        "BenignTraffic2.pcap_Flow.csv", "BenignTraffic3.pcap_Flow.csv"],
    "DDoS-HTTP_Flood": ["DDoS-HTTP_Flood-.pcap_Flow.csv"],
    "DoS-HTTP_Flood":  ["DoS-HTTP_Flood.pcap_Flow.csv", "DoS-HTTP_Flood1.pcap_Flow.csv"],
    "DNS_Spoofing":    ["DNS_Spoofing.pcap_Flow.csv"],
    "XSS":             ["XSS.pcap_Flow.csv"],
    "Brute_Force":     ["DictionaryBruteForce.pcap_Flow.csv"],
}

# Identity columns - carried through but not treated as features.
ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]


def agg_rule(columns):
    """How to merge a flow's segments per column: sum totals, keep max/min, average the rest."""
    rule = {}
    for c in columns:
        if c in ID_COLS or c == "Label":
            continue
        cl = c.lower()
        if cl.startswith(("total", "subflow")) or "duration" in cl:
            rule[c] = "sum"
        elif "max" in cl:
            rule[c] = "max"
        elif "min" in cl:
            rule[c] = "min"
        else:
            rule[c] = "mean"
    return rule


if __name__ == "__main__":
    print("Building flow-level dataset (Task 3.2) ...")
    rng = np.random.default_rng(42)                       # fixed seed = reproducible sample
    attack_total = int(rng.integers(4000, 6200))
    attack_types = [c for c in FLOW_FILES if c != "Benign"]
    per_type = attack_total // len(attack_types)          # even split across the 5 attacks

    pieces = []
    for cls, files in FLOW_FILES.items():
        # load each file for this class and merge segments -> one row per Flow ID
        per_file = []
        for name in files:
            found = list(RAW_FLOW.rglob(name))
            if not found:
                sys.exit(f"Missing flow file: {name} under {RAW_FLOW}")
            df = pd.read_csv(found[0], low_memory=False)
            df["Label"] = cls                             # the raw Label is 'NeedManualLabel'
            keep_ids = {c: "first" for c in ID_COLS if c in df and c != "Flow ID"}
            merged = df.groupby("Flow ID", as_index=False).agg({**keep_ids, **agg_rule(df.columns)})
            merged["Label"] = cls
            per_file.append(merged)
        flows = pd.concat(per_file, ignore_index=True)

        n = 200_000 if cls == "Benign" else per_type      # benign may be fewer -> take all
        picked = flows if len(flows) <= n else flows.sample(n=n, random_state=42)
        print(f"  {cls:16s} flows={len(flows):>7d} sampled={len(picked)}")
        pieces.append(picked)

    data = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    data.to_csv(SAMPLES / "flow_sample.csv", index=False)
    print(f"\nTotal flows: {len(data)}  attack share: {100 * (data['Label'] != 'Benign').mean():.2f}%")
    print("Saved:", SAMPLES / "flow_sample.csv")
