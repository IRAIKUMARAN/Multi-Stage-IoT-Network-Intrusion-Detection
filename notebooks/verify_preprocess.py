import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "phase1_sampling"))

import numpy as np
import pandas as pd
from config import SAMPLES

df = pd.read_csv(SAMPLES / "packet_sample.csv", low_memory=False)
data = np.load(SAMPLES / "packet_preprocessed.npz", allow_pickle=True)
X, names = data["X"], list(data["feature_names"])
book = pd.read_csv(SAMPLES / "packet_bookkeeping.csv", low_memory=False)

print("== CHECK 1: hunt for the dataset's own label column ==")
print(df.columns.tolist())          # read this list with your own eyes!
suspects = [c for c in df.columns if any(k in c.lower()
            for k in ["label", "class", "anomal", "attack", "category", "type"])]
print("suspect columns:", suspects)

print("\n== CHECK 2: nothing forbidden inside the feature matrix ==")
bad = [n for n in names if any(k in n.lower()
       for k in ["label", "attack", "src_ip", "dst_ip", "mac", "port", "oui", "stream"])
       and not n.startswith(("stream_", "src_ip_", "src_ip_mac_"))]  # windowed stats are fine
print("forbidden features found:", bad if bad else "NONE - clean")

print("\n== CHECK 3: alignment and scaling ==")
print("rows match:", X.shape[0] == len(df) == len(book))
print("means ~0 :", np.abs(X.mean(axis=0)).max())   # expect < ~0.01
print("stds  ~1 :", X.std(axis=0).min(), X.std(axis=0).max())

print("\n== CHECK 4: presence flags mean what they claim ==")
i = names.index("has_dns_query_type")
flag_frac = X[:, i] > 0   # careful: scaled, so >0 means 'present' side
print("has_dns fraction:", (df["dns_query_type"].notna()).mean().round(4),
      " (expect ~0.047, i.e. matches raw NaN count)")

print("\n== CHECK 5: does the sample make security sense? ==")
dns_rows = df["attack_type"] == "DNS_Spoofing"
print("DNS-Spoofing packets that are DNS traffic:",
      df.loc[dns_rows, "dns_query_type"].notna().mean().round(3))