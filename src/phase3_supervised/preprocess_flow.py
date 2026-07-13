"""
Phase 3 - flow preprocessing.
Turn flow_sample.csv into a clean feature matrix + labels: drop identifier
columns (leakage), coerce to numeric, fix inf, median-impute. Scaling is done
later in train_supervised.py (fit on the training split only, to avoid leakage).
Saves a bookkeeping file of IPs/ports for matching Phase 2 alerts.
Run: python preprocess_flow.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_sampling"))
from config import SAMPLES

# Identifier columns dropped from the features (they'd let the model memorise hosts).
# Protocol is kept - it is a real feature (6=TCP, 17=UDP), not an identifier.
ID_COLS = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"]

if __name__ == "__main__":
    df = pd.read_csv(SAMPLES / "flow_sample.csv", low_memory=False)

    y = (df["Label"] != "Benign").astype(int).to_numpy()          # 1 = attack, 0 = benign
    attack_type = df["Label"].to_numpy()
    book = df[[c for c in ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port"] if c in df]].copy()

    feats = df.drop(columns=[c for c in ID_COLS if c in df] + ["Label"], errors="ignore")
    X = feats.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = X.median()
    X = X.fillna(medians)                                         # median-impute the gaps
    names = X.columns.tolist()

    np.savez_compressed(SAMPLES / "flow_preprocessed.npz", X=X.values, y=y,
                        attack_type=attack_type, feature_names=np.array(names, dtype=object),
                        medians=medians.values)                   # medians reused at re-check time
    book.to_csv(SAMPLES / "flow_bookkeeping.csv", index=False)

    print(f"flows={X.shape[0]} features={X.shape[1]} attack_rate={y.mean():.4f}")
    print("saved flow_preprocessed.npz, flow_bookkeeping.csv (scaling happens in training)")
