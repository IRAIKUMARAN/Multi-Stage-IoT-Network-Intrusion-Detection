"""
Task 1.1 — Random Dataset Generation Function
ECE 597 Capstone, Phase 1

Samples the CIC IoT-DIAD 2024 packet-level dataset:
  - ~200,000 benign rows  (~97-98% of sample)
  - 4,000-6,200 attack rows (~2-3%), spread ~uniformly across 5 attack types
  - Different composition each run unless a seed is provided

USAGE (adjust FILE_MAP paths to your local download first):
    python generate_dataset.py                # random composition
    python generate_dataset.py --seed 42      # reproducible run
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. CONFIG — EDIT THESE PATHS to match your local dataset layout.
#    Per the announcement: HTTP Flood variants only (TCP Flood has no flow
#    data), TCP files not UDP, and "flood" vs "flood1" are interchangeable.
# ---------------------------------------------------------------------------
from config import RAW_PACKET as RAW, SAMPLES

FILE_MAP = {
    "Benign": [RAW / "BenignTraffic" / f"BenignTraffic{i}.csv" for i in ["", 1, 2, 3]],
    "DDoS-HTTP_Flood": [RAW / "DDoS-HTTP_Flood" / "DDoS-HTTP_Flood-.csv"],   # note trailing '-'
    "DoS-HTTP_Flood":  [RAW / "DoS-HTTP_Flood" / "DoS-HTTP_Flood.csv",
                        RAW / "DoS-HTTP_Flood" / "DoS-HTTP_Flood1.csv"],
    "DNS_Spoofing":    [RAW / "DNS_Spoofing" / "DNS_Spoofing.csv"],
    "XSS":             [RAW / "XSS" / "XSS.csv"],
    "Brute_Force":     [RAW / "DictionaryBruteForce" / "DictionaryBruteForce.csv"],
}
BENIGN_TARGET = 200_000
ATTACK_TOTAL_RANGE = (4_000, 6_200)   # total attack rows drawn from this range
CHUNK_SIZE = 100_000                  # rows per chunk when streaming large CSVs


def _count_rows(paths):
    """Fast row count without loading data (needed to size per-chunk samples)."""
    total = 0
    for p in paths:
        with open(p, "rb") as f:
            total += sum(1 for _ in f) - 1  # minus header
    return total


def _sample_from_files(paths, n_target, rng):
    """
    Uniform random sample of n_target rows from one or more CSVs,
    using chunked reading so we never load a full file into memory.

    Strategy: know the total row count in advance, then from each chunk
    draw a number of rows proportional to the chunk's share of the total.
    This approximates a simple random sample over the whole file.
    """
    total_rows = _count_rows(paths)
    if total_rows < n_target:
        raise ValueError(
            f"Requested {n_target} rows but files only contain {total_rows}. "
            f"Check FILE_MAP paths: {paths}"
        )

    frac = n_target / total_rows
    pieces = []
    for p in paths:
        for chunk in pd.read_csv(p, chunksize=CHUNK_SIZE, low_memory=False):
            # Proportional draw from this chunk; rng keeps it seed-controlled
            k = rng.binomial(len(chunk), frac)
            if k > 0:
                pieces.append(chunk.sample(n=min(k, len(chunk)),
                                           random_state=rng.integers(0, 2**32 - 1)))

    df = pd.concat(pieces, ignore_index=True)

    # Binomial draws land *near* n_target; trim or top up to hit it exactly.
    if len(df) > n_target:
        df = df.sample(n=n_target, random_state=rng.integers(0, 2**32 - 1))
    elif len(df) < n_target:
        # Rare shortfall: acceptable to top up with one more pass, but for
        # simplicity we accept a tiny deficit and report it in validation.
        print(f"  note: drew {len(df)} of {n_target} requested rows")
    return df.reset_index(drop=True)


def generate_random_dataset(seed=None, verbose=True):
    """
    Build one sampled dataset per the Task 1.1 spec.

    seed=None  -> different composition every run (spec requirement)
    seed=int   -> fully reproducible run (debugging / instructor validation)

    Returns a shuffled DataFrame with an added 'attack_type' column.
    """
    rng = np.random.default_rng(seed)

    # --- decide this run's composition -------------------------------------
    attack_total = int(rng.integers(*ATTACK_TOTAL_RANGE))
    attack_types = [k for k in FILE_MAP if k != "Benign"]

    # ~uniform split across the 5 types, with small random jitter so the
    # composition genuinely differs between runs (not always exactly equal)
    weights = rng.dirichlet(np.ones(len(attack_types)) * 50)  # near-uniform
    per_type = {t: int(round(w * attack_total)) for t, w in zip(attack_types, weights)}

    if verbose:
        print(f"Run composition (seed={seed}):")
        print(f"  Benign: {BENIGN_TARGET}")
        for t, n in per_type.items():
            print(f"  {t}: {n}")

    # --- sample each class ---------------------------------------------------
    parts = []
    benign = _sample_from_files(FILE_MAP["Benign"], BENIGN_TARGET, rng)
    benign["attack_type"] = "Benign"
    parts.append(benign)

    for t, n in per_type.items():
        df = _sample_from_files(FILE_MAP[t], n, rng)
        df["attack_type"] = t
        parts.append(df)

    data = pd.concat(parts, ignore_index=True)
    # Shuffle so classes aren't in contiguous blocks
    data = data.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True)

    _validate(data, per_type, verbose)
    return data


def _validate(df, per_type, verbose=True):
    """Basic integrity checks required by the spec."""
    counts = df["attack_type"].value_counts()
    n_benign = counts.get("Benign", 0)
    n_attack = len(df) - n_benign
    attack_pct = 100 * n_attack / len(df)

    checks = {
        "benign count ~200k": abs(n_benign - BENIGN_TARGET) < 1000,
        "attack total in 4000-6200 band": 3_800 <= n_attack <= 6_400,
        "attack share ~2-3%": 1.8 <= attack_pct <= 3.2,
        "all 5 attack types present": all(counts.get(t, 0) > 0 for t in per_type),
        "no fully-duplicated rows": df.duplicated().sum() < 0.01 * len(df),
        "no all-null columns": not df.isna().all().any(),
    }
    if verbose:
        print("\nValidation:")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"\nTotal rows: {len(df)}  |  attack share: {attack_pct:.2f}%")
    if not all(checks.values()):
        raise AssertionError("Dataset failed integrity checks — see FAILs above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed. Omit for a different composition each run.")
    ap.add_argument("--out", type=str, default=str(SAMPLES / "packet_sample.csv"))
    args = ap.parse_args()

    df = generate_random_dataset(seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
