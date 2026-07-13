"""
Task 1.2 - Data Preprocessing (packet-level)
ECE 597 Capstone, Phase 1

Design decisions (each justified in the report):

1. IDENTIFIER / LEAKAGE columns (IPs, MACs, ports, OUIs, stream id, labels)
   are EXCLUDED from the model's feature matrix but KEPT in a bookkeeping
   dataframe: in this testbed specific hosts ran specific attacks, so a
   model could "detect" attacks by memorizing addresses. That doesn't
   generalize. IPs/ports are still needed later to build Flow IDs (Phase 3).

2. PROTOCOL-SPECIFIC missing values are informative, not broken:
   a non-DNS packet has no dns_query_type, a non-HTTP packet has no
   http_request_method. We convert each protocol text field into a binary
   presence flag (has_http, has_dns, ...) and one-hot encode only the
   low-cardinality ones. High-cardinality text (http_uri, user_agent, ...)
   would explode into thousands of one-hot columns, so presence flags only.

3. WINDOWED VARIANCE columns (*_var) are NaN when the time window held a
   single packet -> variance undefined -> correct fill is 0 (no spread).
   Remaining numeric NaNs get median imputation (robust to outliers).

4. INF values (rate features can divide by ~0 time deltas) -> replaced with
   NaN, then imputed like other missing values.

5. CONSTANT columns carry zero information -> dropped (recorded in report).

6. SCALING: StandardScaler, because Phase 2 uses k-means (Euclidean
   distances) and autoencoders (gradient training), both scale-sensitive.
   fit() only ever sees training data; transform() is applied elsewhere.

Usage:
    python preprocess.py            # processes samples/packet_sample.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from config import SAMPLES

# ---------------------------------------------------------------------------
# Column groups (from profiling the real sample, notebooks/profile_sample.py)
# ---------------------------------------------------------------------------
ID_COLS = [
    "stream", "src_mac", "dst_mac", "src_ip", "dst_ip", "src_port",
    "dst_port", "device_mac", "eth_src_oui", "eth_dst_oui",
]

# protocol text fields -> presence flag; ALSO one-hot if low-cardinality
PROTOCOL_TEXT_COLS = [
    "handshake_version", "tls_server", "http_request_method", "http_host",
    "user_agent", "dns_server", "highest_layer", "http_uri",
    "http_content_type", "dns_query_type",
]
ONEHOT_MAX_CARDINALITY = 25   # one-hot only if <= this many unique values

LABEL_HINTS = ["label", "attack_type"]   # auto-detected below


def find_label_cols(df):
    return [c for c in df.columns
            if any(h in c.lower() for h in LABEL_HINTS)]


class PacketPreprocessor:
    """fit on training data only; transform anywhere."""

    def fit(self, df: pd.DataFrame):
        self.label_cols_ = find_label_cols(df)
        self.id_cols_ = [c for c in ID_COLS if c in df.columns]
        self.text_cols_ = [c for c in PROTOCOL_TEXT_COLS if c in df.columns]

        feat = df.drop(columns=self.label_cols_ + self.id_cols_)

        # decide one-hot vs presence-only per text column
        self.onehot_cols_, self.onehot_values_ = [], {}
        for c in self.text_cols_:
            nuniq = feat[c].nunique(dropna=True)
            if 0 < nuniq <= ONEHOT_MAX_CARDINALITY:
                self.onehot_cols_.append(c)
                self.onehot_values_[c] = sorted(feat[c].dropna().unique().tolist())

        num = feat.drop(columns=self.text_cols_, errors="ignore")
        num = num.apply(pd.to_numeric, errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan)

        self.var_cols_ = [c for c in num.columns if c.endswith("_var")]
        # medians for non-var numeric imputation (computed on TRAIN only)
        self.medians_ = num.drop(columns=self.var_cols_).median()
        # constant columns: nothing to learn from
        filled = self._impute(num)
        self.constant_cols_ = filled.columns[filled.nunique() <= 1].tolist()

        X = self._assemble(df)
        self.feature_names_ = X.columns.tolist()
        self.scaler_ = StandardScaler().fit(X.values)
        return self

    def _impute(self, num: pd.DataFrame) -> pd.DataFrame:
        num = num.copy()
        for c in self.var_cols_:
            if c in num.columns:
                num[c] = num[c].fillna(0.0)          # single-packet window
        return num.fillna(self.medians_)             # robust median fill

    def _assemble(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = df.drop(columns=[c for c in self.label_cols_ + self.id_cols_
                                if c in df.columns])
        parts = []

        # numeric block
        num = feat.drop(columns=[c for c in self.text_cols_ if c in feat],
                        errors="ignore")
        num = num.apply(pd.to_numeric, errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan)
        num = self._impute(num)
        num = num.drop(columns=[c for c in self.constant_cols_ if c in num],
                       errors="ignore")
        parts.append(num)

        # protocol presence flags + selective one-hot
        for c in self.text_cols_:
            if c not in df.columns:
                continue
            parts.append(df[c].notna().astype(np.int8).rename(f"has_{c}"))
            if c in self.onehot_cols_:
                col = df[c].astype("string")
                for v in self.onehot_values_[c]:
                    parts.append((col == v).fillna(False).astype(np.int8)
                                 .rename(f"{c}={v}"))

        X = pd.concat(parts, axis=1)
        # tolerate columns missing at transform-time (align to fit schema)
        if hasattr(self, "feature_names_"):
            X = X.reindex(columns=self.feature_names_, fill_value=0)
        return X

    def transform(self, df: pd.DataFrame):
        """Returns (X_scaled ndarray, bookkeeping df, feature names)."""
        X = self._assemble(df)
        keep = [c for c in self.label_cols_ + self.id_cols_ if c in df.columns]
        book = df[keep].copy()
        return self.scaler_.transform(X.values), book, self.feature_names_


if __name__ == "__main__":
    src = SAMPLES / "packet_sample.csv"
    print(f"Loading {src} ...")
    df = pd.read_csv(src, low_memory=False)

    pp = PacketPreprocessor().fit(df)
    X, book, names = pp.transform(df)

    print(f"\nInput shape:   {df.shape}")
    print(f"Feature matrix: {X.shape}  (excluded {len(pp.id_cols_)} id cols, "
          f"{len(pp.label_cols_)} label cols, {len(pp.constant_cols_)} constant cols)")
    print(f"Label cols:    {pp.label_cols_}")
    print(f"Constant cols dropped: {pp.constant_cols_}")
    print(f"One-hot encoded: {pp.onehot_cols_}")
    print(f"NaN/inf remaining in X: {np.isnan(X).sum()} / "
          f"{np.isinf(X).sum()}")

    out = SAMPLES / "packet_preprocessed.npz"
    np.savez_compressed(out, X=X, feature_names=np.array(names, dtype=object))
    book.to_csv(SAMPLES / "packet_bookkeeping.csv", index=False)
    joblib.dump(pp, SAMPLES / "packet_preprocessor.joblib")
    print(f"\nSaved: {out}, packet_bookkeeping.csv, packet_preprocessor.joblib")
