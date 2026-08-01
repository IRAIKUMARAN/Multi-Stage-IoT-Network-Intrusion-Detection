"""Anomaly scoring helpers: normalised reconstruction error, rank fusion, metrics."""

import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

EPS = 1e-9


def feature_error_scale(model, X):
    """Typical per-feature squared error on training data."""
    return np.mean((X - model.predict(X)) ** 2, axis=0) + EPS


def recon_error(model, X):
    """Plain mean squared reconstruction error per row."""
    return np.mean((X - model.predict(X)) ** 2, axis=1)


def norm_recon_error(model, X, scale):
    """Reconstruction error with each feature divided by its typical error.

    Stops a few naturally noisy features from dominating the anomaly score.
    """
    return np.mean(((X - model.predict(X)) ** 2) / scale, axis=1)


def rank_normalize(scores):
    """Map scores to [0, 1] by rank, so detectors on different scales can be fused."""
    order = scores.argsort().argsort()
    return order / max(len(scores) - 1, 1)


def fuse(scores, weights=None):
    """Weighted average of rank-normalised anomaly scores.

    Weights should reflect how informative each detector is (e.g. validation
    AUC - 0.5), so a weak detector cannot drag down a strong one.
    """
    ranks = np.array([rank_normalize(s) for s in scores])
    if weights is None:
        return ranks.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return ranks.mean(axis=0)
    return (ranks * (w / w.sum())[:, None]).sum(axis=0)


def evaluate(y_true, pred, score):
    """Standard detection metrics for one method at one operating point."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "AUC": float(roc_auc_score(y_true, score)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "FPR": float(fp / (fp + tn)),
        "FNR": float(fn / (fn + tp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def per_attack_rate(pred, attack_type, attacks):
    """Fraction of each attack type that was flagged."""
    out = {}
    for a in attacks:
        m = attack_type == a
        if m.sum():
            out[a] = {"n": int(m.sum()), "detection_rate": float(pred[m].mean())}
    return out
