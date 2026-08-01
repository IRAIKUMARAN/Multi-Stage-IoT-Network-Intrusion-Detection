"""Operating-point selection. Thresholds are always chosen on validation, never on test."""

import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix

EPS = 1e-12


def f1_optimal(y, score):
    """Threshold with the best balance of precision and recall."""
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    return float(thr[f1[:-1].argmax()]), float(f1[:-1].max())


def recall_target(y, score, target):
    """Loosest threshold that still catches at least `target` of the attacks."""
    prec, rec, thr = precision_recall_curve(y, score)
    ok = rec[:-1] >= target
    if not ok.any():
        return f1_optimal(y, score)[0]
    return float(thr[ok].max())


def cost_optimal(y, score, cost_fp=1.0, cost_fn=50.0):
    """Threshold minimising analyst cost, given a missed attack costs `cost_fn`
    times more than reviewing one false alarm."""
    best, best_cost = None, np.inf
    for t in np.quantile(score, np.linspace(0.50, 0.999, 60)):
        tn, fp, fn, tp = confusion_matrix(y, (score > t).astype(int)).ravel()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best, best_cost = float(t), cost
    return best, float(best_cost)


def operating_curve(y, score, lo=0.50, hi=0.999, n=25):
    """Recall / FPR / precision across a sweep of thresholds."""
    curve = []
    for q in np.linspace(lo, hi, n):
        t = float(np.quantile(score, q))
        tn, fp, fn, tp = confusion_matrix(y, (score > t).astype(int)).ravel()
        curve.append({
            "threshold": round(t, 6),
            "recall": round(tp / (tp + fn + EPS), 3),
            "FPR": round(fp / (fp + tn + EPS), 3),
            "precision": round(tp / (tp + fp + EPS), 3),
        })
    return curve
