"""Unsupervised detectors for Phase 2. No labels are used anywhere in this file."""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from scoring import recon_error

SEED = 42


def train_autoencoder(X, hidden, max_iter=80):
    """Autoencoder = MLP trained to rebuild its own input through a bottleneck."""
    ae = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                      max_iter=max_iter, early_stopping=True, random_state=SEED)
    ae.fit(X, X)
    return ae


def train_robust_autoencoder(X, hidden, max_iter=80, keep=0.90, rounds=1):
    """Autoencoder retrained on the rows it already considers most normal.

    The sample is ~98% benign, so trimming the worst-reconstructing rows removes
    most contamination and sharpens the model of 'normal'. Uses no labels.
    """
    ae = train_autoencoder(X, hidden, max_iter)
    for _ in range(rounds):
        cutoff = np.quantile(recon_error(ae, X), keep)
        clean = X[recon_error(ae, X) <= cutoff]
        ae = train_autoencoder(clean, hidden, max_iter)
    return ae


def sweep_autoencoder(Xtr, Xva, yva_scorer, archs, max_iter=40):
    """Pick the architecture with the best validation AUC. Labels only score, never train."""
    results = []
    for h in archs:
        ae = train_autoencoder(Xtr, h, max_iter)
        results.append((yva_scorer(recon_error(ae, Xva)), h))
    best_auc, best_arch = max(results)
    return best_arch, best_auc, results


def pca_scores(Xtr, Xs, n_components=20):
    """PCA reconstruction error: cheap, strong linear baseline for anomaly detection."""
    pca = PCA(n_components=n_components, random_state=SEED).fit(Xtr)
    out = []
    for X in Xs:
        rebuilt = pca.inverse_transform(pca.transform(X))
        out.append(np.mean((X - rebuilt) ** 2, axis=1))
    return out


def iforest_scores(Xtr, Xs, n_estimators=200, contamination=0.02):
    """Isolation Forest: outliers need fewer random splits to isolate."""
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=SEED, n_jobs=-1).fit(Xtr)
    return [-iso.score_samples(X) for X in Xs]


def kmeans_split(err_tr, err_te):
    """Two-cluster split of the error distribution: a label-free cutoff."""
    km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(
        np.log1p(err_tr).reshape(-1, 1))
    anom = int(np.argmax([err_tr[km.labels_ == c].mean() for c in (0, 1)]))
    return (km.predict(np.log1p(err_te).reshape(-1, 1)) == anom).astype(int)
