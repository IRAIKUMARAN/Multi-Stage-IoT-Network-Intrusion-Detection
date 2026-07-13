"""Phase 2 - Unsupervised anomaly detection on PACKET-level data.


Task 2.1: train a model that separates normal vs anomalous packets 
           (e.g. autoencoder, k-means).
           DO NOT use labels during training.


Task 2.2: pick alert thresholds, analyze false positives vs negatives.
Metrics: precision/recall/F1, per-attack detection rate, FPR/FNR, AUC-ROC,
         confusion matrix.
         
"""

## Comment guidelines: For all the functions explain its working in own words (1-2 lines max)
def train(X): # change the function when implementing the code
    raise NotImplementedError


def score(model, X): # change the function when implementing the code
    raise NotImplementedError
