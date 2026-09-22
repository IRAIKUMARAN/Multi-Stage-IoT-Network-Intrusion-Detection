# Two-Stage IoT Network Intrusion Detection

**ECE 597 Capstone, Group 7, University of Victoria (Summer 2026)**
Supervised by Dr. Ardeshir Shojaeinasab and Dr. Riham AlTawy

An unsupervised detector screens every network packet, and a supervised classifier re-checks each alert it raises. Built and evaluated on the CIC IoT-DIAD 2024 dataset.

## Results at a glance

All thresholds were tuned on validation data. The test set (61,160 packets) was used once.

| System | Precision | Recall | F1 |
|---|---|---|---|
| Stage 1 alone (robust autoencoder, packets) | 0.348 | 0.468 | 0.400 |
| Stage 2 alone (Random Forest, flows) | 0.676 | 0.376 | 0.483 |
| **Two-stage cascade** | **0.572** | **0.618** | **0.594** |

- The cascade beats the best single stage by **+0.111 F1** (bootstrap 95% CI: +0.085 to +0.139).
- Only **7% of traffic** reaches the expensive flow-level stage, so the screening stage cuts that work by roughly 10x.
- Why it works: the flow classifier needs a threshold of 0.85 on its own, but can run at 0.15 inside the cascade because stage 1 has already removed most benign traffic.

## The problem

Attacks are rare. In our sample only 2.04% of packets are malicious, so a model that calls everything benign scores 98% accuracy while catching nothing. We report precision, recall and F1 instead.

- Unsupervised detectors can catch new attacks but raise many false alarms.
- Supervised classifiers are precise but only know attacks they were trained on.

We combine the two instead of picking one.

## How it works

```
 packets ──► Stage 1: robust autoencoder ──► alerts ──► Stage 2: Random Forest on flows ──► confirmed alerts
             (unsupervised, tuned for recall)            (supervised, tuned for precision)
```

1. **Sampling (Phase 1).** 200,000 benign packets plus 4,000 to 6,200 attack packets across five classes (DDoS-HTTP Flood, DoS-HTTP Flood, DNS Spoofing, XSS, Brute Force). Files are read in 100,000-row chunks so memory stays flat. Fixed seed (42) for reproducibility.
2. **Preprocessing.** IP, MAC and port columns are removed from the features, so the model can't just memorise which machines were attackers. 140 clean features.
3. **Stage 1: packet screening (Phase 2).** Six detectors compared (plain, normalised and robust autoencoders, PCA, Isolation Forest, rank fusion). The chosen **robust autoencoder** trains, drops the 10% of rows it reconstructs worst (mostly attacks), then retrains. F1 goes from 0.349 to 0.415 with no labels used.
4. **Stage 2: flow re-check (Phase 3).** Flow segments are merged into one record per connection. Random Forest, HistGradientBoosting and logistic regression are compared with 5-fold cross-validation, and the Random Forest is selected. A direction-independent flow key raised the packet-to-flow match rate from 61% to 98.1%.
5. **Joint threshold search.** 24 x 19 threshold pairs are evaluated on validation, and the best pair is adopted.

## What didn't work (reported on purpose)

- **Evaluation leak, found and fixed.** The flow model was scoring flows it had trained on. Fixing it dropped its F1 from 0.695 to 0.483, and made the cascade's advantage larger.
- Two of three changes to the unsupervised stage gave no real gain. The winning detector's AUC is statistically the same as the baseline's.
- Flow aggregation made Brute Force detection harder, not easier.
- Results come from one split (seed 42). Variance across seeds was not measured.

## Repository structure

```
597-Group_7/
├── src/
│   ├── phase1_sampling/       download, sample and preprocess the dataset
│   ├── phase2_unsupervised/   stage 1 detectors and packet-level scoring
│   └── phase3_supervised/     flow reassembly, classifier training, alert re-check
├── notebooks/                 exploration and figures
├── results/                   metrics JSON, confusion matrices, ROC curves
├── report/                    final report and slides
├── AI_USAGE_LOG.md            record of AI tool use
└── requirements.txt
```

## How to run

The dataset (~5 GB) is not in the repo. It downloads to `~/ece597-data` on first run (change this with `ECE597_DATA` in `.env`).

```bash
git clone https://github.com/IRAIKUMARAN/597-Group_7.git
cd 597-Group_7
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your CIC cookie (see below)
```

<details>
<summary>Getting the CIC cookie for the dataset download</summary>

1. In Chrome, sign in to the CIC dataset portal and open the IoT dataset browse page.
2. Press F12, open the Network tab, refresh, and click `browse.php`.
3. Under Request Headers, copy the `Cookie:` value (looks like `Token=xxxxxxxx`).
4. Put it in `.env` as `CIC_COOKIE=Token=xxxxxxxx`. Never commit `.env`. If a download returns a registration form, the cookie expired, so grab a fresh one.

</details>

**Phase 0 + 1: download, sample, preprocess**
```bash
cd src/phase1_sampling
python run0_1.py                   # --skip-download, --seed 42, --no-verify
```
Outputs `packet_preprocessed.npz` (for Phase 2) and `packet_bookkeeping.csv` (for flow matching) in `~/ece597-data/samples/`.

**Phase 2: unsupervised packet screening**
```bash
cd ../phase2_unsupervised
python phase2_unsupervised.py
```
Outputs `phase2_metrics.json`, ROC and confusion-matrix plots, and `flagged_packet_ids.csv` (the alerts passed to Phase 3) in `results/`.

**Phase 3: supervised flow re-check**
```bash
cd ../phase3_supervised
python run3.py                     # build flows, preprocess, train, re-check alerts
```
Outputs `phase3_metrics.json`, `phase3_recheck.json` (the headline two-stage result) and the operating curve in `results/`. `analysis.txt` explains each step.

## Team (Group 7)

| Name | GitHub |
|---|---|
| Apoorva Rampal | @apoorvarampal123 |
| Arvind Sharma | @arvindxsharma |
| Irai Kumaran Sivanesan | @IRAIKUMARAN |
| Manivannan Usha Sundaresan | @Concorde-Supernovae |
| Muhammad Aarij | @aarij13406 |
| Nalluraj Babu | @NALLURAJ |
| Vinay Kumar Devarakonda | @VinayKumar-DataScience |

AI tool use is logged in `AI_USAGE_LOG.md`, as the course required.

## Dataset

[CIC IoT-DIAD 2024](https://www.unb.ca/cic/datasets/iot-diad-2024.html), Canadian Institute for Cybersecurity, University of New Brunswick.
