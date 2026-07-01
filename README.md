# ECE 597 Capstone — Multi-Stage IoT Network Intrusion Detection

Using **unsupervised + supervised learning** to detect network attacks in the
CIC IoT-DIAD 2024 dataset. Course project for ECE 597 (Summer 2026),
supervised by Dr. Ardeshir Shojaeinasab & Riham AlTawy.

## Team

| Name                       | GitHub                  |
| -------------------------- | ----------------------- |
| Apoorva Rampal             | @apoorvarampal123       |
| Arvind Sharma              | @arvindxsharma          |
| Irai Kumaran Sivanesan     | @iraikumaranUvic        |
| Manivannan Usha Sundaresan | mvannan@uvic.ca         |
| Muhammad Aarij             | @aarij13406             |
| Naluraj Babu               | @NALLURAJ               |
| Vinay Kumar Devarakonda    | @VinayKumar-DataScience |

## Project overview

The system detects five attack types :

**1. DDoS-HTTP Flood**
**2. DoS-HTTP Flood**
**3. DNS Spoofing**
**4. XSS**
**5. Brute Force — in two stages:**

- **Phase 2 (unsupervised, packet-level):** real-time anomaly detection. Learns
  "normal" traffic and flags outliers. No labels used in training.
- **Phase 3 (supervised, flow-level):** re-checks Phase 2's alerts to reduce false
  positives. Labels used for training only, never at test time.

## Repository structure

```
597-Group_7/
├── README.md            # this file
├── requirements.txt     # dependencies (pick ONE DL framework)
├── AI_USAGE_LOG.md      # log every AI tool use (required)
├── .gitignore           # excludes large dataset files
├── data/
│   ├── raw/             # downloaded CSVs (NOT committed)
│   └── processed/       # sampled/preprocessed data (NOT committed)
├── src/
│   ├── sampling.py             # Task 1.1 - dataset generation
│   ├── phase1_preprocessing.py # Task 1.2 - preprocessing
│   ├── phase2_unsupervised.py# Phase 2 - packet-level anomaly detection
│   ├── phase3_supervised.py  # Phase 3 - flow-level classifier
│   └── utils.py              # shared helpers
├── notebooks/           # exploration & visualizations
├── results/             # figures, metrics, confusion matrices
└── report/              # written report + demo video link
```

## Setup

```bash
# Python 3.11 recommended
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Dataset files are **not** in this repo (too large). See `data/README.md` for the
download link and which files to grab. Store the data in a shared team drive.

## Git workflow

- Commit **small and often**; each member commits from **their own account** so
  contributions are visible (this is graded).
- Push regularly — do not save all work for the final 48 hours.
- Branch convention: **small, frequent direct commits to `main`.** Pull before you
  push (`git pull --rebase`) to avoid conflicts.

## AI tools

Chatbots (ChatGPT, Claude, etc.) are allowed **with citation** — log every use in
`AI_USAGE_LOG.md`. **Autonomous coding agents are not allowed.** You must be able to
explain everything you submit.
