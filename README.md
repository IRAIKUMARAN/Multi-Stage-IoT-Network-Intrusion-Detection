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

The system detects five attack types — **DDoS-HTTP Flood, DoS-HTTP Flood,
DNS Spoofing, XSS, Brute Force** — in two stages:

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
├── .env.example         # template for secrets/paths (copy to .env)
├── src/
│   ├── phase1_sampling/
│   │   ├── config.py           # data paths + .env loading (single source of truth)
│   │   ├── download_dataset.py # Phase 0 - fetch the CIC files
│   │   ├── generate_dataset.py # Task 1.1 - sampling function
│   │   ├── preprocess.py       # Task 1.2 - preprocessing pipeline
│   │   └── run0_1.py           # ONE-CLICK runner: Phase 0 + Phase 1
│   ├── phase2_unsupervised/    # Phase 2 - packet-level anomaly detection (WIP)
│   └── phase3_supervised/      # Phase 3 - flow-level classifier (WIP)
├── notebooks/           # profiling + preprocessing verification
├── results/             # figures, metrics, confusion matrices
└── report/              # written report + demo video link
```

Note: the dataset itself is **not** in the repo. It lives in `~/ece597-data`
(outside the repo, so it's never committed or cloud-synced). See "Running" below.

## Setup

```bash
# Python 3.11 recommended
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running (Phase 0 + Phase 1)

**One-click:** run `src/phase1_sampling/run0_1.py` (press Run in your editor, or
`python run0_1.py` from that folder). It runs, in order and stopping on any error:
download → sample (Task 1.1) → preprocess (Task 1.2) → verify.

First-time setup:

```bash
source .venv/bin/activate                 # activate your venv
cp .env.example .env                       # then edit .env and paste your CIC cookie
cd src/phase1_sampling
python run0_1.py                           # full pipeline
```

Handy flags: `--skip-download` (data already fetched), `--seed 42` (reproducible
sampling), `--no-verify`.

## Data & the .env file

The dataset is **not** committed (too large). It downloads to `~/ece597-data`
by default; override by setting `ECE597_DATA` in your `.env`. The CIC portal needs
a login token — copy your browser cookie into `.env` as `CIC_COOKIE=Token=...`
(the real `.env` is git-ignored; `.env.example` shows the format). Cookies expire,
so refresh it if downloads start returning the registration form. See
`data/README.md` for which files to grab.

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
