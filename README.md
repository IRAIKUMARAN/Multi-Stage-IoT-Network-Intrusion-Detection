# Multi-Stage IoT Network Intrusion Detection

Using **unsupervised + supervised learning** to detect network attacks in the
CIC IoT-DIAD 2024 dataset. Course project for ECE 597 (Summer 2026),
supervised by Dr. Ardeshir Shojaeinasab & Riham AlTawy.

## Team

| Name                       | GitHub                  |
| -------------------------- | ----------------------- |
| Apoorva Rampal             | @apoorvarampal123       |
| Arvind Sharma              | @arvindxsharma          |
| Irai Kumaran Sivanesan     | @iraikumaranUvic        |
| Manivannan Usha Sundaresan | @Concorde-Supernovae    |
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
│   ├── phase2_unsupervised/
│   │   └── phase2_unsupervised.py  # Phase 2 - packet-level anomaly detection
│   └── phase3_supervised/
│       ├── build_flow_dataset.py   # Task 3.2 - build the flow dataset
│       ├── preprocess_flow.py      # clean the flow features
│       ├── train_supervised.py     # Task 3.3 - supervised classifier
│       ├── flow_anomaly.py         # Task 3.1 + novelty experiment
│       ├── recheck_phase2_alerts.py# two-stage re-check (headline result)
│       ├── run3.py                 # ONE-CLICK runner: all of Phase 3
│       └── analysis.txt            # full Phase 3 process write-up
├── notebooks/           # profiling + preprocessing verification
├── results/             # figures, metrics, confusion matrices
└── report/              # written report + demo video link
```

Note: the dataset itself is **not** in the repo. It lives in `~/ece597-data`
(outside the repo, so it's never committed or cloud-synced). See "Running" below.

## How to run (Phase 0 + Phase 1)

Runs the full pipeline: download → sample → preprocess → verify.
Works on macOS and Windows. Do the steps in order.

### Step 1 — Get the latest code

```bash
git pull
```

### Step 2 — Create & activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

You'll see `(.venv)` at the start of your prompt when it's active.

### Step 3 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4 — Get your CIC cookie and put it in `.env`

The dataset portal needs you to be logged in, so each person uses their **own** login token.

1. In Chrome, sign in and open the CIC dataset browse page (`cicresearch.ca/IOTDataset/...`); make sure the file list loads.
2. Press **F12** → **Network** tab.
3. **Refresh** the page, then click the row named **`browse.php`**.
4. Under **Request Headers**, find **`Cookie:`** and copy its full value (looks like `Token=xxxxxxxx`).
5. Copy the template and paste your token in:

```bash
cp .env.example .env               # then edit .env:  CIC_COOKIE=Token=xxxxxxxx
```

> ⚠️ Never commit `.env` — it's git-ignored on purpose. Cookies expire, so if a download later
> returns a "registration form", just redo Step 4 with a fresh cookie.

### Step 5 — Run the pipeline (one command)

```bash
cd src/phase1_sampling
python run0_1.py                   # download → sample → preprocess → verify
```

Handy flags:

```bash
python run0_1.py --skip-download   # data already downloaded; just re-sample + preprocess
python run0_1.py --seed 42         # reproducible sample (same mix every run)
python run0_1.py --no-verify       # skip the verification step
```

### Notes

The dataset (~5 GB) downloads to `~/ece597-data` — **outside** the repo, so it's never committed;
the folder is created automatically on first run (override the location by setting `ECE597_DATA` in
your `.env`). Outputs land in `~/ece597-data/samples/`: `packet_sample.csv`,
`packet_preprocessed.npz` (input for Phase 2), and `packet_bookkeeping.csv` (for Phase 3 flow
matching). See `data/README.md` for which files are fetched.

## Running Phase 2 (unsupervised, packet-level)

After Phase 1 has produced `packet_preprocessed.npz`:

```bash
cd src/phase2_unsupervised
python phase2_unsupervised.py
```

Outputs to `results/`: `phase2_metrics.json`, `phase2_confusion_matrix.png`,
`phase2_roc.png`, and `flagged_packet_ids.csv` (the alerts handed to Phase 3).

## Running Phase 3 (supervised, flow-level)

Needs the data in `~/ece597-data` and Phase 2's `results/flagged_packet_ids.csv`.
One command runs the whole stage:

```bash
cd src/phase3_supervised
python run3.py     # build flow dataset → preprocess → train → 3.1 → re-check
```

Outputs to `results/`: `phase3_metrics.json` (classifier scores),
`phase3_recheck.json` (the two-stage false-positive-reduction headline),
`phase3_operating_curve.json/.png` (the trade-off curve), `phase3_anomaly.json`
(Task 3.1), plus confusion-matrix and ROC plots. See `analysis.txt` in the
Phase 3 folder for the full write-up of what each step does and why.

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
