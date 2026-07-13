#!/usr/bin/env python3
"""
run0_1.py - one-click runner for Phase 0 (download) + Phase 1 (sample + preprocess).

Runs these steps in order, stopping immediately if any step fails:
  1. download_dataset.py --download   Phase 0 : fetch the project's CIC files
  2. generate_dataset.py              Task 1.1: build one sampled dataset
  3. preprocess.py                    Task 1.2: clean + scale it
  4. verify_preprocess.py             sanity checks (optional)

HOW TO RUN
  - Press the Run button in your editor, OR
  - Terminal:  python run0_1.py

FLAGS
  --skip-download   force-skip step 1 even if some files are missing
  --force-download  re-download even if the data is already present
  --seed N          reproducible sampling (default: different mix each run)
  --no-verify       skip the final verification step

By default the download step is skipped AUTOMATICALLY when all required data
files are already present in the data folder (no need to pass --skip-download).
"""
import argparse
import subprocess
import sys
from pathlib import Path

from config import RAW                             # data folder (~/ece597-data/raw)

HERE = Path(__file__).resolve().parent            # src/phase1_sampling
REPO = HERE.parents[1]                             # repo root
VERIFY = REPO / "notebooks" / "verify_preprocess.py"
PY = sys.executable                                # the same Python running this file

# Packet CSVs the sampler needs. If all are found under RAW, the download is skipped.
REQUIRED_FILES = [
    "BenignTraffic.csv", "DDoS-HTTP_Flood-.csv", "DoS-HTTP_Flood.csv",
    "DNS_Spoofing.csv", "XSS.csv", "DictionaryBruteForce.csv",
]


def data_present():
    """True only if every required packet CSV already exists somewhere under RAW."""
    if not RAW.exists():
        return False
    missing = [f for f in REQUIRED_FILES if not any(RAW.rglob(f))]
    if missing:
        print(f"Missing {len(missing)} data file(s), e.g. {missing[:3]} -> will download.")
        return False
    return True


def run(title, args, cwd=HERE):
    """Run one script as a subprocess; abort the whole pipeline if it fails."""
    print("\n" + "=" * 72)
    print(f">>> {title}")
    print("=" * 72, flush=True)
    result = subprocess.run([PY, *args], cwd=str(cwd))
    if result.returncode != 0:
        sys.exit(f"\n[STOPPED] '{title}' failed (exit code {result.returncode}). "
                 f"Fix the error shown above, then run again.")


def main():
    ap = argparse.ArgumentParser(description="Run Phase 0 + Phase 1 end to end.")
    ap.add_argument("--skip-download", action="store_true",
                    help="Force-skip the download step even if files are missing.")
    ap.add_argument("--force-download", action="store_true",
                    help="Download even if the data is already present.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for sampling. Omit for a different mix each run.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the verification step.")
    args = ap.parse_args()

    # 1. Phase 0 - download (auto-skipped when the data is already there)
    if args.skip_download:
        print("Skipping download (--skip-download).")
    elif data_present() and not args.force_download:
        print(f"Dataset already present in {RAW} - skipping download.")
    else:
        run("Phase 0  -  Download dataset", ["download_dataset.py", "--download"])

    # 2. Task 1.1 - generate sampled dataset
    gen = ["generate_dataset.py"]
    if args.seed is not None:
        gen += ["--seed", str(args.seed)]
    run("Task 1.1  -  Generate sampled dataset", gen)

    # 3. Task 1.2 - preprocess
    run("Task 1.2  -  Preprocess", ["preprocess.py"])

    # 4. Verify (optional)
    if not args.no_verify and VERIFY.exists():
        run("Verify  -  Preprocessing sanity checks", [str(VERIFY)])

    print("\n" + "=" * 72)
    print("ALL DONE. Outputs are in your data folder (see config.DATA_ROOT):")
    print("  - samples/packet_sample.csv")
    print("  - samples/packet_preprocessed.npz  (feature matrix for Phase 2)")
    print("  - samples/packet_bookkeeping.csv   (IPs/ports for Phase 3 flow matching)")
    print("=" * 72)


if __name__ == "__main__":
    main()
