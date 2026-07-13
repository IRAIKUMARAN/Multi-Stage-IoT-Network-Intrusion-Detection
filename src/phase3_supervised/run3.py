"""
One-click runner for Phase 3: build flow dataset -> preprocess -> train -> re-check.
Stops immediately if any step fails.
Run: python run3.py            (needs Phase 1 outputs + Phase 2 flagged_packet_ids.csv)
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("Task 3.2  -  Build flow dataset", "build_flow_dataset.py"),
    ("Preprocess flow features",        "preprocess_flow.py"),
    ("Task 3.3  -  Train supervised model", "train_supervised.py"),
    ("Task 3.1  -  Flow anomaly + novelty", "flow_anomaly.py"),
    ("Re-check Phase 2 alerts",         "recheck_phase2_alerts.py"),
]


def run(title, script):
    """Run one Phase 3 script as a subprocess; abort the pipeline if it fails."""
    print("\n" + "=" * 72 + f"\n>>> {title}\n" + "=" * 72, flush=True)
    if subprocess.run([PY, script], cwd=str(HERE)).returncode != 0:
        sys.exit(f"\n[STOPPED] '{title}' failed. Fix the error above and re-run.")


if __name__ == "__main__":
    for title, script in STEPS:
        run(title, script)
    print("\nPhase 3 complete. See results/phase3_metrics.json and results/phase3_recheck.json")
