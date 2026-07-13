"""Central config: where the raw dataset lives, and shared settings.

Loads the repo-root .env, then resolves the data location:
  - If the ECE597_DATA environment variable is set (e.g. in .env), use it.
  - Otherwise default to ~/ece597-data — your home folder on ANY OS
    (/Users/<you>/ece597-data on macOS, C:\\Users\\<you>\\ece597-data on Windows).

This keeps the (large) dataset OUTSIDE the repo, so it is never committed to git
and is not synced by OneDrive. The dataset itself is never committed.
"""
import os
from pathlib import Path


def _load_env():
    """Load KEY=VALUE lines from the repo-root .env into os.environ.
    No external dependency; skips comments and blank lines."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


_load_env()

# ~/ece597-data by default; override with the ECE597_DATA env var (see .env.example)
DEFAULT = Path.home() / "ece597-data"
DATA_ROOT = Path(os.environ.get("ECE597_DATA", DEFAULT)).expanduser()

RAW        = DATA_ROOT / "raw"                # download target (packet/ + flow/ live here)
RAW_PACKET = RAW / "packet"
RAW_FLOW   = RAW / "flow"
SAMPLES    = DATA_ROOT / "samples"            # generated datasets also stay local

# The session token for the CIC download portal (set CIC_COOKIE in .env).
COOKIE = os.environ.get("CIC_COOKIE", "")
