"""Central config: where the raw dataset lives on THIS machine.
Each teammate sets the ECE597_DATA environment variable, or edits DEFAULT.
The dataset itself is never committed to git."""
import os
from pathlib import Path

DEFAULT = r"C:\ece597-data"
DATA_ROOT = Path(os.environ.get("ECE597_DATA", DEFAULT))

RAW_PACKET = DATA_ROOT / "raw" / "packet"
RAW_FLOW   = DATA_ROOT / "raw" / "flow"
SAMPLES    = DATA_ROOT / "samples"          # generated datasets also stay local