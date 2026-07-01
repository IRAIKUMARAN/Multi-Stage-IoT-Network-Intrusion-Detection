Please follow these guidlines for best commits

# data/

DO NOT COMMIT ANY DATA FILES - they are large and excluded via `.gitignore`.

# On your local system

- `raw/` — original CSVs downloaded from CIC IoT-DIAD 2024 (packet + flow, all 5 attack types + benign)

- `processed/` — output of the sampling/preprocessing steps

## Find the dataset from :

CIC IoT-DIAD 2024: https://www.unb.ca/cic/datasets/iot-diad-2024.html
Get BOTH feature sets for each of the 5 attack types AND benign traffic:

- `DI_AD_Packet-based-features` -> packet-level (Phase 2, unsupervised)
- `AD_Flow-based-features` -> flow-level (Phase 3, supervised)

Attack file rules (from the drop-in Q&A): use HTTP Flood (not TCP Flood) for
DoS/DDoS; prefer TCP over UDP; for near-duplicate `...flood`/`...flood1` files,
either is fine.

Keep the actual data in a shared Drive/cloud bucket for the team, never in Git. (We can discuss this later meeting where we can store the processed dataset for common access point)
