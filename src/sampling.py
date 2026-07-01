"""Task 1.1 - Random Dataset Generation Function.

Builds ONE sampled dataset per run:
  - ~200,000 benign rows        (97-98% of the sample)
  - 4,000-6,200 attack rows     (2-3% of the sample)
  - attacks spread ~uniformly across the 5 categories, chosen at random
  - configurable random seed  -> different composition each run
  - basic validation + integrity checks
"""

ATTACK_CATEGORIES = [
    "DDoS-HTTP_Flood",
    "DoS-HTTP_Flood",
    "DNS_Spoofing",
    "XSS",
    "Brute_Force",
]

# Basic code structure (this is temp), can be changed as needed.
def build_dataset(seed: int = 42, n_benign: int = 200_000, n_attack: int = 5_000):
    """Return a sampled dataframe. TODO: implement.

    Steps:
      1. Load benign source, randomly sample n_benign rows (seeded).
      2. For each of the 5 attack categories, sample ~n_attack/5 rows (seeded).
      3. Concatenate, shuffle, validate counts/ratios, return.
    """
    raise NotImplementedError


def validate(df):
    """Check row counts, benign/attack ratio (~97-98% / 2-3%), and per-class balance."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
