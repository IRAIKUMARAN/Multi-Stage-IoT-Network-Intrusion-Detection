import sys
from pathlib import Path
# make src/phase1_sampling importable from the notebooks folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "phase1_sampling"))

import pandas as pd
from config import SAMPLES

df = pd.read_csv(SAMPLES / "packet_sample.csv", low_memory=False)
print(df.shape)
print(df.dtypes.value_counts())
print(df.select_dtypes('object').columns.tolist())
na = df.isna().sum()
print(na[na > 0].sort_values(ascending=False).head(30))
print(df['attack_type'].value_counts())