"""
data_loader.py
--------------
Generic CSV loader driven by *datasets.json*.
Keeps the legacy helpers (load_temperature_data, …) so no downstream file breaks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np


ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "data"
MANIFEST  = json.loads((ROOT_DIR / "datasets.json").read_text(encoding="utf-8"))

_DEFAULT_X, _DEFAULT_Y = MANIFEST.get("default_columns", ["year", "temperature_anomaly"])
_DATASETS              = MANIFEST["datasets"]
_COLUMN_OVERRIDES      = MANIFEST.get("columns", {})        # optional


def load_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for *name* defined in the manifest."""
    if name not in _DATASETS:
        raise KeyError(f"{name!r} not found in datasets.json")

    csv_file = DATA_DIR / _DATASETS[name]

    xcol, ycol = _COLUMN_OVERRIDES.get(name, [_DEFAULT_X, _DEFAULT_Y])
    df = pd.read_csv(csv_file).rename(str.lower, axis=1)

    return df[xcol].to_numpy(), df[ycol].to_numpy()


# ------------------------------------------------------------------ #
#  Convenience one-liner accessors to preserve old API
# ------------------------------------------------------------------ #
globals().update({
    f"load_{k}_data": (lambda key=k: load_data(key))          # default-arg “key” freezes the loop variable
    for k in _DATASETS
})
