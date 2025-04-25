# data_loader.py
"""CSV loader helpers returns (x, y) NumPy arrays for the three datasets."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).with_suffix('').parent / 'data'   # ./data alongside this file

# dataset name(csv filename, xcolumn, ycolumn)
_FILES = {
    'temperature': ('Temperature_Data.csv', 'year', 'temperature_anomaly'),
    'co2':         ('merged_co2_temp.csv', 'year', 'temperature_anomaly'),  # rename cols if different
    'gis':         ('gistemp.csv',        'year', 'temperature_anomaly'),
    'long':        ('long.csv',          'year', 'temperature_anomaly')
}

def _load(key):
    """Internal: read CSV, lowercase headers, return requested columns as NumPy arrays."""
    csv, xcol, ycol = _FILES[key]
    df = pd.read_csv(DATA_DIR / csv).rename(str.lower, axis=1)
    return df[xcol].to_numpy(), df[ycol].to_numpy()

# public loaders
load_temperature_data = lambda: _load('temperature')
load_co2_data         = lambda: _load('co2')
load_gis_data         = lambda: _load('gis')
load_long_data        = lambda: _load('long')


