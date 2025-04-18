# data_loader.py
import os
import pandas as pd

# ── locate the data directory ─────────────────────────────────
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# ── full paths to your CSVs ────────────────────────────────────
TEMP_CSV   = os.path.join(DATA_DIR, 'Temperature_Data.csv')
MERGED_CSV = os.path.join(DATA_DIR, 'merged_co2_temp.csv')
GIS_CSV    = os.path.join(DATA_DIR, 'gistemp.csv')    # adjust if your GIS filename differs

# ── load each file exactly once ───────────────────────────────
df_temp   = pd.read_csv(TEMP_CSV)
df_merged = pd.read_csv(MERGED_CSV)
df_gis    = pd.read_csv(GIS_CSV)

# ── normalize column names to lowercase to avoid typos ─────────
df_temp.columns   = df_temp.columns.str.lower()
df_merged.columns = df_merged.columns.str.lower()
df_gis.columns    = df_gis.columns.str.lower()

# ── loader functions ──────────────────────────────────────────
def load_temperature_data():
    x = df_temp['year'].to_numpy()
    y = df_temp['temperature_anomaly'].to_numpy()
    return x, y

def load_co2_data():
    x = df_merged['year'].to_numpy()
    # replace 'co2' with whatever your merged CSV calls it (e.g. 'co2_ppm')
    y = df_merged['temperature_anomaly'].to_numpy()
    return x, y

def load_gis_data():
    x = df_gis['year'].to_numpy()
    y = df_gis['temperature_anomaly'].to_numpy()
    return x, y
