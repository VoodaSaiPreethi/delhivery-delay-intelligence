"""Shared constants and preprocessing for the Delay Intelligence pipeline.

Both training (Parts A-C) and scoring (predict_delay.py) use these two functions,
so that preprocessing is identical in training and in production.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100  # anything above this is a scanner fault (network max is 40 kg)


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians)."""
    df = raw.copy()
    # 1. exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # 2. service_type spelling
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    # 3. payment_mode placeholder
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    # 4. declared value text -> number
    dv = df["declared_value"].astype("string").str.replace("₹", "", regex=False)
    dv = dv.str.replace(",", "", regex=False).str.strip()
    df["declared_value_inr"] = pd.to_numeric(dv, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])
    # 5. dispatch time
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    # 6. weight sentinel -> missing
    df["parcel_weight_kg"] = pd.to_numeric(df["parcel_weight_kg"], errors="coerce").astype(float)
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").astype(float)
    # 7. median fill
    if medians is None:
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    else:
        medians = {c: float(medians[c]) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])
    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and engineered features. Never uses delivery_hours."""
    df = clean.merge(hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
                     on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
