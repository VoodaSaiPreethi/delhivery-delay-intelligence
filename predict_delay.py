"""Score new parcels and turn them into an operations action list."""
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.shipment_prep import build_features, clean_shipments  # noqa: E402

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = dict(zip(imp["column"], imp["median"].astype(float)))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new, medians=medians)
feats = build_features(clean, hubs)
assert feats[feature_order].notna().all().all(), "missing values in model inputs"

out = feats[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = model.predict_proba(feats[feature_order])[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments, on="hub_id", how="left")


def action(row):
    if row["flagged"] != 1:
        return "No action"
    return "Escalate to hub manager" if row["segment"] == "Escalate" else "Proactive customer call"


out["action"] = out.apply(action, axis=1)
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
os.makedirs("outputs", exist_ok=True)
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(f"Scored {len(out)} parcels: {int(out['flagged'].sum())} flagged, "
      f"{int((out['action'] == 'Escalate to hub manager').sum())} escalations.")
