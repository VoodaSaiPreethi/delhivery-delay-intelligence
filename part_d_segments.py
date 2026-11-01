"""Part D - hub profile, K-Means segments, separation evidence and tier check."""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile (ascending hub_id)
profile = (df.groupby("hub_id")
             .agg(shipments=("is_delayed", "size"),
                  mean_delivery_hours=("delivery_hours", "mean"),
                  delay_rate=("is_delayed", "mean"),
                  mean_distance_km=("distance_km", "mean"),
                  mean_n_stops=("n_stops", "mean"))
             .reset_index().sort_values("hub_id").reset_index(drop=True))
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. scale and cluster (hubs in ascending hub_id order)
PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[PROFILE_COLS])
km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)


# 3. separation evidence
def across_over_within(Z, labels):
    across, within = [], []
    for i, j in combinations(range(len(Z)), 2):
        d = float(np.linalg.norm(Z[i] - Z[j]))
        (within if labels[i] == labels[j] else across).append(d)
    return np.mean(across) / np.mean(within)


sep_rows = []
for k in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    sizes = sorted(np.bincount(lab), reverse=True)
    sep_rows.append({"K": k, "across_over_within": round(across_over_within(Z, lab), 4),
                     "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

# 4. name segments by mean delay rate and compare with tier labels
profile["cluster"] = km3.labels_
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/hub_segments.csv", index=False)

seg_profile = (seg.groupby("segment")
                  .agg(hubs=("hub_id", "size"),
                       mean_delivery_hours=("mean_delivery_hours", "mean"),
                       delay_rate=("delay_rate", "mean"),
                       mean_distance_km=("mean_distance_km", "mean"),
                       mean_n_stops=("mean_n_stops", "mean"))
                  .reindex(["Standard", "Watch", "Escalate"]).reset_index())
seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

TIER_IMPLIES = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = seg[seg["segment"] != seg["hub_tier"].map(TIER_IMPLIES)]
mismatch[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv(
    "outputs/tier_vs_segment_mismatch.csv", index=False)

print("Part D done.")
print(pd.DataFrame(sep_rows).to_string(index=False))
print(mismatch[["hub_id", "hub_city", "hub_tier", "segment"]].to_string(index=False))
