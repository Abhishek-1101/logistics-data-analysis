# === STEP 1: Data collection (simulated shipment records) ===
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n = 1000
cities = ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Mumbai", "Pune", "Jaipur", "Delhi"]
raw = pd.DataFrame({
    "shipment_id": np.arange(1, n + 1),
    "origin_city": rng.choice(cities, n),
    "dest_city": rng.choice(cities, n),
    "vehicle_type": rng.choice(["Truck", "Mini Truck", "Van"], n, p=[0.4, 0.35, 0.25]),
    "ship_date": pd.date_range("2026-01-01", periods=n, freq="8h").strftime("%Y-%m-%d"),
    "distance_km": rng.uniform(20, 1200, n).round(1),
    "weight_kg": rng.uniform(50, 5000, n).round(1),
})
raw["transit_days"] = (raw.distance_km / 450 + rng.normal(1, 0.3, n)).clip(0.5).round(1)
raw["freight_cost"] = (raw.distance_km * 14 + raw.weight_kg * 0.9 + rng.normal(0, 800, n)).round(0)

# --- inject realistic data quality problems ---
raw.loc[rng.choice(n, 60, replace=False), "weight_kg"] = np.nan
raw.loc[rng.choice(n, 45, replace=False), "transit_days"] = np.nan
raw.loc[rng.choice(n, 30, replace=False), "vehicle_type"] = np.nan
raw.loc[rng.choice(n, 25, replace=False), "origin_city"] += " "          # trailing space
raw.loc[rng.choice(n, 25, replace=False), "dest_city"] = raw.dest_city.str.upper()
raw.loc[rng.choice(n, 8, replace=False), "distance_km"] = 99999          # sensor error
raw.loc[rng.choice(n, 6, replace=False), "freight_cost"] *= 15            # extreme cost
raw.loc[rng.choice(n, 5, replace=False), "transit_days"] = -2             # impossible value
raw.loc[rng.choice(n, 4, replace=False), "ship_date"] = "31/02/2026"      # invalid date
raw = pd.concat([raw, raw.sample(20, random_state=1)], ignore_index=True)  # duplicate rows

print("Raw shape:", raw.shape)

# === STEP 2: Data inspection ===
print(raw.dtypes)
print("\nMissing values per column:\n", raw.isna().sum())
print("\nDuplicate rows:", raw.duplicated().sum())
print("\nSummary statistics:\n", raw[["distance_km", "weight_kg", "transit_days", "freight_cost"]].describe().round(1))
cost_before = raw["freight_cost"].copy()

# === STEP 3: Remove duplicates ===
df = raw.drop_duplicates().copy()
print("\nAfter removing duplicates:", df.shape)

# === STEP 4: Fix text and date formats ===
for col in ["origin_city", "dest_city"]:
    df[col] = df[col].str.strip().str.title()
df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")   # invalid -> NaT
print("Invalid dates found:", df["ship_date"].isna().sum())
df = df.dropna(subset=["ship_date"])
print("Unique cities now:", sorted(df.origin_city.unique()))

# === STEP 5: Handle impossible values ===
bad_days = (df.transit_days <= 0).sum()
bad_dist = (df.distance_km > 3000).sum()
bad_cost = (df.freight_cost <= 0).sum()
df.loc[df.transit_days <= 0, "transit_days"] = np.nan
df.loc[df.distance_km > 3000, "distance_km"] = np.nan
df.loc[df.freight_cost <= 0, "freight_cost"] = np.nan
print(f"Impossible transit_days: {bad_days}, distance_km: {bad_dist}, freight_cost: {bad_cost} -> set to NaN")

# === STEP 6: Handle missing values ===
print("\nMissing before imputation:\n", df.isna().sum()[lambda s: s > 0])
df["vehicle_type"] = df["vehicle_type"].fillna(df["vehicle_type"].mode()[0])
df["weight_kg"] = df["weight_kg"].fillna(df.groupby("vehicle_type")["weight_kg"].transform("median"))
df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
df["transit_days"] = df["transit_days"].fillna(df["transit_days"].median())
df["freight_cost"] = df["freight_cost"].fillna(df["freight_cost"].median())
print("Missing after imputation:", int(df.isna().sum().sum()))

# === STEP 7: Detect and treat outliers (IQR method) ===
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

for col in ["freight_cost", "weight_kg", "distance_km", "transit_days"]:
    lo, hi = iqr_bounds(df[col])
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)          # winsorise (cap) instead of deleting rows
    print(f"{col}: {n_out} outliers capped to [{lo:.1f}, {hi:.1f}]")

# === STEP 8: Feature engineering ===
df["cost_per_km"] = (df.freight_cost / df.distance_km).round(2)
df["month"] = df.ship_date.dt.month

# === STEP 9: Normalisation / scaling ===
num_cols = ["distance_km", "weight_kg", "transit_days", "freight_cost"]
for c in num_cols:
    df[c + "_minmax"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())   # 0 to 1
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()                        # mean 0, std 1
print("\nMin-max range check:", df["freight_cost_minmax"].min(), "to", df["freight_cost_minmax"].max())
print("Z-score check: mean =", round(df["freight_cost_z"].mean(), 3), ", std =", round(df["freight_cost_z"].std(), 3))

# === STEP 10: Encode categorical variables ===
df = pd.get_dummies(df, columns=["vehicle_type"], prefix="veh", dtype=int)

# === STEP 11: Validate and save ===
assert df.isna().sum().sum() == 0 and df.duplicated().sum() == 0
df.to_csv("clean_logistics_data.csv", index=False)
print("\nFinal clean shape:", df.shape)
print(df.describe().loc[["min", "max"], num_cols].round(1))

# --- before/after chart ---
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
ax[0].boxplot(cost_before.dropna()); ax[0].set_title("Freight cost - before")
ax[1].boxplot(df["freight_cost"]);    ax[1].set_title("Freight cost - after")
for a in ax: a.set_ylabel("INR")
plt.tight_layout(); plt.savefig("before_after.png", dpi=150)
