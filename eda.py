# === STEP 1: Simulate the logistics dataset ===
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", font_scale=0.95)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
rng = np.random.default_rng(21)
n = 2400

hubs = ["Ahmedabad", "Surat", "Mumbai", "Pune", "Delhi"]
df = pd.DataFrame({
    "shipment_id": np.arange(1, n + 1),
    "date": pd.to_datetime("2025-11-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "hub": rng.choice(hubs, n, p=[0.25, 0.2, 0.25, 0.15, 0.15]),
    "vehicle_type": rng.choice(["Truck", "Mini Truck", "Van"], n, p=[0.4, 0.35, 0.25]),
    "distance_km": rng.gamma(3.0, 130, n).clip(15, 1500).round(1),
})
df["month"] = df.date.dt.month
df["weight_kg"] = np.where(df.vehicle_type == "Truck", rng.uniform(1500, 5000, n),
                  np.where(df.vehicle_type == "Mini Truck", rng.uniform(500, 2000, n),
                           rng.uniform(50, 800, n))).round(0)
df["volume_cbm"] = (df.weight_kg / rng.uniform(150, 250, n)).round(1)

# cost drivers: distance, weight, vehicle type, festive-season surcharge
rate = df.vehicle_type.map({"Truck": 11, "Mini Truck": 14, "Van": 19})
festive = df.month.isin([10, 11, 12]).astype(int)
df["freight_cost"] = (df.distance_km * rate + df.weight_kg * 0.6
                      + festive * 1200 + rng.normal(0, 700, n)).clip(300).round(0)

# delivery time: distance, festive peak, congested Mumbai hub (built-in bottleneck)
hub_delay = df.hub.map({"Ahmedabad": 0, "Surat": 0.1, "Mumbai": 0.9, "Pune": 0.2, "Delhi": 0.4})
df["delivery_days"] = (df.distance_km / 420 + 1 + festive * 0.6 + hub_delay
                       + rng.normal(0, 0.35, n)).clip(0.5).round(1)
df["promised_days"] = (df.distance_km / 420 + 1.8).round(1)
df["delay_days"] = (df.delivery_days - df.promised_days).round(1)
df["on_time"] = (df.delay_days <= 0).astype(int)
df["cost_per_km"] = (df.freight_cost / df.distance_km).round(2)
print(df.head(3).to_string())
print("Shape:", df.shape, "| missing values:", int(df.isna().sum().sum()))

# === STEP 2: Descriptive statistics (central tendency and spread) ===
num = ["distance_km", "weight_kg", "delivery_days", "freight_cost", "cost_per_km"]
stats = df[num].agg(["mean", "median", "std", "min", "max", "skew"]).round(2)
print("\n", stats)
print("\nMode of vehicle type:", df.vehicle_type.mode()[0])
print("Overall on-time rate: %.1f%%" % (df.on_time.mean() * 100))

# === STEP 3: Correlation analysis ===
corr = df[["distance_km", "weight_kg", "volume_cbm", "delivery_days", "freight_cost", "delay_days"]].corr().round(2)
print("\n", corr)

# === STEP 4: Group analysis (efficiency, cost drivers, bottlenecks) ===
print("\nBy vehicle type:\n", df.groupby("vehicle_type")[["freight_cost", "cost_per_km", "delivery_days"]].mean().round(2))
by_hub = df.groupby("hub").agg(shipments=("shipment_id", "count"), avg_days=("delivery_days", "mean"),
                              avg_delay=("delay_days", "mean"), on_time_pct=("on_time", "mean")).round(2)
by_hub["on_time_pct"] = (by_hub.on_time_pct * 100).round(1)
print("\nBy hub:\n", by_hub.sort_values("on_time_pct"))
monthly = df.groupby("month").agg(shipments=("shipment_id", "count"), avg_cost=("freight_cost", "mean"),
                                  avg_days=("delivery_days", "mean"), on_time_pct=("on_time", "mean")).round(2)
monthly["on_time_pct"] = (monthly.on_time_pct * 100).round(1)
print("\nBy month:\n", monthly)

# === STEP 5: Chart 1 - Distribution of delivery time (histogram + KDE) ===
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df.delivery_days, bins=30, kde=True, color="#2E5597", ax=ax)
ax.axvline(df.delivery_days.mean(), color="red", ls="--", label=f"Mean = {df.delivery_days.mean():.2f}")
ax.axvline(df.delivery_days.median(), color="green", ls=":", label=f"Median = {df.delivery_days.median():.2f}")
ax.set(title="Distribution of Delivery Time", xlabel="Delivery time (days)", ylabel="Number of shipments")
ax.legend(); plt.tight_layout(); plt.savefig("chart1_delivery_hist.png", dpi=150); plt.close()

# === STEP 6: Chart 2 - Monthly trend (line chart, dual axis) ===
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.bar(monthly.index, monthly.shipments, color="#B7C9E8", label="Shipments")
ax1.set(xlabel="Month", ylabel="Number of shipments", title="Monthly Shipment Volume and Average Freight Cost")
ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly.avg_cost, color="#C0392B", marker="o", label="Avg freight cost")
ax2.set_ylabel("Average freight cost (INR)"); ax2.grid(False)
fig.legend(loc="lower center", ncol=2)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig("chart2_monthly_trend.png", dpi=150); plt.close()

# === STEP 7: Chart 3 - Cost per km by vehicle type (box plot) ===
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="vehicle_type", y="cost_per_km", hue="vehicle_type", palette="Blues", legend=False, ax=ax)
ax.set(title="Cost per Kilometre by Vehicle Type", xlabel="Vehicle type", ylabel="Cost per km (INR)")
plt.tight_layout(); plt.savefig("chart3_cost_boxplot.png", dpi=150); plt.close()

# === STEP 8: Chart 4 - Correlation heatmap ===
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f", ax=ax)
ax.set_title("Correlation Between Key Variables")
plt.tight_layout(); plt.savefig("chart4_corr_heatmap.png", dpi=150); plt.close()

# === STEP 9: Chart 5 - Distance vs cost (scatter with trend line) ===
fig, ax = plt.subplots(figsize=(7, 4.4))
sns.scatterplot(data=df.sample(900, random_state=1), x="distance_km", y="freight_cost",
                hue="vehicle_type", alpha=0.6, ax=ax)
sns.regplot(data=df, x="distance_km", y="freight_cost", scatter=False, color="black", line_kws={"lw": 1.5}, ax=ax)
ax.set(title="Freight Cost vs Distance", xlabel="Distance (km)", ylabel="Freight cost (INR)")
plt.tight_layout(); plt.savefig("chart5_distance_cost.png", dpi=150); plt.close()

# === STEP 10: Chart 6 - On-time delivery rate by hub (bar chart) ===
fig, ax = plt.subplots(figsize=(7, 4))
order = by_hub.sort_values("on_time_pct")
colors = ["#C0392B" if v < 70 else "#2E5597" for v in order.on_time_pct]
ax.barh(order.index, order.on_time_pct, color=colors)
ax.axvline(85, color="green", ls="--", label="Target 85%")
for i, v in enumerate(order.on_time_pct): ax.text(v + 1, i, f"{v}%", va="center")
ax.set(title="On-Time Delivery Rate by Hub", xlabel="On-time deliveries (%)", xlim=(0, 112))
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig("chart6_ontime_by_hub.png", dpi=150); plt.close()

# === STEP 11: Chart 7 - Average delay by hub and month (heatmap) ===
pivot = df.pivot_table(index="hub", columns="month", values="delay_days", aggfunc="mean").round(2)
fig, ax = plt.subplots(figsize=(8, 3.8))
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax)
ax.set(title="Average Delay (days) by Hub and Month", xlabel="Month", ylabel="")
plt.tight_layout(); plt.savefig("chart7_delay_heatmap.png", dpi=150); plt.close()
print("\nAll charts saved.")

# === STEP 12: Extra numbers used in the insights ===
peak = df.month.isin([10, 11, 12])
print("\nAvg cost Jan-Sep: %.0f | Oct-Dec: %.0f | increase: %.1f%%" % (
    df[~peak].freight_cost.mean(), df[peak].freight_cost.mean(),
    (df[peak].freight_cost.mean() / df[~peak].freight_cost.mean() - 1) * 100))
print("On-time Jan-Sep: %.1f%% | Oct-Dec: %.1f%%" % (df[~peak].on_time.mean() * 100, df[peak].on_time.mean() * 100))
late = df[df.on_time == 0]
print("Share of all late shipments from Mumbai: %.1f%%" % (late.hub.eq("Mumbai").mean() * 100))
print("On-time rate excluding Mumbai: %.1f%%" % (df[df.hub != "Mumbai"].on_time.mean() * 100))
print("Cost per km, short (<150 km) vs long (>600 km): %.1f vs %.1f" % (
    df[df.distance_km < 150].cost_per_km.mean(), df[df.distance_km > 600].cost_per_km.mean()))
