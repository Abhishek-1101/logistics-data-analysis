import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------- 1. Data collection (simulated delivery records) ----------
rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "order_id": range(1, n + 1),
    "distance_km": rng.uniform(2, 120, n).round(1),
    "load_kg": rng.uniform(5, 500, n).round(1),
    "stops": rng.integers(1, 12, n),
    "traffic_index": rng.uniform(0.8, 2.0, n).round(2),
})
df["delivery_hours"] = (0.9 * df.distance_km / 40 * df.traffic_index
                        + 0.15 * df.stops + rng.normal(0, 0.4, n)).clip(0.5)
df["cost_inr"] = 12 * df.distance_km + 0.8 * df.load_kg + 40 * df.stops
df["promised_hours"] = df.delivery_hours * rng.uniform(0.8, 1.3, n)
df.loc[rng.choice(n, 10, replace=False), "load_kg"] = np.nan   # missing values

# ---------- 2. Cleaning ----------
df["load_kg"] = df["load_kg"].fillna(df["load_kg"].median())
df = df.drop_duplicates("order_id")

# ---------- 3. KPIs ----------
df["on_time"] = df.delivery_hours <= df.promised_hours
kpis = {
    "On-time delivery rate (%)": round(df.on_time.mean() * 100, 1),
    "Avg cost per km (INR)": round((df.cost_inr / df.distance_km).mean(), 1),
    "Avg delivery time (hrs)": round(df.delivery_hours.mean(), 2),
}
print("KPIs:", kpis)

# ---------- 4. Exploratory analysis ----------
print(df[["distance_km", "load_kg", "delivery_hours", "cost_inr"]].corr().round(2))

# ---------- 5. Regression: predict delivery time ----------
X = df[["distance_km", "load_kg", "stops", "traffic_index"]]
y = df["delivery_hours"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)
model = LinearRegression().fit(X_tr, y_tr)
print("MAE (hours):", round(mean_absolute_error(y_te, model.predict(X_te)), 2))

# ---------- 6. Clustering: group deliveries into zones ----------
df["zone"] = KMeans(n_clusters=3, n_init=10, random_state=1).fit_predict(
    df[["distance_km", "stops"]])
print(df.groupby("zone")[["distance_km", "stops", "cost_inr"]].mean().round(1))

# ---------- 7. Optimization: nearest-neighbour route ----------
def nearest_neighbour(points):
    unvisited, route = list(range(1, len(points))), [0]
    while unvisited:
        last = points[route[-1]]
        nxt = min(unvisited, key=lambda i: np.linalg.norm(points[i] - last))
        route.append(nxt)
        unvisited.remove(nxt)
    return route

pts = rng.uniform(0, 50, (8, 2))           # depot + 7 customers (x, y in km)
route = nearest_neighbour(pts)
dist = sum(np.linalg.norm(pts[route[i]] - pts[route[i + 1]])
           for i in range(len(route) - 1))
print("Route:", route, "| total km:", round(dist, 1))
