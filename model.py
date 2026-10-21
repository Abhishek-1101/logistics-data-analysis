# === STEP 1: Problem definition and data simulation ===
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid", font_scale=0.95)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
rng = np.random.default_rng(42)
n = 3000
hubs = ["Ahmedabad", "Surat", "Mumbai", "Pune", "Delhi"]
df = pd.DataFrame({
    "hub": rng.choice(hubs, n, p=[0.25, 0.2, 0.25, 0.15, 0.15]),
    "vehicle_type": rng.choice(["Truck", "Mini Truck", "Van"], n, p=[0.4, 0.35, 0.25]),
    "distance_km": rng.gamma(3.0, 130, n).clip(15, 1500).round(1),
    "month": rng.integers(1, 13, n),
})
df["weight_kg"] = np.where(df.vehicle_type == "Truck", rng.uniform(1500, 5000, n),
                  np.where(df.vehicle_type == "Mini Truck", rng.uniform(500, 2000, n),
                           rng.uniform(50, 800, n))).round(0)
df["stops"] = rng.integers(1, 8, n)
df["festive"] = df.month.isin([10, 11, 12]).astype(int)
hub_delay = df.hub.map({"Ahmedabad": 0, "Surat": 0.1, "Mumbai": 0.9, "Pune": 0.2, "Delhi": 0.4})
# target: delivery time in days (non-linear effects: festive x Mumbai congestion, stops)
df["delivery_days"] = (df.distance_km / 420 + 1 + df.festive * 0.6 + hub_delay
                       + 0.08 * df.stops + df.festive * (df.hub == "Mumbai") * 0.5
                       + rng.normal(0, 0.3, n)).clip(0.5).round(2)
df["promised_days"] = (df.distance_km / 420 + 1.8).round(2)
print("Dataset shape:", df.shape)
print(df.drop(columns="promised_days").describe().round(2).T[["mean", "std", "min", "max"]])

# === STEP 2: Data preparation (features, target, train/test split) ===
features = ["hub", "vehicle_type", "distance_km", "weight_kg", "stops", "month", "festive"]
X, y = df[features], df["delivery_days"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cat_cols = ["hub", "vehicle_type"]
num_cols = ["distance_km", "weight_kg", "stops", "month", "festive"]
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                          ("num", StandardScaler(), num_cols)])
print("Train rows:", len(X_train), "| Test rows:", len(X_test))

# === STEP 3: Train and compare models (5-fold cross-validation) ===
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    pipe = Pipeline([("prep", prep), ("model", m)])
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    rows.append([name, cv_rmse.mean(), cv_rmse.std(), cv_r2.mean()])
cv_table = pd.DataFrame(rows, columns=["Model", "CV RMSE", "CV RMSE std", "CV R2"]).round(3)
print("\n5-fold cross-validation on training data:\n", cv_table.to_string(index=False))

# === STEP 4: Hyperparameter tuning (GridSearchCV) ===
pipe = Pipeline([("prep", prep), ("model", GradientBoostingRegressor(random_state=42))])
grid = {"model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.05, 0.1]}
search = GridSearchCV(pipe, grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)
print("\nBest parameters:", search.best_params_)
print("Best CV RMSE: %.3f" % -search.best_score_)

# === STEP 5: Final evaluation on the unseen test set ===
def evaluate(name, model):
    p = model.predict(X_test)
    return [name, mean_absolute_error(y_test, p), np.sqrt(mean_squared_error(y_test, p)), r2_score(y_test, p)]
final = []
for name, m in models.items():
    final.append(evaluate(name, Pipeline([("prep", prep), ("model", m)]).fit(X_train, y_train)))
best = search.best_estimator_
final.append(evaluate("Tuned Gradient Boosting", best))
test_table = pd.DataFrame(final, columns=["Model", "MAE (days)", "RMSE (days)", "R2"]).round(3)
print("\nTest-set performance:\n", test_table.to_string(index=False))
pred = best.predict(X_test)

# === STEP 6: Chart 1 - model comparison ===
fig, ax = plt.subplots(figsize=(7, 3.8))
order = test_table.sort_values("RMSE (days)", ascending=False)
ax.barh(order.Model, order["RMSE (days)"], color=["#2E5597" if "Tuned" not in m else "#C0392B" for m in order.Model])
for i, v in enumerate(order["RMSE (days)"]): ax.text(v + 0.005, i, f"{v:.3f}", va="center")
ax.set(title="Test RMSE by Model (lower is better)", xlabel="RMSE (days)", xlim=(0, order["RMSE (days)"].max() * 1.15))
plt.tight_layout(); plt.savefig("chart1_model_comparison.png", dpi=150); plt.close()

# === STEP 7: Chart 2 - actual vs predicted ===
fig, ax = plt.subplots(figsize=(5.8, 5))
ax.scatter(y_test, pred, alpha=0.4, s=14, color="#2E5597")
lim = [0.5, max(y_test.max(), pred.max()) + 0.2]
ax.plot(lim, lim, "r--", label="Perfect prediction")
ax.set(title="Actual vs Predicted Delivery Time", xlabel="Actual (days)", ylabel="Predicted (days)")
ax.legend(); plt.tight_layout(); plt.savefig("chart2_actual_vs_pred.png", dpi=150); plt.close()

# === STEP 8: Chart 3 - feature importance ===
gb = best.named_steps["model"]
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(gb.feature_importances_, index=[s.split("__")[1] for s in names]).sort_values()
print("\nTop feature importances:\n", imp.sort_values(ascending=False).head(6).round(3))
fig, ax = plt.subplots(figsize=(7, 4.2))
imp.tail(10).plot.barh(color="#2E5597", ax=ax)
ax.set(title="Feature Importance (Tuned Gradient Boosting)", xlabel="Importance")
plt.tight_layout(); plt.savefig("chart3_feature_importance.png", dpi=150); plt.close()

# === STEP 9: Optimization 1 - data-driven promised delivery time ===
resid = y_test.values - pred
margin = np.quantile(resid, 0.90)                # 90th percentile of under-prediction
test = X_test.copy()
test["actual"], test["pred"] = y_test.values, pred
test["old_promise"] = df.loc[X_test.index, "promised_days"].values
test["new_promise"] = test.pred + margin
old_ot = (test.actual <= test.old_promise).mean() * 100
new_ot = (test.actual <= test.new_promise).mean() * 100
print("\nSafety margin (90th percentile of residuals): %.2f days" % margin)
print("On-time rate with old promise rule: %.1f%% | with model-based promise: %.1f%%" % (old_ot, new_ot))
print("Average promised days: old %.2f | new %.2f" % (test.old_promise.mean(), test.new_promise.mean()))
mum = test[test.hub == "Mumbai"]
print("Mumbai on-time: old %.1f%% | new %.1f%%" % ((mum.actual <= mum.old_promise).mean() * 100,
                                                    (mum.actual <= mum.new_promise).mean() * 100))

# === STEP 10: Optimization 2 - what-if hub reallocation (resource allocation) ===
mumbai = X_test[X_test.hub == "Mumbai"].copy()
base = best.predict(mumbai)
shift = mumbai.copy(); shift["hub"] = "Pune"
alt = best.predict(shift)
print("\nMumbai shipments in test set:", len(mumbai))
print("Avg predicted days at Mumbai: %.2f | if handled by Pune: %.2f | saving: %.2f days (%.1f%%)" % (
    base.mean(), alt.mean(), base.mean() - alt.mean(), (base.mean() - alt.mean()) / base.mean() * 100))
for share in (0.25, 0.5):
    k = int(len(mumbai) * share)
    moved = np.concatenate([alt[:k], base[k:]])
    print("Move %d%% of Mumbai load to Pune -> avg days %.2f" % (share * 100, moved.mean()))

# === STEP 11: Optimization 3 - route planning (nearest neighbour + 2-opt) ===
pts = np.vstack([[25, 25], rng.uniform(0, 100, (14, 2))])       # depot + 14 customers (km grid)
def dist(a, b): return float(np.linalg.norm(pts[a] - pts[b]))
def route_len(r): return sum(dist(r[i], r[i + 1]) for i in range(len(r) - 1))
def nearest_neighbour():
    left, r = list(range(1, len(pts))), [0]
    while left:
        nxt = min(left, key=lambda j: dist(r[-1], j)); r.append(nxt); left.remove(nxt)
    return r + [0]
def two_opt(r):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(r) - 2):
            for j in range(i + 1, len(r) - 1):
                new = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                if route_len(new) < route_len(r) - 1e-9:
                    r, improved = new, True
    return r
manual = [0] + list(range(1, len(pts))) + [0]                    # visiting stops in list order
nn = nearest_neighbour(); opt = two_opt(nn)
print("\nRoute length (km): list order %.1f | nearest neighbour %.1f | after 2-opt %.1f" % (
    route_len(manual), route_len(nn), route_len(opt)))
print("Saving vs list order: %.1f%%" % ((route_len(manual) - route_len(opt)) / route_len(manual) * 100))
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, r, t in zip(ax, [manual, opt], ["Unplanned route", "Optimised route (NN + 2-opt)"]):
    a.plot(pts[r, 0], pts[r, 1], "-o", color="#2E5597", ms=4)
    a.scatter(*pts[0], color="red", s=70, zorder=3, label="Depot")
    a.set_title(f"{t}\n{route_len(r):.1f} km"); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
ax[0].legend(); plt.tight_layout(); plt.savefig("chart4_route.png", dpi=150); plt.close()
print("\nAll charts saved.")
