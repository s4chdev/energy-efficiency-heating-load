"""Generate the figures used in the report and print EDA facts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

df = pd.read_excel(ROOT / "data" / "ENB2012_data.xlsx")

# --- EDA facts -------------------------------------------------------------
print("rows,cols:", df.shape)
print("missing:", int(df.isna().sum().sum()), "dupes:", int(df.duplicated().sum()))
print("nunique:\n", df.nunique())
print("Y1 range:", df.Y1.min(), df.Y1.max(), "mean", round(df.Y1.mean(), 2))
print("corr Y1/Y2:", round(df.Y1.corr(df.Y2), 4))
print("corr with Y1:\n", df.corr()["Y1"].drop(["Y1", "Y2"]).round(3).to_string())
print("X5 counts:\n", df.X5.value_counts().to_string())
print("X6 counts:\n", df.X6.value_counts().sort_index().to_string())
print("X7 counts:\n", df.X7.value_counts().sort_index().to_string())
print("X8 counts:\n", df.X8.value_counts().sort_index().to_string())
print("X4 nunique:", df.X4.nunique(), sorted(df.X4.unique()))

# --- Figure 1: feature correlations ---------------------------------------
ax = df.corr()["Y1"].drop(["Y1", "Y2"]).sort_values().plot(
    kind="barh", figsize=(5.2, 2.8), color="#4a7ebb"
)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("Pearson correlation with heating load")
ax.set_title("Linear association of each feature with Y1", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "fig1_correlations.png")
plt.close()

# --- Figure 2: target distribution + relationship -------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
df.Y1.plot(kind="hist", bins=30, ax=axes[0], color="#4a7ebb", edgecolor="white")
axes[0].set_title("Heating load distribution")
axes[0].set_xlabel("Y1 (kWh/m²)")
axes[1].scatter(df.X1, df.Y1, s=5, c=df.X5, cmap="viridis")
axes[1].set_title("Y1 vs relative compactness\n(colour = overall height)")
axes[1].set_xlabel("X1 relative compactness")
axes[1].set_ylabel("Y1")
axes[2].scatter(df.X7, df.Y1, s=5, c=df.X5, cmap="viridis")
axes[2].set_title("Y1 vs glazing area\n(colour = overall height)")
axes[2].set_xlabel("X7 glazing area")
plt.tight_layout()
plt.savefig(FIG / "fig2_target.png")
plt.close()

# --- Figure 3: predicted vs actual ----------------------------------------
model = joblib.load(ROOT / "models" / "heating_model.joblib")
RENAMED = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
}
X, y = df[FEATURES].rename(columns=RENAMED), df.Y1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pred = model.predict(X_te)
print("re-check test R2:", round(r2_score(y_te, pred), 4))

fig, ax = plt.subplots(figsize=(4.2, 4))
ax.scatter(y_te, pred, s=12, alpha=0.75, color="#4a7ebb")
lims = [y_te.min() - 1, y_te.max() + 1]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlabel("Actual heating load (kWh/m²)")
ax.set_ylabel("Predicted heating load (kWh/m²)")
ax.set_title("Random Forest on the held-out test set", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "fig3_pred_vs_actual.png")
plt.close()

print("figures written to", FIG)
