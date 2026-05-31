"""
make_figure.py

Reproduces the keystone figure: the distribution of test R2 across repeated
80/20 splits for each predictor configuration, with the single-split value
(seed 42) marked. The point of the figure is that the single-split value can
sit far from the centre of a model's own distribution, so single-split
accuracy is an unreliable basis for ranking models.

This script recomputes the repeated-holdout distributions directly from the
GEE export CSVs, so it is self-contained. Edit the FILES paths to match your
exports, then:

    python make_figure.py
    python make_figure.py --trees 400 --reps 15   # faster, for a quick look

Requirements: numpy, pandas, scikit-learn, matplotlib
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA = "data"
FILES = {
    "M1": f"{DATA}/M1_DE_LUCAS2018_S2_2022_S1_2022_training.csv",
    "M2": f"{DATA}/M2_DE_LUCAS2018_S2S1_2022_BARESOIL_training.csv",
    "M3": f"{DATA}/M3_DE_LUCAS2018_S1S2_ERA5_2022_training.csv",
    "M4": f"{DATA}/M4_DE_LUCAS2018_S1S2_DEM_2022_training.csv",
    "M5": f"{DATA}/M5_DE_LUCAS_spring_autumn_DEM.csv",
    "M6": f"{DATA}/M6_DE_LUCAS_S1S2_DEM_ERA5seasonal.csv",
}
ORDER = ["M1", "M2", "M3", "M4", "M5", "M6"]
LABELS = {
    "M1": "M1\nS2+S1\n+NDVI", "M2": "M2\n+bare soil\nmask", "M3": "M3\n+ERA5\nannual",
    "M4": "M4\n+DEM", "M5": "M5\n+seas S2\n+BSI", "M6": "M6\n+DEM\n+ERA5",
}
COLORS = {"M1": "#cdd9e3", "M2": "#d99a9a", "M3": "#d6c2a8",
          "M4": "#a9c6b0", "M5": "#c9b3cf", "M6": "#e0b48f"}

NON_FEATURE = {"system:index", ".geo", "OC", "lon", "lat"}


def load_xy(path):
    df = pd.read_csv(path)
    df["OC"] = pd.to_numeric(df["OC"].astype(str).str.replace(",", "."), errors="coerce")
    features = [c for c in df.columns if c not in NON_FEATURE]
    df = df.dropna(subset=features + ["OC"]).reset_index(drop=True)
    X = df[features].astype(float).values
    y = np.log1p(df["OC"].astype(float)).values
    return X, y


def repeated_holdout(X, y, n_trees, reps):
    scores = []
    for seed in range(reps):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                                      random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        scores.append(r2_score(y_te, model.predict(X_te)))
    return np.array(scores)


def single_split(X, y, n_trees, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                                  random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
    return r2_score(y_te, model.predict(X_te))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trees", type=int, default=800)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="figure_split_instability.png")
    args = ap.parse_args()

    dists, singles = {}, {}
    for m in ORDER:
        X, y = load_xy(FILES[m])
        dists[m] = repeated_holdout(X, y, args.trees, args.reps)
        singles[m] = single_split(X, y, args.trees)
        print(f"{m}: single={singles[m]:.3f}  holdout={dists[m].mean():.3f} +/- {dists[m].std():.3f}")

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    pos = range(1, len(ORDER) + 1)
    data = [dists[m] for m in ORDER]
    bp = ax.boxplot(data, positions=list(pos), widths=0.58, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222", lw=1.4),
                    whiskerprops=dict(color="#999"), capprops=dict(color="#999"))
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_edgecolor("#555")

    rng = np.random.RandomState(1)
    for p, m in zip(pos, ORDER):
        d = dists[m]
        ax.scatter(p + rng.uniform(-0.14, 0.14, len(d)), d, s=14, color="#444", alpha=0.4, zorder=3)
        ax.scatter([p], [singles[m]], marker="X", s=120, color="#c0392b",
                   zorder=6, edgecolor="white", lw=0.8)
    ax.scatter([], [], marker="X", s=110, color="#c0392b", label="single split (seed 42)")

    ax.axhspan(-0.02, 0.05, color="#d99a9a", alpha=0.12)
    ax.axhline(0, color="#aaa", lw=0.7, ls=":")
    ax.set_xticks(list(pos)); ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=8.5)
    ax.set_ylabel("Test R\u00b2 (log scale)", fontsize=11)
    ax.set_title(f"Repeated holdout vs single split ({args.trees}-tree RF, {args.reps} splits)",
                 fontsize=11.5, pad=10)
    ax.set_ylim(-0.02, 0.50); ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    plt.tight_layout(); plt.savefig(args.out, dpi=220)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
