"""
evaluate_models.py

Compares predictor configurations for SOC prediction across Germany under four
evaluation protocols of increasing stringency:

  1. single   - one 80/20 train/test split (the conventional report)
  2. holdout  - repeated 80/20 splits over many seeds (variance of #1)
  3. randomcv - repeated random K-fold cross-validation
  4. spatialcv- spatially blocked K-fold (blocks, not points, assigned to folds)

It also reports an ERA5 pixel-sharing diagnostic for climate configurations,
quantifying how many sampling points fall in the same coarse climate pixel.

Each input CSV is a GEE export with: predictor columns, an 'OC' column
(topsoil organic carbon, g/kg), and a '.geo' column holding point geometry
(requires geometries:true in the GEE export). Predictor columns are detected
automatically as every column except system:index, .geo, and OC.

Usage:
    python evaluate_models.py path/to/M4_*.csv
    python evaluate_models.py path/to/M6_*.csv --trees 800 --holdout-reps 20

Requirements: numpy, pandas, scikit-learn, pyproj
"""

import argparse
import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

# Columns that are never predictors
NON_FEATURE = {"system:index", ".geo", "OC", "lon", "lat"}

# ERA5-Land predictor names used across the climate configurations, for the
# pixel-sharing diagnostic. Any of these present in a file triggers the check.
ERA5_COLS = ["sm1", "sm2", "t2m_C", "tp_sum", "sm_annual", "t2m_summer", "tp_winter"]


def load(path):
    """Load a GEE export, parse coordinates from .geo, drop incomplete rows."""
    df = pd.read_csv(path)
    # OC may use a comma decimal separator depending on locale
    df["OC"] = pd.to_numeric(df["OC"].astype(str).str.replace(",", "."), errors="coerce")

    coords = df[".geo"].apply(lambda g: json.loads(g)["coordinates"])
    df["lon"] = coords.apply(lambda c: c[0])
    df["lat"] = coords.apply(lambda c: c[1])

    features = [c for c in df.columns if c not in NON_FEATURE]
    df = df.dropna(subset=features + ["OC", "lon", "lat"]).reset_index(drop=True)
    return df, features


def make_model(n_trees, seed):
    return RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def single_split(X, y, n_trees, seed=42):
    """One 80/20 split; the conventional single-number report."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = make_model(n_trees, seed).fit(X_tr, y_tr)
    return r2_score(y_te, model.predict(X_te))


def repeated_holdout(X, y, n_trees, reps):
    """Repeated 80/20 splits; characterises the variance of a single split."""
    scores = []
    for seed in range(reps):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = make_model(n_trees, 42).fit(X_tr, y_tr)
        scores.append(r2_score(y_te, model.predict(X_te)))
    return np.array(scores)


def random_cv(X, y, n_trees, reps, k=5):
    """Repeated random K-fold cross-validation."""
    scores = []
    for seed in range(reps):
        for tr_idx, te_idx in KFold(k, shuffle=True, random_state=seed).split(X):
            model = make_model(n_trees, 42).fit(X[tr_idx], y[tr_idx])
            scores.append(r2_score(y[te_idx], model.predict(X[te_idx])))
    return np.array(scores)


def spatial_blocks(lon, lat, block_km):
    """Assign each point to a square block in EPSG:3035 (metric, equal-area)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = transformer.transform(lon, lat)
    size = block_km * 1000.0
    bx = np.floor(np.asarray(x) / size).astype(int)
    by = np.floor(np.asarray(y) / size).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(bx, by)])


def spatial_cv(X, y, block_ids, n_trees, reps, k=5):
    """Spatially blocked K-fold: whole blocks are held out together."""
    unique_blocks = np.unique(block_ids)
    scores = []
    for seed in range(reps):
        rng = np.random.RandomState(seed)
        block_fold = {b: f for b, f in zip(unique_blocks, rng.randint(0, k, len(unique_blocks)))}
        fold = np.array([block_fold[b] for b in block_ids])
        for f in range(k):
            test = fold == f
            if test.sum() < 5 or (~test).sum() < 5:
                continue
            model = make_model(n_trees, 42).fit(X[~test], y[~test])
            scores.append(r2_score(y[test], model.predict(X[test])))
    return np.array(scores)


def era5_pixel_sharing(df, features):
    """Count points sharing an identical ERA5 predictor tuple (a shared pixel)."""
    cols = [c for c in ERA5_COLS if c in features]
    if not cols:
        return None
    tuples = df[cols].round(6).apply(tuple, axis=1)
    counts = tuples.value_counts()
    return {
        "era5_columns": cols,
        "n_points": len(df),
        "n_unique_pixels": int(counts.size),
        "mean_points_per_pixel": round(len(df) / counts.size, 2),
        "max_points_in_one_pixel": int(counts.max()),
        "pct_points_sharing": round(100 * (tuples.map(counts) > 1).mean(), 1),
    }


def fmt(arr):
    return f"{arr.mean():.3f} +/- {arr.std():.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="GEE export CSV (geometries:true) for one configuration")
    ap.add_argument("--trees", type=int, default=800, help="number of trees (default 800)")
    ap.add_argument("--holdout-reps", type=int, default=20)
    ap.add_argument("--cv-reps", type=int, default=5)
    ap.add_argument("--spatial-reps", type=int, default=10)
    ap.add_argument("--block-km", type=float, default=100.0)
    args = ap.parse_args()

    df, features = load(args.csv)
    X = df[features].astype(float).values
    y = np.log1p(df["OC"].astype(float)).values  # model is trained in log space
    blocks = spatial_blocks(df["lon"].values, df["lat"].values, args.block_km)

    print(f"file:      {args.csv}")
    print(f"n:         {len(df)}")
    print(f"features:  {len(features)}  {features}")
    print(f"trees:     {args.trees}")
    print(f"blocks:    {len(np.unique(blocks))} populated at {args.block_km:.0f} km\n")

    s = single_split(X, y, args.trees)
    ho = repeated_holdout(X, y, args.trees, args.holdout_reps)
    rc = random_cv(X, y, args.trees, args.cv_reps)
    sc = spatial_cv(X, y, blocks, args.trees, args.spatial_reps)

    print("Test R2 (log scale):")
    print(f"  single split (seed 42):     {s:.3f}")
    print(f"  repeated holdout (n={args.holdout_reps}):     {fmt(ho)}")
    print(f"  random {args.cv_reps}x5-fold CV:          {fmt(rc)}")
    print(f"  spatial {args.spatial_reps}x5-fold CV ({args.block_km:.0f} km): {fmt(sc)}")

    diag = era5_pixel_sharing(df, features)
    if diag:
        print("\nERA5 pixel-sharing diagnostic:")
        for key, val in diag.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
