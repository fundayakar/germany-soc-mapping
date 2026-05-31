# Germany SOC mapping: evaluation protocol and predictor comparison

Code and data for a study of soil organic carbon (SOC) prediction across Germany
from remote sensing predictors, with a focus on how the **evaluation protocol**,
rather than the predictor set, governs the conclusions drawn from a Random Forest
workflow.

A Random Forest is trained on LUCAS 2018 topsoil observations using Sentinel-2,
Sentinel-1, SRTM terrain, and ERA5-Land climate predictors. Six predictor
configurations (M1 to M6) are each evaluated under four protocols of increasing
stringency. The central finding is that a single random train/test split is too
variable to rank models at this performance level: for a fixed model the
single-split test R² ranges by roughly 0.2, and conclusions reached from a single
split reverse under repeated and spatial evaluation.

## Predictor configurations

| Model | Predictors |
|-------|------------|
| M1 | Sentinel-2 + Sentinel-1 + NDVI |
| M2 | M1 restricted to bare soil pixels (NDVI < 0.45) |
| M3 | M1 + ERA5-Land annual climate |
| M4 | M1 + SRTM terrain (elevation, slope, aspect) |
| M5 | M1 + terrain + seasonal Sentinel-2 composites + bare soil index |
| M6 | M1 + terrain + ERA5-Land seasonal climate |

## Evaluation protocols

1. **Single split** – one 80/20 train/test split (the conventional report).
2. **Repeated holdout** – the 80/20 split regenerated over many seeds, to
   characterise the variance of the single split.
3. **Random cross-validation** – repeated random five-fold CV.
4. **Spatially blocked cross-validation** – points projected to EPSG:3035 and
   assigned to 100 km blocks; whole blocks, not individual points, are held out.

For the climate configurations, an ERA5 pixel-sharing diagnostic counts how many
sampling points fall in the same coarse climate pixel, testing directly whether
combining coarse climate with fine imagery inflates accuracy through shared pixels.

## Repository structure

```
germany-soc-mapping/
├── data/                  # GEE-exported training tables (one per configuration)
│   ├── M1_DE_LUCAS2018_S2_2022_S1_2022_training.csv
│   ├── M2_DE_LUCAS2018_S2S1_2022_BARESOIL_training.csv
│   ├── M3_DE_LUCAS2018_S1S2_ERA5_2022_training.csv
│   ├── M4_DE_LUCAS2018_S1S2_DEM_2022_training.csv
│   ├── M5_DE_LUCAS_spring_autumn_DEM.csv
│   └── M6_DE_LUCAS_S1S2_DEM_ERA5seasonal.csv
├── gee/                   # Google Earth Engine scripts: predictor export + mapping
├── python/                # Legacy per-model training scripts (see note below)
├── Figures/
│   ├── make_figure.py     # regenerates the keystone figure
│   └── split_instability.png
├── evaluate_models.py     # main analysis: four protocols + leakage diagnostic
├── requirements.txt
└── README.md
```

Each CSV in `data/` is exported from the corresponding script in `gee/` with
`geometries: true`, so it contains the predictor columns, an `OC` column
(topsoil organic carbon, g/kg), and a `.geo` column holding point coordinates.
Coordinates are required for the spatially blocked cross-validation.

### Note on the `python/` scripts

The scripts in `python/` are the original, per-model training notebooks. Each one
trains a single configuration and evaluates it with one 80/20 split and one random
five-fold cross-validation. They are kept as a record of the initial approach,
which is exactly the single-split evaluation the study shows to be unreliable. The
current analysis supersedes them with `evaluate_models.py`.

## Reproducing the results

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all four protocols for one configuration:

```bash
python evaluate_models.py data/M6_DE_LUCAS_S1S2_DEM_ERA5seasonal.csv
```

Optional arguments: `--trees` (default 800), `--holdout-reps` (default 20),
`--cv-reps`, `--spatial-reps`, `--block-km` (default 100).

To reproduce the full results table, run the command above for each CSV in
`data/`. Regenerate the keystone figure with:

```bash
python Figures/make_figure.py
```

(Adjust the `DATA` path at the top of `make_figure.py` if you run it from inside
the `Figures/` directory.)

## Data sources

- **LUCAS 2018** topsoil survey (European Soil Data Centre, ESDAC).
- **Sentinel-1 / Sentinel-2** (Copernicus, via Google Earth Engine).
- **SRTM** terrain (via Google Earth Engine).
- **ERA5-Land** climate reanalysis (Copernicus Climate Data Store, via GEE).

## License

Released for review and reproducibility. See repository settings for license terms.
