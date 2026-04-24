# Germany SOC Mapping Workflow

This repository contains the code and workflow developed for soil organic carbon (SOC) mapping in Germany using remote sensing and machine learning.

## Overview

The workflow combines:
- Google Earth Engine (GEE) for data extraction and spatial prediction
- Python (Colab) for model training and evaluation
- Post-processing (bias correction) applied externally

## Repository Structure
gee/
m1_lucas_training_s1s2_ndvi.js
m4_soc_mapping.js

notebooks/
m1_rf_model_training_s1_s2_ndvi.ipynb

python/
m1_rf_model_training_s1_s2_ndvi.py

## Workflow

1. **Training data generation (GEE)**  
   Run:
   gee/m1_lucas_training_s1s2_ndvi.js

This script extracts predictor variables and LUCAS soil data and exports a training dataset.

2. **Model training (Python / Colab)**  
Run:

notebooks/m1_rf_model_training_s1_s2_ndvi.ipynb

or the Python script in `/python/`.

The script expects the training dataset exported from GEE.

3. **SOC mapping (GEE)**  
Run:

gee/soc_mapping.js

This script trains a Random Forest model and produces a spatial SOC prediction map.

4. **Post-processing (external)**  
Additional bias correction and refinement steps were applied using GIS tools (e.g., QGIS).

## Data Sources

- Sentinel-2 (optical imagery)
- Sentinel-1 (radar)
- SRTM DEM
- LUCAS soil database
- ERA5

## Notes

- Large datasets and outputs are not included in this repository.
- All data can be reproduced using the provided scripts.

## Status

Prepared as a reproducibility resource for manuscript submission and review.

## Model Variants

- M1: S2 + S1 + NDVI
- M2: M1 + bare soil mask
- M3: M1 + ERA5 annual
- M4: M1 + DEM
- M5: M1 + DEM + seasonal S2 + BSI
- M6: M1 + DEM + ERA5 seasonal
