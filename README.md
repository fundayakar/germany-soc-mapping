# Germany SOC Mapping Workflow

This repository contains the code and workflow developed for soil organic carbon (SOC) mapping in Germany using remote sensing and machine learning.

## Overview

The workflow combines:
- Google Earth Engine (GEE) for data extraction and spatial prediction
- Python (Colab) for model training and evaluation
- Post-processing (bias correction) applied externally

## Repository Structure
gee/
lucas_training_export.js
soc_mapping_initial.js

notebooks/
rf_model_training_s1_s2_dem.ipynb

python/
rf_model_training_s1_s2_dem.py

## Workflow

1. **Training data generation (GEE)**  
   Run:
   gee/lucas_training_export.js

This script extracts predictor variables and LUCAS soil data and exports a training dataset.

2. **Model training (Python / Colab)**  
Run:

notebooks/rf_model_training_s1_s2_dem.ipynb

or the Python script in `/python/`.

The script expects the training dataset exported from GEE.

3. **SOC mapping (GEE)**  
Run:

gee/soc_mapping_initial.js

This script trains a Random Forest model and produces a spatial SOC prediction map.

4. **Post-processing (external)**  
Additional bias correction and refinement steps were applied using GIS tools (e.g., QGIS).

## Data Sources

- Sentinel-2 (optical imagery)
- Sentinel-1 (radar)
- SRTM DEM
- LUCAS soil database

## Notes

- Large datasets and outputs are not included in this repository.
- All data can be reproduced using the provided scripts.

## Status

Prepared as a reproducibility resource for manuscript submission and review.
