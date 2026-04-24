# Germany SOC Mapping Workflow

This repository contains the code and workflow developed for soil organic carbon (SOC) mapping in Germany using remote sensing and machine learning.

## Overview

The workflow combines:
- Google Earth Engine (GEE) for data extraction and spatial prediction
- Python (Colab) for model training and evaluation
- Post-processing (bias correction) applied externally
  
## File naming convention:
Each script is prefixed with the model identifier (e.g., m3_, m4_) to indicate the corresponding feature combination.

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

- Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR), April–October 2022  
- Sentinel-1 GRD (COPERNICUS/S1_GRD), 2022 (VV and VH polarizations)  
- SRTM Digital Elevation Model (USGS/SRTMGL1_003)  
- LUCAS topsoil dataset (2018)  
- ERA5-Land climate data (annual and seasonal aggregates, 2022)

## Notes

- Large datasets and outputs are not included in this repository.
- All data can be reproduced using the provided scripts.

## Status

Prepared as a reproducibility resource for manuscript submission and review.

## Model Variants

The following feature combinations were tested:

- **M1**: Sentinel-2 + Sentinel-1 + NDVI  
- **M2**: M1 + bare soil mask  
- **M3**: M1 + ERA5 annual variables  
- **M4**: M1 + DEM  
- **M5**: M1 + DEM + seasonal Sentinel-2 + BSI  
- **M6**: M1 + DEM + ERA5 seasonal variables  

The best-performing model in this workflow is **M4**.
