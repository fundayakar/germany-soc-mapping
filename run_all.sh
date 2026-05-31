#!/usr/bin/env bash
# run_all.sh - reproduce the full evaluation table for all six configurations.
#
# Edit the paths below to match your exported CSV filenames, then:
#   bash run_all.sh
#
# Each call prints the four-protocol results and (for climate models) the
# ERA5 pixel-sharing diagnostic. Defaults to 800 trees; lower --trees for a
# quick check on modest hardware.

set -euo pipefail

DATA="data"   # directory holding the GEE exports

declare -A FILES=(
  [M1]="$DATA/M1_DE_LUCAS2018_S2_2022_S1_2022_training.csv"
  [M2]="$DATA/M2_DE_LUCAS2018_S2S1_2022_BARESOIL_training.csv"
  [M3]="$DATA/M3_DE_LUCAS2018_S1S2_ERA5_2022_training.csv"
  [M4]="$DATA/M4_DE_LUCAS2018_S1S2_DEM_2022_training.csv"
  [M5]="$DATA/M5_DE_LUCAS_spring_autumn_DEM.csv"
  [M6]="$DATA/M6_DE_LUCAS_S1S2_DEM_ERA5seasonal.csv"
)

for m in M1 M2 M3 M4 M5 M6; do
  echo "==================== $m ===================="
  python evaluate_models.py "${FILES[$m]}" --trees 800
  echo
done
