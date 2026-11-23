#!/usr/bin/env bash
# TBX11K (Liu et al., CVPR 2020). Licencia oficial CC BY 4.0 (https://mmcheng.net/tb/), lo que permite la copia de Kaggle
# usmanshams/tbx-11 con atribucion. La pagina de Kaggle declara licencia "Unknown" y totalBytes 4 252 079 748.
# La integridad frente a la version oficial (Google Drive) NO esta verificada: ver docs/02_datos.md Section 5.
DATASET=tbx11k; source "$(dirname "$0")/_common.sh"
kaggle_dataset usmanshams/tbx-11 "$RAW_ROOT/tbx11k" 4252079748 10
