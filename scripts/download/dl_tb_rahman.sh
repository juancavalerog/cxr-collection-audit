#!/usr/bin/env bash
# Tuberculosis (TB) Chest X-ray Database (Rahman et al.), Kaggle tawsifurrahman/tuberculosis-tb-chest-xray-dataset.
# Base compuesta (TB de NLM, Belarus y NIAID; normales de RSNA y NLM). totalBytes: 694 671 204.
DATASET=tb_rahman; source "$(dirname "$0")/_common.sh"
kaggle_dataset tawsifurrahman/tuberculosis-tb-chest-xray-dataset "$RAW_ROOT/tb_rahman" 694671204 3
