#!/usr/bin/env bash
# COVID-19 Radiography Database (Chowdhury, Rahman et al.), Kaggle tawsifurrahman/covid19-radiography-database.
# Licencia en Kaggle: "Data files (c) Original Authors"; uso academico y no comercial. totalBytes: 806 838 951.
DATASET=covid19_radiography; source "$(dirname "$0")/_common.sh"
kaggle_dataset tawsifurrahman/covid19-radiography-database "$RAW_ROOT/covid19_radiography" 806838951 3
