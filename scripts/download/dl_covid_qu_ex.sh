#!/usr/bin/env bash
# COVID-QU-Ex (Tahir et al.), Kaggle anasmohammedtahir/covidqu, CC BY-SA 4.0. 33 920 CXR con mascaras pulmonares.
# totalBytes publicado por la API de Kaggle: 1 211 011 783.
DATASET=covid_qu_ex; source "$(dirname "$0")/_common.sh"
kaggle_dataset anasmohammedtahir/covidqu "$RAW_ROOT/covid_qu_ex" 1211011783 4
