# cxr-collection-audit

Code, configurations, split manifests and result tables for the article

**A Class-Blind Collection Detector Matches Nearly All Methods in the Internal Evaluation of COVIDx CXR-4**

Juan Carlos Valero Gómez, Juan Carlos Clares Perca, Alex Peter Zuñiga Incalla
Universidad Nacional de Moquegua, Peru

Repository: [repository URL]
Archived version: [DOI to be added after the Zenodo deposit]

The study is a retrospective evaluation on public chest radiography data, carried out under a prespecified, internally frozen analysis plan (the plan, the code, the configurations and the selected hyperparameters were frozen with their SHA-256 hashes before any test set was opened; the plan was not deposited in a public registry). It audits the internal evaluation of COVIDx CXR-4 with a reference that has never seen a class label (a collection detector), measures how eleven methods behave when the correlation between class and source is fixed by construction (pleural effusion benchmark built from NIH ChestX-ray14 and CheXpert, with a tuberculosis evaluation), and quantifies the cost of model selection with a validation set drawn from the training distribution.

This repository lets a reader (a) see which code and configuration produced each number, (b) rebuild the data splits from the public databases, and (c) regenerate the supplementary tables and the figures of the article from the result files included here, without retraining. **No image, mask, preprocessed cache, network checkpoint or feature array is redistributed.**

One change was made to the analysis after the test sets were opened: the calibration factor of the seed term of the hierarchical intervals was corrected. It changes no decision. The original and the corrected tables are both included; see "Post hoc correction of the seed-term calibration factor".

## Contents

| Path | Content |
|---|---|
| `src/` | Library code: data loading and preprocessing, lung masks, the twelve compared methods (`src/baselines/`), training loop, evaluation and statistics (`src/eval/`), unblinding guard (`src/eval/desbloqueo.py`), plotting style (`src/viz/`). `src/model/` and `src/train/csdnet_*` belong to an architecture that was piloted and abandoned before the freeze of the analysis plan; they are kept because they are part of the frozen code. |
| `scripts/data/` | Builders and checkers of the split manifests. |
| `scripts/download/` | Download helpers for the public databases (they need your own credentials where the provider requires them). |
| `scripts/jobs/` | Generators of the training and evaluation jobs (one JSON per run) and the job chain driver. |
| `scripts/eval/` | Evaluation programs, the freeze tool (`congelar.py`) and the report builders. |
| `scripts/sensibilidad/` | Prespecified sensitivity analyses, computed after the test sets were opened. |
| `scripts/correccion/` | Post hoc correction of the seed-term calibration factor: corrected factor (`ec6.py`), recomputation of the affected tables (`lanzador.py`), simulation check (`simulacion.py`), impact table (`impacto.py`) and supplementary calculations (`complementos.py`). |
| `scripts/diag/` | Exploration and audit scripts from which the collection audit derives. |
| `scripts/suplemento/` | Builder and verifier of the supplementary tables (`tablas.py`), the English translation dictionary (`traduccion_en.py`), the check that the supplementary tables carry the corrected intervals (`comprobar_correccion.py`) and the builder of the segmentation-overlap source tables (`solape_segmentador.py`). |
| `scripts/figs/`, `scripts/figs/en/` | Figure scripts (Spanish originals and English versions) and their shared reading and saving utilities (`scripts/figs/comun.py`). |
| `scripts/refs/`, `scripts/inventario/`, `scripts/informes/`, `scripts/model/` | Bibliography helper, inventory utilities, Spanish report builder, model complexity counter. |
| `scripts/release/check_ast.py` | Release check: compares the Python files of this folder with an original tree (see "Freeze record"). |
| `configs/` | YAML configurations of every benchmark, the freeze list (`gate7_congelacion.yaml`) and the closed list of analyses. |
| `tests/` | Unit tests (pytest). |
| `runner/` | Job executor and GPU/CPU thermal guards used on the training server. |
| `docker/` | Dockerfile, requested requirements and the exact lock of the container. |
| `docs/08_evaluacion.md` | Frozen analysis plan with its dated amendments (Spanish). |
| `data/manifests/` | Split manifests (file names, public identifiers, labels, split and subset flags, perceptual hashes). See `data/manifests/README.md` (Spanish). |
| `results/` | Result tables (CSV, JSON, MD), frozen hyperparameters, run summaries, execution chain logs, figures. |
| `predictions/` | Per-image predicted probabilities, compressed (see below). |
| `manuscript/srep/correspondencia_tablas.csv` | Correspondence between the table numbers used by the programs and the Supplementary Table numbers of the article (see "Table numbering"). |
| `MANIFEST.sha256` | SHA-256 hash, size in bytes and path of every file in this folder (one line per file: `hash size path`). Check with `awk '{print $1"  "$3}' MANIFEST.sha256 \| sha256sum -c --quiet`. |

### Results directories

| Directory | Content |
|---|---|
| `results/c1/test`, `results/c1/val` | COVIDx CXR-4 audit on the test and validation sets (confirmatory and secondary families, references, collections, acquisition-only classifier, masked ERM, COVIDGR, AP/PA strata). |
| `results/b1` | COVIDx hyperparameter search tables, validation of the final runs, frozen hyperparameters (`congelados/`). |
| `results/b2`, `results/b2/informe_test` | Tuberculosis evaluation: frozen hyperparameters and test report. |
| `results/b3a` | Pleural effusion benchmark: search tables and frozen hyperparameters per level. |
| `results/b3/test_A_best`, `results/b3/test_A_bestdc` | Pleural effusion test reports with the same-ρ checkpoint and with the oracle checkpoint (original calibration factor). |
| `results/c2c3` | Flipped test set, per-source AUC, specificity gap, reverse orientation, epoch-selection and hyperparameter-selection analyses (original calibration factor). |
| `results/correccion_ec6` | Tables recomputed with the corrected calibration factor, under the same relative paths, file names and columns as the originals (`b3/`, `c2c3/`, `sensibilidad/`); impact of the correction (`antes_despues*.csv`, `.md`); simulation check (`simulacion/`); post hoc supplementary calculations (`complementos/`). |
| `results/c3_busqueda` | Hyperparameter-selection rerun with test evaluation: frozen configurations and run summaries. |
| `results/sensibilidad`, `results/sensibilidad_v28`, `results/sensibilidad_intervalos` | Prespecified sensitivity analyses (search stability, second backbone, second label, audit variants, alternative intervals). |
| `results/sens_tronco`, `results/sens_etiqueta`, `results/estabilidad_b3a` | Frozen configurations and run summaries of the ResNet-50, cardiomegaly and search-stability runs. |
| `results/train/<experiment>/seed<k>/` | `config_resolved.json` (full resolved configuration) and `summary.json` (training history, early stopping, metrics) of every training run, except three runs of the abandoned architecture. |
| `results/cadenas` | Execution chain logs with start and end times of each run (source of Supplementary Table S13); `corr_cadena.json` is the chain of the correction. |
| `results/suplemento` | The 38 supplementary tables built by `scripts/suplemento/tablas.py`, as CSV with Spanish labels (`tabla_S01.csv` to `tabla_S38.csv`, numbered as in the programs; see "Table numbering"), and their cell-to-source traces (`trazas.json`). |
| `results/suplemento_fuentes` | Source tables of the segmentation-overlap table (Supplementary Table S4), written by `scripts/suplemento/solape_segmentador.py`. |
| `results/figs`, `results/figs/en` | Figures (PDF, PNG, TIFF) and the CSV data behind each one; `en/` holds the English versions used in the article, and `en/correspondencia.csv` relates each English figure to its Spanish original. |
| `results/gate6`, `results/gate7`, `results/audit`, `results/informe_plan_b` | Pilot table (Supplementary Fig. S1), freeze record, preliminary audit tables, Spanish summary report. |

## Requirements

All experiments ran inside the container defined in `docker/` on a server with two NVIDIA A100 80 GB GPUs (driver with CUDA 12.6). Versions recorded in `results/gate7/entorno_contenedor.json`:

| Package | Version |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.14.0+cu126 (CUDA 12.6, cuDNN 9.10.2) |
| torchvision | 0.29.0+cu126 |
| timm | 1.0.30 (weights `convnext_tiny.in12k_ft_in1k`, `resnet50.a1_in1k`) |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| Pillow | 12.3.0 |
| opencv-python-headless | 4.10.0.84 |
| PyWavelets | 1.8.0 |
| PyYAML | 6.0.3 |

The full package list is in `docker/requirements.lock.txt` and in `results/gate7/entorno_contenedor.json`.

Regenerating the tables needs only Python 3.10 and pandas; the figures also need NumPy and matplotlib. Neither needs a GPU, PyTorch or any image.

## Obtaining the public databases

The images are not redistributed. They can be obtained from their original providers under their own licences and access conditions, which the user must accept. Sources used by the project (URLs as recorded in the project's data notes):

| Database | Role | Source |
|---|---|---|
| COVIDx CXR-4 (Wu et al., arXiv 2311.17677), which bundles BIMCV, RSNA, RICORD, Cohen, ActMed, Fig1, SIRM and Stony Brook images | audit | Kaggle dataset `andyczhao/covidx-cxr2`, version 9: https://www.kaggle.com/datasets/andyczhao/covidx-cxr2 ; paper: https://arxiv.org/abs/2311.17677 |
| BIMCV-COVID19 (original provider of the BIMCV images) | audit | https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/ |
| RSNA Pneumonia Detection Challenge mapping to NIH (used to link RSNA images to NIH patients) | clean split | https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018 ; mapping file: https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json |
| COVIDGR 1.0 (Tabik et al.) | external cohort | https://github.com/ari-dasci/OD-covidgr (release 1.0) ; paper: https://arxiv.org/abs/2006.01409 |
| NIH ChestX-ray14 | effusion and cardiomegaly benchmarks; tuberculosis normals | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| CheXpert, small version (CheXpert-v1.0-small) | effusion and cardiomegaly benchmarks | https://stanfordmlgroup.github.io/competitions/chexpert/ |
| TB Chest X-ray Database (Rahman et al.), source of the Belarus positives | tuberculosis | https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset ; paper: https://arxiv.org/abs/2007.14895 |
| TBX11K (Liu et al.) | tuberculosis | https://mmcheng.net/tb/ |
| Shenzhen Hospital CXR Set | tuberculosis test | https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html |
| Montgomery County CXR Set | tuberculosis test | https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html |
| CheXmask 1.0.0 | training of the lung segmentation network | https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/ |

The download helpers in `scripts/download/` (Spanish comments) automate part of this; read `scripts/download/README.md` first. They also contain helpers for databases that were evaluated and not used (for example VinDr-CXR).

## Rebuilding the splits

The manifests in `data/manifests/` list, for every image, the dataset, the source, the image path relative to a data root, the patient identifier, the label, the split and the subset flags. Column `raiz` says which root a path is relative to:

- `datos`: the data root, set with the environment variable `PAPER_NEW_DATA_ROOT` (see `src/data/paths.py`). Expected layout: `data/dataset/<file>` for COVIDx CXR-4, `nih/images/<file>` for NIH ChestX-ray14, `CheXpert/CheXpert-v1.0-small/...` for CheXpert.
- `proyecto`: the repository root, with each database under `data/raw/<dataset>/` (Montgomery, Shenzhen, TBX11K, the Rahman database, COVIDGR).

Main files:

- `covidx_cxr4.csv` (official COVIDx splits) and `covidx_cxr4_clean.csv` (clean split: column `split_clean`, exclusion reason, `val_robust` and `test_subset`), built with `scripts/data/build_manifests.py` and `scripts/data/build_covidx_clean.py`; `covidx_cxr4_colecciones.csv` gives the BIMCV collection of each image (`scripts/data/build_colecciones_covidx.py`).
- `bench/b3a/apos/rho<level>_s<seed>.csv` (effusion, NIH as positive source), `bench/b3a/bpos/` (reverse orientation), `bench/b3c/apos/` (cardiomegaly), `bench/b2/b2_varB_s<seed>.csv` (tuberculosis), `bench/covidgr/covidgr.csv`, each with a JSON sidecar and an index (`indice.json`). Builders: `scripts/data/build_b3a.py`, `build_b3a_extra.py`, `build_b3c.py`, `build_b2.py`, `build_covidgr.py`; checker: `scripts/data/check_bench.py`.
- `bench/hashes/` and `duplicados_hash.csv`: perceptual hashes used to find near-duplicates.

To use the splits, download the databases, place them under the two roots above, and check that every path resolves (for example with `scripts/data/check_manifests.py` or `scripts/data/check_bench.py`). The manifests are used as distributed: their SHA-256 hashes are those of the freeze record, so they must not be regenerated if the aim is to reproduce the article. Preprocessed caches are built with `scripts/build_cache.py` and lung masks with `src/seg/`; training jobs are produced by `scripts/jobs/make_jobs_*.py` and run with `runner/job_runner.py` inside the container.

No file in this folder is larger than 50 MB (the largest is `data/manifests/chexpert_small.csv`, 45.7 MB), so no manifest had to be compressed or split.

## Regenerating the supplementary tables and the figures

Run from the repository root.

```bash
# Spanish version: rewrites results/suplemento/tabla_S01.csv ... tabla_S38.csv and trazas.json,
# and writes manuscript/es/07_suplemento.md
mkdir -p manuscript/es manuscript/en
python3 scripts/suplemento/tablas.py
# Rebuilds every table from the sources and checks each printed number against its raw value
python3 scripts/suplemento/tablas.py --verificar

# English version: writes manuscript/en/07_supplementary_material.md
python3 scripts/suplemento/tablas.py --idioma en
python3 scripts/suplemento/tablas.py --idioma en --verificar

# Checks that no interval or p value in the supplementary tables is the one computed with the original factor
python3 scripts/suplemento/comprobar_correccion.py

# Figures 1 to 7 and S1 to S8, English labels: write results/figs/en/Fig*.{pdf,png,tiff,csv}
python3 scripts/figs/en/figs_principales_en.py          # or with figure numbers, e.g. 1 5 7
python3 scripts/figs/en/figs_suplementarias_en.py       # or 1 ... 8
# Spanish versions: write results/figs/figB*.* and figSB*.*
python3 scripts/figs/figB1_colecciones.py               # likewise figB2_bosque_c1.py ... figB7_seleccion.py,
python3 scripts/figs/figSB_lote1.py                     # figSB_lote1.py and figSB_lote2.py
```

Each `--verificar` run needs the Markdown file written by the run without it, so keep the order shown. The `manuscript/` folders are created only for these outputs and are ignored by `.gitignore`.

Where a table has a version recomputed with the corrected calibration factor, both `tablas.py` and the figure scripts read that version: `tablas.py` reads `results/correccion_ec6/<path>` whenever it exists instead of `results/<path>` (method `Fuentes.ruta`), and the trace of each cell records which file was read; the figure scripts do the same through `scripts/figs/comun.py` (function `leer`).

These commands read only CSV and JSON files under `results/` (and, for the search-space table, the literal search spaces in `src/baselines/`; for the Spanish Figure 1, `data/manifests/covidx_cxr4_clean.csv` and `covidx_cxr4_colecciones.csv`), never an image, a checkpoint or a prediction. They were run on this folder: `tablas.py --verificar` and `--idioma en --verificar` reported 0 failures over 5644 cells and 6017 numbers, the 38 CSV files and `trazas.json` were reproduced byte for byte, `comprobar_correccion.py` reported 0 failures, and the figures regenerated by the scripts above (CSV, PNG, TIFF and PDF) are byte-identical to the included ones.

`scripts/suplemento/solape_segmentador.py` cannot be rerun from this folder, because it reads the list of images seen by the segmentation network from working files that are not distributed; its two output tables are included in `results/suplemento_fuentes/`.

## Post hoc correction of the seed-term calibration factor

After the test sets were opened, an error was found in the calibration factor $\lambda_r$ of the seed term of the hierarchical seed-and-patient bootstrap (equation (7) of the article; equation (17) of `docs/08_evaluacion.md`; the comments of the code call it equation (6)). The frozen code resamples the seeds within strata defined by the levels at which each seed exists, so that a seed is resampled in the same way at every level; in the pleural effusion benchmark seeds 0 to 4 exist at every level and seeds 5 to 9 only at $\rho$ of 0.9, 0.95 and 0.99. The frozen factor had been derived for resampling all the seeds of a level together. At levels with two strata the variance of the seed term was therefore the stated variance multiplied by the ratio of the within-stratum to the total sum of squares (between 0.43 and 1, median 0.97, for the differences from ERM). The corrected factor calibrates the term with the variance that the stratified resampling actually produces, so that the variance of the term equals $\hat\sigma^2_{S,r}/K_r$ at every level that can be calibrated.

No model was retrained, and no estimand, hypothesis family, margin, level, Holm correction, number of resamples, random seed, model selection rule or data set changed. Of the 786 intervals reviewed, 303 change, and no confirmatory or secondary decision changes. The intervals and p values of the article use the corrected factor.

- **Code.** `scripts/correccion/`. `ec6.py` holds the corrected factor; it imports the frozen module `src/eval/jerarquico.py` and replaces two of its functions at run time, only inside an explicit context, so the frozen code is not modified. `lanzador.py` calls the frozen report functions twice for each affected table: with the original factor, as a control that must reproduce the published table exactly (it did, with a maximum difference of 0 in every numeric column), and with the corrected factor. `simulacion.py` checks the calibration by simulation with the seed structure of the benchmark, `impacto.py` builds the before-and-after table, and `complementos.py` computes three post hoc supplementary calculations (seed-and-patient interval for the primary reference of the audit, Holm correction with the alternative intervals, and the epoch-selection loss on the scale of the area). The execution chain is logged in `results/cadenas/corr_cadena.json`.
- **Original tables.** In their original places: `results/b3/test_A_best/`, `results/b3/test_A_bestdc/`, `results/c2c3/` (`c3-epoca.csv`, `c3_epoca_por_metodo.csv`, `fuente.csv`, `orientacion.csv`, `orientacion_b_curva.csv`) and `results/sensibilidad/tronco/`, `results/sensibilidad/etiqueta/`.
- **Corrected tables.** `results/correccion_ec6/`, with the same relative path, file name and columns as each original (for example `results/correccion_ec6/b3/test_A_best/comparacion_erm.csv`). Tables whose intervals use a single stratum (flipped test set, specificity gap, tuberculosis evaluation, the own curves of the reverse orientation, the second backbone and the second label) are bit-identical with either factor and have no corrected copy.
- **Impact and checks.** `results/correccion_ec6/antes_despues_todas_las_filas.csv` gives, for the 786 rows, the estimate, the interval and the p values before and after, the ratio of widths and any change of decision (Supplementary Table S10); `antes_despues.csv` and `.md` list the rows that change. `results/correccion_ec6/simulacion/` holds the simulation (Supplementary Table S11), and `results/correccion_ec6/complementos/` the supplementary calculations, with their control files.

The programs in `scripts/correccion/` read test predictions, so they first call the freeze check (`src/eval/desbloqueo.py`), which requires every hashed file of the original project, network checkpoints included. They therefore cannot be rerun from this folder; the tables they wrote can be checked against the predictions in `predictions/` with the functions of `src/eval/`.

## Table numbering

The programs use their own table numbers: nine tables of the main text (Tables 1 to 9) and 38 supplementary tables (Tables S1 to S38, the numbers used by `scripts/suplemento/tablas.py`, by `results/suplemento/tabla_SNN.csv` and in the comments of the code). In the article all 47 tables are Supplementary Tables S1 to S47. `manuscript/srep/correspondencia_tablas.csv` gives the correspondence: column `antigua` is the number used by the programs, `nueva` the number in the article, `origen` whether the table was in the main text (`texto principal`) or in the supplementary material (`material suplementario`), and `primera_cita` the section of the article where it is first cited. Figures keep their numbers (Figures 1 to 7 and Supplementary Figures S1 to S8).

## Where each table and figure comes from

Paths in the last column are relative to `results/` unless they start with `src/`, `data/`, `configs/` or `docs/`. A path that starts with `correccion_ec6/` is the version recomputed with the corrected calibration factor of the file with the same relative path, which is also included (see "Post hoc correction of the seed-term calibration factor").

| Article (Supplementary Table or Figure) | Number in the programs | Script | Source files |
|---|---|---|---|
| S1 (compared methods) | Table 3 |  | `src/baselines/`, `configs/` |
| S2 (datasets) | Table 1 | `scripts/data/build_*.py` | `data/manifests/` and their JSON sidecars |
| S3 (shared preprocessing and training protocol) | Table 2 |  | `configs/`, `src/baselines/` |
| S4 (overlap of the lung segmentation network's training set with the controlled benchmarks) | Table S1 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S01.csv`) | `suplemento_fuentes/solape_segmentador.csv`, `suplemento_fuentes/solape_segmentador_calidad.csv` |
| S5 (hyperparameter search space) | Table S2 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S02.csv`) | `b1/busqueda_todas.csv`, `src/baselines/` (read as text) |
| S6 (configurations selected in COVIDx CXR-4 and tuberculosis) | Table S3 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S03.csv`) | `b1/congelados/*.json`, `b2/congelados/*.json`, `src/baselines/` (read as text) |
| S7 (configurations selected in the pleural effusion benchmark) | Table S4 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S04.csv`) | `b3a/congelados/*.json`, `src/baselines/` (read as text) |
| S8 (validation of the final COVIDx runs) | Table S5 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S05.csv`) | `b1/finales_validacion.csv` |
| S9 (hypothesis families) | Table 4 |  | `docs/08_evaluacion.md` |
| S10 (effect of the correction of the seed-term calibration factor) | Table S6 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S06.csv`) | `correccion_ec6/antes_despues_todas_las_filas.csv` |
| S11 (simulation check of the calibration factor) | Table S7 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S07.csv`) | `correccion_ec6/simulacion/simulacion.csv` |
| S12 (software environment) | Table S8 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S08.csv`) | `gate7/congelacion.json`, `gate7/entorno_contenedor.json` |
| S13 (number of training runs) | Table S9 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S09.csv`) | `cadenas/*.json` |
| S14 (BIMCV part of the test set and COVIDGR) | Table 5 | `scripts/eval/auditoria_covidx.py` | `c1/test/f_c1.csv`, `c1/test/f_c1b.csv`, `c1/test/detectores.csv`, `c1/test/c1_vi_covidgr.csv` |
| S15 (confirmatory family of the audit) | Table S10 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S10.csv`) | `c1/test/f_c1.csv` |
| S16 (secondary family of the audit) | Table S11 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S11.csv`) | `c1/test/f_c1b.csv`, `sensibilidad_intervalos/c1b_jerarquico.csv` |
| S17 (class-blind references, backbone by backbone) | Table S12 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S12.csv`) | `c1/test/detectores.csv` |
| S18 (AP and PA strata) | Table S13 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S13.csv`) | `c1/test/c1_vii_estratos.csv`, `c1/test/f_c1.csv` |
| S19 (AUC by collection of origin) | Table S14 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S14.csv`) | `c1/test/c1_iii_por_coleccion.csv` |
| S20 (linear probes across collections) | Table S15 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S15.csv`) | `c1/test/c1_ii_reservado.csv` |
| S21 (acquisition-only classifier and masked ERM) | Table S16 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S16.csv`) | `c1/test/c1_iv_adquisicion.csv`, `c1/test/c1_v_mascara.csv` |
| S22 (external cohort COVIDGR) | Table S17 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S17.csv`) | `c1/test/c1_vi_covidgr.csv` |
| S23 (sensitivity by radiological severity) | Table S18 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S18.csv`) | `c1/test/c1_vi_gravedad.csv` |
| S24 (AUC by confounding level) | Table 6 | `scripts/eval/informe_b3.py` | `correccion_ec6/b3/test_A_best/curva.csv` |
| S25 (dose-response curve, pleural effusion benchmark) | Table S19 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S19.csv`) | `correccion_ec6/b3/test_A_best/curva.csv` |
| S26 (differences from ERM) | Table 7 | `scripts/eval/informe_b3.py` | `correccion_ec6/b3/test_A_best/comparacion_erm.csv` |
| S27 (full comparison with ERM) | Table S20 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S20.csv`) | `correccion_ec6/b3/test_A_best/comparacion_erm.csv` |
| S28 (flipped test set) | Table S21 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S21.csv`) | `c2c3/invertido.csv` |
| S29 (AUC within each source) | Table S22 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S22.csv`) | `correccion_ec6/c2c3/fuente.csv` |
| S30 (specificity gap) | Table S23 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S23.csv`) | `c2c3/brecha.csv` |
| S31 (reverse orientation) | Table S24 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S24.csv`) | `correccion_ec6/c2c3/orientacion.csv`, `correccion_ec6/c2c3/orientacion_b_curva.csv` |
| S32 (tuberculosis evaluation) | Table S25 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S25.csv`) | `b2/informe_test/comparacion_erm.csv`, `b2/informe_test/curva.csv` |
| S33 (epoch-selection loss) | Table 8 | `scripts/eval/informe_c2_c3.py`; `scripts/figs/figB7_seleccion.py` writes `figs/figB7_seleccion.csv` | `correccion_ec6/c2c3/c3-epoca.csv`, `figs/figB7_seleccion.csv`, `correccion_ec6/b3/test_A_best/comparacion_erm.csv` |
| S34 (epoch-selection loss by method) | Table S26 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S26.csv`) | `correccion_ec6/c2c3/c3-epoca.csv`, `correccion_ec6/c2c3/c3_epoca_por_metodo.csv` |
| S35 (epoch-selection loss on the area scale) | Table S27 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S27.csv`) | `correccion_ec6/complementos/e2_diferencias_area_frente_erm.csv`, `correccion_ec6/complementos/e2_perdida_seleccion_area.csv` |
| S36 (hyperparameter selection) | Table S28 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S28.csv`) | `c2c3/c3-hiper.csv` |
| S37 (stability of the hyperparameter search) | Table S29 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S29.csv`) | `sensibilidad/estabilidad/estabilidad.csv` |
| S38 (epoch-selection loss in tuberculosis) | Table S30 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S30.csv`) | `b2/informe_test/seleccion_epoca.csv` |
| S39 (agreement of the sensitivity analyses) | Table 9 | `scripts/eval/informe_sensibilidad.py` | `correccion_ec6/sensibilidad/tronco/concordancia.csv`, `correccion_ec6/sensibilidad/etiqueta/concordancia.csv`, `correccion_ec6/b3/test_A_bestdc/comparacion_erm.csv`, `correccion_ec6/b3/test_A_bestdc/resumen.csv` |
| S40 (second backbone and second label: AUC) | Table S31 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S31.csv`) | `correccion_ec6/sensibilidad/etiqueta/curva.csv`, `correccion_ec6/sensibilidad/tronco/curva.csv` |
| S41 (second backbone and second label: differences from ERM) | Table S32 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S32.csv`) | `correccion_ec6/sensibilidad/etiqueta/comparacion_erm.csv`, `correccion_ec6/sensibilidad/etiqueta/resumen.csv`, `correccion_ec6/sensibilidad/tronco/comparacion_erm.csv`, `correccion_ec6/sensibilidad/tronco/resumen.csv` |
| S42 (oracle selection: curve) | Table S33 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S33.csv`) | `correccion_ec6/b3/test_A_bestdc/curva.csv` |
| S43 (oracle selection: comparison with ERM) | Table S34 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S34.csv`) | `correccion_ec6/b3/test_A_bestdc/comparacion_erm.csv` |
| S44 (sensitivity analyses of the audit) | Table S35 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S35.csv`) | `c1/test/detectores.csv`, `c1/test/f_c1.csv`, `sensibilidad_v28/resumen_familias.csv`, `sensibilidad_v28/s7_composicion.csv` |
| S45 (alternative intervals for the area) | Table S36 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S36.csv`) | `correccion_ec6/b3/test_A_best/comparacion_erm.csv`, `correccion_ec6/complementos/e3_area_alternativos.csv` |
| S46 (alternative intervals for the epoch-selection loss) | Table S37 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S37.csv`) | `correccion_ec6/c2c3/c3-epoca.csv`, `correccion_ec6/c2c3/c3_epoca_por_metodo.csv`, `correccion_ec6/complementos/e3_seleccion_alternativos.csv`, `sensibilidad_intervalos/c3_epoca_media_alternativos.csv`, `sensibilidad_intervalos/c3_epoca_por_metodo_alternativos.csv` |
| S47 (audit with the seed-and-patient interval) | Table S38 | `scripts/suplemento/tablas.py` (`results/suplemento/tabla_S38.csv`) | `correccion_ec6/complementos/e1_c1_din_jerarquico.csv` |
| Fig. 1 (BIMCV collections by split) | | `scripts/figs/en/figs_principales_en.py 1` | `figs/figB1_colecciones.csv` (written by `scripts/figs/figB1_colecciones.py` from the manifests) |
| Fig. 2 (audit forest plot) | | `... figs_principales_en.py 2` | `c1/test/f_c1.csv`, `c1/test/f_c1b.csv` |
| Fig. 3 (collection signal) | | `... figs_principales_en.py 3` | `c1/test/c1_iii_resumen.csv`, `c1/test/c1_ii_reservado.csv` |
| Fig. 4 (COVIDGR) | | `... figs_principales_en.py 4` | `c1/test/c1_vi_covidgr.csv`, `c1/test/c1_iii_resumen.csv` |
| Fig. 5 (dose-response curves) | | `... figs_principales_en.py 5` | `correccion_ec6/b3/test_A_best/curva.csv`, `correccion_ec6/b3/test_A_best/comparacion_erm.csv` |
| Fig. 6 (area and cost without confounding) | | `... figs_principales_en.py 6` | `correccion_ec6/b3/test_A_best/comparacion_erm.csv` |
| Fig. 7 (model selection) | | `... figs_principales_en.py 7` | `correccion_ec6/c2c3/c3-epoca.csv`, `correccion_ec6/c2c3/c3_epoca_por_metodo.csv`, `correccion_ec6/b3/test_A_best/comparacion_erm.csv` |
| Fig. S1 (pilot) | | `scripts/figs/en/figs_suplementarias_en.py 1` | `gate6/tabla_supervisor.csv` (written by `scripts/eval/puerta6.py`) |
| Fig. S2 (AP and PA strata) | | `... figs_suplementarias_en.py 2` | `c1/test/c1_vii_estratos.csv` |
| Fig. S3 (acquisition-only and masked) | | `... figs_suplementarias_en.py 3` | `c1/test/c1_iii_resumen.csv`, `c1/test/c1_iv_adquisicion.csv`, `c1/test/c1_v_mascara.csv` |
| Fig. S4 (flipped test set) | | `... figs_suplementarias_en.py 4` | `c2c3/invertido.csv`, `correccion_ec6/b3/test_A_best/comparacion_erm.csv` |
| Fig. S5 (reverse orientation) | | `... figs_suplementarias_en.py 5` | `correccion_ec6/b3/test_A_best/curva.csv`, `correccion_ec6/c2c3/orientacion_b_curva.csv`, `correccion_ec6/c2c3/orientacion.csv` |
| Fig. S6 (tuberculosis evaluation) | | `... figs_suplementarias_en.py 6` | `b2/informe_test/comparacion_erm.csv`, `b2/informe_test/curva.csv` |
| Fig. S7 (epoch selection by method) | | `... figs_suplementarias_en.py 7` | `correccion_ec6/c2c3/c3_epoca_por_metodo.csv` |
| Fig. S8 (hyperparameter selection) | | `... figs_suplementarias_en.py 8` | `c2c3/c3-hiper.csv` |

Each figure is also saved with the CSV of the values it plots (`results/figs/en/Fig*.csv`). The Spanish figure scripts in `scripts/figs/` produce `results/figs/figB*` and `figSB*` (same data, Spanish labels). The tables under `results/` were produced from the per-image predictions by the evaluation programs in `scripts/eval/`, `scripts/sensibilidad/` and `scripts/correccion/`, using the statistics in `src/eval/` (`stats.py`, `jerarquico.py`, `dosis.py`, `c1_auditoria.py`, `c2.py`, `c3.py`).

## Per-image predictions

`predictions/` holds the predicted probabilities of every training run used in the article, as CSV files packed in seven `tar.xz` archives (about 77 MB in total, 1.8 GB uncompressed). They allow the intervals and tests to be recomputed. Each archive keeps the original relative paths, so extracting from the repository root restores them under `results/`:

```bash
for f in predictions/*.tar.xz; do tar -xJf "$f"; done
```

| Archive | Content |
|---|---|
| `predictions_01_covidx_test.tar.xz` | COVIDx test predictions of the 60 final runs (best, EMA and last checkpoints; full clean test and BIMCV part) and COVIDGR predictions (`results/b1/test`, `results/b1/covidgr`) |
| `predictions_02_covidx_val.tar.xz` | Validation predictions of the final COVIDx runs and of the held-out and pilot ERM backbones (`results/train/baseline_*_final`, `expl_erm_reserva*`, `baseline_erm`) |
| `predictions_03_effusion_a.tar.xz`, `predictions_04_effusion_b.tar.xz` | Pleural effusion benchmark, final runs, NIH as positive source (validation, deconfounded validation, deconfounded test and flipped test; best and oracle checkpoints) |
| `predictions_05_reverse_tb_search.tar.xz` | Reverse orientation (`b3a_B_*`), tuberculosis final runs (`b2_*`, tuberculosis evaluation) and the hyperparameter-selection rerun with test evaluation (`results/c3_busqueda`) |
| `predictions_06_sensitivity.tar.xz` | Predictions without horizontal-flip averaging (`results/sensibilidad_v28/sin_espejo_pred`), ResNet-50, cardiomegaly and search-stability runs |
| `predictions_07_search_trials_other.tar.xz` | Hyperparameter search trials of the three benchmarks (validation only) and the earlier ERM and baseline pilots, including the reproduction runs of published figures |

Columns: `image_id`, `label`, `source` (index), `patient`, `prob` (probability of the positive class), and in some files the checkpoint or extra identifiers. The predictions of three pilot runs of the abandoned architecture are not included.

## Freeze record

On 17 November 2026 (2026-11-17T23:32:48Z) the prespecified analysis plan, the code, the configurations and the selected hyperparameters were frozen internally by recording the SHA-256 hashes of 1129 files with `scripts/eval/congelar.py` and the list in `configs/gate7_congelacion.yaml`. The record is `results/gate7/congelacion.json` (groups of files, hash of each file, analysis environment, container environment); `results/gate7/congelacion.sha256sum` is the same list in `sha256sum` format. The SHA-256 of `congelacion.json` itself is `db617b620c2b9a6c7ef588898982006f0d4731457bd63ae0f6d367c10f6e24dc`. Every program that reads a test set checks the hashes first (`src/eval/desbloqueo.py`). In the original project each such check was appended to `results/gate7/desbloqueos.jsonl`; that log is not included here because it records host names, user names and paths of the working environment.

Not every hashed file is redistributed. Of the 1129 files, 858 are included here; the other 271 are network checkpoints of the final COVIDx runs and of the exploration backbones (201 files), feature arrays and their index files (64), acquisition statistics (4), one internal working document, and the run report of the search-stability analysis (`results/sensibilidad/estabilidad/informe.json`), withheld because it records working-environment paths (its results are in `results/sensibilidad/estabilidad/estabilidad.csv`).

**Edited copy.** The comments and the documentation of this copy were edited for release: internal notes of the working process were removed. The executable code is the frozen code: for every Python file the syntax tree, once docstrings are removed, is identical to that of the original file, except for 12 files (6 of them frozen) in which informative text strings lost such a note (a log message, an error message, free-text fields of job descriptions, the reason passed to the unblinding check, and the header comment written by the bibliography helper); those strings take no part in any computation, path, key or comparison. One further frozen file, `src/data/paths.py`, has the name of a server folder changed in `saved_root()`, a path that only the scripts reproducing the figures of an earlier study (`scripts/jobs/gate2a_inference.py`, `gate2b_compare.py`) use and that none of the analyses of this article reads. The SHA-256 hashes of the record therefore correspond to the original files, not to the files of this copy. Of the 858 frozen files included, 784 keep their recorded hash and 74 differ: 53 Python files only in comments and docstrings, 7 Python files also in text strings (the 6 frozen files with informative strings just described and `src/data/paths.py`), 11 YAML files only in comments, 1 YAML file (`configs/gate7_congelacion.yaml`) in comments and in two free-text `nota` fields (the loaded content is otherwise identical), and 2 Markdown documents (`docs/08_evaluacion.md`, `data/manifests/README.md`) in wording only. The code added after the freeze (`scripts/correccion/` and the later versions of `scripts/suplemento/`, `scripts/figs/` and `scripts/refs/`) is not part of the record. Given a copy of the original tree, `python3 scripts/release/check_ast.py <original tree>` repeats the comparison for every Python file.

To check the files that are present (the 74 edited files are reported as FAILED):

```bash
sha256sum -c --ignore-missing results/gate7/congelacion.sha256sum
```

`python3 scripts/eval/congelar.py --verificar` requires all 1129 files and therefore fails on this repository by design.

## Absolute paths

Some scripts and documents contain absolute paths of the original server or of the working environment (for example `/home/administrador/...` in `src/data/paths.py`, `runner/`, `scripts/data/`, `scripts/download/`, `tests/test_runner.py` and `data/manifests/README.md`). These paths are part of the executable code (default values, path candidates) or document it, so they were not edited. Paths are those of the original server; adjust them with the environment variables provided (`PAPER_NEW_DATA_ROOT`, `PAPER_NEW_DATA_DIR`, `RAW_ROOT`, `LOG_DIR`) or edit a local copy, knowing that an edited file will no longer match the freeze record.

## Language note and glossary

The code, comments, documentation inside the code, column names and directory names are in Spanish. The article and this README are in English. Frequent terms:

| Spanish | English |
|---|---|
| `metodo`, `semilla`, `semillas`, `n_semillas` | method, seed, seeds, number of seeds |
| `rho`, `nivel` | confounding level ρ, level |
| `auc_media_semillas`, `dt_semillas` | mean AUC over seeds, standard deviation over seeds |
| `delta`, `ic95_inf`, `ic95_sup` | difference, lower and upper bound of the 95% CI |
| `p_unilateral`, `p_holm`, `rechaza_holm` | one-sided p value, Holm-adjusted p value, rejected after Holm correction |
| `no_inferior`, `margen` | non-inferior, margin |
| `familia`, `papel`, `confirmatoria` | hypothesis family, role, confirmatory |
| `pacientes`, `doble`, `jerarquico` | patient bootstrap, double bootstrap, hierarchical bootstrap over seeds and patients |
| `referencia`, `detector`, `D_IN`, `D_ERM`, `D_MASK`, `D_RES` | reference; collection detector on the ImageNet network (primary reference), on the final ERM backbones (secondary reference), on the masked ERM backbones, on the backbones trained with a held-out set |
| `coleccion`, `antigua`, `nueva` | collection; `bimcv_pos_antigua` = five-digit collection (BIMCV positives with five-digit subject identifiers), `bimcv_pos_nueva` = six-digit collection (six-digit identifiers) |
| `fuente`, `control_fuente` | source, source control |
| `erm_mascara`, `erm_cuotas`, `dfa_fuente`, `fourier` | masked ERM, quota ERM, source-aware DFA, Fourier amplitude mixing |
| `conjunto`, `particion`, `entrenamiento`, `validacion` | set, split, training, validation |
| `test_desconf`, `val_desconf`, `test_invertido` | deconfounded test set, deconfounded validation, flipped test set |
| `best`, `bestdc`, `ema`, `last` | checkpoint selected by same-ρ validation, by deconfounded validation (oracle), exponential moving average of the weights, last epoch |
| `reserva`, `reservado` | held-out set |
| `adquisicion`, `mascara`, `estratos`, `gravedad` | acquisition-only classifier, lung mask, AP and PA strata, radiological severity |
| `curva`, `area`, `caida`, `brecha`, `invertido`, `orientacion` | dose-response curve, area, drop, specificity gap, flipped, orientation |
| `seleccion`, `epoca`, `hiper`, `ensayo` | model selection, epoch, hyperparameters, trial |
| `busqueda`, `congelados`, `finales` | search, frozen configurations, final runs |
| `cadenas`, `corrida` | execution chains, run |
| `sensibilidad`, `tronco`, `etiqueta`, `estabilidad` | sensitivity analysis, backbone (ResNet-50), label (cardiomegaly), search stability |
| `congelacion`, `desbloqueo` | freeze of the analysis plan, unblinding check |
| `correccion`, `ec6`, `factor`, `antes`, `despues` | correction (of the seed-term calibration factor), before, after |
| `registrado`, `registrada` (in comments and file names) | prespecified in the frozen plan |

Internal codes used in file and directory names: `gate6`, `puerta6` and "Puerta 6" = pilot of the pleural effusion benchmark; `gate7` and "Puerta 7" = freeze of the analysis plan (both names remain in file names and in messages of the code); "Puerta 2a", "Puerta 2b", "Puerta 3" = checkpoints of the pilot of the abandoned architecture; `b1` = COVIDx CXR-4; `b2` = tuberculosis evaluation; `b3a` = pleural effusion benchmark (`A` or `apos`: NIH as positive source; `B` or `bpos`: reverse orientation); `b3c` = cardiomegaly benchmark; `c1` = audit; `c2` and `c3` = controlled-confounding and model-selection questions; `s0` to `s7` in `results/sensibilidad_v28` = the prespecified audit variants of Supplementary Table S44.

## Licence

Code: MIT License (`LICENSE`). Result tables, split manifests, predictions and figures: Creative Commons Attribution 4.0 International (`LICENSE-DATA.md`). The public databases keep their own licences; nothing here grants rights over the images.

## How to cite

Please cite the article and this archive. Citation metadata is in `CITATION.cff`.

Valero Gómez JC, Clares Perca JC, Zuñiga Incalla AP. A Class-Blind Collection Detector Matches Nearly All Methods in the Internal Evaluation of COVIDx CXR-4. Code and data: [repository URL], archived at [DOI to be added after the Zenodo deposit].
