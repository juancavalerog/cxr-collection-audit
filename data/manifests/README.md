# Manifiestos de los datos ya presentes en disco

Generados el 12 de noviembre de 2026 con `scripts/data/build_manifests.py`
y comprobados con `scripts/data/check_manifests.py` (salida completa en `check_manifests.log`).

**Raiz de las rutas.** Toda columna `image_path` es relativa a `/home/administrador/notebooks/covid/`
(en la VM de trabajo, `$HOME/mnt/notebooks/covid/`). Ejemplo: `nih/images/00000001_000.png`.

Regenerar:

```bash
python3 scripts/data/build_manifests.py --data-root /home/administrador/notebooks/covid \
    --out data/manifests --check-exists
python3 scripts/data/check_manifests.py --manifests data/manifests \
    --data-root /home/administrador/notebooks/covid --full > data/manifests/check_manifests.log
```

## covidx_cxr4.csv (84 818 filas)

| Columna | Definicion |
|---|---|
| dataset | `covidx_cxr4` |
| source | fuente declarada en los .txt oficiales: actmed, bimcv, cohen, fig1, ricord, rsna, sirm, stonybrook |
| image_path | `data/dataset/<fichero>`; todas las imagenes estan en `data/dataset/` (las carpetas `val/` y `test/` estan vacias) |
| patient_id | primer campo de los .txt oficiales (en RSNA es un UUID por imagen, ver mas abajo) |
| label | 1 = COVID-19 positivo, 0 = negativo |
| split | train, val, test |

Formato real de `dataset.txt`, `val.txt` y `test.txt`: cuatro campos separados por un espacio y sin cabecera,
`<patient_id> <fichero> <positive|negative> <fuente>`. `dataset.txt` contiene las 84 818 imagenes, incluidas
las de val y test; el train se obtiene como `dataset.txt` menos `val.txt` y `test.txt` emparejando por nombre
de fichero (los nombres son unicos). Las filas de val y test coinciden con las de `dataset.txt` en etiqueta,
fuente y paciente (0 discrepancias).

Conteos (reproducen la Tabla 1 de Wu et al., arXiv 2311.17677):

| split | imagenes | positivas | negativas | pacientes |
|---|---|---|---|---|
| train | 67 863 | 57 199 | 10 664 | 35 457 |
| val | 8 473 | 4 241 | 4 232 | 5 163 |
| test | 8 482 | 4 241 | 4 241 | 4 722 |

Negativos de train: RSNA 9 719, BIMCV 647, Cohen 220, ActMed 78. La tabla clase por fuente completa esta en
`check_manifests.log`.

## nih_cxr14.csv (112 120 filas)

Desde `nih/Data_Entry_2017.csv`. Columnas: `dataset` (`nih_cxr14`), `source` (`nih_cc`), `image_path`,
`patient_id` (`nih_<Patient ID>`), `view` (PA o AP), `labels` (cadena original separada por `|`), `age`, `sex`,
`followup`, una columna binaria `y_<etiqueta>` por cada una de las 14 patologias y `y_no_finding`, y `split`
(vacia: los ficheros oficiales `train_val_list.txt` y `test_list.txt` no estan en disco). 30 805 pacientes;
PA 67 310, AP 44 810. Positivos de interes: derrame (`y_effusion`) 13 317 imagenes de 4 275 pacientes;
cardiomegalia 2 776 de 1 566; sin hallazgos 60 361 de 24 907.

## chexpert_small.csv (223 648 filas)

Desde `CheXpert-v1.0-small/train.csv` (223 414) y `valid.csv` (234). Columnas: `dataset` (`chexpert_small`),
`source` (`stanford`), `image_path` (`CheXpert/<Path>`), `patient_id` (`chx_patientNNNNN`), `study_id`, `view`
(Frontal o Lateral), `ap_pa`, `age`, `sex`, `y_<etiqueta>` para las 14 observaciones con la convencion original
(1 positivo, 0 negativo, -1 incierto, vacio no mencionado), `labels` (observaciones con valor 1) y `split`
(train o valid, la particion oficial). 64 740 pacientes; frontales 191 229, laterales 32 419.

## covidx_cxr4_clean.csv (84 818 filas): versión limpia del Benchmark 1

Generada por `scripts/data/build_covidx_clean.py` (registro en `logs/tmp_datos/build_covidx_clean.log`) a partir de
`covidx_cxr4.csv`, `covidx_rsna_to_nih_map.csv`, `covidx_cross_split_duplicates.csv` y
`covidx_label_conflicts.csv`. Conserva todas las filas: las retiradas llevan `split_clean = excluded`.

| Columna | Definición |
|---|---|
| dataset, source, image_path, patient_id, label, split | como en `covidx_cxr4.csv` (`split` es la partición oficial) |
| nih_patient_id | paciente NIH enlazado por hash (`nih_<id>`), solo en las 13 756 imágenes RSNA con copia confirmada; vacío en el resto |
| split_clean | train, val, test o excluded |
| exclude_reason | vacío, o razones separadas por `;` (ver reglas) |
| val_robust | 1 si la imagen está en la validación limpia y es de BIMCV |
| test_subset | solo en test oficial: bimcv, ricord, rsna_sin_fuga, rsna_con_fuga |

Reglas, aplicadas en este orden:

1. `rsna_nih_patient_in_test`: imagen RSNA de train o val oficial cuyo paciente NIH tiene alguna imagen en el test
   oficial. Se excluye. 655 imágenes (591 de train, 64 de val).
2. `nih_patient_regrouped_to_train` o `nih_patient_regrouped_to_val`: paciente NIH con imágenes en train y en val
   (tras la regla 1). Todas sus imágenes pasan a la partición donde tiene más; en empate, a train. No se excluyen.
   886 imágenes pasan de val a train y 143 de train a val.
3. `label_conflict`: los dos miembros de cada par casi idéntico con etiqueta opuesta se excluyen, también en test.
   8 imágenes (4 pares; 3 imágenes del test).
4. `duplicate_cross_split`: en cada par casi idéntico (con la misma etiqueta) que sigue en particiones distintas
   tras la regla 2, se excluye el miembro de train si el otro está en val o test, y el de val si el otro está en
   test. 80 imágenes. De los 86 pares, 2 caen en la regla 3 y el resto quedan resueltos por esta regla o por la 2.
5. `val_robust` y `test_subset` se calculan después. `rsna_con_fuga` marca las imágenes RSNA del test cuyo paciente
   NIH tiene imágenes en train o val **oficiales** (129); tras la regla 1 ninguna conserva esos pacientes en la
   versión limpia.

Recuentos (negativas / positivas):

| Fuente | train limpio | val limpio | test limpio | excluidas (partición oficial) |
|---|---|---|---|---|
| bimcv | 647 / 42 493 | 257 / 3 150 | 4 039 / 4 040 | 2 de train, 1 de val, 3 de test |
| rsna | 9 871 / 0 | 3 062 / 0 | 200 / 0 | 591 de train, 64 de val |
| stonybrook | 0 / 12 696 | 0 / 940 | ausente | 0 |
| ricord | 0 / 834 | 0 / 62 | 0 / 200 | 0 |
| sirm | 0 / 804 | 0 / 65 | ausente | 74 de train |
| cohen | 219 / 246 | 76 / 19 | ausente | 6 de train, 1 de val |
| actmed | 78 / 23 | 29 / 2 | ausente | 0 |
| fig1 | 0 / 21 | 0 / 2 | ausente | 1 de train |
| **Total** | **67 932** | **7 664** | **8 479** | **743** |

RSNA de train crece de 9 719 a 9 871 porque la regla 2 mueve a train 886 imágenes de val y saca 143, después de
haber retirado 591 por la regla 1.

Validación robusta (`val_robust = 1`): 3 407 imágenes BIMCV de 999 pacientes (257 negativas y 3 150 positivas).
Si se amplía a las otras fuentes con ambas clases en val, Cohen añadiría 95 imágenes de 59 pacientes (76 y 19) y
ActMed 31 de 28 pacientes (29 y 2).

Subconjuntos del test oficial (`test_subset`): bimcv 8 082, ricord 200, rsna_con_fuga 129, rsna_sin_fuga 71.

Pares con etiqueta opuesta (`covidx_label_conflicts.csv`; todas las rutas relativas a la raíz de datos):

| # | Imagen A (partición, fuente, etiqueta) | Imagen B (partición, fuente, etiqueta) | Correlación | Propuesta |
|---|---|---|---|---|
| 1 | `data/dataset/sub-S322642_ses-E45761_run-1_bp-chest_vp-pa_cr-corrected.png` (train, bimcv, 1) | `data/dataset/sub-S06286_ses-E11310_run-1_bp-chest_vp-pa_cr-corrected.png` (test, bimcv, 0) | 0.977 | correlación en el límite: revisión visual; si son imágenes distintas, restaurar ambas |
| 2 | `data/dataset/COVID(267).png` (train, sirm, 1) | `data/dataset/F051E018-DAD1-4506-AD43-BE4CA29E960B.jpeg` (val, cohen, 0) | 0.9999 | misma imagen con etiqueta distinta en dos repositorios: excluir ambas y consultar los metadatos de Cohen |
| 3 | `data/dataset/sub-S09331_ses-E20067_run-1_bp-chest_vp-pa_dx-corrected.png` (test, bimcv, 1) | `data/dataset/sub-S06833_ses-E13728_run-2_bp-chest_vp-pa_dx-corrected.png` (test, bimcv, 0) | 1.000 | misma imagen asignada a dos sujetos BIMCV con etiqueta opuesta: excluir ambas del test limpio e informarlo |
| 4 | `data/dataset/covid-19-pneumonia-58-prior.jpg` (train, cohen, 0) | `data/dataset/COVID(673).png` (train, sirm, 1) | 1.000 | Cohen la marca como estudio previo (negativo) y SIRM como positiva: excluir ambas |

La versión limpia aplica la exclusión de los cuatro pares; la propuesta 1 queda pendiente de revisión visual.

## Manifiestos derivados de las comprobaciones de solapamiento

- `covidx_rsna_to_nih_map.csv`: 13 756 imagenes RSNA de COVIDx con su copia confirmada en NIH ChestX-ray14
  (columnas `path_covidx`, `path_nih`, `nih_patient_id`, `split`, `nih_labels`, `phash_dist`, `thumb_corr`).
- `covidx_cross_split_duplicates.csv`: 86 pares de imagenes casi identicas de COVIDx en particiones distintas.
- `covidx_label_conflicts.csv`: 4 pares casi identicos con etiqueta opuesta (2 de ellos dentro de una misma particion).

Metodo y cifras en `docs/02_datos.md` (Section 4) y `logs/tmp_datos/overlap_summary.txt`.

## Resultado de las comprobaciones (check_manifests.log)

- Conteos de COVIDx: coinciden con los esperados (train 67 863 con 57 199 positivas y 10 664 negativas; val
  8 473; test 8 482; negativos de train RSNA 9 719 y BIMCV 647).
- Existencia de ficheros: comprobacion completa, 0 ausentes en los tres manifiestos.
- Pacientes compartidos entre particiones con el `patient_id` oficial: 0 en COVIDx (con clave fuente+paciente y
  con paciente solo) y 0 en CheXpert (train contra valid).
- Fuga oculta (AVISO, no cuenta como fallo de la particion oficial): mediante el enlace RSNA a NIH, 923
  pacientes NIH aparecen en train y val, 114 en train y test, 44 en val y test. El `patient_id` oficial de RSNA
  es un UUID por imagen y no permite verlo.
- Versión limpia (`covidx_cxr4_clean.csv`): 655 exclusiones por la regla 1; 0 pacientes NIH y 0 pares con clave
  fuente+paciente compartidos entre particiones limpias; 0 pares casi idénticos repartidos entre particiones; las 8
  imágenes en conflicto de etiqueta excluidas; `val_robust` solo BIMCV; `test_subset` con los recuentos esperados.
- `check_manifests.log` termina con: todas las comprobaciones pasan.
