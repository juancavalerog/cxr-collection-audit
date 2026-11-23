# COVIDx CXR-4: totales por partición y comprobación contra las cifras publicadas

Generado el 12 de noviembre de 2026 con
`scripts/inventario/manifest_covidx.py`. Registro de la ejecución en
`logs/tmp_inventario/manifest_log.txt`. Tabla detallada en
`data/manifests/covidx_cxr4_clase_fuente.csv` (columnas `split,source,label,n_images,n_patients`).

## Origen y método

- Ficheros: `~/mnt/notebooks/covid/data/dataset.txt` (84 818 filas), `val.txt` (8 473) y `test.txt` (8 482).
- Formato real, comprobado antes de procesar: sin cabecera, separador espacio, 4 columnas en todas las filas
  (`patient_id filename label source`). Etiquetas `positive` y `negative`. Fuentes: actmed, bimcv, cohen, fig1,
  ricord, rsna, sirm, stonybrook.
- Train = `dataset.txt` menos los ficheros de `val.txt` y `test.txt`, emparejando por nombre de fichero.
- `n_patients` cuenta valores distintos de `patient_id` dentro de cada combinación de partición, fuente y clase.
  Un paciente con imágenes de ambas clases cuenta en las dos filas, así que las sumas de `n_patients` por partición
  pueden superar los pacientes únicos de esa partición.

## Comprobaciones de integridad

| Comprobación | Resultado |
|---|---|
| Ficheros de val ausentes de `dataset.txt` | 0 |
| Ficheros de test ausentes de `dataset.txt` | 0 |
| Ficheros compartidos entre val y test | 0 |
| Filas de val o test cuya etiqueta, fuente o paciente difiere de `dataset.txt` | 0 en las tres columnas |
| Pacientes compartidos train-val, train-test y val-test | 0, 0 y 0 |
| Ficheros de `dataset.txt` presentes en `data/dataset/` | 84 818 de 84 818; ningún fichero sobrante |
| Ficheros en `data/val/` y `data/test/` | 0 y 0 (carpetas vacías; todo vive en `data/dataset/`) |
| Pacientes únicos | 45 342 en total; 5 163 en val; 4 722 en test |

## Totales por partición

| Partición | Total | Positivas | Negativas |
|---|---|---|---|
| Train | 67 863 | 57 199 | 10 664 |
| Val | 8 473 | 4 241 | 4 232 |
| Test | 8 482 | 4 241 | 4 241 |
| Total | 84 818 | 65 681 | 19 137 |

## Clase por fuente en train

| Fuente | Positivas | Negativas |
|---|---|---|
| bimcv | 42 495 | 647 |
| stonybrook | 12 696 | 0 |
| rsna | 0 | 9 719 |
| sirm | 878 | 0 |
| ricord | 834 | 0 |
| cohen | 251 | 220 |
| actmed | 23 | 78 |
| fig1 | 22 | 0 |

Solo bimcv, cohen y actmed aportan las dos clases en train. Val repite el patrón (negativos: rsna 3 869, bimcv 257,
cohen 77, actmed 29). Test contiene bimcv 4 041 / 4 041, ricord 200 positivas y rsna 200 negativas.

## Comprobación contra las cifras publicadas

Referencias: manuscrito del artículo 1 (`~/mnt/covid2026/envios/jiim/03_Manuscript_full.docx`, Section 2.1 y 2.2,
Table 1) y criterios del PLAN de `paper_new`.

| Cifra | Publicada | Calculada | Estado |
|---|---|---|---|
| Train total | 67 863 | 67 863 | coincide |
| Train positivas / negativas | 57 199 / 10 664 | 57 199 / 10 664 | coincide |
| Val total (positivas / negativas) | 8 473 (4 241 / 4 232) | 8 473 (4 241 / 4 232) | coincide |
| Test total (positivas / negativas) | 8 482 (4 241 / 4 241) | 8 482 (4 241 / 4 241) | coincide |
| Negativos de train RSNA / BIMCV / Cohen / ActMed | 9 719 / 647 / 220 / 78 | 9 719 / 647 / 220 / 78 | coincide |
| Positivos BIMCV de train | 42 495 | 42 495 | coincide |
| Régimen controlado (solo BIMCV) | 43 142 | 42 495 + 647 = 43 142 | coincide |
| Test BIMCV positivas / negativas; RICORD; RSNA | 4 041 / 4 041; 200; 200 | 4 041 / 4 041; 200; 200 | coincide |
| Imágenes y pacientes del dataset completo | 84 818 de 45 342 pacientes | 84 818; 45 342 | coincide |
| Proporción de negativos de train que son RSNA | "about 91 per cent" | 9 719 / 10 664 = 91.1 % | coincide |
| Relación positivas / negativas en train | "about 5.4 to 1" | 5.36 a 1 | coincide |
| Proporción BIMCV del test | "95 per cent" | 8 082 / 8 482 = 95.3 % | coincide |

No hay discrepancias. Las cuentas se obtuvieron sin forzar ningún valor.
