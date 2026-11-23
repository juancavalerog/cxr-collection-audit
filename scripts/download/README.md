# Scripts de descarga de datos nuevos

> **AVISO IMPORTANTE: VinDr-CXR.** Antes de ejecutar `dl_vindr_subset.sh` (o `download_all.sh` sin `--no-vindr`),
> el autor debe entrar con su cuenta en https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection,
> **aceptar las reglas de la competición en la web** y **confirmar que esas reglas permiten el uso de los datos en
> investigación académica con publicación**. No se pudieron leer (página dinámica). Si no lo permiten, se usa
> la **alternativa B (PadChest)** con acuerdo de BIMCV y no se ejecuta `dl_vindr_subset.sh`.
>
> | Alternativa | Tercera fuente del benchmark controlado | Tamaño total de la selección |
> |---|---|---|
> | A | VinDr-CXR (subconjunto, tope de 45 GB) + CheXmask VinDr-CXR.csv (0.9 GB) | **72.1 GB** |
> | B | PadChest (ficheros parciales, tope de 45 GB) + CheXmask Padchest.csv (4.3 GB) | **75.5 GB** |
>
> Ambas caben en el límite de 90 GB. La versión de Kaggle de VinDr-CXR no trae la etiqueta de tuberculosis; solo se
> usa para derrame pleural, cardiomegalia y "No finding".

Destino: `/home/administrador/Proyectos/paper_new/data/raw/<dataset>/` (variable `RAW_ROOT`). Registros en
`/home/administrador/Proyectos/paper_new/logs/download/` (variable `LOG_DIR`). Justificación de la selección en
`docs/02_datos.md` (Section 2). Ninguno requiere credenciales de PhysioNet. No se han ejecutado: solo se comprobó
su sintaxis con `bash -n` (y `python3 -m py_compile` para `vindr_select.py`, que además se probó con un CSV sintético).

## Requisitos

- `bash`, `curl`, `wget`, `unzip`, `python3` con `pandas`, `sha256sum`, `md5sum`, `df`, `du`.
- CLI de Kaggle: `pip install --user kaggle` y el token en `~/.kaggle/kaggle.json` (Kaggle, Settings, API,
  Create New Token; después `chmod 600 ~/.kaggle/kaggle.json`).
- VinDr-CXR: aceptar antes las reglas de la competición en
  https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection con la misma cuenta, y leer sus
  condiciones de uso de datos (no verificadas).

## Comportamiento común (`_common.sh`)

- Comprueba el espacio libre antes de empezar (lo necesario más 10 GB de margen) y aborta si no alcanza.
- Idempotente: al terminar bien escribe `<destino>/.done`; si existe, el script no hace nada. Las descargas usan
  reanudación (`curl -C -`, `wget -c`; VinDr omite los DICOM ya presentes).
- Verificación al terminar: Kaggle, suma de bytes extraídos frente a `totalBytes` publicado (tolerancia 2 %;
  `SKIP_SIZE_CHECK=1` para forzar si Kaggle cambia de versión); COVIDGR, tamaño exacto del zip; NLM, número de
  PNG; CheXmask, SHA256 oficial. Se guarda una suma local (`*.sha256`, `MD5SUMS.local` o `SHA256SUMS.local`).
- Los zips de Kaggle se borran tras verificarlos.

## Paso previo: contenido de los datasets de Kaggle

`bash check_kaggle_contents.sh` lista los ficheros de un dataset de Kaggle (con `kaggle datasets files`, paginado)
y cuenta imágenes por carpeta sin descargar nada. Por defecto solo lista la base de Rahman; otros datasets, solo si
se pasan como argumento (`bash check_kaggle_contents.sh usmanshams/tbx-11`). Hace una pausa entre páginas
(`KAGGLE_PAGE_PAUSE`, 3 s), reintenta con espera creciente ante 429 y 5xx, escribe a un temporal y solo sustituye
el CSV previo si el listado termina bien. **Nunca detiene el orquestador**: si falla, deja un AVISO en
`logs/kaggle_contents.log` y sale con código 0.

**Resultado del 12-11-2026 en el servidor:** `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` tiene 3 500
imágenes normales y 700 de tuberculosis (4 200 en total). La descarga no incluye las 2 800 imágenes de NIAID, así
que **aplica la variante B del Benchmark 2**. La variante A queda condicional a que el autor obtenga acceso a NIAID
TB Portals (`docs/02_datos.md`, Section 3.2).

## Credenciales y reintentos

- Antes de cualquier descarga se validan las credenciales (`require_kaggle` en `_common.sh`). Se aceptan
  `KAGGLE_USERNAME` y `KAGGLE_KEY` en el entorno, `KAGGLE_API_TOKEN` o `~/.kaggle/access_token` (versiones recientes
  de la CLI), o `kaggle.json` en `$KAGGLE_CONFIG_DIR` o `~/.kaggle` con los campos `username` y `key`. Si falta algo,
  el script se detiene con un mensaje en español y el remedio, sin traza de Python. El fallo del 12-11-2026
  (`KeyError: 'username'`) se debía a un `kaggle.json` sin el campo `username`.
- Todas las llamadas a la CLI pasan por `kaggle_retry`: ante 429, 5xx o cortes de red reintentan con espera creciente
  (`KAGGLE_RETRIES` 6, `KAGGLE_RETRY_BASE` 30 s, que se duplica en cada intento hasta `KAGGLE_RETRY_MAX` 600 s).
  Entre datasets hay una pausa de `KAGGLE_PAUSE` (10 s). VinDr usa por defecto 2 descargas en paralelo.
- `FREE_MARGIN_GB` (10 por defecto) fija el margen de espacio libre exigido además de lo necesario.
- Pruebas sin red: `scripts/download/test_download_scripts.sh`, con el simulador `test_fixtures/kaggle_simulado`;
  resultado en `logs/tmp_datos/test_download_scripts.log` (7 de 7 pruebas correctas).

## Orden y espacio

| Orden | Script | Dataset | Espacio final | Comprobación |
|---|---|---|---|---|
| 1 | `dl_rsna_mapping.sh` | tabla RSNA a NIH | menos de 0.1 GB | JSON legible |
| 2 | `dl_tb_rahman.sh` | TB Chest X-ray Database | 0.69 GB | totalBytes 694 671 204 |
| 3 | `dl_covid19_radiography.sh` | COVID-19 Radiography Database | 0.81 GB | totalBytes 806 838 951 |
| 4 | `dl_covidgr.sh` | COVIDGR 1.0 | 1.0 GB | zip de 995 465 872 bytes; 852 imágenes |
| 5 | `dl_covid_qu_ex.sh` | COVID-QU-Ex | 1.21 GB | totalBytes 1 211 011 783 |
| 6 | `dl_nlm_tb.sh all` | Montgomery y Shenzhen (NLM oficial) | unos 0.6 y 3.8 GB (estimado) | 138 y 662 PNG |
| 7 | `dl_tbx11k.sh` | TBX11K (copia de Kaggle) | 4.25 GB | totalBytes 4 252 079 748 |
| 8 | `dl_chexmask.sh` | CheXmask: ChestX-Ray8, CheXpert, VinDr-CXR | 11.7 GB | SHA256SUMS.txt oficial |
| 9 | `dl_tb_pakistan_mendeley.sh` | TB Pakistán (Mendeley, CC BY 4.0) | 3 GB de reserva (no verificado) | 3 008 imágenes |
| 10 | `dl_vindr_subset.sh` | VinDr-CXR, subconjunto de train (solo alternativa A) | 45 GB como tope (`MAX_GB`) | recuento de DICOM, MD5 local |
| | **Total alternativa A** | | **72.1 GB** | |
| | **Total alternativa B** (sin el paso 10; con `dl_chexmask.sh --padchest` y 45 GB de PadChest) | | **75.5 GB** | |

`dl_tb_pakistan_mendeley.sh` necesita la URL del botón "Download All" de https://data.mendeley.com/datasets/8j2g3csprk/2
en `ZIP_URL`, o un zip descargado a mano en `ZIP_PATH`. PadChest (alternativa B) se pide con el formulario de
https://bimcv.cipf.es/bimcv-projects/padchest/ y no tiene script: el enlace llega tras aceptar el acuerdo.

Todo seguido: `bash scripts/download/download_all.sh` (o `--no-vindr` para dejar VinDr-CXR para después).
Para ejecutar solo algunos pasos: `ONLY_STEPS=dl_tb_rahman,dl_covidgr bash scripts/download/download_all.sh`.
Opcional: `dl_chexmask.sh --padchest` añade `Padchest.csv` (4.3 GB) si se elige la alternativa PadChest.

Variables de `dl_vindr_subset.sh`: `MAX_GB` (45), `N_NOFINDING` (3 000), `JOBS` (4 descargas en paralelo). La lista
de imágenes la genera `vindr_select.py` desde `train.csv` (positivas para derrame o cardiomegalia por al menos un
lector, más una muestra sin hallazgos unánime con semilla 20261112). Es reanudable: si se corta o
llega al tope, se puede relanzar.

## Después de descargar

```bash
python3 scripts/data/hash_images.py --dir data/raw/<dataset> --out data/hashes/<dataset>.csv --workers 32
python3 scripts/data/find_overlaps.py --a data/hashes/<dataset>.csv --b data/hashes/nih_cxr14.csv \
    --name-a <dataset> --name-b nih --verify --root-a data/raw/<dataset> \
    --root-b /home/administrador/notebooks/covid --out results/overlaps/<dataset>_nih.csv
```

## Cambios del 12-11-2026 (tras la segunda ejecución real)

- `dl_nlm_tb.sh` ya no usa `wget -r`. El servidor de NLM publica cada carpeta como `<carpeta>/index.html` y responde 403 a la URL de la carpeta, de modo que la primera petición fallaba y, por `set -e`, el script y el orquestador terminaban sin dejar mensaje. Ahora descarga `nlm_crawl.py`, que recorre los `index.html`, reanuda parciales con `Range`, reintenta ante errores transitorios y devuelve un código distinto de cero si falta algo.
- `_common.sh` registra en el log cualquier aborto por `set -e` (orden, línea y código).
- `download_all.sh` ya no se detiene en el primer paso que falla: lo anota, sigue con los demás, y al final lista los pasos fallidos y sale con código 1. Al relanzarlo, los pasos completados se saltan y los parciales se reanudan.
- `dl_chexmask.sh` limita esperas y reintentos de `wget`.
- Pruebas sin red: `bash scripts/download/test_nlm_crawl.sh` (servidor simulado en `test_fixtures/nlm_fake_server.py`; 7 pruebas). Estructura real comprobada el mismo día desde el navegador: enlaces relativos, 138 PNG en Montgomery y 662 PNG (3.77 GB) en Shenzhen.
- TBX11K y la suma de bytes (12-11-2026, tercera ejecución real). La descarga de `usmanshams/tbx-11` terminó bien en 26 minutos, pero el paso falló en la verificación de tamaño: `bytes_of` sumaba con `awk` y, con mawk y configuración regional en español, un total mayor que 2^31 salía como `4,25208e+09`, que la comparación no podía leer. Ahora imprime el entero exacto. Los datos extraídos están completos (4 252 079 748 bytes, igual que lo publicado; 12 279 imágenes); al relanzar, el paso solo termina la verificación y borra el zip.
- Vigilancia de inactividad de Kaggle. La CLI de Kaggle no tiene tiempo de espera propio, así que una llamada sin respuesta quedaría colgada sin fin. `kaggle_run` (en `_common.sh`) vigila cada llamada: si el proceso no lee ni escribe nada (`/proc/<pid>/io`) y no cambian ni su salida ni el destino durante `KAGGLE_STALL_SEC` segundos (3 600 por defecto), mata la llamada; `kaggle_retry` la reintenta una vez y, si se repite, abandona el paso para que el orquestador siga. El directorio de destino no sirve por sí solo como señal: la CLI baja el zip a una ubicación temporal y lo mueve al final, de modo que el destino estuvo vacío durante los 26 minutos de la descarga de TBX11K. La primera versión de esta vigilancia, con 900 s y sin la señal del proceso, la habría cortado; se corrigió antes de que llegara a ejecutarse en el servidor. Pruebas sin red: `bash scripts/download/test_kaggle_stall.sh` (7 pruebas).
