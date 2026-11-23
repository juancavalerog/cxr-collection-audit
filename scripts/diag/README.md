# Diagnósticos de la firma de adquisición (13-11-2026)

Scripts para estudiar, en CPU y con datos reales de validación, por qué la firma de adquisición no
aprendió la fuente en el primer intento del piloto de CSD-Net. No forman parte del entrenamiento ni de la
evaluación y no tocan el test.

- `diag_independencia_gradientes.py PESO PASOS`: entrena solo el codificador de adquisición y la cabeza de fuente
  con 0.5 L_src + PESO L_ind (la función real de `src/model/losses.py`) sobre lotes con la composición del
  muestreador, y compara las normas de los gradientes de ambas pérdidas en el codificador. Construye la primera
  vez un caché de entradas de adquisición en `/tmp/diag_cache2.pt` (662 imágenes de validación).
- `diag_independencia_estimadores.py {sesgado|insesgado} PESO PASOS`: lo mismo con el estimador sesgado (el del
  código) o con el insesgado, y mide la AUC de clase en 80 imágenes retenidas de bimcv y la dispersión de la firma.
  Necesita el caché del script anterior.
- `diag_aumento.py {0|1} PASOS`: entrena el codificador solo con L_src, sin o con el aumento del entrenamiento.

Resultados del 13-11-2026 (60 a 90 pasos, lotes de 32 a 36): sin L_ind, L_src baja de 2.08 a 1.25; con L_ind
sesgado de peso 1.0 o 0.1, L_src se queda en la entropía de la distribución de fuentes del lote (1.565) y la
desviación de la firma cae a 0.003; con L_ind insesgado de peso 1.0 la firma no colapsa (desviación 0.34) pero
L_src tampoco baja en 90 pasos. La razón entre las normas de los gradientes de L_ind y de 0.5 L_src en el
codificador pasa de 2 a más de 110 en 30 pasos. Con el aumento del entrenamiento, L_src sigue bajando (1.82 a 1.61
en 90 pasos), de modo que el aumento no es la causa.

## Diagnóstico de la firma (13-11-2026)

Scripts `r2_*.py`; análisis completo y cifras en `docs/10_correccion_independencia.md`. Los temporales van a
`/tmp/r2` (cachés y puntos de control) y los resultados a `logs/diag_r2/`.

- `r2_cache.py {train|val}`: caché compacta de imágenes y máscaras (7 872 de entrenamiento, 4 262 de validación).
- `r2_lib.py`: lotes con la composición del muestreador real, penalizaciones candidatas y medidas.
- `r2_train.py VARIANTE`: entrena la vía de adquisición con una variante de la penalización y mide L_src, la entropía
  de la composición, la exactitud de fuente, la dispersión de la firma y V1 en bimcv de val.
- `r2_origen_clase.py CK`: de dónde sale la clase dentro de bimcv de val (geometría frente a contenido).
- `r2_asociacion_particiones.py` y `r2_dentro_fuera.py CK...`: asociación clase-adquisición en entrenamiento frente a
  validación.
- `r2_tabla.py`: tabla comparativa (`logs/diag_r2/tabla.txt`).
