Edited for release: internal task-assignment notes were removed; the registered content is unchanged.

# 08. Plan de análisis estadístico y evaluación

Versión 1.1, 12 de noviembre de 2026 (enmiendas al final). Este plan se fija antes de que exista ningún resultado de
entrenamiento con datos reales: no se ha ejecutado ninguna corrida del piloto ni de la evaluación final, y todas
las cifras que producen los programas descritos aquí se han obtenido, por ahora, sobre árboles de resultados
sintéticos. La evaluación del piloto (Puerta 3) se rige por docs/07_piloto.md, que este documento no modifica; lo
que sigue fija la evaluación final sobre el test, cuando se desbloquee con los hiperparámetros congelados
(docs/06_lineas_base.md, Section 4), y la manera de informar los tres benchmarks de docs/02_datos.md, Section 3.
Cualquier cambio posterior se añadirá como enmienda fechada al final, con su motivo, y no podrá apoyarse en
resultados del test.

## 1. Hipótesis y criterio principal

El artículo sostiene que CSD-Net clasifica mejor que el control de fuente [valero2026learning] cuando clase y
fuente están confundidas por construcción, sin reabrir la dependencia de la fuente que el control cierra. La
comparación primaria es única y se declara aquí: AUC-ROC de CSD-Net frente a la del control de fuente reentrenado
con el protocolo nuevo, en la parte bimcv del test limpio de COVIDx CXR-4 [wu2023covidx] (8 079 imágenes, 4 039
negativas y 4 040 positivas), con modelos individuales: la cifra principal es la media de las cinco semillas,
cada una con su estado de mejor época y TTA de espejo. Se elige la parte bimcv porque es la única fuente del test con ambas clases: en el test completo las 200
negativas de rsna y las 200 positivas de ricord permiten acertar reconociendo la fuente, de modo que su AUC mezcla
discriminación de la enfermedad y lectura de la firma de adquisición. Se toma del test limpio, y no del
subconjunto `test_bimcv` del manifiesto, porque este último se define sobre el test oficial y conserva tres
imágenes que la versión limpia excluye por pertenecer a pares casi idénticos con etiqueta opuesta
(data/manifests/README.md, regla 3).

La diferencia que se contrasta es la de la ecuación (1), calculada sobre las mismas imágenes.

$$\Delta = \frac{1}{5}\sum_{s=1}^{5}\Big[\mathrm{AUC}\big(p_s^{\,\text{CSD}}\big) - \mathrm{AUC}\big(p_s^{\,\text{control}}\big)\Big] \qquad (1)$$

En la ecuación (1), p_s es la probabilidad que da el modelo entrenado con la semilla s, las dos AUC de cada término
se calculan sobre las mismas imágenes y la AUC es el estadístico de Mann y Whitney con rangos medios para los
empates. La hipótesis nula es Δ ≤ 0 y la alternativa Δ > 0. Se contrasta la media por semilla, y no la AUC del
conjunto que promedia las probabilidades de las cinco semillas, por cuatro razones: lo que el artículo afirma es
que el método entrena mejores modelos, y promediar modelos mejora a cualquier método y más a los inestables; es
el formato en que se informan las líneas base con las que se compara; deja a la vista la variabilidad del
entrenamiento; y evita que la ventaja dependa de un promedio que multiplica por cinco el coste de inferencia. El
conjunto medio se informa como dato secundario (Section 2, S6). El intervalo de confianza al 95 % es el de
percentiles del bootstrap pareado de pacientes de la ecuación (2).

$$\mathrm{IC95}(\Delta) = \big[\, Q_{0.025}\{\Delta^{*b}\}_{b=1}^{B},\ Q_{0.975}\{\Delta^{*b}\}_{b=1}^{B} \,\big] \qquad (2)$$

En la ecuación (2), Δ^{*b} es la media de las cinco diferencias recalculadas con los mismos pacientes en el
remuestreo b y Q el cuantil empírico. Este intervalo recoge la incertidumbre del muestreo de pacientes con las cinco
semillas fijadas; la variabilidad entre semillas se informa aparte, con la diferencia de cada semilla, su desviación
típica y el número de semillas en que gana CSD-Net. El valor p unilateral es la fracción de remuestreos con diferencia no positiva,
corregida con una unidad en el numerador y en el denominador para que nunca valga cero, según la ecuación (3).

$$p = \frac{1 + \big|\{\, b : \Delta^{*b} \le 0 \,\}\big|}{B_v + 1} \qquad (3)$$

En la ecuación (3), B_v es el número de remuestreos en los que las dos AUC están definidas, que se registra junto
a B en cada salida. La mejora se declara si el extremo inferior del IC95 % es mayor que cero, lo que equivale, salvo
por la discreción de los cuantiles, a p < 0.025 con la ecuación (3); ese es el nivel unilateral de todas las
pruebas confirmatorias del plan. El PLAN (Section
12.3, punto 5) estima en unas 0.01 la semianchura del IC95 % de la AUC; una diferencia menor de
0.02 puede ser estadísticamente detectable con la prueba pareada y, aun así, se discutirá como de relevancia
clínica incierta. Como comprobación secundaria se calcula, sobre el conjunto medio, la prueba de DeLong pareada con el algoritmo rápido de
Sun y Xu [delong1988comparing, sun2014fast], que no tiene en cuenta la agrupación por paciente; si las dos pruebas
discrepan en la decisión, manda el bootstrap de pacientes y la discrepancia se informa.

## 2. Criterios secundarios

1. S1. CSD-Net frente a cada una de las demás líneas base del nivel 1 (ERM, GroupDRO, DANN, CORAL, MixStyle,
   Fourier, LISA, DFA con fuente, DFR y ERM con enmascarado pulmonar) y del nivel 2 que lleguen a la evaluación
   final (CDANN, IRM, la variante de Lai et al. y la sonda de RAD-DINO), con la misma prueba y el mismo conjunto que
   el criterio principal.
2. S2. Las mismas comparaciones en el test limpio completo, que se informan pero no sostienen la afirmación
   principal por la razón de la Section 1.
3. S3. Brecha de especificidad rsna menos bimcv en los negativos del test limpio, al umbral de Youden de la
   validación robusta, para cada método (V6 en su versión final).
4. S4. Calibración: ECE de 15 bins y puntuación de Brier, en el test limpio y en su parte bimcv.
5. S5. Sensibilidad, especificidad y exactitud balanceada por fuente y por subconjunto del test al umbral de Youden
   de la validación robusta; AUC-PR.
6. S6. La misma comparación con el conjunto medio de las cinco semillas (probabilidades promediadas) y el análisis
   con las semillas como réplicas (Section 6).
7. S7. Verificaciones V1 a V7 en su versión final, con las cinco semillas (docs/03_arquitectura.md, Section 3).
8. S8. Ablaciones A1 a A15 frente a la CSD-Net completa (A0).

El umbral de decisión se fija siempre fuera del conjunto evaluado, con el criterio de Youden de la ecuación (4).

$$t^{*} = \arg\max_{t}\ \big\{\mathrm{Sens}(t) + \mathrm{Esp}(t) - 1\big\} \quad \text{sobre } p_s \text{ en la validación robusta} \qquad (4)$$

En la ecuación (4), la sensibilidad cuenta los positivos con probabilidad mayor o igual que t y la especificidad
los negativos con probabilidad menor que t. Para DFR la validación robusta disponible es val_robust_sel, porque la
otra mitad interviene en su reajuste; los demás métodos usan val_robust completa. Cada semilla usa el umbral de
Youden de su propia validación robusta; el conjunto medio, que es secundario, usa el de sus probabilidades
promediadas. La calibración se resume con la ECE de la ecuación (5).

$$\mathrm{ECE} = \sum_{k=1}^{15} \frac{n_k}{N}\,\big|\,\bar y_k - \bar p_k\,\big| \qquad (5)$$

En la ecuación (5), los 15 bins tienen igual anchura en [0, 1], el último incluye el 1, n_k es el número de imágenes
del bin k, ȳ_k su frecuencia observada de positivos y p̄_k su probabilidad media. La brecha de especificidad es la
de la ecuación (6).

$$G = \mathrm{Esp}_{\text{rsna}}(t^{*}) - \mathrm{Esp}_{\text{bimcv}}(t^{*}) \qquad (6)$$

En la ecuación (6), las dos especificidades se calculan en los negativos del conjunto evaluado. Un valor positivo
indica que el modelo acierta más en los negativos de rsna que en los de bimcv, que es el patrón del atajo de fuente;
en el artículo 1 la brecha fue 0.739 en el ingenuo y −0.131 en el control, con otro protocolo (docs/00_inventario.md,
Section 5.8).

## 3. Jerarquía de pruebas y corrección por multiplicidad

La familia confirmatoria tiene una sola prueba, la principal, que se contrasta al nivel unilateral completo de
0.025. Las
pruebas S1 forman la primera familia secundaria y solo se interpretan como confirmatorias si la principal se
rechaza (procedimiento secuencial de puerta). Las ablaciones S8 forman la segunda familia secundaria. Dentro de
cada familia se aplica la corrección de Holm de la ecuación (7).

$$\tilde p_{(i)} = \max_{j \le i}\ \min\big\{1,\ (m - j + 1)\, p_{(j)}\big\} \qquad (7)$$

En la ecuación (7), p_(1) ≤ ... ≤ p_(m) son los valores p ordenados de la familia de m pruebas y p̃_(i) el ajustado.
Las familias son:

| Familia | Pruebas | Conjunto | Prueba de cada comparación | Corrección |
|---|---|---|---|---|
| F0 (confirmatoria) | CSD-Net frente al control de fuente | parte bimcv del test limpio | media por semilla de la diferencia de AUC, bootstrap pareado de pacientes, unilateral | ninguna (una sola prueba) |
| F1 | CSD-Net frente a cada línea base disponible, control incluido | parte bimcv del test limpio | igual que F0; conjunto medio y DeLong como comprobación | Holm sobre todos los comparadores del conjunto |
| F1' | las mismas comparaciones en cada uno de los demás conjuntos del test con ambas clases | test limpio completo y subconjunto bimcv oficial | igual que F1 | Holm dentro de cada conjunto, descriptivo |
| F2 | cada ablación frente a A0 | parte bimcv del test limpio | igual que F0, sobre las semillas comunes | Holm sobre las ablaciones disponibles |
| F3 | comparaciones con las semillas como réplicas | los mismos de F1 | Welch unilateral y t pareada | Holm dentro de cada conjunto, descriptivo |

En las tablas, el valor p de Holm de F1 incluye también al control de fuente, de modo que la cifra publicada para
la comparación principal en esa tabla es la más conservadora de las dos; la decisión confirmatoria es la de F0. Las
DeLong se corrigen por Holm en paralelo con el bootstrap, dentro de la misma familia. Las métricas S3 a S5 y S7 se
informan con su IC95 % y sin valor p, para no multiplicar contrastes que el artículo no necesita.

## 4. Unidad de remuestreo

Todos los intervalos se obtienen remuestreando pacientes con reemplazo y conservando todas las imágenes de cada
paciente elegido, con percentiles 2.5 y 97.5. La razón es que un paciente de bimcv aporta varias imágenes (en la
validación robusta, 3 407 imágenes de 999 pacientes) cuyas probabilidades están correlacionadas; remuestrear
imágenes trataría como independientes observaciones que no lo son y daría intervalos demasiado estrechos (la
prueba `test_cluster_bootstrap_wider_than_image_bootstrap_with_correlated_patients` lo muestra con datos
sintéticos). En rsna el identificador oficial es un UUID por imagen; cuando el manifiesto limpio está disponible,
las imágenes rsna se agrupan por el paciente NIH enlazado por hash (docs/02_datos.md, Section 4.1), que es la unidad
real de dependencia. El remuestreo no se estratifica por clase ni por fuente: el número de negativos y positivos
varía entre remuestreos, como varía entre muestras de pacientes, y los remuestreos en los que una métrica no está
definida se descuentan y se registran. El umbral de Youden no se recalcula en cada remuestreo, porque viene de
otro conjunto.

## 5. Número de remuestreos y semillas

B = 2 000 remuestreos con la semilla 2026 en todas las pruebas de este plan y en la Puerta 3. El número declarado y
el usado coinciden y se escriben en cada salida (campos `B`, `B_validos` y `semilla_bootstrap`), a diferencia del
artículo 1, que declaró 2 000 y usó 1 000 (docs/00_inventario.md, Section 4.6). Si se cambia B por línea de órdenes,
el informe lo registra como desviación del protocolo. Los probes de las verificaciones usan el número de
remuestreos de la configuración de CSD-Net (`probe_n_boot`) y la semilla de `src/eval/probes.py`, que se informan
junto a cada verificación. Semillas de entrenamiento: 42, 123, 2024, 7 y 999 para todos los métodos y ablaciones
con la configuración congelada, igual presupuesto de búsqueda para todos (PLAN, Section 12.8).

## 6. Semillas: modelo individual, conjunto medio y réplicas

La afirmación principal se hace sobre modelos individuales: la cifra de cada método es la media de sus cinco
semillas, con su desviación típica muestral, y la prueba principal contrasta la media por semilla de la diferencia
(ecuación 1). Para cada comparación se informan además la diferencia de cada semilla y el número de semillas en que
gana CSD-Net. Como datos secundarios se informan dos análisis. El primero es el conjunto medio de las cinco
semillas, con las probabilidades promediadas, su IC95 % por bootstrap de pacientes y la prueba de DeLong; va en una
tabla aparte (`tabla_conjunto_medio`). El segundo (F3) trata las semillas como réplicas y contrasta que la AUC media
de CSD-Net supere a la de cada método con la prueba de Welch unilateral y, como las semillas coinciden entre
métodos, con la t pareada por semilla. Con cinco réplicas la potencia de estas pruebas es baja; una diferencia
media que no se acompañe de una diferencia consistente entre semillas se discutirá como frágil. Ninguna semilla se
descarta ni se sustituye: si una corrida falla, se repite con la misma semilla; si no puede completarse, la tabla
lo indica y el programa se niega a continuar salvo que se pida expresamente, y entonces lo deja escrito en los
avisos.

## 7. DFR

DFR reajusta su última capa con val_robust_fit y elige la regularización en val_robust_sel [kirichenko2023last].
En el test no hay conflicto: todos los métodos se comparan sobre las mismas imágenes. Las dos consecuencias que se
declaran son que su umbral de Youden se calcula en val_robust_sel (1 615 imágenes, 130 negativas) en lugar de la
validación robusta completa, con más varianza, y que su estado evaluado es `dfr_best`, el reajuste sobre el estado
de mejor época. En el piloto, la comparación con DFR se hizo en val_robust_sel (docs/07_piloto.md, Section 4).

## 8. Subconjuntos del test con y sin fuga

Los subconjuntos `rsna_con_fuga` (129 imágenes) y `rsna_sin_fuga` (71) separan las imágenes rsna del test cuyo
paciente NIH tenía imágenes en el train o la validación oficiales. En la versión limpia ninguna conserva esos
pacientes en el train ni en la validación, pero los modelos preentrenados en NIH completo pudieron verlos
(docs/02_datos.md, Section 4.1). Como solo contienen negativos, de cada uno se informa la especificidad con su
IC95 %; una especificidad claramente mayor en `rsna_con_fuga` que en `rsna_sin_fuga` se leería como memoria de
pacientes. Con 71 y 129 imágenes los intervalos serán anchos y la comparación es descriptiva. De ricord (200
positivas) se informa la sensibilidad. Estos subconjuntos se definen sobre el test oficial; la tabla principal los
presenta junto al test limpio y su parte bimcv, que es la que sostiene la afirmación principal.

## 9. Qué se informa de cada benchmark y de los conjuntos externos

| Benchmark | Conjunto | Qué se informa | Papel |
|---|---|---|---|
| 1, COVIDx CXR-4 limpio | parte bimcv del test limpio | AUC con IC95 %, Δ frente a cada método, sensibilidad, especificidad, ECE | confirmatorio (F0 y F1) |
| 1 | test limpio completo; subconjunto bimcv oficial | las mismas métricas | descriptivo (F1') |
| 1 | ricord, rsna con y sin fuga | sensibilidad o especificidad con IC95 % | descriptivo |
| 1 | negativos del test limpio | brecha de especificidad rsna menos bimcv | secundario (S3) |
| 1, externo | COVIDGR 1.0 (426 y 426) | AUC, sensibilidad y especificidad al umbral de Youden de la validación robusta | descriptivo, sin selección ni umbral propios |
| 2, tuberculosis (variante B) | Shenzhen y Montgomery, por fuente | AUC por fuente con IC95 % y Δ frente al control y ERM | secundario, con intervalos anchos por los 906 positivos de entrenamiento |
| 2, externo | Mendeley (Pakistán) | AUC y especificidad | descriptivo y de procedencia débil |
| 3, controlado | test desconfundido por nivel ρ | AUC frente a ρ (curva dosis-respuesta), test invertido como peor caso, AUC por fuente y brecha de especificidad | secundario; Holm sobre los niveles de ρ para CSD-Net frente a ERM |

MIMIC no se usa mientras su procedencia no esté aclarada (PLAN, Section 12.6). Los benchmarks 2 y 3 usarán las
mismas funciones de `src/eval/stats.py`; el informe final implementado (Section 13) cubre por ahora el benchmark 1 y
admite conjuntos externos descriptivos por especificación.

## 10. Análisis de sensibilidad

1. Estados EMA y de última época, además del de mejor época.
2. Conjunto medio frente a modelo individual, y umbral por semilla frente al umbral del conjunto medio.
3. Test oficial frente a test limpio, para enlazar con las cifras del artículo 1 sin mezclarlas con las nuevas.
4. Remuestreo por imagen frente a remuestreo por paciente, para mostrar el efecto de la agrupación.
5. DeLong frente al bootstrap de pacientes en todas las comparaciones de F1.
6. Inicialización: ERM, control de fuente y CSD-Net con un tronco sin NIH frente al preentreno NIH (PLAN, Section
   12.4); la sonda de RAD-DINO, que vio NIH completo, también sin rsna en el test (docs/01_novedad.md, Section 5).
7. Del piloto, la AUC en val_robust_fit (enmienda 3 de docs/07_piloto.md), que no interviene en la selección de los
   métodos distintos de DFR.

## 11. Lo que no se hará

1. Elegir umbrales, estados, semillas, épocas, hiperparámetros o modelos con el test; el umbral viene siempre de la
   validación robusta.
2. Quitar, sustituir o añadir semillas después de ver el test; tampoco sustituir la media por semilla por la mejor
   semilla ni por el conjunto medio.
3. Cambiar el criterio principal, el conjunto primario, el comparador primario, el margen o la dirección de la
   prueba a posteriori.
4. Presentar como confirmatorias comparaciones fuera de F0 a F2, o declarar diferencias por el solo hecho de que el
   valor p sin corregir sea menor que 0.025.
5. Poner las cifras del artículo 1 en la tabla principal: van en una tabla aparte, rotulada como de otro protocolo.
6. Volver a elegir la mejor línea base del piloto después de abrir CSD-Net (el programa de la Puerta 3 lo impide).
7. Leer el test durante el piloto o la búsqueda (los programas lo impiden con error).

## 12. Afirmaciones previstas y pruebas que las sostienen

| Afirmación prevista del artículo | Prueba o métrica | Conjunto | Criterio |
|---|---|---|---|
| CSD-Net discrimina mejor que el control de fuente | F0: media por semilla de la Δ de AUC, bootstrap pareado de pacientes; conjunto medio y DeLong como comprobación | parte bimcv del test limpio | extremo inferior del IC95 % > 0 |
| CSD-Net supera (o no es inferior) a las demás líneas base | F1: Δ de AUC con Holm | parte bimcv del test limpio | p de Holm < 0.025 para superioridad; extremo inferior > −0.01 para no inferioridad, que se informa pero no se contrasta como hipótesis propia |
| La mejora no depende de una semilla afortunada | Δ de cada semilla y número de semillas a favor; F3: Welch y t pareada | parte bimcv del test limpio | coherencia en dirección; p informado con Holm |
| CSD-Net no reabre la brecha de especificidad | S3: brecha rsna menos bimcv con IC95 % | negativos del test limpio | |G| ≤ 0.25 como referencia del piloto, comparada con el control |
| La calibración no empeora | S4: ECE y Brier con IC95 % | test limpio y parte bimcv | intervalos comparados, sin prueba |
| Los supuestos del contrafactual se cumplen | S7: V1 a V5 con cinco semillas; V4 con Welch frente al control | val y test | umbrales de la Tabla 7 de docs/03_arquitectura.md |
| La mejora viene del intercambio dirigido y no de la compuerta ni de perturbar el estilo | F2: A1, A2 y A4 frente a A0 | parte bimcv del test limpio | p de Holm < 0.025 |
| Completar la tabla aporta frente a normalizar el estilo | F2: A14 frente a A0 | parte bimcv del test limpio | p de Holm < 0.025 |
| El preprocesado homogéneo reduce la firma trivial | V7 y ablación PRE | val | reducción de exactitud balanceada, descriptiva |
| El fenómeno se repite en otra enfermedad y en confusión controlada | benchmarks 2 y 3 | Section 9 | descriptivo con IC95 %; curva frente a ρ |

## 13. Implementación y reproducibilidad

| Fichero | Contenido |
|---|---|
| `src/eval/stats.py` | Métricas, bootstrap de pacientes (reutiliza `cluster_bootstrap` de `src/eval/probes.py`), prueba primaria de la media por semilla (`paired_auc_test_seedmean`), DeLong de Sun y Xu, no inferioridad, semillas, Holm, por fuente, brecha (reutiliza `v6_specificity_gap`), lectura de corridas e informes y alineación estricta |
| `scripts/jobs/gate3_evaluar.py` | Puerta 3 en dos pasos (Section 14) |
| `scripts/eval/informe_final.py` | Evaluación final: tablas en CSV, Markdown y LaTeX con booktabs y figuras en PDF y PNG a 300 ppp |
| `src/eval/synthetic_tree.py` | Árboles de resultados sintéticos con la estructura de los entrenadores |
| `tests/test_eval_stats.py` | Pruebas (salida en `logs/tests_eval.log`) |

Toda tabla y toda figura se regeneran con una orden desde las predicciones guardadas:
`python scripts/eval/informe_final.py`, que escribe en `results/final/` las tablas `tabla_principal`,
`tabla_conjunto_medio`, `tabla_semillas`, `tabla_calibracion`, `tabla_brecha_especificidad`, `tabla_ablaciones`, `tabla_verificaciones` y
`tabla_otro_protocolo_articulo1`, las figuras `fig_roc_test_bimcv_limpio`, `fig_fiabilidad_test`,
`fig_bosque_delta_auc` y `fig_brecha_especificidad`, y `informe_final.json` con todas las cifras, B, las semillas y la
huella SHA256 de cada fichero de entrada. El programa exige que cada corrida tenga `protocol.test_bloqueado: false` y
un `protocol.congelado` existente, y que exista el informe de la Puerta 3; si no, termina con error antes de abrir
ningún test. Las figuras usan una paleta apta para daltonismo (Okabe e Ito, completada con la de Paul Tol) con el
mismo color y marcador por método en todas, tipografía de 8 pt para un ancho de columna de unos 8.6 cm, ejes con
unidades y sin título dentro de la figura, que irá en el pie. Los nombres de experimento finales se fijan en la
especificación del programa (`DEFAULT_SPEC`, o un JSON con `--spec`) y deben coincidir con los de las
configuraciones finales cuando se escriban.

## 14. Puerta 3 implementada

`python scripts/jobs/gate3_evaluar.py paso1` elige la mejor línea base del nivel 1 sin abrir CSD-Net: mayor AUC media
por semilla en la validación robusta sobre el estado de mejor época (DFR en val_robust_sel con `dfr_best`), entre
los métodos del nivel 1 distintos del control de fuente. Escribe `results/gate3/mejor_linea_base.json` con la fecha,
el criterio, la tabla de candidatos y la huella SHA256 de cada fichero leído, y se niega a rehacerla una vez
escrito el informe del paso 2. `python scripts/jobs/gate3_evaluar.py paso2` comprueba esas huellas, verifica que
todas las corridas comparten caché, selección en val_robust_sel y test bloqueado (si no, la Puerta 3 no se evalúa),
calcula D1, D2 y D3 con B = 2 000 y semilla 2026, lee V1, V2, V2b, V3 y V5 de `probes/final_best.json` y V1 al final
de A2 de `summary.json` en cada semilla, calcula V6 con las probabilidades medias, aplica la regla de la Section 5 de
docs/07_piloto.md, indica la rama de su Section 6 y añade el dato de la enmienda 3, las ablaciones sin umbral y los
estados que no deciden. Nunca lee un conjunto test: pedirlo termina con error. Los trabajos
`jobs/staged/p3_13_gate3_mejor_linea_base.json` y `jobs/staged/p3_14_gate3_evaluar.json` ejecutan los dos pasos y
declaran sus dependencias en "notas".

Decisiones de implementación que el autor debe conocer, porque docs/07_piloto.md no las fija de forma unívoca: la
mejor línea base se busca entre los métodos del nivel 1 sin el control de fuente; la "AUC media" es la media de las
AUC por semilla (se registra también la del conjunto medio); DFR entra en esa elección con su AUC en
val_robust_sel, que no es el mismo conjunto que el de los demás (se registra también la clasificación con todos en
val_robust_sel y si coincide); V1 no concluyente al final de A2 cuenta como verificación no aprobada (rama 2 si la
discriminación se aprueba) y solo V1 con veredicto de fallo al final de A2 lleva a la rama 3; si fallan a la vez V6 y
otra verificación, el programa lo marca como rama "2+V6" y remite al autor; val_robust_fit se obtiene restando
val_robust_sel de val_robust por image_id, porque los entrenadores no la evalúan; V6 se calcula con 2 000 remuestreos
y la semilla 2026, que el protocolo no fija para V6.

## Referencias verificadas y pendientes

Verificadas el 12-11-2026 e incorporadas a refs/refs.bib (estado y enlace de comprobación en
refs/verificacion_refs.csv). Entre corchetes va la clave con la que se citan.

- Comparación de AUC correlacionadas [delong1988comparing] y su algoritmo rápido [sun2014fast].
- Procedimiento secuencial de rechazo [holm1979simple]. No tiene DOI; se cita con la ficha de JSTOR.
- Prueba t con varianzas distintas [welch1947generalization].
- Intervalos de percentiles del bootstrap [efron1994introduction] y bootstrap de conglomerados
  [field2007bootstrapping].
- Corrección del valor p de remuestreo para que nunca valga cero [phipson2010permutation].
- Error de calibración esperado [naeini2015obtaining, guo2017calibration]. La segunda no tiene DOI; se cita con
  la página de PMLR.
- Índice J para el umbral [youden1950index].
- Guías de informe: CLAIM en su actualización de 2024 [tejani2024checklist] y TRIPOD+AI [collins2024tripod].

Pendientes, sin verificar todavía y por tanto sin citar: Okabe e Ito (Color Universal Design, 2008) y Tol (nota
técnica de paletas), para la paleta apta para daltonismo de las figuras. Solo harán falta si el artículo justifica
la paleta con una cita.

## Enmiendas

Las dos son anteriores a cualquier resultado: no se ha entrenado ni evaluado ningún modelo con datos reales.

1. **Valor p del bootstrap** (12-11-2026). El valor p unilateral de la ecuación (3) lleva una unidad en
   el numerador y en el denominador para que nunca valga cero [phipson2010permutation]. Con B = 2 000 el mínimo es
   0.0005.
2. **Comparación primaria con la media por semilla** (decisión del autor, 12-11-2026). La versión 1.0 de este plan
   contrastaba la AUC del conjunto medio de las cinco semillas. La cifra principal pasa a ser la del modelo
   individual, como media de las cinco semillas, y la prueba principal la de la ecuación (1); el conjunto medio
   queda como dato secundario en una tabla aparte. Los motivos están en la Section 1. Cambian con ello las
   ecuaciones (1) y (4), las familias F0 a F2 de la Section 3, la Section 6, la tabla de la Section 12, la función
   `paired_auc_test_seedmean` de `src/eval/stats.py` y las tablas de `scripts/eval/informe_final.py`. No cambian el
   conjunto primario, el comparador, la dirección de la prueba, el nivel, el número de remuestreos ni los criterios
   del piloto (docs/07_piloto.md), que siguen usando las dos semillas del piloto como están escritos.

# Versión 2 (plan B)

14 de noviembre de 2026. Esta parte formaliza el protocolo del artículo de evaluación decidido en
PLAN.md, Section 13. Se escribe antes de que exista ningún resultado del plan B: no hay corridas de B3a ni de B2, no
se ha extraído ninguna característica de COVIDGR y el test de COVIDx sigue bloqueado. Las únicas cifras reales que
aparecen son las ya registradas en docs/07_piloto.md (Tablas E1 a E4) y las que reproduce
`python scripts/eval/auditoria_covidx.py` con los ficheros de validación existentes (Section V2.3.9); todas las demás
proceden de árboles sintéticos o de cálculos de precisión esperada y se rotulan así.

## V2.0 Relación con la versión 1

La versión 1 se escribió para un artículo de modelo cuya comparación primaria era CSD-Net frente al control de
fuente. Con el plan B esa comparación desaparece y CSD-Net pasa al material suplementario como resultado negativo
(PLAN.md, Section 13.1, C4).

Quedan sustituidas por esta parte:

- Section 1 (hipótesis y criterio principal) y Section 2 (criterios secundarios S1 a S8): sustituidas por las
  Sections V2.1 a V2.5.
- Section 3 (familias F0 a F3): sustituida por la Section V2.2.4 y las familias de cada aporte.
- Section 9 (qué se informa de cada benchmark): sustituida por las Sections V2.3 a V2.5.
- Section 11 (lo que no se hará) y Section 12 (afirmaciones previstas): sustituidas por las Sections V2.9 y V2.10.
- Section 14 (Puerta 3): queda como registro histórico; la Puerta 3 se evaluó y no se aprobó (docs/07_piloto.md).

Siguen vigentes, y esta parte se apoya en ellas:

- Section 4 (unidad de remuestreo: el paciente, sin estratificar; remuestreos no definidos descontados y registrados).
- Section 5 (B = 2 000 remuestreos con la semilla 2026; número declarado igual al usado y escrito en cada salida).
  Las semillas de entrenamiento de COVIDx siguen siendo 42, 123, 2024, 7 y 999; en B2 y B3a son 0 a 4, porque en esos
  benchmarks la semilla fija a la vez la submuestra y el entrenamiento (docs/02_datos.md, Section 3.3, punto 6).
- Section 6 en lo que dice sobre las semillas: la cifra principal es la media por semilla del modelo individual
  (PLAN.md, Section 12.10); el conjunto de cinco semillas se informa aparte como dato secundario; ninguna semilla se
  descarta ni se sustituye.
- Las definiciones de las ecuaciones (2) a (7): intervalo de percentiles, valor p de remuestreo con la corrección de
  una unidad, umbral de Youden fijado fuera del conjunto evaluado, ECE de 15 bins, brecha de especificidad y Holm.
- Section 7 (DFR en COVIDx: reajuste con val_robust_fit, comparación en val_robust_sel o en el test).
- Section 8 (subconjuntos rsna con y sin fuga, descriptivos).
- Section 10, puntos 1 a 5 (análisis de sensibilidad), ampliados en la Section V2.8.
- Las referencias verificadas y las dos enmiendas.

## V2.1 Preguntas y estructura de la inferencia

Las cuatro preguntas de PLAN.md, Section 13.2, se responden así.

| Pregunta | Aporte | Análisis que la responde | Papel |
|---|---|---|---|
| 1. En la evaluación interna de COVIDx (bimcv), ¿una referencia sin información de clase es no inferior a los métodos entrenados con etiquetas? | C1 | C1-i: detector de colección frente a cada método, no inferioridad con margen 0.02, en la parte bimcv del test limpio | confirmatorio (familia F-C1) |
| 2. ¿Qué rendimiento conservan esos modelos en una cohorte externa de un solo hospital? | C1 | C1-vi: AUC en COVIDGR con IC95 y caída frente a la evaluación interna | descriptivo con intervalo |
| 3. Con confusión controlada, ¿algún método supera a ERM en el test desconfundido, a qué rho y con qué coste? | C2 | C2: área bajo la curva dosis-respuesta frente a ERM (primaria), AUC en rho = 0.99 y 1.0 y coste en rho = 0.5 | confirmatorio (familia F-C2-area) y secundario |
| 4. ¿Se repite el patrón en tuberculosis? | C3 (C2-R desde el 17-11-2026, enmienda E0) | C3: mismos estimandos en Shenzhen y Montgomery | secundario, con advertencia de potencia |

Hay dos familias confirmatorias independientes, una por aporte, y cada una se contrasta con su propio nivel
unilateral de 0.025 con Holm dentro de la familia: F-C1 (doce comparaciones) y F-C2-area (once comparaciones; diez, (17-11-2026, enmienda E5 de las "Enmiendas previas a la congelación")). Desde el 17-11-2026 hay una tercera, F-C3, del aporte C3 (enmienda E2). No se
reparte el nivel entre ellas porque responden a preguntas distintas sobre datos distintos y cada afirmación del
artículo se apoya en una sola familia; la tabla de la Section V2.10 dice cuál.

## V2.2 Convenciones comunes

### V2.2.1 Cifra principal y emparejamiento

Para un método m con semillas s = 1, ..., S, la cifra que se informa en cualquier conjunto es la media por semilla de
la ecuación (8).

$$\overline{\mathrm{AUC}}_m = \frac{1}{S}\sum_{s=1}^{S} \mathrm{AUC}\big(p_{m,s}\big) \qquad (8)$$

En la ecuación (8), p_{m,s} son las probabilidades del modelo entrenado con la semilla s y la AUC es el estadístico de
Mann y Whitney con rangos medios. Junto a ella se informan la desviación típica entre semillas y las AUC de cada
semilla. Toda comparación entre dos métodos a y b sobre las mismas imágenes usa la diferencia de la ecuación (9),
emparejada por semilla.

$$\Delta_{a,b} = \frac{1}{S}\sum_{s=1}^{S}\Big[\mathrm{AUC}\big(p_{a,s}\big) - \mathrm{AUC}\big(p_{b,s}\big)\Big] \qquad (9)$$

En la ecuación (9), las dos AUC de cada término se calculan con las mismas imágenes. Si a es una referencia única que
no depende de la semilla (el detector de colección sobre un tronco fijo), su AUC se repite en cada término. El IC95
es el de la ecuación (2) de la versión 1, con remuestreo de pacientes que recalcula en cada remuestreo las 2S AUC con
los mismos pacientes; el valor p unilateral es el de la ecuación (3). Como análisis principal, este intervalo queda sustituido por el remuestreo jerárquico de semillas y pacientes (17-11-2026, enmienda E1 de las "Enmiendas previas a la congelación"); el de pacientes se conserva como sensibilidad. El conjunto de cinco semillas, con las
probabilidades promediadas, se informa en una tabla aparte y nunca sustituye a la cifra principal.

### V2.2.2 No inferioridad

Cuando la pregunta es si a no es peor que b en más de un margen δ, la hipótesis nula es Δ_{a,b} ≤ -δ y la alternativa
Δ_{a,b} > -δ. El valor p es el de la ecuación (10).

$$p_{\mathrm{NI}} = \frac{1 + \big|\{\, b : \Delta^{*b}_{a,b} \le -\delta \,\}\big|}{B_v + 1} \qquad (10)$$

En la ecuación (10), Δ^{*b} es la diferencia del remuestreo b y B_v el número de remuestreos válidos. La no
inferioridad se declara si el p de Holm de la familia es menor que 0.025, lo que equivale, salvo por la discreción
de los cuantiles, a que el extremo inferior del IC95 supere -δ con la corrección por multiplicidad aplicada. Se
informan siempre la diferencia, su IC95 y los dos valores p (sin corregir y de Holm).

### V2.2.3 Selección, estados y TTA

En COVIDx rige el protocolo común (docs/06_lineas_base.md, Section 4): estado de mejor época elegido en
val_robust_sel, TTA de espejo, hiperparámetros fijados con el presupuesto común y DFR con `dfr_best`. En B3a y B2 la
regla de selección es la de la Section V2.4.5. Las referencias sin información de clase no usan TTA en la versión
actual de la extracción de características; la congelación del protocolo fija si se extraen con TTA, y la decisión se aplica igual a
todas. Fijado (17-11-2026, enmienda E3 de las "Enmiendas previas a la congelación"): con promediado de volteos, igual para todas.

### V2.2.4 Multiplicidad

Holm, ecuación (7) de la versión 1, al nivel unilateral de 0.025 dentro de cada familia. Las familias son:

| Familia | Contenido | Conjunto | Papel |
|---|---|---|---|
| F-C1 | detector D_IN frente a cada uno de los doce métodos, no inferioridad con δ = 0.02 | parte bimcv del test limpio | confirmatoria |
| F-C1b | detector D_ERM frente a los doce métodos, emparejado por semilla, misma prueba | parte bimcv del test limpio | secundaria |
| F-C2-area | área bajo la curva dosis-respuesta, cada método frente a ERM | test_desconf de B3a, orientación A | confirmatoria |
| F-C2-099 | AUC en rho = 0.99, cada método frente a ERM | igual | secundaria |
| F-C2-100 | AUC en rho = 1.0, cada método frente a ERM (sin DFR en la variante principal) | igual | secundaria |
| F-C2-coste | no inferioridad en rho = 0.5 con δ = 0.01, cada método frente a ERM | igual | secundaria |
| F-C3 | AUC de cada método frente a ERM, por fuente | Shenzhen y Montgomery | secundaria (renombrada F-C2R-shenzhen y F-C2R-montgomery (17-11-2026, enmienda E0 de las "Enmiendas previas a la congelación")) |
| F-C3 (nueva, 17-11-2026) | selección de época, Δsel medio sobre métodos en rho 0.9, 0.95 y 0.99 (enmienda E2) | test_desconf de B3a | confirmatoria del aporte C3 |

Las familias secundarias solo se leen como apoyo de la confirmatoria del mismo aporte; los demás análisis son
descriptivos y se informan con su IC95 y sin valor p.

### V2.2.5 Resultados adversos

Toda tabla registrada aquí se publica completa con independencia de su resultado, en el texto principal o en el
material suplementario según la Section V2.10. Un resultado adverso para una afirmación prevista se informa con la
misma tabla y la misma prueba, y la afirmación se retira o se reformula en los términos que permita la Section
V2.3.8 o V2.4.8; no se sustituye la prueba, el conjunto, el margen ni la familia.

## V2.3 Auditoría de COVIDx CXR-4 (C1)

Datos: partición limpia de COVIDx CXR-4 [wu2023covidx], parte bimcv. La fuente bimcv reúne tres colecciones con una
sola clase cada una (docs/07_piloto.md, Tabla E1): negativos; positivos de la colección antigua (identificadores de
sujeto de cinco cifras), que son todos los positivos de validación y de test; y positivos de la colección nueva
(seis cifras), solo en entrenamiento. La colección se lee del nombre de fichero una vez conocida la clase (corregido el 17-11-2026, enmienda 5: los negativos de bimcv del test también tienen identificadores de cinco cifras). Lista cerrada de
siete análisis; no se añadirá ninguno a la familia confirmatoria después de abrir el test.

### V2.3.1 C1-i. Detector de colección sin información de clase

Definición. Regresión logística L2 con C = 0.05, pesos de clase equilibrados y 4 000 iteraciones de lbfgs, sobre las
768 características agrupadas del tronco, estandarizadas con la media y la desviación del propio conjunto de ajuste.
El conjunto de ajuste contiene solo positivos de bimcv de entrenamiento, con diana 1 para la colección antigua y 0
para la nueva; la etiqueta de clase se usa únicamente para restringir el ajuste a positivos, y el detector no ve
ningún negativo ni contrasta clases. La puntuación es la función de decisión, orientada de modo que un valor mayor
significa más parecido a la colección antigua. Los hiperparámetros son los del prototipo exploratorio
(`scripts/diag/expl_colecciones.py`) y no se ajustan con ningún conjunto de evaluación. Se registran tres troncos:

- D_IN (principal): ConvNeXt-Tiny con los pesos de ImageNet con los que se inicializan todos los métodos, congelado y
  sin ningún ajuste con imágenes médicas. Es el único tronco que no ha visto ninguna etiqueta de clase, de modo que el
  detector es estrictamente una referencia sin información de clase. Conjunto de ajuste: la misma muestra de 6 000
  positivos de bimcv de entrenamiento que usa la extracción existente (semilla de muestreo 20261113). Sus
  características aún no se han extraído (Section V2.11, pendientes). Es el principal con cualquier resultado de su
  ensayo en validación (17-11-2026, enmienda E3 de las "Enmiendas previas a la congelación").
- D_ERM (secundario): tronco ERM final de cada semilla, emparejado por semilla con los métodos. Es el detector con
  el que se obtuvieron las cifras de la Tabla E4; su tronco se entrenó con etiquetas, pero el detector no.
- D_RES (sensibilidad): tronco ERM entrenado sin el conjunto reservado de 3 248 imágenes de bimcv
  (docs/07_piloto.md, "Hipótesis de diseño y corrida exploratoria"), con el detector ajustado sobre los positivos
  reservados, que el tronco no vio.

Evaluación. Ahora, en val_robust (3 407 imágenes de 999 pacientes, 257 negativas) como ensayo; tras la congelación del protocolo, en
la parte bimcv del test limpio (4 039 negativas y 4 040 positivas), que es el conjunto confirmatorio.

Estimando. Δ_{D,m} de la ecuación (9) con a = D y b = m, para cada uno de los doce métodos de docs/06_lineas_base.md
(ERM, control de fuente, GroupDRO, DANN, CORAL, MixStyle, Fourier, LISA, DFA con fuente, DFR, ERM con enmascarado y
ERM con cuotas), con las cinco semillas del protocolo final.

Prueba y margen. No inferioridad de la Section V2.2.2 con δ = 0.02, familia F-C1 (Holm sobre las doce
comparaciones). El margen se justifica por tres razones fijadas antes de este plan. Primero, 0.02 es la menor
diferencia de AUC que el proyecto considera declarable con una prueba pareada (PLAN.md, Section 12.3, punto 5, y
criterio D1 de docs/07_piloto.md). Segundo, la afirmación de C1 no es que el detector sea tan bueno como un
clasificador clínico, sino que la evaluación interna no puede separar a los métodos de una referencia que no sabe
nada de la enfermedad; una diferencia menor que la que el propio protocolo exige para declarar una mejora entre
métodos no permite esa separación. Tercero, con el tamaño de la parte bimcv del test la semianchura esperada del IC95
de una diferencia pareada está entre 0.006 y 0.010 (Section V2.6), de modo que el margen no queda absorbido por la
imprecisión. Se informa también la prueba de superioridad (H1: Δ > 0) como descriptiva, sin Holm.

Unidad de remuestreo: paciente. Semillas: D_IN es una sola referencia y se compara con la media de las cinco
semillas de cada método; D_ERM se empareja semilla a semilla.

Qué se informa siempre: para cada método, las dos AUC medias, Δ, su IC95, el p de no inferioridad sin corregir y de
Holm y la decisión; las AUC de D_IN, D_ERM y D_RES con su IC95; el estrato AP y PA (C1-vii).

DFR. Su reajuste usa val_robust_fit, que contiene negativos y positivos de la colección antigua, es decir,
información del destino que ni el detector ni los demás métodos usan. En el test todos se comparan sobre las mismas
imágenes; en validación DFR solo es comparable en val_robust_sel. Si DFR resulta superior al detector, se informa así
y se discute como el efecto de ver etiquetas de la colección evaluada, no como prueba de que aprende la enfermedad.

### V2.3.2 C1-ii. Transferencia de una lectura entre colecciones de positivos

Solo en los troncos con conjunto reservado (D_RES: semillas 42 y 123 de las dos variantes de reserva). Estimandos:
(a) AUC fuera de pliegue, con cinco pliegues agrupados por paciente, de tres contrastes dentro del reservado:
negativos frente a positivos antiguos, negativos frente a positivos nuevos y positivos antiguos frente a nuevos; (b)
transferencia: la lectura negativos frente a antiguos, ajustada con la mitad de los negativos (partición por hash
del paciente), aplicada a la otra mitad frente a los nuevos, y la recíproca; (c) la lectura negativos frente a nuevos
aplicada a val_robust. Regresión logística L2 con C = 0.05 y pesos equilibrados, estandarización con todo el
reservado. IC95 por remuestreo de pacientes en (b) y (c). Descriptivo, sin valor p.

Interpretación registrada. Si la señal de la enfermedad dominara, la lectura ajustada en una colección de positivos
debería conservar en la otra una AUC cercana a la de dentro del contraste. Se considerará que la lectura no se
transfiere si la AUC transferida queda más de 0.10 por debajo de la del contraste de origen y su IC95 no alcanza a
esta. Este umbral es descriptivo y no forma parte de ninguna familia.

### V2.3.3 C1-iii. Evaluación de cada método por colección

Estimando: para cada método (media por semilla) y para cada referencia sin clase, la AUC de cada pareja (colección de
negativos, colección de positivos) con al menos 20 imágenes por lado, con IC95 por pacientes. En validación las
colecciones de negativos son bimcv, rsna, cohen y actmed, y las de positivos la colección antigua de bimcv,
stonybrook, ricord, sirm, cohen y actmed; en el test, las que existan con el mismo mínimo. En DFR se excluye la
mitad val_robust_fit. Descriptivo.

Lectura registrada. Dos contrastes resumen se destacan en la tabla: (a) negativos de bimcv frente a positivos de
bimcv (la evaluación interna habitual) y (b) negativos de bimcv frente a positivos de otras fuentes (stonybrook,
ricord y sirm), que no comparten colección con los negativos y donde el detector de colección no tiene por qué
funcionar. Un método que aprende la enfermedad debería mantener (b) próximo a (a); un detector de colección no. Las
parejas con negativos de rsna o de positivos de fuentes de una sola clase mezclan fuente y clase por construcción y
se informan como medida del atajo de fuente, no de la enfermedad.

### V2.3.4 C1-iv. Clasificador de solo adquisición

Definición, tomada del prototipo del diagnóstico de la firma del 13-11-2026 (`scripts/diag/r2_origen_clase.py`; docs/10_correccion_independencia.md,
Section 4, Tabla 1): los 14 estadísticos de las seis sub-bandas de detalle en la zona visible y de LL1 en la banda
periférica visible, sin ninguna información anatómica de la zona pulmonar, con la sonda de V1 (estandarización y
regresión logística). Dos variantes registradas: (a) ajustada con bimcv de entrenamiento (todos los negativos y la
muestra de positivos de la caché del prototipo) y evaluada en val_robust y, tras la congelación del protocolo, en la parte bimcv del
test; (b) validación cruzada de cinco pliegues por paciente dentro del conjunto evaluado, como en el prototipo, que
mide cuánta clase hay en la adquisición de ese conjunto y no es un clasificador desplegable. Estimando: AUC con IC95 y
Δ frente a ERM (media por semilla), descriptivo. La variante (a) usa etiquetas de clase, pero solo ve la adquisición;
por eso no entra en F-C1, cuya referencia es sin clase.

Advertencia que se informa con el resultado: en el prototipo la asociación entre clase y adquisición era más débil
en entrenamiento que en validación (docs/10_correccion_independencia.md, Section 4), de modo que la variante (a) puede
quedar muy por debajo de la (b); las dos se publican.

### V2.3.5 C1-v. Pulmones enmascarados

Dos estimandos. (a) Δ de la ecuación (9) entre ERM con enmascarado pulmonar y ERM, en la parte bimcv del test,
emparejado por semilla; descriptivo con IC95. (b) AUC del detector de colección ajustado con las características del
tronco de ERM con enmascarado (definición de la Section V2.3.1 con ese tronco), con IC95. Exige extraer esas
características (Section V2.11). Lectura: si (b) sigue cerca de D_ERM, la diferencia entre colecciones está también
dentro del campo pulmonar, lo que no permite atribuirla a marcas o rótulos fuera del tórax; si (b) cae y (a) es
negativo, una parte de la AUC interna procede de fuera de los pulmones.

### V2.3.6 C1-vi. Cohorte externa COVIDGR

COVIDGR 1.0 (426 positivas y 426 negativas de un solo hospital). Ningún modelo se
ajusta, se selecciona ni se recalibra con COVIDGR. Estimandos por método: AUC media por semilla con IC95 por
remuestreo de pacientes (o de imágenes si el manifiesto no trae paciente, lo que se declara); sensibilidad y
especificidad al umbral de Youden de la validación robusta de cada semilla (en DFR, de val_robust_sel); y la caída
de la ecuación (11).

$$C_m = \overline{\mathrm{AUC}}_m^{\,\text{bimcv test}} - \overline{\mathrm{AUC}}_m^{\,\text{COVIDGR}} \qquad (11)$$

En la ecuación (11), los dos términos son medias por semilla sobre conjuntos distintos, de modo que el IC95 de C_m se
obtiene con remuestreos independientes de pacientes en cada conjunto (los mismos B en los dos) y no es pareado. Las
mismas cifras se dan para las referencias sin clase D_IN y D_ERM, cuya AUC en COVIDGR no tiene ninguna razón para
ser alta si su rendimiento interno procede de la colección. Si el manifiesto incluye la gravedad radiológica que
publica la base, se informa la sensibilidad por grado. Todo es descriptivo, sin valor p.

### V2.3.7 C1-vii. Proyección AP o PA como estrato

En validación la proyección se asocia a la clase dentro de bimcv: el 51.6 % de los negativos y el 61.6 % de los
positivos de val_robust son AP, y la proyección sola da una AUC de 0.550 (cifras reproducidas, Section V2.3.9). Por
eso todo análisis de C1-i y C1-iii se repite dentro de cada estrato AP y PA, y se informa la AUC del clasificador de
solo proyección. La proyección no se usa para ajustar nada. Lectura: si el detector conserva su AUC dentro de cada
estrato, la proyección no explica su rendimiento.

### V2.3.8 Qué permite concluir cada resultado

| Resultado | Conclusión permitida | Conclusión no permitida |
|---|---|---|
| F-C1: D_IN no inferior a todos o casi todos los métodos | La evaluación interna de bimcv no distingue a esos métodos de una referencia que no ha visto ninguna etiqueta de clase; sus AUC internas no sostienen que hayan aprendido la enfermedad | Que los métodos no aprendan nada de la enfermedad; que la colección antigua difiera por la adquisición y no por la gravedad clínica de sus pacientes |
| F-C1: D_IN inferior, D_ERM no inferior | Las características aprendidas con etiquetas separan las colecciones de positivos tan bien como la clase; la afirmación se limita a referencias sobre esos troncos | Que una referencia sin ninguna información de clase iguale a los métodos |
| F-C1: ni D_IN ni D_ERM no inferiores | La evaluación interna sí separa a los métodos de las referencias sin clase; C1 se reformula como descripción de la composición de bimcv | Que la composición no afecte a la evaluación |
| C1-ii: lecturas que no se transfieren | La señal que se lee dentro de bimcv depende de la colección de los positivos | Que la señal transferida (lo que sí se transfiere) sea la enfermedad pura: también puede ser gravedad o adquisición compartida |
| C1-iii: métodos con (b) muy por debajo de (a) | En esos métodos la AUC interna sobrestima la discriminación fuera de la colección antigua | Una ordenación de los métodos por capacidad clínica |
| C1-vi: caída grande en COVIDGR | El rendimiento interno no se conserva en un hospital externo | Que la causa sea la colección, porque COVIDGR difiere también en hospital, equipo y población |
| C1-vii: detector estable en AP y PA | La proyección no explica el detector | Que ningún otro factor de adquisición lo explique |

Con estos datos no se puede distinguir si la colección antigua difiere por la adquisición o por la gravedad clínica
de los pacientes de la primera ola: las dos explicaciones predicen lo mismo en todos los análisis de esta lista. El
artículo lo dirá así.

### V2.3.9 Ensayo en validación ya ejecutado

`python scripts/eval/auditoria_covidx.py` (B = 2 000, semilla 2026) reproduce sin diferencias apreciables las cifras
de la Tabla E4 (`results/audit/reproduccion_tabla_E4.csv`: las nueve AUC del detector y los seis contrastes del
reservado, todas a menos de 0.0005 de lo registrado, tolerancia 0.005). La única decisión que hubo que fijar para
reproducirlas es la estandarización del detector, que se hace con el propio conjunto de ajuste (solo positivos). El
ensayo de C1-i con D_ERM en val_robust da la no inferioridad con δ = 0.02 y Holm frente a los once métodos
comparables en ese conjunto y, en val_robust_sel, frente a todos menos DFR, cuya AUC media (0.924) supera a la del
detector (0.900; Δ = -0.024, IC95 de -0.046 a -0.003). Estas cifras de validación se conocían al escribir este plan
y por eso no son confirmatorias: la prueba de la familia F-C1 es la del test tras la congelación del protocolo, con D_IN como detector
principal.

## V2.4 Comparación bajo confusión controlada (C2, benchmark B3a)

Diseño (docs/02_datos.md, Section 3.3): derrame pleural, fuente A = NIH y fuente B = CheXpert, prevalencia rho en A y
1 - rho en B, rho en {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0}, 2 800 imágenes de entrenamiento por fuente y nivel,
una imagen por paciente, fracción AP igualada entre clases dentro de cada fuente, cinco semillas (0 a 4) en la
orientación A positiva y dos en la B positiva. El test desconfundido (`test_desconf`) tiene 500 positivas y 500
negativas por fuente y es el mismo para todos los niveles y semillas; los modelos con preentreno en NIH no se evalúan
en él (docs/02_datos.md, Section 4.1, consecuencia 3).

### V2.4.1 Estimandos

La curva dosis-respuesta de un método es la AUC media por semilla en test_desconf, ecuación (8), en función de rho,
con IC95 puntual por remuestreo de pacientes. Las medidas resumen son las de la ecuación (12).

$$\begin{aligned} A_m &= \frac{1}{\rho_K - \rho_1}\sum_{k=1}^{K-1} \frac{\overline{\mathrm{AUC}}_m(\rho_k) + \overline{\mathrm{AUC}}_m(\rho_{k+1})}{2}\,(\rho_{k+1} - \rho_k), \\ D_m &= \overline{\mathrm{AUC}}_m(0.5) - \overline{\mathrm{AUC}}_m(1.0) \end{aligned} \qquad (12)$$

En la ecuación (12), A_m es el área normalizada bajo la curva entre rho = 0.5 y rho = 0.99 (los siete niveles de 0.5 a
0.99, K = 7), que equivale a la AUC media en ese intervalo, y D_m es la caída entre ausencia de confusión y confusión
total. El área se toma hasta 0.99 y no hasta 1.0 porque en rho = 1.0 DFR no está definido en su variante principal
(Section V2.4.6) y porque 0.99 es el régimen más parecido a COVIDx (PLAN.md, Section 13.3). Las demás medidas son la
AUC en rho = 0.99 y en rho = 1.0 y el coste en rho = 0.5, que es Δ_{m,ERM}(0.5). Como los niveles comparten test, los
remuestreos de pacientes son los mismos en todos los niveles, semillas y métodos, y las medidas que combinan niveles
se remuestrean de una vez.

### V2.4.2 Comparación con ERM

Para cada método m distinto de ERM, la diferencia de cada medida con la de ERM, emparejada por semilla (la semilla fija
la misma submuestra de entrenamiento para todos los métodos) y por imágenes, con IC95 de pacientes. Pruebas:

- F-C2-area (confirmatoria): H1: A_m - A_ERM > 0, Holm sobre los once métodos (diez aplicables (17-11-2026, enmienda E5 de las "Enmiendas previas a la congelación")).
- F-C2-099 y F-C2-100 (secundarias): H1: Δ_{m,ERM}(rho) > 0 en rho = 0.99 y en rho = 1.0.
- F-C2-coste (secundaria): no inferioridad en rho = 0.5 con δ = 0.01, ecuación (10). El margen es la mitad del de C1
  porque en ausencia de confusión todos los métodos resuelven la misma tarea y la semianchura esperada del IC95 de la
  diferencia en test_desconf es de unas 0.011 a 0.015 (Section V2.6); un coste mayor que 0.01 es ya apreciable para
  un método que se propone como sustituto de ERM.
- Descriptivo: diferencia de la caída D_m - D_ERM y Δ en cada uno de los demás niveles, con IC95.

Se declara que un método supera a ERM bajo confusión si se rechaza su hipótesis en F-C2-area; el nivel de rho a
partir del cual lo hace se describe con la curva de Δ y su banda, sin prueba por nivel.

### V2.4.3 Test invertido, por fuente y brecha

El test invertido (prevalencia 1 - rho en A y rho en B) es el peor caso: se informa su AUC media por semilla en cada
nivel y la diferencia con test_desconf, con IC95; también la AUC por fuente y la brecha de especificidad entre fuentes
al umbral de Youden de la validación con la distribución de entrenamiento, ecuación (6). Descriptivo.

### V2.4.4 Orientación B positiva

Comprobación con dos semillas y rho en {0.5, 0.8, 1.0}: curva de ERM y de los métodos que se entrenen, y diferencia
entre orientaciones de la caída de ERM, con IC95 (remuestreos independientes porque los modelos son distintos, pero
el test es el mismo; se usan los mismos pacientes remuestreados). Si la caída difiere entre orientaciones en más de
0.05, las conclusiones de C2 se limitan a la orientación A y se informa la asimetría.

### V2.4.5 Regla de selección de modelo

Regla principal: selección de hiperparámetros, de época (estado `best`) y del umbral con la validación que tiene la
misma distribución que el entrenamiento (`val`, misma rho). Análisis de sensibilidad: la misma corrida con la época
elegida en la validación desconfundida (`val_desconf`), que se guarda como estado `bestdc`, y la misma búsqueda de
hiperparámetros repetida con ese criterio si el presupuesto lo permite. La justificación es el protocolo de DomainBed
[gulrajani2021search]: un método de generalización solo puede usar en la selección datos que tendría en la práctica,
y la selección con datos del destino (el "oráculo") favorece a los métodos más inestables y debe informarse aparte.
En B3a nadie dispone en la práctica de una validación desconfundida, porque si la hubiera el problema no existiría.
Las dos selecciones se presentan en tablas separadas y la familia confirmatoria usa solo la principal.

### V2.4.6 DFR en rho = 1.0

DFR reajusta la última capa con una submuestra equilibrada por celdas de una mitad de la validación con la
distribución de entrenamiento [kirichenko2023last]. Con rho = 1.0 esa validación no tiene celdas minoritarias (A solo
positiva, B solo negativa) y el reajuste equilibrado no está definido. Regla: en la variante principal DFR se informa
como no aplicable en rho = 1.0, sin imputar ningún valor; entra en F-C2-area (definida hasta 0.99) y en F-C2-099, y no
en F-C2-100. Con rho = 0.99 las celdas minoritarias de la validación son muy pequeñas (unas cuatro imágenes por celda
con 400 imágenes por fuente); se usan tal cual, se informa su recuento y se marca la cifra como inestable. Como
análisis de sensibilidad, DFR-dc reajusta con una mitad de val_desconf en todos los niveles, incluido 1.0; usa
etiquetas desconfundidas que los demás métodos no ven, de modo que va en la tabla de selección con oráculo y no en
ninguna familia.

### V2.4.7 Unidad y semillas

Una imagen por paciente: el remuestreo de pacientes coincide con el de imágenes. Cinco semillas por nivel en la
orientación A. Si falta alguna corrida, el programa la lista como ausente y la medida se calcula con las semillas
presentes en todos los niveles del método; la tabla indica el número de semillas. Ampliado (17-11-2026, enmienda E1 de las "Enmiendas previas a la congelación"): cada nivel usa todas las semillas que tiene.

### V2.4.8 Qué permite concluir

| Resultado | Conclusión permitida | No permitida |
|---|---|---|
| Ningún método supera a ERM en F-C2-area | Con confusión controlada entre dos fuentes, ninguno de los doce mejora de forma detectable la discriminación desconfundida de ERM en el intervalo de rho estudiado | Que ningún método pueda hacerlo; que el resultado valga para una fuente mixta como bimcv (B3b no se ejecuta) |
| Algún método supera a ERM | Ese método reduce la dependencia de la fuente en este benchmark, con el coste que diga F-C2-coste | Que lo haga en COVIDx, donde C1 muestra que la evaluación interna no lo puede comprobar |
| Mejora solo con la selección oráculo | La ventaja depende de datos de validación que en la práctica no existen | Que el método mejore en las condiciones de uso |

## V2.5 Réplica en tuberculosis (C3, benchmark B2, variante B; C2-R desde el 17-11-2026, enmienda E0)

Mismos estimandos que C2 sin dosis: AUC media por semilla de cada método en el test robusto, por fuente (Shenzhen,
336 TB y 326 normales; Montgomery, 58 y 80), con Δ frente a ERM emparejada por semilla y por imágenes y Holm en la
familia F-C3 (once comparaciones por fuente; familias F-C2R-shenzhen y F-C2R-montgomery (17-11-2026, enmienda E0 de las "Enmiendas previas a la congelación")). Se añade la referencia sin clase análoga a C1 si la estructura lo
permite (un detector entre fuentes de positivos ajustado sin negativos), como descriptivo. La cohorte de Mendeley se
informa con AUC y especificidad, descriptiva y con la advertencia de procedencia débil de docs/02_datos.md, Section
3.2. Ni Shenzhen ni Montgomery publican identificador de paciente; la unidad de remuestreo es la imagen tras la
deduplicación por hash, y se declara.

Advertencia de potencia, que acompaña a toda tabla de C3: con 906 positivos de entrenamiento y los tamaños del test,
la semianchura esperada del IC95 de una AUC es de unas 0.03 en Shenzhen y de unas 0.07 en Montgomery (Section V2.6),
de modo que solo diferencias grandes serán detectables. C3 se presenta como coherente o no coherente en dirección con
C2, no como confirmación independiente.

## V2.6 Precisión esperada

Las semianchuras se calculan con la aproximación de Hanley y McNeil para la varianza de una AUC (referencia por
verificar al final), ecuación (13), y las de las diferencias pareadas con simulación binormal; son órdenes de magnitud
para planificar y no resultados.

$$\mathrm{Var}(\widehat{A}) \approx \frac{A(1 - A) + (n_1 - 1)(Q_1 - A^2) + (n_0 - 1)(Q_2 - A^2)}{n_1\,n_0}, \quad Q_1 = \frac{A}{2 - A}, \quad Q_2 = \frac{2A^2}{1 + A} \qquad (13)$$

En la ecuación (13), n_1 y n_0 son los positivos y negativos y A la AUC. La semianchura es 1.96 veces la raíz.

| Conjunto | n_1 / n_0 | Semianchura de una AUC (A = 0.75 / 0.85 / 0.95) | Semianchura de una diferencia pareada |
|---|---|---|---|
| Parte bimcv del test limpio de COVIDx | 4 040 / 4 039 | 0.011 / 0.008 / 0.005 | entre 0.006 y 0.010 (extrapolada del ensayo en val_robust, donde fue de unas 0.021 con 257 negativas) |
| val_robust (ensayo) | 3 150 / 257 | 0.026 / 0.018 / 0.008 | unas 0.021 observada (detector frente a ERM) |
| COVIDGR | 426 / 426 | 0.033 / 0.026 / 0.015 | no aplica (descriptivo) |
| test_desconf de B3a | 1 000 / 1 000 | 0.021 / 0.017 / 0.010 | 0.011 a 0.015 con correlación de 0.8 a 0.6 entre métodos, AUC 0.85 |
| val_desconf de B3a (piloto, supuesto de 400 / 400) | 400 / 400 | 0.034 / 0.027 / 0.016 | 0.018 a 0.025 |
| Shenzhen | 336 / 326 | 0.037 / 0.029 / 0.017 | unas 0.03 |
| Montgomery | 58 / 80 | 0.085 / 0.069 / 0.041 | unas 0.07 |

Consecuencias. Con potencia del 80 % y nivel unilateral de 0.025, la diferencia mínima detectable es unas 1.43 veces
la semianchura de la diferencia sin corrección, y unas 1.87 veces con Holm sobre once o doce comparaciones (valor z
de unos 2.83 para el comparador más exigente). En B3a eso da unas 0.02 a 0.03 de AUC en una comparación de un nivel;
el área bajo la curva promedia siete niveles con errores correlacionados y será algo más precisa. En la parte bimcv
del test de COVIDx la no inferioridad con δ = 0.02 tiene margen holgado frente a la imprecisión. En Montgomery solo
serán detectables diferencias de 0.10 o más. La media por semilla no reduce la incertidumbre del muestreo de
pacientes, que es la que miden estos intervalos; la variabilidad entre semillas se informa aparte. Corregido (17-11-2026, enmienda E1 de las "Enmiendas previas a la congelación"): las semianchuras de las diferencias en test_desconf y esta última frase quedan sustituidas por la enmienda E1.1.

## V2.7 Puertas

### V2.7.1 Piloto de B3a, criterio ejecutable

Entrada: corridas `b3a_A_<metodo>_rho<RRR>` de ERM, DFR y GroupDRO con dos semillas y rho en {0.5, 0.9, 1.0}
(PLAN.md, Section 13.9), predicciones del estado `best` en `val_desconf`. Nunca se usa el test. Criterio, aplicado a
ERM:

$$D_{\text{ERM}} = \frac{1}{S}\sum_{s}\Big[\mathrm{AUC}_s^{\,\text{val\_desconf}}(0.5) - \mathrm{AUC}_s^{\,\text{val\_desconf}}(1.0)\Big] \qquad (14)$$

En la ecuación (14), S es el número de semillas presentes en los dos niveles (al menos dos). El benchmark es
informativo si D_ERM ≥ 0.05, el extremo inferior de su IC95 por pacientes es mayor que cero y la caída es positiva en
cada semilla; no informativo si el extremo superior del IC95 es menor que 0.05; en cualquier otro caso es
indeterminado y decide el autor, con la opción registrada de añadir una semilla o los niveles 0.95 y 0.99 antes de
decidir. Un benchmark no informativo se rediseña (otra etiqueta u otras fuentes) antes de gastar el presupuesto
completo. DFR, GroupDRO y el nivel 0.9 se informan como descripción y no intervienen. Orden:
`python scripts/eval/puerta6.py` (salida en `results/gate6/`, código 0 informativo, 2 no informativo, 3
indeterminado, 4 no evaluable).

### V2.7.2 Congelación del protocolo

Antes de abrir el test de COVIDx, de B2 o de B3a se congelan y se registran con su SHA256 en
`results/gate7/congelacion.json`, con la fecha y la orden usada: (1) esta versión de docs/08_evaluacion.md; (2)
`src/eval/stats.py`, `src/eval/audit.py`, `src/eval/dosis.py`, `scripts/eval/auditoria_covidx.py`,
`scripts/eval/informe_b3.py`, `scripts/eval/puerta6.py` y el informe final que se adapte; (3) los manifiestos y sus
JSON de B1, B2, B3a y COVIDGR; (4) las configuraciones finales de cada método y benchmark; (5) los ficheros de
características de los troncos de D_IN, D_ERM, D_RES y del ERM con enmascarado, y la lista de imágenes del conjunto de
ajuste del detector; (6) la lista de experimentos y semillas que entran en cada tabla. `informe_b3.py` se niega a leer
un test sin ese fichero. Toda modificación posterior se añade como enmienda fechada y no puede apoyarse en resultados
del test. La lista de análisis de esta parte es cerrada: un análisis nuevo después de la congelación del protocolo se rotula como
posterior y no entra en ninguna familia.

## V2.8 Análisis de sensibilidad

1. Detector: C en {0.01, 0.3} y características de la penúltima capa en lugar de las agrupadas.
2. D_RES y D_ERM además de D_IN (Section V2.3.1).
3. Selección con val_desconf en B3a y DFR-dc (Sections V2.4.5 y V2.4.6).
4. Estados EMA y de última época (solo en COVIDx (17-11-2026, enmienda E4 de las "Enmiendas previas a la congelación")).
5. Conjunto de cinco semillas frente a modelo individual.
6. Remuestreo por imagen frente a remuestreo por paciente en COVIDx.
7. En COVIDx, test limpio completo además de su parte bimcv, con la advertencia de la Section 1 de la versión 1
   (las fuentes de una sola clase permiten acertar reconociendo la fuente).

## V2.9 Lo que no se hará

1. Abrir el test de COVIDx, de B2 o de B3a antes de la congelación del protocolo, ni usar sus etiquetas para elegir umbrales, estados,
   semillas, hiperparámetros, márgenes o análisis. El piloto de B3a usa solo val_desconf.
2. Cambiar después de la congelación del protocolo el detector principal, sus hiperparámetros, su conjunto de ajuste, el margen de 0.02,
   el de 0.01 del coste, el intervalo de rho del área o la composición de las familias.
3. Usar negativos, o cualquier contraste entre clases, en el ajuste de las referencias sin información de clase.
4. Imputar un valor de DFR en rho = 1.0, ni excluir a DFR de F-C2-area.
5. Sustituir la media por semilla por la mejor semilla o por el conjunto de semillas; descartar o sustituir semillas.
6. Presentar como confirmatorias las comparaciones de las familias secundarias o los análisis descriptivos, o
   declarar diferencias por un valor p sin corregir.
7. Presentar la selección con oráculo (val_desconf, DFR-dc) como resultado principal.
8. Interpretar la diferencia entre colecciones como prueba de un artefacto de adquisición: con estos datos no se
   distingue de la gravedad clínica (Section V2.3.8).
9. Ajustar, seleccionar o recalibrar ningún modelo con COVIDGR o con Mendeley.
10. Lanzar corridas nuevas de CSD-Net; su material es el del piloto (PLAN.md, Section 13.1, C4).

## V2.10 Afirmaciones previstas y pruebas que las sostienen

| Afirmación prevista | Prueba o métrica | Conjunto | Criterio | Dónde va |
|---|---|---|---|---|
| La evaluación interna de COVIDx no distingue a los métodos de una referencia sin información de clase | F-C1: no inferioridad de D_IN, δ = 0.02, Holm | parte bimcv del test limpio | p de Holm < 0.025 para cada método citado | texto principal |
| Lo mismo ocurre con referencias sobre troncos entrenados | F-C1b con D_ERM; D_RES | parte bimcv del test limpio | igual | texto principal o suplemento |
| La señal interna depende de la colección de los positivos | C1-ii, transferencia; C1-iii, contraste (b) frente a (a) | reservado; validación y test | descriptivo con IC95 (umbral de 0.10 en C1-ii) | texto principal |
| La proyección no lo explica | C1-vii | val_robust y test bimcv | AUC del detector dentro de AP y de PA con IC95 | suplemento |
| La adquisición sola lleva parte de la clase | C1-iv | val_robust y test bimcv | descriptivo | suplemento |
| Parte de la señal está dentro de los pulmones | C1-v | test bimcv | descriptivo | suplemento |
| El rendimiento interno no se conserva en un hospital externo | C1-vi: AUC en COVIDGR y caída C_m | COVIDGR | descriptivo con IC95 | texto principal |
| Con confusión controlada, ningún método (o el método X) supera a ERM | F-C2-area | test_desconf de B3a | p de Holm < 0.025 | texto principal |
| La ventaja (o su ausencia) se mantiene en el régimen de COVIDx y en confusión total | F-C2-099 y F-C2-100 | test_desconf | p de Holm < 0.025, secundario | texto principal |
| El método X no cuesta nada sin confusión | F-C2-coste, δ = 0.01 | test_desconf, rho = 0.5 | p de Holm < 0.025 | texto principal |
| El patrón no depende de la orientación | Orientación B, diferencia de caídas | test_desconf | diferencia ≤ 0.05 | suplemento |
| El patrón se repite en tuberculosis | F-C3 (F-C2R-shenzhen y F-C2R-montgomery desde el 17-11-2026) y dirección frente a C2 | Shenzhen y Montgomery | coherencia de dirección; p de Holm como secundario | texto principal, con advertencia |
| CSD-Net no cumple sus supuestos | verificaciones del piloto | val | registradas en docs/07_piloto.md | suplemento |

## V2.11 Implementación, reproducibilidad y pendientes

| Fichero | Contenido |
|---|---|
| `src/eval/audit.py` | Colección y proyección desde el nombre, carga de características, detector de colección, diferencias emparejadas con no inferioridad, contrastes del reservado y transferencia, evaluación por colección |
| `src/eval/dosis.py` | Remuestreo rápido de AUC con pesos por paciente (mismos remuestreos que `cluster_bootstrap`), inventario de corridas por nombre, curva, medidas de la ecuación (12), comparación con ERM, familias de Holm, criterio del piloto de B3a y árbol sintético para pruebas |
| `scripts/eval/auditoria_covidx.py` | Auditoría en validación; comprueba la Tabla E4 y termina con código 1 si no se reproduce |
| `scripts/eval/informe_b3.py` | Informe de B3a; por defecto en val_desconf; el test solo con `--permitir-test` y la congelación del protocolo |
| `scripts/eval/puerta6.py` | Piloto de B3a |
| `tests/test_eval_plan_b.py` | Pruebas rápidas en CPU |

Órdenes: `python scripts/eval/auditoria_covidx.py` (unos 100 s en la VM, salida en `results/audit/`),
`python scripts/eval/informe_b3.py [--estado bestdc]`, `python scripts/eval/puerta6.py` y, para ver el formato con
datos sintéticos, `python scripts/eval/informe_b3.py --demo uno_mejor --out /tmp/b3_demo`.

Contrato que debe respetar el entrenador de B2 y B3a: nombres de experimento
`<bench>_<orient>_<metodo>_rho<RRR>` con RRR = round(100 rho) en tres cifras, `rho` en `config_resolved.json`
(clave `bench.rho` o `data.rho`), estados `best` (selección en `val`) y `bestdc` (selección en `val_desconf`) y, en
DFR, `dfr_best` y `dfr_bestdc`. Los dos estados nuevos se añadieron a `KNOWN_STATES` de `src/eval/stats.py` para que
la guarda del test reconozca ficheros como `bestdc_test_desconf.csv`.

Pendientes antes de la congelación del protocolo: (1) extraer las características de D_IN (tronco de ImageNet congelado) y del ERM con
enmascarado con `scripts/diag/expl_lectura_features.py` o un trabajo equivalente, y las del test bimcv y de COVIDGR
para todos los troncos; (2) pasar el clasificador de solo adquisición de los guiones del diagnóstico de la firma a la auditoría;
(3) adaptar el informe final al plan B (tablas de C1 en el test y de COVIDGR); (4) fijar el
contrato de nombres y estados de arriba; (5) confirmar la composición de val_desconf.

## Referencias por verificar (versión 2)

No están en refs/refs.bib y no se citan por clave hasta que se verifiquen.

- Hanley J. A. y McNeil B. J., "The meaning and use of the area under a receiver operating characteristic (ROC)
  curve", Radiology, 1982. Para la ecuación (13). DOI sin comprobar.
- Tabik S. et al., artículo de presentación de COVIDGR (IEEE Journal of Biomedical and Health Informatics, 2020,
  según lo que se recuerda). Para describir la cohorte externa. Datos bibliográficos y DOI sin comprobar.
- Una referencia metodológica sobre ensayos de no inferioridad y elección del margen (por ejemplo, la extensión de
  CONSORT para no inferioridad). Sin identificar ni comprobar.

## Enmiendas a la versión 2 (14-11-2026, antes de cualquier resultado de B3a o de B2)

Proceden de la revisión del entrenamiento del plan B (docs/13_entrenamiento_plan_b.md) y no cambian
ningún estimando.

1. V2.7.1. El piloto de B3a escribe en `results/piloto_b3a/` para no coincidir con las corridas finales, que
   llevan el mismo nombre de experimento. La orden de evaluación es `python scripts/eval/puerta6.py --root
   results/piloto_b3a`.
2. V2.4.2. El control de fuente no es aplicable en B3a: con rho menor que 1 las dos fuentes son mixtas y la corrida
   sería idéntica a ERM, y con rho igual a 1 no hay ninguna fuente mixta. La familia de Holm frente a ERM queda con
   los diez métodos aplicables.
3. V2.4.6. En `val_desconf` el estado `dfr_bestdc` no es comparable, porque ese conjunto contiene sus datos de
   ajuste; la tabla con selección de oráculo se calcula en `test_desconf`.
4. V2.5. En B2 el estado `best` se elige en `val`, el estado `bestdc` en `val_robust` y DFR reajusta con mitades de
   `val`. El campo `seleccion` del JSON de los manifiestos de B2 (`val_robust`) queda sin efecto frente a esta regla.
5. V2.3 (corrección, 17-11-2026, hallazgo M2 del revisor independiente). La colección de una imagen de bimcv no se
   obtiene solo del nombre de fichero: los negativos del test también llevan identificadores de cinco cifras, de modo
   que hace falta además la etiqueta de clase del manifiesto. La Tabla E1 de docs/07_piloto.md usa, por tanto, las
   etiquetas del test para contar su composición, sin evaluar ningún modelo.
6. Usos previos de información del test de COVIDx (declaración, 17-11-2026). Antes del plan B, y sin que ninguno
   eligiera un modelo, un umbral o un hiperparámetro de este artículo, el test oficial de COVIDx se usó en cuatro
   ocasiones: (a) en la Puerta 2, para comprobar que el pipeline reproduce cifras ya publicadas (probabilidades de
   los modelos convertidos sobre las 8 482 imágenes del test oficial y evaluación de las réplicas, con las que se
   dibujó el borrador del panel b de la figura 1); (b) en la auditoría de las máscaras pulmonares, que revisó las
   84 075 imágenes con su clase, test incluido; (c) en la Tabla E1, que cuenta la composición del test por colección
   y clase; y (d) en la formulación de la hipótesis de C1 y en la elección de la parte bimcv del test como conjunto
   confirmatorio, hechas conociendo esa composición. La expresión "test bloqueado" de los documentos del proyecto
   significa que ningún modelo del plan B se ha evaluado ni seleccionado con el test; el manuscrito declarará estos
   cuatro usos en métodos, y el panel b de la figura 1 se rehará con los resultados finales tras la congelación del protocolo.

# Enmiendas previas a la congelación (17-11-2026)

17 de noviembre de 2026. Estas enmiendas responden a los hallazgos B2, B3, M3 (parte de texto), M4, m1,
m2, m4, m8 y m9 del revisor independiente (review/informe_puerta7.md) y se escriben antes de abrir ningún test: no se
ha leído ninguna predicción ni etiqueta de test de COVIDx, B2 ni B3a. Las únicas cifras nuevas son de validación y se
regeneran con las órdenes que se citan en cada caso. Donde corrigen el texto anterior, la corrección se ha marcado en
su sitio con la fecha y sin borrar lo registrado.

## E0. Numeración de los aportes

La decisión del autor del 17-11-2026 (PLAN.md, Section 13.11) fija tres hallazgos. En todo este documento la
numeración queda así: C1, auditoría de COVIDx CXR-4 (Section V2.3); C2, curva dosis-respuesta de los doce métodos en
B3a (Section V2.4), con su réplica en tuberculosis, que pasa a llamarse C2-R (Section V2.5, antes C3); y C3, la
afirmación de que la validación con la distribución de entrenamiento no sirve para elegir modelo cuando la confusión
es fuerte (enmienda E2, nueva). La familia de la réplica, antes F-C3, se divide en F-C2R-shenzhen y F-C2R-montgomery;
el nombre F-C3 designa desde ahora la familia de C3. El resultado negativo de CSD-Net sigue como material
suplementario (PLAN.md, Section 13.1, C4) y no cambia.

## E1. Remuestreo jerárquico de semillas y pacientes (hallazgo B2)

Motivo. Los intervalos de V2.2.1 y V2.4.2 remuestrean pacientes con los modelos fijos. En B3a la semilla fija a la vez
la submuestra de entrenamiento y la inicialización, y su variación forma parte del estimando "método en este
benchmark". En el piloto de B3a, con rho 0.9, la diferencia de GroupDRO frente a ERM en val_desconf vale 0.004
y 0.023 según la semilla, y el intervalo de pacientes de su media (0.004 a 0.023) excluye el cero mientras que el
jerárquico que se registra aquí (-0.005 a 0.032) no lo excluye (`python scripts/eval/informe_b3.py --root
results/piloto_b3a --out logs/eval_r2/piloto_jerarquico`; validación, dos semillas, ilustración del método y no
resultado).

Modelo. El diseño es cruzado: todas las semillas se evalúan sobre las mismas imágenes. Para un nivel r, la diferencia
de la semilla s se escribe Δ_s = μ + a_s + b + c_s, donde a_s es el efecto de la semilla, b el del muestreo de
pacientes, común a todas las semillas, y c_s la interacción entre semilla y pacientes. La varianza de la media por
semilla, que es la cifra principal de la ecuación (9), es la de la ecuación (15).

$$\mathrm{Var}\big(\overline{\Delta}_r\big) = \frac{\sigma^2_{S,r}}{K_r} + V_{P,r}, \qquad V_{P,r} = \mathrm{Var}(b) + \frac{\mathrm{Var}(c)}{K_r} \qquad (15)$$

En la ecuación (15), K_r es el número de semillas del nivel, σ²_S la varianza del efecto de semilla y V_P la varianza
por pacientes de la media de las K_r semillas, que es la que estima el remuestreo de pacientes de la versión 2. La
varianza observada entre las diferencias por semilla, s², no estima σ²_S, porque incluye la interacción. Esta se
obtiene comparando la varianza por pacientes de una semilla sola, V_1, con la de la media, y de ahí el estimador de la
ecuación (16).

$$\widehat{\sigma}^2_{S,r} = \max\Big(0,\; s^2_r - \frac{K_r}{K_r - 1}\big(\overline{V}_{1,r} - V_{P,r}\big)\Big) \qquad (16)$$

En la ecuación (16), V̄_1 es la media sobre las semillas de la varianza por pacientes de cada una. Cada remuestreo b
suma a la media remuestreada por pacientes un término de semillas, como en la ecuación (17).

$$T^{*}_{b,r} = \overline{\Delta}^{*P}_{b,r} + \lambda_r\, \overline{e}^{*}_{b,r}, \qquad \lambda_r = \sqrt{\widehat{\sigma}^2_{S,r} / s^2_r}\;\sqrt{K_r/(K_r - 1)} \qquad (17)$$

En la ecuación (17), Δ̄*P es la media por semilla calculada con los pesos de pacientes del remuestreo b (los mismos
para todos los métodos, niveles y semillas, con la semilla 2026 de la Section 5 de la versión 1), y ē* es la media,
con las semillas remuestreadas con reemplazo, de los efectos centrados e_s = Δ_s - Δ̄. El factor λ hace que la
varianza del término de semillas sea σ̂²_S / K, de modo que la interacción no se cuenta dos veces y se corrige el
factor (K - 1) / K del remuestreo de K semillas. El IC95 es el de percentiles de T* y el valor p unilateral el de la
ecuación (3) de la versión 1 (o el de la ecuación (10) en las pruebas de no inferioridad) aplicado a T*; la cifra
puntual no cambia. El remuestreo de conglomerados que se sigue es el de [field2007bootstrapping] y
[efron1994introduction].

Semillas como conglomerados. El identificador de semilla remuestreado es el mismo para todos los métodos, porque la
semilla fija la misma submuestra para todos y la comparación es pareada, y para todos los niveles en que existe:
las submuestras de entrenamiento de una misma semilla en dos niveles contiguos comparten el 95 % de las imágenes,
frente al 57 % entre semillas distintas (manifiestos de rho 0.90 y 0.95, semillas 0 y 1). Para admitir cualquier
número de semillas por nivel, las semillas se agrupan en estratos según el conjunto de niveles en que existen y se
remuestrean con reemplazo dentro de cada estrato, con su tamaño. La semilla del generador de semillas es 20261117 y
es independiente del de pacientes. Con una sola semilla en un nivel no hay término de semillas y el intervalo se
reduce al de pacientes, lo que la tabla indica con el número de semillas.

Alcance. Es el análisis principal de todas las familias y descriptivos de C2 (F-C2-area, F-C2-099, F-C2-100,
F-C2-coste, curva, caída, demás niveles), de C2-R (en B2 la unidad es la imagen deduplicada) y de C3 (enmienda E2),
y se informa también para C1-i con D_ERM, emparejado por semilla. En C1 con D_IN, que no depende de la semilla, el
término de semillas es el de los métodos. Cambia V2.4.7: cada nivel usa todas las semillas que tiene, no solo las
presentes en todos los niveles del método; el área y la caída se calculan con las medias por nivel y su prueba t de
sensibilidad con las semillas presentes en todos los niveles.

Sensibilidades registradas, que se publican junto al principal y no lo sustituyen: (i) el intervalo de pacientes con
los modelos fijos de la versión 2; (ii) el remuestreo doble sin corrección (semillas y pacientes a la vez, que cuenta
dos veces la interacción y es conservador); (iii) la prueba t pareada entre semillas, con K - 1 grados de libertad.
El número de semillas se declara como limitación en el artículo.

Comprobación con datos sintéticos de varianza conocida (`pytest -q tests/test_eval_jerarquico.py`). Con 400
imágenes, cinco semillas y un efecto de semilla de desviación 0.3 en la escala logit, el error típico del
jerárquico corregido es 0.92 veces el verdadero (Monte Carlo sobre semillas y pacientes nuevos), el de pacientes 0.35
veces y el doble 0.94 veces; sin efecto de semilla, 1.19, 1.00 y 1.66 veces. Sin variación de pacientes, el término
de semillas reproduce la varianza entre semillas dividida por K, y con varios números de semillas por nivel cada
nivel recibe exactamente las suyas.

Implementación: `src/eval/jerarquico.py` (estratos, multiplicidades, componentes de la ecuación (16) y remuestreos
de la ecuación (17)) y `src/eval/c2.py` (`curva_jerarquica`). `scripts/eval/informe_b3.py` usa desde hoy este
análisis como principal; `src/eval/dosis.py` conserva el de pacientes, que sigue siendo el del piloto de B3a (su
veredicto se reproduce igual).

### E1.1 Semianchuras esperadas (corrección de V2.6)

Las cifras de V2.6 para las diferencias pareadas en test_desconf suponían una correlación de 0.6 a 0.8 entre métodos
y no incluían la variación entre semillas. Con los ficheros de validación existentes (`python
scripts/eval/precision_b3.py`, salida en `logs/eval_r2/precision/`; B = 1 000) se obtiene lo siguiente. (1) La
correlación entre las probabilidades de cada método y las de ERM en val_desconf, con los ensayos elegidos, va de 0.72
a 0.99. (2) El error típico por pacientes de una diferencia frente a ERM con una semilla, reescalado de 800 a 2 000
imágenes, va de 0.0016 a 0.0075 (mediana de 0.0024 en rho 0.5 a 0.0040 en rho 0.95), es decir, una semianchura por
pacientes de 0.003 a 0.015, menor que la de V2.6. En el área (rho 0.5 a 0.99) es de unas 0.0013. (3) En el piloto,
con dos semillas, la desviación del efecto de semilla en la diferencia, descontada la interacción con la ecuación
(16), es prácticamente nula en rho 0.5 y en rho 1.0 (0.000 y 0.002 en GroupDRO) y vale 0.012 en GroupDRO y 0.018 en
DFR con rho 0.9 (esta última con val_desconf completo, que contiene su reajuste; sin las imágenes compartidas con val
el subconjunto queda desequilibrado y da 0.044, que no se usa); la combinación en rho 0.9 es 0.015, con un grado de
libertad por método. La interacción estimada es despreciable (menos de 0.0002 de varianza): el ruido de pacientes es
casi todo común a las semillas.

Con esos componentes, la semianchura del IC95 jerárquico y la diferencia mínima detectable con potencia del 80 % son
las de la tabla siguiente. Como en rho 0.95 y 0.99 no hay ninguna estimación de la variación entre semillas, se dan
tres supuestos: la observada en rho 0.9, 1.5 veces y 2 veces. La columna de Holm usa el comparador más exigente de
una familia de diez (z de 2.81).

| Nivel | σ_S supuesta | K | Semianchura del IC95 | Diferencia mínima detectable, sin corrección | Con Holm sobre diez |
|---|---|---|---|---|---|
| rho 0.9, 0.95 y 0.99 | 0.015 (observada en 0.9) | 5 | 0.015 | 0.021 a 0.022 | 0.027 a 0.029 |
| rho 0.9, 0.95 y 0.99 | 0.015 (observada en 0.9) | 10 | 0.011 a 0.012 | 0.016 a 0.017 | 0.021 a 0.023 |
| rho 0.9, 0.95 y 0.99 | 0.022 (1.5 veces) | 5 | 0.021 | 0.030 | 0.039 |
| rho 0.9, 0.95 y 0.99 | 0.022 (1.5 veces) | 10 | 0.016 | 0.022 a 0.023 | 0.029 a 0.030 |
| rho 0.9, 0.95 y 0.99 | 0.030 (2 veces) | 5 | 0.027 | 0.039 | 0.050 a 0.051 |
| rho 0.9, 0.95 y 0.99 | 0.030 (2 veces) | 10 | 0.020 | 0.028 a 0.029 | 0.037 a 0.038 |
| Área de 0.5 a 0.99 | perfil de los tres supuestos | 5 | 0.008 a 0.010 | 0.012 a 0.014 | 0.015 a 0.018 |
| Área de 0.5 a 0.99 | perfil de los tres supuestos | 10 | 0.006 a 0.007 | 0.009 a 0.010 | 0.011 a 0.013 |

En el área, la desviación de semilla de cada nivel se toma lineal entre cero en rho 0.5 y la de rho 0.9, igual al
supuesto en 0.95 y 0.99, y perfectamente correlacionada entre niveles, lo que da una cota superior. Los pesos del
trapecio de la ecuación (12) son 0.10, 0.20, 0.20, 0.20, 0.15, 0.09 y 0.04 para los niveles de 0.5 a 0.99, de modo
que los tres niveles de confusión fuerte pesan el 29 % del área: una ventaja limitada a rho de 0.9 a 0.99 se diluye en
la medida confirmatoria. No se cambia la medida, que se registró antes de cualquier dato de B3a; el artículo lo dirá
al interpretar F-C2-area junto con F-C2-099.

Lectura para el plan. Con cinco semillas, F-C2-099 detectará con Holm diferencias de unas 0.03 si la variación entre
semillas en rho 0.99 se parece a la de rho 0.9, y de unas 0.05 si es el doble. El coste en rho 0.5, con variación de
semilla casi nula, tiene una semianchura de unas 0.005 frente al margen de 0.01, de modo que la no inferioridad de
F-C2-coste tiene potencia alta. La frase final de V2.6 ("la media por semilla no reduce la incertidumbre del muestreo
de pacientes...") queda sustituida por esta sección.

### E1.2 Opción que decide el autor: diez semillas en rho 0.9, 0.95 y 0.99

Ampliación: semillas 5 a 9 en los niveles 0.9, 0.95 y 0.99 de la orientación A para los once métodos aplicables de
B3a. ERM debe entrar, porque toda comparación es pareada por semilla, de modo que son 11 x 3 x 5 = 165 corridas (el
revisor contó 150 con diez métodos), unas 14 horas de ranura de GPU con los 5 minutos por corrida medidos en la
búsqueda, más los manifiestos rho0.90, rho0.95 y rho0.99 de las semillas 5 a 9 y su caché, que construye el
constructor de datos de B3a. Beneficio calculado: la diferencia mínima detectable con Holm en esos niveles baja de
0.028 a 0.022 (el 21 %) con la variación observada y de 0.051 a 0.037 (el 27 %) con el doble; la del área baja de
0.015 a 0.011 (el 26 %). Si se aprueba, la decisión se anota aquí antes de la congelación del protocolo y las semillas 5 a 9 entran en
todas las familias y descriptivos de sus niveles; el área y la caída usan las medias por nivel. Si no se aprueba, el
plan sigue con cinco semillas sin ningún cambio: el código admite cualquier número de semillas por nivel
(`pytest -q tests/test_eval_jerarquico.py -k semillas_por_nivel`). En ningún caso se añaden semillas después de abrir
el test.

## E2. Aporte C3: la selección de modelo (hallazgo B3)

Afirmación. Cuando la confusión entre clase y fuente es fuerte, la validación con la distribución de entrenamiento
no permite elegir el modelo: ni la época ni los hiperparámetros que elige se acercan a los que elegiría una validación
desconfundida, y la pérdida que causa es del orden de las diferencias entre métodos. La afirmación surgió del análisis
exploratorio de la búsqueda (docs/13, Tabla H3 y correlaciones de Spearman), ya visto en val_desconf, y por eso se
registra con estimandos que solo pueden calcularse en el test.

(a) Selección de época (confirmatoria, familia F-C3). Con las corridas finales de B3a, para cada método m, nivel rho y
semilla s se define la ecuación (18).

$$\Delta^{\mathrm{sel}}_{m,s}(\rho) = \mathrm{AUC}^{\,\mathrm{test\_desconf}}\big(\mathrm{bestdc}_{m,s,\rho}\big) - \mathrm{AUC}^{\,\mathrm{test\_desconf}}\big(\mathrm{best}_{m,s,\rho}\big) \qquad (18)$$

En la ecuación (18), best es el estado elegido con la validación de la misma rho (regla principal de V2.4.5) y bestdc
el elegido entre las mismas épocas con val_desconf, que es un oráculo: Δsel mide lo que se pierde por no disponer de
una validación desconfundida. En DFR se usan los estados de su etapa ERM (best y bestdc), porque dfr_bestdc cambia
además los datos del reajuste. El estimando de la familia es la media sobre los once métodos aplicables, Δ̄sel(rho),
con el IC95 jerárquico de la enmienda E1 (las mismas semillas remuestreadas para todos los métodos). Hipótesis:
H1: Δ̄sel(rho) > 0 en rho 0.9, 0.95 y 0.99, con Holm sobre los tres niveles al 0.025 unilateral. Los niveles 0.5 a 0.8
y 1.0 se informan como descriptivos, igual que Δsel por método y nivel, con su IC95 jerárquico y la prueba t de
sensibilidad. Para juzgar la relevancia se informa junto a cada nivel la mayor diferencia frente a ERM de C2 en ese
nivel, sin prueba. En B2 se informa la misma diferencia (bestdc elegido en val_robust frente a best) por fuente, como
descriptivo de C2-R.

(b) Selección de hiperparámetros (análisis registrado que solo se ejecuta si el autor lo aprueba). Las 704 corridas
de la búsqueda no conservan estados ni evaluaron el test, de modo que esta parte solo es contrastable repitiéndolas.
Diseño: ERM, LISA, GroupDRO y DFR; rho 0.9, 0.95 y 0.99; los ocho ensayos de la búsqueda con sus mismos
hiperparámetros (índices 0 a 7 del protocolo común); semilla 0; evaluación de test_desconf dentro de la corrida tras
la congelación del protocolo, con salida en `results/c3_busqueda/` para no tocar la búsqueda. Son 4 x 3 x 8 = 96 corridas, unas 8 horas
de GPU. Para cada método y nivel, con la regla registrada (máxima early_stopping.best, empate al ensayo menor; si la
repetición no reproduce exactamente las métricas de la búsqueda, se usan las de la repetición), se definen las
cantidades de la ecuación (19).

$$G = \mathrm{AUC}^{\,\mathrm{test}}(t_{\mathrm{regla}}) - \frac{1}{8}\sum_{t=0}^{7} \mathrm{AUC}^{\,\mathrm{test}}(t), \qquad L = \mathrm{AUC}^{\,\mathrm{test}}(t_{\mathrm{oráculo}}) - \mathrm{AUC}^{\,\mathrm{test}}(t_{\mathrm{regla}}) \qquad (19)$$

En la ecuación (19), G es la ganancia de la regla sobre elegir un ensayo al azar, L la pérdida frente al ensayo que
elegiría la AUC en val_desconf del estado best (en DFR, sin las imágenes de val, que contienen su reajuste), y la
AUC se calcula en test_desconf. Se informan G, L y la correlación de Spearman entre la métrica de selección y la AUC
en test_desconf entre los ocho ensayos, por método y nivel y su media por nivel, con IC95 de pacientes; con una
semilla no se recoge la variación entre semillas, y por eso esta parte es descriptiva y no entra en ninguna familia.
Lectura registrada: si G es próxima a cero o negativa y L positiva en los tres niveles, la regla no elige
hiperparámetros mejor que el azar cuando la confusión es fuerte. Si el autor no aprueba la repetición, la parte de
hiperparámetros de C3 se presenta solo como exploratoria (Tabla H3 y correlaciones de docs/13, con la advertencia de
que se calcularon en val_desconf tras la búsqueda).

(c) Correlación de rangos (descriptiva). Con las corridas finales, por nivel, Spearman entre métodos de la métrica de
selección media por semilla (early_stopping.best) y la AUC media por semilla en test_desconf, con IC95 del remuestreo
doble. Con rho 1.0 la métrica de selección está saturada (1.000 en todos los métodos) y la correlación no está
definida; se informa así.

Qué permite concluir:

| Resultado | Conclusión permitida | No permitida |
|---|---|---|
| F-C3 rechaza en los niveles de confusión fuerte | La época elegida con la validación de la misma distribución pierde de forma detectable frente a la que elegiría una validación desconfundida, y la pérdida se compara con las diferencias entre métodos de C2 | Que exista en la práctica una validación desconfundida; que la pérdida sea recuperable con otra regla sin datos del destino |
| F-C3 no rechaza | La selección de época no explica las diferencias de C2 en este benchmark | Que la validación sirva para elegir hiperparámetros |
| (b) con G próxima a cero y L positiva | La regla no elige hiperparámetros mejor que el azar con confusión fuerte | Generalizar a otras reglas de selección |

Filas que se añaden a la tabla de V2.10:

| Afirmación prevista | Prueba o métrica | Conjunto | Criterio | Dónde va |
|---|---|---|---|---|
| Con confusión fuerte, la época elegida con la validación de la misma distribución pierde frente a la de una validación desconfundida | F-C3: Δ̄sel en rho 0.9, 0.95 y 0.99, IC95 jerárquico, Holm sobre tres | test_desconf de B3a | p de Holm < 0.025 | texto principal |
| La pérdida es del orden de las diferencias entre métodos | Δ̄sel frente a la mayor diferencia de C2 en el mismo nivel | test_desconf | descriptivo con IC95 | texto principal |
| La regla no elige hiperparámetros mejor que el azar con confusión fuerte | G, L y Spearman de la ecuación (19), solo si se aprueba la repetición | test_desconf | descriptivo con IC95 de pacientes | texto principal o suplemento |
| La métrica de selección no ordena a los métodos como el rendimiento desconfundido | Spearman entre métodos (c) | test_desconf | descriptivo | suplemento |

Implementación: `src/eval/c3.py` y `python scripts/eval/informe_c2_c3.py c3-epoca`, `c3-spearman` y `c3-hiper
--root-c3 results/c3_busqueda`.

## E3. Detector principal y promediado de volteos en C1 (hallazgos M3, parte de texto, y m9)

1. D_IN es el detector principal de F-C1 sea cual sea el resultado de su ensayo en val_robust, que todavía no existe.
   Si ese ensayo resulta desfavorable, D_IN sigue siendo el principal, la prueba tras la congelación del protocolo es la misma y la
   conclusión se lee con la tabla de V2.3.8; D_ERM nunca pasa a principal. El margen de 0.02 se fijó conociendo el
   ensayo con D_ERM en val_robust (V2.3.9), lo que el artículo declara.
2. Sensibilidad registrada: F-C1 con δ = 0.01, la misma prueba de la ecuación (10) y Holm sobre las doce
   comparaciones. Se publica junto a la principal y no la sustituye. Para cada comparación se informa siempre el IC95
   de Δ, de modo que el lector pueda aplicar su propio margen.
3. Promediado de volteos. Todas las referencias sin información de clase (D_IN, D_ERM, D_RES y el detector sobre el
   tronco de ERM con enmascarado) y el clasificador de solo adquisición se evalúan con el mismo promediado de espejo
   horizontal que los métodos: la puntuación es la media de la función de decisión sobre las características de la
   imagen original y de la volteada. El ajuste del detector se hace solo con las imágenes originales, como está
   registrado. Las cifras de validación ya publicadas sin volteo (Tablas E4 y E5) se conservan como están. La misma
   ERM aparece como 0.872 sin volteo (Tabla E4) y 0.879 con volteo: en las tablas de C1 se usa siempre la versión con
   volteo. Sensibilidad: todo sin volteo, referencias y métodos (los best.pt de las finales de COVIDx se conservan).
   La extracción de características debe, por tanto, guardar también las de la imagen volteada; es trabajo del
   código de C1.

## E4. Estados EMA y de última época (hallazgo M4)

Las corridas finales de COVIDx conservan los estados best, last y ema (cambio del 17-11-2026, anotado
en los trabajos `jobs/staged/_pasos_b1_final/*.json`), de modo que la sensibilidad 4 de V2.8 se ejecuta en COVIDx: los
doce métodos con sus cinco semillas se evalúan en la parte bimcv del test también con ema y last (en DFR, dfr_ema y
dfr_last si la corrida los conserva; si no, DFR queda fuera de esta sensibilidad y se declara), y se recalculan F-C1 y
F-C1b como sensibilidad, sin familia propia. En B2 y B3a solo existen y se evalúan best y bestdc, porque el test se
evalúa dentro de la corrida y las semillas 1 a 4 no conservan estados (docs/13, Section 5): allí la sensibilidad 4 no
se aplica y la de selección es la 3 (bestdc). `scripts/eval/informe_b3.py` solo admite desde hoy los estados best y
bestdc.

## E5. Familias, recuentos y marcas de no comparable (hallazgos m1, m2 y m4)

Recuentos, que corrigen los de V2.1 y V2.4.2. F-C1 y F-C1b, doce comparaciones. F-C2-area, F-C2-099 y F-C2-coste, diez
(los once métodos aplicables de B3a menos ERM; el control de fuente no es aplicable, enmienda 2). F-C2-100, nueve
(sin DFR, V2.4.6). F-C2R-shenzhen y F-C2R-montgomery, once cada una (los doce métodos menos ERM). F-C3, tres niveles.
Solo F-C1 y F-C2-area son confirmatorias en sus aportes, y F-C3 en el suyo; las demás son secundarias. La caída
D_m - D_ERM es descriptiva y no forma familia: `familias_holm` de `src/eval/dosis.py` marcaba además F-C2-100, F-C2-099
y F-C2-coste como confirmatorias y aplicaba Holm a una familia F-C2-caida no registrada; está corregido, y el análisis
principal usa `familias_c2` de `src/eval/c2.py`, con una columna de papel (confirmatoria, secundaria o descriptiva).

DFR. Un estado reajustado no es comparable en ningún conjunto que contenga imágenes de su conjunto de reajuste o las
comparta con él. Regla completa: en B3a, dfr_best y dfr_bestdc no son comparables ni en val ni en val_desconf (val
contiene val_fit y comparte de 168 a 283 de sus 800 imágenes con val_desconf, de 85 a 150 de val_fit; val_desconf
contiene val_desconf_fit); en B2, ni en val ni en val_robust (val_robust está dentro de val y 896 de sus 1 791 imágenes
están en val_fit); en COVIDx, ni en val ni en val_robust completo. Los summary.json solo marcan dfr_best/val y
dfr_bestdc/val_desconf (o val_robust en B2); el entrenador no se modifica mientras corren las finales, y la regla se
aplica en la evaluación (`src.eval.c2.no_comparable`, que excluye esos ficheros por defecto). En test no hay
solapamiento (revisor, Section 3). Cifras de validación ya publicadas afectadas, que deben llevar nota: en
docs/13, la última columna de la Tabla P6 ("DFR reajustado con val") y la frase "DFR recupera 0.035 (IC95 de 0.025 a
0.045)"; la fila de DFR de la Tabla H1 y las cifras de DFR de la "lectura prudente" (0.036, 0.051 y 0.040); las cuatro
columnas de la Tabla H3, en las que DFR entra en la media, y en particular la de "AUC media dentro de cada fuente",
cuyo criterio para DFR se calcula con dfr_best en val; la columna val de DFR en la Tabla H4 (0.957); y, en
`results/`, la fila de DFR de `b3a/busqueda_val_elegido.csv`, `busqueda_val_desconf_elegido.csv`,
`busqueda_criterios_*.csv` y las columnas val y val_robust de DFR en `b2/busqueda_todas.csv`. El revisor comprobó que
quitar val_fit no reduce la ventaja de DFR sobre su etapa ERM (en rho 0.99, 0.040 con todo y 0.060 sin val_fit), de
modo que no hay inflación apreciable, pero las cifras no son comparables con las de los demás métodos.

## E6. Reproducibilidad de las tablas de la búsqueda (hallazgos M5, M6, m3 y m8)

Órdenes. `python scripts/eval/tablas_busqueda.py todo --comparar` regenera en `logs/eval_r2/regenerado/` los ocho CSV
de `results/b3a/`, el de `results/b2/`, los dos de `results/b1/` y las Tablas H1 a H4 con las correlaciones de
Spearman, y los compara valor a valor con los ficheros actuales y celda a celda con docs/13. `python
scripts/eval/elegir_b1.py` reproduce byte a byte `results/b1/congelados/elegidos.json` y escribe los doce ficheros de
congelación de COVIDx, `results/b1/congelados/baseline_<metodo>.json`, con el formato de `results/b3a/congelados/`
(method, params, lr_B, trial, metrica_seleccion, corrida_busqueda), comprobando que los hiperparámetros coinciden con
los del protocolo común para ese ensayo y con los de las corridas finales ya iniciadas.

Resultado de la comparación. Todo coincide, con dos excepciones explicadas. (1) En
`busqueda_elegidos_detalle.csv`, 74 celdas difieren solo en el último dígito de coma flotante (como máximo 1.1e-16),
porque el fichero se escribió con otra implementación de la AUC. (2) En `busqueda_ensayo_elegido.csv`, la columna de
rho 1.0 difiere en diez de los once métodos: se escribió con el máximo de la métrica en el orden en que el sistema de
ficheros devuelve los directorios, y no con la regla registrada, que en los empates elige el ensayo menor. Los
ficheros congelados (`results/b3a/congelados/`), la Tabla H1 y el resto de CSV usan la regla registrada. La opción
`--regla-antigua` reproduce el fichero tal como está. Se recomienda sustituirlo por el regenerado.

Empates. Con rho 1.0, nueve de los once métodos empatan exactamente en los ocho ensayos y ERM con máscara en seis;
en los diez se elige el ensayo 0 (valores por defecto) con una época temprana, de modo que F-C2-100 compara, en la
práctica, configuraciones por defecto; el artículo lo dirá. La excepción es DFA con fuente: su ensayo 1 tiene una métrica de 1.0000000000000002 frente a 1.0 en
los demás, una diferencia de redondeo, y la regla registrada, aplicada literalmente, lo elige (`congelados/
b3a_A_dfa_fuente_rho100.json`, ensayo 1). Se mantiene la regla tal como se aplicó y se declara. Con rho 0.99 el
recorrido medio de la métrica entre ensayos es de 0.0009.

Definiciones de las tablas exploratorias (m8). Tabla H2: para cada método y nivel, el ensayo se elige con
early_stopping_dc.best (AUC sin volteo en val_desconf, o en val_desconf_sel en DFR; empate al ensayo menor) y la
cifra es la AUC con volteo en val_desconf del estado bestdc de ese ensayo; en DFR, el bestdc de su etapa ERM, sin
reajuste. Tabla H3: media sobre los métodos (once, diez en rho 1.0, donde DFR no está definido) de la AUC con volteo
en val_desconf del estado best (dfr_best en DFR) del ensayo elegido por cada criterio: "peor" y "mejor ensayo" son el
mínimo y el máximo sobre los ocho ensayos; "regla registrada", el de la máxima early_stopping.best; "dentro de cada
fuente", el de la máxima media de las AUC con volteo en val calculadas dentro de cada fuente que tiene las dos clases
(en DFR, con dfr_best en val, que contiene val_fit; no definida en rho 1.0). Las correlaciones de Spearman se calculan
entre los ocho ensayos de cada método, con los métodos cuya métrica no es constante, y se resumen por nivel con su
media, mínimo y máximo.

## E7. Código para los análisis de test de C2, C2-R y C3 (parte de B1 del revisor)

| Orden | Tabla |
|---|---|
| `python scripts/eval/informe_b3.py --conjunto test_desconf --permitir-test --out results/b3/test_A_best` | Curva, medidas y F-C2 (Tablas 2 y 3) |
| `python scripts/eval/informe_b3.py --conjunto test_desconf --permitir-test --estado bestdc --out results/b3/test_A_bestdc` | Selección con oráculo (V2.4.5) |
| `python scripts/eval/informe_c2_c3.py invertido`, `fuente`, `brecha` | V2.4.3 |
| `python scripts/eval/informe_c2_c3.py orientacion` | V2.4.4 |
| `python scripts/eval/informe_c2_c3.py c3-epoca`, `c3-spearman`, `c3-hiper` | C3, enmienda E2 |
| `python scripts/eval/informe_b2.py --test --delta-c2 results/b3/test_A_best/comparacion_erm.csv` | C2-R |
| `python scripts/eval/informe_plan_b.py` | Informe que reúne las tablas de C2, C2-R y C3 |

Precisiones. La brecha de especificidad de V2.4.3 es la especificidad en la fuente con menos positivos en
entrenamiento menos la de la fuente con más positivos (en la orientación A, CheXpert menos NIH), al umbral de Youden de
val de cada semilla (val_sel en DFR); el test invertido se compara con test_desconf remuestreando los pacientes sobre
la unión de los dos conjuntos. En V2.4.4 las semillas se remuestrean como conglomerados que atraviesan las dos
orientaciones. Todo camino que lee un test llama antes a `exigir_congelacion` de `src/eval/desbloqueo.py`; el código se
prueba solo con árboles sintéticos (`pytest -q tests/test_eval_jerarquico.py`, `python
scripts/eval/informe_plan_b.py --demo --out /tmp/pb_demo`).

## E8. Decisiones del autor del 17-11-2026 sobre los experimentos adicionales (registro previo a su ejecución)

El 17-11-2026 se aprobaron todos los experimentos adicionales que propuso el revisor independiente
(`review/informe_puerta7.md`, tabla de amenazas), con la condición de que se ejecuten de forma automática. Quedan
registrados aquí antes de lanzar ninguna de sus corridas. Solo el primero toca una familia confirmatoria, y lo hace
como ya preveía E1.2: aumenta el número de semillas de tres niveles. Los demás son análisis de sensibilidad o
descriptivos, no entran en ninguna corrección por multiplicidad y se publican sea cual sea su resultado.

1. Diez semillas en rho 0.9, 0.95 y 0.99 (E1.2): aprobada. Semillas 5 a 9 de la orientación A para los once métodos
   aplicables, 165 corridas con la configuración congelada de cada método y nivel. Los manifiestos de las semillas 5 a
   9 se construyen con el constructor de B3a y sus mismas reglas (solo cambian `train` y `val`; `val_desconf`,
   `test_desconf` y `test_invertido` son los mismos de los 46 manifiestos existentes) antes de la congelación del protocolo y entran en
   la congelación. Las diez semillas entran en todas las familias y descriptivos de esos tres niveles; el área y la
   caída usan las medias por nivel.
2. Aporte C3: se registra como contrastable. La familia F-C3 de la parte (a) de E2 queda como confirmatoria, y la
   repetición de la búsqueda de la parte (b) de E2 queda aprobada con el diseño allí escrito (ERM, LISA, GroupDRO y
   DFR; rho 0.9, 0.95 y 0.99; ocho ensayos; semilla 0; 96 corridas tras la congelación del protocolo, salida en
   `results/c3_busqueda/`).
3. Sensibilidad al tronco (S-T): aprobada. ResNet-50 con pesos de ImageNet (la etiqueta efectiva de los pesos se
   registra en el resumen de cada corrida) en B3a, orientación A, con ERM, GroupDRO, LISA y DFR, en rho 0.5, 0.9, 0.99
   y 1.0, semillas 0 a 2, con los valores por defecto de cada método (ensayo 0) y la misma regla de selección de época.
   Son 45 corridas, porque DFR no es aplicable en rho 1.0. Se informa, con el intervalo jerárquico de E1 y sin prueba
   de hipótesis: la caída de ERM entre rho 0.5 y rho 1.0 en `test_desconf`, la diferencia de cada método frente a ERM
   en rho 0.9 y 0.99, y si el signo de cada diferencia coincide con el del tronco principal. Los hiperparámetros no se
   buscan para este tronco, lo que el artículo declara como límite de la comparación.
4. Sensibilidad a la etiqueta (S-E): aprobada. Cardiomegalia con las mismas dos fuentes (NIH como A y CheXpert como B),
   construida con las reglas de B3a (reservas de pacientes fijas por fuente, mismas definiciones de etiqueta positiva y
   negativa en cada fuente, mismos conjuntos y tamaños; si los recuentos disponibles no alcanzan, los tamaños se
   reducen con una regla escrita en `docs/11_benchmarks_plan_b.md` antes de entrenar nada). ConvNeXt-Tiny, orientación
   A, ERM, GroupDRO, LISA y DFR, rho 0.5, 0.9, 0.99 y 1.0, semillas 0 a 2, valores por defecto: 45 corridas. Se informa
   lo mismo que en S-T. Los manifiestos se construyen antes de la congelación del protocolo y entran en la congelación.
5. Estabilidad de la búsqueda (S-B): aprobada. Se repite la búsqueda de ERM y de LISA en rho 0.9 y 0.99 con el
   manifiesto y la semilla de entrenamiento 1 (ocho ensayos, 32 corridas, solo validación, sin estados). Se informa,
   para cada método y nivel: si el ensayo elegido con la regla registrada coincide con el de la semilla 0, la
   correlación de rangos de Spearman entre las métricas de selección de los ocho ensayos en las dos semillas y la
   diferencia de AUC en `val_desconf` entre el ensayo elegido con cada semilla. Es descriptivo y no cambia las
   configuraciones congeladas, que siguen siendo las elegidas con la semilla 0. Al usar solo validación, puede
   ejecutarse antes de la congelación del protocolo.
6. Orden y condiciones. S-B se ejecuta antes de la congelación. Los puntos 1 a 4 evalúan el test dentro de la corrida
   y solo se lanzan con la congelación del protocolo válida y el visto bueno del autor para abrir los tests; su
   código, sus configuraciones y sus manifiestos entran en la lista de la congelación. Después de abrir el test no se
   añade ningún experimento ni ninguna semilla.
7. Otras decisiones que quedan fijadas con esta enmienda. El intervalo principal de C2, C2-R y C3 es el jerárquico
   corregido de E1. En rho 1.0 la métrica de selección empata entre los ocho ensayos y la regla registrada se
   mantiene tal como se aplicó, incluido el caso de DFA con fuente, donde resultó elegido el ensayo 1 por una
   diferencia en el último dígito; el artículo lo declara. Las referencias sin información de clase se evalúan con
   promediado de espejo, como fija E3, y `configs/gate7_lista_c1.yaml` se ajusta a ello.

## Referencias por verificar (enmiendas previas a la congelación)

- Owen A. B., artículo sobre el remuestreo de diseños cruzados ("pigeonhole bootstrap"), Annals of Applied
  Statistics, 2007, según lo que se recuerda. Para justificar que el remuestreo doble de la sensibilidad (ii) es
  conservador. Datos y DOI sin comprobar.
