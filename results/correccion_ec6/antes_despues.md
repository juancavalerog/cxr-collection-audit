# Impacto de la corrección del factor del término de semillas

Filas cuyo IC95 cambia (303 de 786 filas revisadas). Las 483 filas restantes son idénticas bit a bit: niveles con un solo estrato de semillas (rho 0.5 a 0.8 y 1.0), la caída entre rho 0.5 y 1.0 y el coste en rho 0.5.

## Decisiones de familia que cambian: 0

Ninguna.

## Comparaciones de familia con IC95 distinto

| tabla | familia | metodo | medida | estimacion | ic95_inf_antes | ic95_sup_antes | ic95_inf_despues | ic95_sup_despues | p_antes_txt | p_despues_txt | p_holm_antes_txt | p_holm_despues_txt | razon_anchuras | decision_antes | decision_despues | cambia_decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B3a selección principal, comparación con ERM | F-C2-099 | coral | delta_auc_rho099 | -0.0148 | -0.0240 | -0.0035 | -0.0250 | -0.0025 | 0.9935 | 0.9880 | 1.0000 | 1.0000 | 1.0970 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | coral | delta_area_dosis | 0.0019 | -0.0008 | 0.0047 | -0.0009 | 0.0048 | 0.0980 | 0.1074 | 0.4423 | 0.5372 | 1.0391 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-099 | dann | delta_auc_rho099 | -0.0658 | -0.0860 | -0.0447 | -0.0960 | -0.0345 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.4854 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | dann | delta_area_dosis | 0.0020 | -0.0010 | 0.0050 | -0.0012 | 0.0052 | 0.0885 | 0.1089 | 0.4423 | 0.5372 | 1.0798 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-099 | dfa_fuente | delta_auc_rho099 | -0.0289 | -0.0526 | -0.0066 | -0.0565 | -0.0030 | 0.9975 | 0.9890 | 1.0000 | 1.0000 | 1.1640 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | dfa_fuente | delta_area_dosis | -0.0014 | -0.0050 | 0.0027 | -0.0052 | 0.0029 | 0.7351 | 0.7296 | 1.0000 | 1.0000 | 1.0507 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-099 | dfr | delta_auc_rho099 | 0.0483 | 0.0091 | 0.0873 | 0.0091 | 0.0873 | 0.0070 | 0.0075 | 0.0700 | 0.0750 | 1.0004 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | dfr | delta_area_dosis | 0.0104 | 0.0047 | 0.0158 | 0.0047 | 0.0158 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0030 | rechaza H0 | rechaza H0 | no |
| B3a selección principal, comparación con ERM | F-C2-099 | erm_cuotas | delta_auc_rho099 | -0.0008 | -0.0131 | 0.0139 | -0.0131 | 0.0139 | 0.5707 | 0.5707 | 1.0000 | 1.0000 | 1.0003 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | erm_cuotas | delta_area_dosis | 0.0072 | 0.0045 | 0.0099 | 0.0045 | 0.0099 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 0.9993 | rechaza H0 | rechaza H0 | no |
| B3a selección principal, comparación con ERM | F-C2-099 | erm_mascara | delta_auc_rho099 | -0.0153 | -0.0268 | -0.0027 | -0.0272 | -0.0023 | 0.9925 | 0.9920 | 1.0000 | 1.0000 | 1.0329 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | erm_mascara | delta_area_dosis | -0.0217 | -0.0285 | -0.0141 | -0.0286 | -0.0140 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0056 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-099 | fourier | delta_auc_rho099 | 0.0029 | -0.0089 | 0.0175 | -0.0107 | 0.0198 | 0.3778 | 0.3948 | 1.0000 | 1.0000 | 1.1497 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | fourier | delta_area_dosis | -0.0049 | -0.0078 | -0.0022 | -0.0078 | -0.0021 | 0.9995 | 0.9995 | 1.0000 | 1.0000 | 1.0040 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-099 | groupdro | delta_auc_rho099 | 0.0090 | -0.0045 | 0.0254 | -0.0046 | 0.0256 | 0.1189 | 0.1219 | 1.0000 | 1.0000 | 1.0127 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | groupdro | delta_area_dosis | 0.0143 | 0.0099 | 0.0186 | 0.0099 | 0.0186 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 0.9984 | rechaza H0 | rechaza H0 | no |
| B3a selección principal, comparación con ERM | F-C2-099 | lisa | delta_auc_rho099 | -0.0041 | -0.0185 | 0.0122 | -0.0185 | 0.0122 | 0.7031 | 0.7031 | 1.0000 | 1.0000 | 1.0001 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | lisa | delta_area_dosis | 0.0147 | 0.0093 | 0.0200 | 0.0093 | 0.0199 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 0.9946 | rechaza H0 | rechaza H0 | no |
| B3a selección principal, comparación con ERM | F-C2-099 | mixstyle | delta_auc_rho099 | 0.0001 | -0.0097 | 0.0115 | -0.0100 | 0.0119 | 0.5107 | 0.5112 | 1.0000 | 1.0000 | 1.0333 | no rechaza | no rechaza | no |
| B3a selección principal, comparación con ERM | F-C2-area | mixstyle | delta_area_dosis | 0.0023 | -0.0002 | 0.0049 | -0.0002 | 0.0050 | 0.0375 | 0.0405 | 0.2249 | 0.2429 | 1.0116 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | coral | delta_auc_rho099 | -0.0202 | -0.0296 | -0.0115 | -0.0296 | -0.0115 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0059 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | coral | delta_area_dosis | -0.0043 | -0.0064 | -0.0022 | -0.0064 | -0.0022 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0004 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | dann | delta_auc_rho099 | -0.0610 | -0.0836 | -0.0405 | -0.0938 | -0.0318 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.4397 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | dann | delta_area_dosis | 0.0040 | 0.0005 | 0.0073 | 0.0003 | 0.0076 | 0.0120 | 0.0165 | 0.0720 | 0.0990 | 1.0644 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | dfa_fuente | delta_auc_rho099 | -0.0407 | -0.0651 | -0.0178 | -0.0660 | -0.0170 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0368 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | dfa_fuente | delta_area_dosis | -0.0055 | -0.0081 | -0.0029 | -0.0082 | -0.0028 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0266 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | dfr | delta_auc_rho099 | 0.1530 | 0.1319 | 0.1757 | 0.1309 | 0.1764 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0386 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | dfr | delta_area_dosis | 0.0228 | 0.0185 | 0.0270 | 0.0184 | 0.0270 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0169 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | erm_cuotas | delta_auc_rho099 | 0.0517 | 0.0394 | 0.0639 | 0.0373 | 0.0657 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.1609 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | erm_cuotas | delta_area_dosis | 0.0064 | 0.0040 | 0.0089 | 0.0039 | 0.0090 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0302 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | erm_mascara | delta_auc_rho099 | -0.0194 | -0.0334 | -0.0058 | -0.0343 | -0.0051 | 0.9980 | 0.9980 | 1.0000 | 1.0000 | 1.0574 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | erm_mascara | delta_area_dosis | -0.0238 | -0.0306 | -0.0168 | -0.0309 | -0.0166 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0318 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | fourier | delta_auc_rho099 | -0.0007 | -0.0097 | 0.0087 | -0.0102 | 0.0091 | 0.5542 | 0.5517 | 1.0000 | 1.0000 | 1.0453 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | fourier | delta_area_dosis | -0.0083 | -0.0110 | -0.0057 | -0.0110 | -0.0057 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9976 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | groupdro | delta_auc_rho099 | 0.1071 | 0.0919 | 0.1237 | 0.0919 | 0.1237 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0001 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | groupdro | delta_area_dosis | 0.0243 | 0.0188 | 0.0292 | 0.0188 | 0.0293 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0154 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | lisa | delta_auc_rho099 | 0.0438 | 0.0240 | 0.0647 | 0.0240 | 0.0647 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0011 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | lisa | delta_area_dosis | 0.0260 | 0.0206 | 0.0312 | 0.0205 | 0.0313 | 1/2001 (suelo) | 1/2001 (suelo) | 10/2001 (suelo de Holm) | 10/2001 (suelo de Holm) | 1.0172 | rechaza H0 | rechaza H0 | no |
| B3a selección de oráculo, comparación con ERM | F-C2-099 | mixstyle | delta_auc_rho099 | -0.0096 | -0.0178 | -0.0020 | -0.0178 | -0.0020 | 0.9950 | 0.9950 | 1.0000 | 1.0000 | 1.0024 | no rechaza | no rechaza | no |
| B3a selección de oráculo, comparación con ERM | F-C2-area | mixstyle | delta_area_dosis | -0.0003 | -0.0031 | 0.0020 | -0.0031 | 0.0020 | 0.5857 | 0.5862 | 1.0000 | 1.0000 | 1.0030 | no rechaza | no rechaza | no |
| c3-epoca (media de los once métodos) | F-C3 | media de once | delta_sel_rho090 | 0.0408 | 0.0337 | 0.0482 | 0.0335 | 0.0483 | 1/2001 (suelo) | 1/2001 (suelo) | 3/2001 (suelo de Holm) | 3/2001 (suelo de Holm) | 1.0153 | rechaza H0 | rechaza H0 | no |
| c3-epoca (media de los once métodos) | F-C3 | media de once | delta_sel_rho095 | 0.0600 | 0.0518 | 0.0688 | 0.0517 | 0.0689 | 1/2001 (suelo) | 1/2001 (suelo) | 3/2001 (suelo de Holm) | 3/2001 (suelo de Holm) | 1.0166 | rechaza H0 | rechaza H0 | no |
| c3-epoca (media de los once métodos) | F-C3 | media de once | delta_sel_rho099 | 0.0366 | 0.0305 | 0.0423 | 0.0302 | 0.0424 | 1/2001 (suelo) | 1/2001 (suelo) | 3/2001 (suelo de Holm) | 3/2001 (suelo de Holm) | 1.0336 | rechaza H0 | rechaza H0 | no |

## Lecturas descriptivas del IC95 que cambian (sin prueba de hipótesis): 1

| tabla | metodo | medida | estimacion | ic95_inf_antes | ic95_sup_antes | ic95_inf_despues | ic95_sup_despues | ic_antes | ic_despues |
|---|---|---|---|---|---|---|---|---|---|
| B3a selección principal, comparación con ERM | erm_mascara | delta_auc_rho095 | -0.0224 | -0.0437 | -0.0000 | -0.0438 | 0.0000 | por debajo | contiene el cero |

## Resumen de la razón de anchuras (después / antes) por tabla

| tabla | count | min | 50% | max |
|---|---|---|---|---|
| AUC por fuente | 64.000 | 0.998 | 1.005 | 1.671 |
| B3a curva, selección oráculo | 33.000 | 0.999 | 1.006 | 1.282 |
| B3a curva, selección principal | 33.000 | 1.000 | 1.008 | 1.353 |
| B3a medidas resumen, selección oráculo | 22.000 | 0.992 | 1.003 | 1.282 |
| B3a medidas resumen, selección principal | 22.000 | 0.996 | 1.002 | 1.353 |
| B3a selección de oráculo, comparación con ERM | 40.000 | 0.998 | 1.023 | 1.440 |
| B3a selección principal, comparación con ERM | 40.000 | 0.995 | 1.012 | 1.485 |
| c3-epoca (media de los once métodos) | 3.000 | 1.015 | 1.017 | 1.034 |
| c3_epoca_por_metodo | 33.000 | 1.001 | 1.023 | 1.249 |
| concordancia etiqueta, columnas del análisis principal | 6.000 | 1.000 | 1.005 | 1.193 |
| concordancia tronco, columnas del análisis principal | 6.000 | 1.000 | 1.005 | 1.193 |
| orientacion | 1.000 | 1.002 | 1.002 | 1.002 |

