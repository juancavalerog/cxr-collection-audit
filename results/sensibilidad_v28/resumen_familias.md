# Resumen de las familias F-C1 y F-C1b

Generado por `python scripts/sensibilidad/v28.py resumen` a partir de las tablas ya escritas.

| analisis | referencia | variante | auc_referencia | no_inferior_0.02 | no_inferior_0.01 | n_comparaciones | ic_entero_sobre_cero | fallan_0.02 | fallan_0.01 | delta_erm | ic95_inf_erm | ic95_sup_erm | delta_dfr | ic95_inf_dfr | ic95_sup_dfr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| principal (results/c1/test/f_c1.csv) | D_IN | base | 0.821 | 11 | 11 | 12 | 10 | dfr | dfr | 0.012 | -0.002 | 0.026 | -0.029 | -0.041 | -0.018 |
| principal (results/c1/test/f_c1b.csv) | D_ERM | base | 0.853 | 12 | 12 | 12 | 11 |  |  | 0.044 | 0.031 | 0.057 | 0.003 | -0.007 | 0.013 |
| 1: detector | D_IN | C0.01 | 0.829 | 11 | 11 | 12 | 11 | dfr | dfr | 0.020 | 0.006 | 0.034 | -0.021 | -0.032 | -0.010 |
| 1: detector | D_IN | C0.3 | 0.807 | 11 | 10 | 12 | 9 | dfr | erm dfr | -0.002 | -0.016 | 0.013 | -0.043 | -0.055 | -0.031 |
| 1: detector | D_ERM | C0.01 | 0.854 | 12 | 12 | 12 | 11 |  |  | 0.046 | 0.033 | 0.059 | 0.004 | -0.006 | 0.014 |
| 1: detector | D_ERM | C0.3 | 0.847 | 12 | 11 | 12 | 11 |  | dfr | 0.038 | 0.025 | 0.051 | -0.003 | -0.013 | 0.006 |
| 1: detector | D_ERM | penult | 0.825 | 11 | 11 | 12 | 11 | dfr | dfr | 0.016 | 0.005 | 0.027 | -0.025 | -0.035 | -0.017 |
| 4: estados | D_IN | ema | 0.821 | 11 | 11 | 12 | 11 | dfr | dfr | 0.084 | 0.067 | 0.102 | -0.035 | -0.046 | -0.024 |
| 4: estados | D_ERM | ema | 0.853 | 12 | 11 | 12 | 11 |  | dfr | 0.117 | 0.100 | 0.133 | -0.003 | -0.012 | 0.006 |
| 4: estados | D_IN | last | 0.821 | 11 | 11 | 12 | 11 | dfr | dfr | 0.094 | 0.078 | 0.111 | -0.031 | -0.042 | -0.020 |
| 4: estados | D_ERM | last | 0.853 | 12 | 12 | 12 | 11 |  |  | 0.127 | 0.111 | 0.142 | 0.002 | -0.008 | 0.011 |
| 5: conjunto de semillas | D_IN | conjunto | 0.821 | 10 | 9 | 12 | 7 | erm dfr | erm dfr erm_cuotas | -0.005 | -0.020 | 0.009 | -0.046 | -0.058 | -0.035 |
| 6: unidad de remuestreo | D_IN | paciente | 0.821 | 11 | 11 | 12 | 10 | dfr | dfr | 0.012 | -0.002 | 0.026 | -0.029 | -0.041 | -0.018 |
| 6: unidad de remuestreo | D_ERM | paciente | 0.853 | 12 | 12 | 12 | 11 |  |  | 0.044 | 0.031 | 0.057 | 0.003 | -0.007 | 0.013 |
| 6: unidad de remuestreo | D_IN | imagen | 0.821 | 11 | 11 | 12 | 11 | dfr | dfr | 0.012 | 0.002 | 0.022 | -0.029 | -0.038 | -0.021 |
| 6: unidad de remuestreo | D_ERM | imagen | 0.853 | 12 | 12 | 12 | 11 |  |  | 0.044 | 0.036 | 0.053 | 0.003 | -0.003 | 0.009 |
| 7: test completo | D_IN | test_completo | 0.818 | 11 | 11 | 12 | 10 | dfr | dfr | 0.007 | -0.006 | 0.022 | -0.032 | -0.044 | -0.020 |
| 7: test completo | D_ERM | test_completo | 0.844 | 12 | 11 | 12 | 11 |  | dfr | 0.032 | 0.021 | 0.045 | -0.007 | -0.016 | 0.003 |
| E3: sin espejo | D_IN | sin_espejo | 0.819 | 11 | 11 | 12 | 10 | dfr | dfr | 0.011 | -0.003 | 0.025 | -0.030 | -0.042 | -0.018 |
| E3: sin espejo | D_ERM | sin_espejo | 0.850 | 12 | 12 | 12 | 11 |  |  | 0.043 | 0.029 | 0.056 | 0.001 | -0.009 | 0.011 |
