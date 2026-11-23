# s2_referencias

Generado por `python scripts/sensibilidad/v28.py s2` (2026-11-20 16:39:24 UTC). B = 2000, semilla 2026. Código posterior a la congelación para un análisis registrado antes de ella (docs/16_sensibilidad_registrada.md). Congelación: db617b620c2b9a6c7ef588898982006f0d4731457bd63ae0f6d367c10f6e24dc.

- Extracto sin cálculo nuevo de results/c1/test/detectores.csv y f_c1b.csv (python scripts/eval/auditoria_covidx.py --c1 test).

## s2_detectores_resumen

| variante | detector | n_troncos | auc_media | auc_min | auc_max | ic95_inf_min | ic95_sup_max |
|---|---|---|---|---|---|---|---|
| base | D_IN | 1 | 0.821 | 0.821 | 0.821 | 0.808 | 0.834 |
| base | D_ERM | 5 | 0.853 | 0.846 | 0.858 | 0.833 | 0.870 |
| base | D_MASK | 5 | 0.829 | 0.821 | 0.833 | 0.808 | 0.846 |
| base | D_RES | 4 | 0.850 | 0.847 | 0.853 | 0.835 | 0.864 |


## s2_f_c1b

| metodo | auc_referencia | auc_metodo | delta | ic95_inf | ic95_sup | p_holm_0.02 | no_inferior_0.02 | p_holm_0.01 | no_inferior_0.01 |
|---|---|---|---|---|---|---|---|---|---|
| erm | 0.853 | 0.809 | 0.044 | 0.031 | 0.057 | 0.006 | sí | 0.006 | sí |
| control_fuente | 0.853 | 0.758 | 0.095 | 0.080 | 0.110 | 0.006 | sí | 0.006 | sí |
| groupdro | 0.853 | 0.773 | 0.080 | 0.065 | 0.095 | 0.006 | sí | 0.006 | sí |
| dann | 0.853 | 0.782 | 0.071 | 0.057 | 0.084 | 0.006 | sí | 0.006 | sí |
| coral | 0.853 | 0.784 | 0.069 | 0.055 | 0.084 | 0.006 | sí | 0.006 | sí |
| mixstyle | 0.853 | 0.781 | 0.072 | 0.058 | 0.086 | 0.006 | sí | 0.006 | sí |
| fourier | 0.853 | 0.793 | 0.060 | 0.045 | 0.075 | 0.006 | sí | 0.006 | sí |
| lisa | 0.853 | 0.779 | 0.074 | 0.060 | 0.088 | 0.006 | sí | 0.006 | sí |
| dfa_fuente | 0.853 | 0.786 | 0.067 | 0.052 | 0.082 | 0.006 | sí | 0.006 | sí |
| dfr | 0.853 | 0.850 | 0.003 | -0.007 | 0.013 | 0.006 | sí | 0.007 | sí |
| erm_mascara | 0.853 | 0.779 | 0.074 | 0.059 | 0.089 | 0.006 | sí | 0.006 | sí |
| erm_cuotas | 0.853 | 0.789 | 0.064 | 0.049 | 0.079 | 0.006 | sí | 0.006 | sí |

