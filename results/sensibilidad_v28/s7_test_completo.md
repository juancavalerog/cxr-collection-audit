# s7_test_completo

Generado por `python scripts/sensibilidad/v28.py s7` (2026-11-20 16:47:40 UTC). B = 2000, semilla 2026. Código posterior a la congelación para un análisis registrado antes de ella (docs/16_sensibilidad_registrada.md). Congelación: db617b620c2b9a6c7ef588898982006f0d4731457bd63ae0f6d367c10f6e24dc.

- Test limpio completo: 8 479 imágenes, la parte bimcv (8 079) más 200 positivas de ricord y 200 negativas de rsna. Advertencia registrada (docs/08, versión 1, Section 1): las fuentes de una sola clase permiten acertar reconociendo la fuente, de modo que estas cifras no miden mejor la enfermedad que las de bimcv.
- En las filas de detectores, ic95_inf e ic95_sup son el mínimo y el máximo de los extremos entre troncos cuando hay varios; el IC de cada tronco está en s7_detectores.

## s7_auc

| metodo | auc_test_completo | ic95_inf | ic95_sup | auc_test_bimcv | completo_menos_bimcv | n_troncos |
|---|---|---|---|---|---|---|
| erm | 0.812 | 0.799 | 0.823 | 0.809 | 0.003 |  |
| control_fuente | 0.755 | 0.740 | 0.768 | 0.758 | -0.004 |  |
| groupdro | 0.776 | 0.762 | 0.789 | 0.773 | 0.003 |  |
| dann | 0.787 | 0.776 | 0.799 | 0.782 | 0.005 |  |
| coral | 0.789 | 0.776 | 0.802 | 0.784 | 0.006 |  |
| mixstyle | 0.788 | 0.774 | 0.799 | 0.781 | 0.006 |  |
| fourier | 0.799 | 0.785 | 0.812 | 0.793 | 0.006 |  |
| lisa | 0.791 | 0.779 | 0.802 | 0.779 | 0.012 |  |
| dfa_fuente | 0.791 | 0.777 | 0.804 | 0.786 | 0.006 |  |
| dfr | 0.851 | 0.840 | 0.861 | 0.850 | 0.001 |  |
| erm_mascara | 0.782 | 0.769 | 0.794 | 0.779 | 0.003 |  |
| erm_cuotas | 0.799 | 0.785 | 0.811 | 0.789 | 0.010 |  |
| D_IN | 0.818 | 0.805 | 0.831 | 0.821 | -0.002 | 1.000 |
| D_ERM | 0.844 | 0.817 | 0.865 | 0.853 | -0.009 | 5.000 |
| D_MASK | 0.824 | 0.805 | 0.840 | 0.829 | -0.005 | 5.000 |
| D_RES | 0.844 | 0.830 | 0.859 | 0.850 | -0.006 | 4.000 |


## s7_familias

| referencia | variante | metodo | n_semillas | semillas | auc_referencia | auc_metodo | delta | ic95_inf | ic95_sup | p_superioridad | comparable | n | p_ni_0.02 | p_ni_0.01 | p_holm_0.02 | no_inferior_0.02 | p_holm_0.01 | no_inferior_0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_IN | test_completo | erm | 5 | 42,123,2024,7,999 | 0.818 | 0.812 | 0.007 | -0.006 | 0.022 | 0.147 | sí | 8479 | 0.000 | 0.011 | 0.006 | sí | 0.022 | sí |
| D_IN | test_completo | control_fuente | 5 | 42,123,2024,7,999 | 0.818 | 0.755 | 0.064 | 0.048 | 0.081 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | groupdro | 5 | 42,123,2024,7,999 | 0.818 | 0.776 | 0.042 | 0.027 | 0.059 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | dann | 5 | 42,123,2024,7,999 | 0.818 | 0.787 | 0.031 | 0.017 | 0.046 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | coral | 5 | 42,123,2024,7,999 | 0.818 | 0.789 | 0.029 | 0.014 | 0.045 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | mixstyle | 5 | 42,123,2024,7,999 | 0.818 | 0.788 | 0.031 | 0.016 | 0.046 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | fourier | 5 | 42,123,2024,7,999 | 0.818 | 0.799 | 0.019 | 0.004 | 0.035 | 0.006 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | lisa | 5 | 42,123,2024,7,999 | 0.818 | 0.791 | 0.027 | 0.014 | 0.042 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | dfa_fuente | 5 | 42,123,2024,7,999 | 0.818 | 0.791 | 0.027 | 0.012 | 0.044 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | dfr | 5 | 42,123,2024,7,999 | 0.818 | 0.851 | -0.032 | -0.044 | -0.020 | 1.000 | sí | 8479 | 0.980 | 1.000 | 0.980 | no | 1.000 | no |
| D_IN | test_completo | erm_mascara | 5 | 42,123,2024,7,999 | 0.818 | 0.782 | 0.036 | 0.022 | 0.052 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | test_completo | erm_cuotas | 5 | 42,123,2024,7,999 | 0.818 | 0.799 | 0.020 | 0.005 | 0.035 | 0.005 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | erm | 5 | 42,123,2024,7,999 | 0.844 | 0.812 | 0.032 | 0.021 | 0.045 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | control_fuente | 5 | 42,123,2024,7,999 | 0.844 | 0.755 | 0.090 | 0.075 | 0.105 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | groupdro | 5 | 42,123,2024,7,999 | 0.844 | 0.776 | 0.068 | 0.055 | 0.083 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | dann | 5 | 42,123,2024,7,999 | 0.844 | 0.787 | 0.057 | 0.044 | 0.070 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | coral | 5 | 42,123,2024,7,999 | 0.844 | 0.789 | 0.055 | 0.041 | 0.069 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | mixstyle | 5 | 42,123,2024,7,999 | 0.844 | 0.788 | 0.057 | 0.043 | 0.071 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | fourier | 5 | 42,123,2024,7,999 | 0.844 | 0.799 | 0.045 | 0.031 | 0.060 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | lisa | 5 | 42,123,2024,7,999 | 0.844 | 0.791 | 0.053 | 0.041 | 0.066 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | dfa_fuente | 5 | 42,123,2024,7,999 | 0.844 | 0.791 | 0.053 | 0.039 | 0.067 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | dfr | 5 | 42,123,2024,7,999 | 0.844 | 0.851 | -0.007 | -0.016 | 0.003 | 0.900 | sí | 8479 | 0.002 | 0.232 | 0.006 | sí | 0.232 | no |
| D_ERM | test_completo | erm_mascara | 5 | 42,123,2024,7,999 | 0.844 | 0.782 | 0.062 | 0.049 | 0.076 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | test_completo | erm_cuotas | 5 | 42,123,2024,7,999 | 0.844 | 0.799 | 0.046 | 0.032 | 0.059 | 0.000 | sí | 8479 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |


## s7_detectores

| variante | detector | tag | C | caracteristicas | tta | conjunto | unidad | auc | ic95_inf | ic95_sup | n | n_neg | n_unidades | B_validos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | D_IN | din_imagenet | 0.050 | pooled | espejo | test | paciente | 0.818 | 0.805 | 0.831 | 8479 | 4239 | 4721 | 2000 |
| base | D_ERM | baseline_erm_final_seed42_best | 0.050 | pooled | espejo | test | paciente | 0.847 | 0.834 | 0.859 | 8479 | 4239 | 4721 | 2000 |
| base | D_ERM | baseline_erm_final_seed123_best | 0.050 | pooled | espejo | test | paciente | 0.831 | 0.817 | 0.844 | 8479 | 4239 | 4721 | 2000 |
| base | D_ERM | baseline_erm_final_seed2024_best | 0.050 | pooled | espejo | test | paciente | 0.853 | 0.840 | 0.865 | 8479 | 4239 | 4721 | 2000 |
| base | D_ERM | baseline_erm_final_seed7_best | 0.050 | pooled | espejo | test | paciente | 0.850 | 0.836 | 0.863 | 8479 | 4239 | 4721 | 2000 |
| base | D_ERM | baseline_erm_final_seed999_best | 0.050 | pooled | espejo | test | paciente | 0.840 | 0.828 | 0.851 | 8479 | 4239 | 4721 | 2000 |
| base | D_MASK | baseline_erm_mascara_final_seed42_best | 0.050 | pooled | espejo | test | paciente | 0.826 | 0.814 | 0.838 | 8479 | 4239 | 4721 | 2000 |
| base | D_MASK | baseline_erm_mascara_final_seed123_best | 0.050 | pooled | espejo | test | paciente | 0.827 | 0.815 | 0.839 | 8479 | 4239 | 4721 | 2000 |
| base | D_MASK | baseline_erm_mascara_final_seed2024_best | 0.050 | pooled | espejo | test | paciente | 0.817 | 0.805 | 0.829 | 8479 | 4239 | 4721 | 2000 |
| base | D_MASK | baseline_erm_mascara_final_seed7_best | 0.050 | pooled | espejo | test | paciente | 0.823 | 0.811 | 0.836 | 8479 | 4239 | 4721 | 2000 |
| base | D_MASK | baseline_erm_mascara_final_seed999_best | 0.050 | pooled | espejo | test | paciente | 0.828 | 0.815 | 0.840 | 8479 | 4239 | 4721 | 2000 |
| base | D_RES | expl_erm_reserva_seed42_best | 0.050 | pooled | espejo | test | paciente | 0.846 | 0.834 | 0.857 | 8479 | 4239 | 4721 | 2000 |
| base | D_RES | expl_erm_reserva_seed123_best | 0.050 | pooled | espejo | test | paciente | 0.842 | 0.830 | 0.853 | 8479 | 4239 | 4721 | 2000 |
| base | D_RES | expl_erm_reserva20_seed42_best | 0.050 | pooled | espejo | test | paciente | 0.841 | 0.830 | 0.852 | 8479 | 4239 | 4721 | 2000 |
| base | D_RES | expl_erm_reserva20_seed123_best | 0.050 | pooled | espejo | test | paciente | 0.848 | 0.837 | 0.859 | 8479 | 4239 | 4721 | 2000 |


## s7_composicion

| fuente | n | positivas |
|---|---|---|
| bimcv | 8079 | 4040 |
| ricord | 200 | 200 |
| rsna | 200 | 0 |

