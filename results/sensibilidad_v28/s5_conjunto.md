# s5_conjunto

Generado por `python scripts/sensibilidad/v28.py s5` (2026-11-20 16:42:11 UTC). B = 2000, semilla 2026. Código posterior a la congelación para un análisis registrado antes de ella (docs/16_sensibilidad_registrada.md). Congelación: db617b620c2b9a6c7ef588898982006f0d4731457bd63ae0f6d367c10f6e24dc.

- Conjunto: media de las probabilidades de las cinco semillas (42, 123, 2024, 7, 999) del estado principal.
- La comparación con D_IN repite la prueba de F-C1 con el conjunto en lugar de la media por semilla; es una sensibilidad sin familia propia y no sustituye a la cifra principal (V2.2.1, V2.9 punto 5).
- No se forma un conjunto de D_ERM: sus puntuaciones son funciones de decisión de cinco detectores con escalas distintas y el registro no lo prevé.

## s5_auc_conjunto

| metodo | auc_individual_media | ic95_inf_individual | ic95_sup_individual | auc_conjunto | ic95_inf_conjunto | ic95_sup_conjunto | conjunto_menos_individual |
|---|---|---|---|---|---|---|---|
| erm | 0.809 | 0.795 | 0.821 | 0.826 | 0.812 | 0.839 | 0.017 |
| control_fuente | 0.758 | 0.744 | 0.772 | 0.776 | 0.760 | 0.792 | 0.018 |
| groupdro | 0.773 | 0.758 | 0.788 | 0.791 | 0.775 | 0.806 | 0.018 |
| dann | 0.782 | 0.770 | 0.794 | 0.796 | 0.783 | 0.809 | 0.014 |
| coral | 0.784 | 0.769 | 0.796 | 0.805 | 0.790 | 0.819 | 0.021 |
| mixstyle | 0.781 | 0.767 | 0.794 | 0.791 | 0.775 | 0.805 | 0.010 |
| fourier | 0.793 | 0.778 | 0.807 | 0.804 | 0.789 | 0.818 | 0.011 |
| lisa | 0.779 | 0.766 | 0.792 | 0.805 | 0.791 | 0.818 | 0.026 |
| dfa_fuente | 0.786 | 0.771 | 0.799 | 0.809 | 0.794 | 0.824 | 0.024 |
| dfr | 0.850 | 0.839 | 0.861 | 0.867 | 0.856 | 0.878 | 0.017 |
| erm_mascara | 0.779 | 0.765 | 0.793 | 0.803 | 0.788 | 0.818 | 0.024 |
| erm_cuotas | 0.789 | 0.774 | 0.802 | 0.813 | 0.799 | 0.827 | 0.025 |


## s5_d_in_frente_conjunto

| referencia | variante | metodo | n_semillas | semillas | auc_referencia | auc_metodo | delta | ic95_inf | ic95_sup | p_superioridad | comparable | n | p_ni_0.02 | p_ni_0.01 | p_holm_0.02 | no_inferior_0.02 | p_holm_0.01 | no_inferior_0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_IN | conjunto | erm | 1 | conjunto | 0.821 | 0.826 | -0.005 | -0.020 | 0.009 | 0.752 | sí | 8079 | 0.020 | 0.239 | 0.040 | no | 0.479 | no |
| D_IN | conjunto | control_fuente | 1 | conjunto | 0.821 | 0.776 | 0.044 | 0.027 | 0.062 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | groupdro | 1 | conjunto | 0.821 | 0.791 | 0.030 | 0.013 | 0.047 | 0.001 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | dann | 1 | conjunto | 0.821 | 0.796 | 0.024 | 0.009 | 0.040 | 0.001 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | coral | 1 | conjunto | 0.821 | 0.805 | 0.016 | -0.001 | 0.032 | 0.028 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | mixstyle | 1 | conjunto | 0.821 | 0.791 | 0.030 | 0.014 | 0.046 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | fourier | 1 | conjunto | 0.821 | 0.804 | 0.017 | 0.000 | 0.032 | 0.023 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | lisa | 1 | conjunto | 0.821 | 0.805 | 0.016 | 0.001 | 0.031 | 0.016 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | dfa_fuente | 1 | conjunto | 0.821 | 0.809 | 0.011 | -0.005 | 0.027 | 0.096 | sí | 8079 | 0.000 | 0.005 | 0.006 | sí | 0.020 | sí |
| D_IN | conjunto | dfr | 1 | conjunto | 0.821 | 0.867 | -0.046 | -0.058 | -0.035 | 1.000 | sí | 8079 | 1.000 | 1.000 | 1.000 | no | 1.000 | no |
| D_IN | conjunto | erm_mascara | 1 | conjunto | 0.821 | 0.803 | 0.017 | 0.002 | 0.033 | 0.009 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | conjunto | erm_cuotas | 1 | conjunto | 0.821 | 0.813 | 0.008 | -0.008 | 0.023 | 0.167 | sí | 8079 | 0.000 | 0.015 | 0.006 | sí | 0.045 | no |

