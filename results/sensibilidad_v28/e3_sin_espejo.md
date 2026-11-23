# e3_sin_espejo

Generado por `python scripts/sensibilidad/v28.py e3` (2026-11-20 16:49:48 UTC). B = 2000, semilla 2026. Código posterior a la congelación para un análisis registrado antes de ella (docs/16_sensibilidad_registrada.md). Congelación: db617b620c2b9a6c7ef588898982006f0d4731457bd63ae0f6d367c10f6e24dc.

- Referencias: mismo ajuste del detector; la puntuación es la función de decisión de la imagen original (tta 'no'), frente a la media con su espejo de la cifra principal.

## e3_detectores

| variante | detector | tag | C | caracteristicas | tta | conjunto | unidad | auc | ic95_inf | ic95_sup | n | n_neg | n_unidades | B_validos | auc_AP | ic95_inf_AP | ic95_sup_AP | auc_PA | ic95_inf_PA | ic95_sup_PA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | D_IN | din_imagenet | 0.050 | pooled | espejo | test_bimcv | paciente | 0.821 | 0.808 | 0.834 | 8079 | 4039 | 4343 | 2000 | 0.821 | 0.801 | 0.840 | 0.820 | 0.802 | 0.838 |
| base | D_ERM | baseline_erm_final_seed42_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.855 | 0.842 | 0.866 | 8079 | 4039 | 4343 | 2000 | 0.864 | 0.847 | 0.881 | 0.848 | 0.831 | 0.866 |
| base | D_ERM | baseline_erm_final_seed123_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.846 | 0.833 | 0.859 | 8079 | 4039 | 4343 | 2000 | 0.857 | 0.838 | 0.876 | 0.843 | 0.825 | 0.861 |
| base | D_ERM | baseline_erm_final_seed2024_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.854 | 0.841 | 0.866 | 8079 | 4039 | 4343 | 2000 | 0.854 | 0.835 | 0.873 | 0.850 | 0.834 | 0.867 |
| base | D_ERM | baseline_erm_final_seed7_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.853 | 0.839 | 0.866 | 8079 | 4039 | 4343 | 2000 | 0.856 | 0.836 | 0.876 | 0.844 | 0.825 | 0.862 |
| base | D_ERM | baseline_erm_final_seed999_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.858 | 0.846 | 0.870 | 8079 | 4039 | 4343 | 2000 | 0.864 | 0.847 | 0.882 | 0.846 | 0.828 | 0.863 |
| base | D_MASK | baseline_erm_mascara_final_seed42_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.829 | 0.816 | 0.841 | 8079 | 4039 | 4343 | 2000 | 0.821 | 0.801 | 0.841 | 0.840 | 0.824 | 0.857 |
| base | D_MASK | baseline_erm_mascara_final_seed123_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.830 | 0.817 | 0.842 | 8079 | 4039 | 4343 | 2000 | 0.827 | 0.808 | 0.845 | 0.838 | 0.821 | 0.855 |
| base | D_MASK | baseline_erm_mascara_final_seed2024_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.821 | 0.808 | 0.834 | 8079 | 4039 | 4343 | 2000 | 0.813 | 0.793 | 0.833 | 0.832 | 0.814 | 0.850 |
| base | D_MASK | baseline_erm_mascara_final_seed7_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.831 | 0.817 | 0.844 | 8079 | 4039 | 4343 | 2000 | 0.827 | 0.808 | 0.846 | 0.840 | 0.822 | 0.858 |
| base | D_MASK | baseline_erm_mascara_final_seed999_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.833 | 0.820 | 0.846 | 8079 | 4039 | 4343 | 2000 | 0.828 | 0.808 | 0.847 | 0.846 | 0.829 | 0.862 |
| base | D_RES | expl_erm_reserva_seed42_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.853 | 0.841 | 0.864 | 8079 | 4039 | 4343 | 2000 | 0.864 | 0.847 | 0.880 | 0.843 | 0.826 | 0.860 |
| base | D_RES | expl_erm_reserva_seed123_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.847 | 0.835 | 0.859 | 8079 | 4039 | 4343 | 2000 | 0.859 | 0.841 | 0.876 | 0.838 | 0.819 | 0.856 |
| base | D_RES | expl_erm_reserva20_seed42_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.850 | 0.838 | 0.861 | 8079 | 4039 | 4343 | 2000 | 0.861 | 0.845 | 0.878 | 0.842 | 0.825 | 0.859 |
| base | D_RES | expl_erm_reserva20_seed123_best | 0.050 | pooled | espejo | test_bimcv | paciente | 0.852 | 0.840 | 0.863 | 8079 | 4039 | 4343 | 2000 | 0.858 | 0.841 | 0.875 | 0.845 | 0.828 | 0.862 |
| sin_espejo | D_IN | din_imagenet | 0.050 | pooled | no | test_bimcv | paciente | 0.819 | 0.806 | 0.833 | 8079 | 4039 | 4343 | 2000 | 0.821 | 0.801 | 0.839 | 0.818 | 0.800 | 0.836 |
| sin_espejo | D_ERM | baseline_erm_final_seed42_best | 0.050 | pooled | no | test_bimcv | paciente | 0.853 | 0.841 | 0.865 | 8079 | 4039 | 4343 | 2000 | 0.862 | 0.845 | 0.878 | 0.850 | 0.833 | 0.867 |
| sin_espejo | D_ERM | baseline_erm_final_seed123_best | 0.050 | pooled | no | test_bimcv | paciente | 0.841 | 0.828 | 0.854 | 8079 | 4039 | 4343 | 2000 | 0.852 | 0.833 | 0.871 | 0.839 | 0.821 | 0.858 |
| sin_espejo | D_ERM | baseline_erm_final_seed2024_best | 0.050 | pooled | no | test_bimcv | paciente | 0.851 | 0.839 | 0.864 | 8079 | 4039 | 4343 | 2000 | 0.850 | 0.831 | 0.870 | 0.848 | 0.831 | 0.865 |
| sin_espejo | D_ERM | baseline_erm_final_seed7_best | 0.050 | pooled | no | test_bimcv | paciente | 0.850 | 0.836 | 0.864 | 8079 | 4039 | 4343 | 2000 | 0.854 | 0.835 | 0.874 | 0.841 | 0.821 | 0.859 |
| sin_espejo | D_ERM | baseline_erm_final_seed999_best | 0.050 | pooled | no | test_bimcv | paciente | 0.856 | 0.844 | 0.868 | 8079 | 4039 | 4343 | 2000 | 0.862 | 0.845 | 0.880 | 0.844 | 0.826 | 0.861 |
| sin_espejo | D_MASK | baseline_erm_mascara_final_seed42_best | 0.050 | pooled | no | test_bimcv | paciente | 0.824 | 0.811 | 0.837 | 8079 | 4039 | 4343 | 2000 | 0.815 | 0.794 | 0.835 | 0.839 | 0.823 | 0.856 |
| sin_espejo | D_MASK | baseline_erm_mascara_final_seed123_best | 0.050 | pooled | no | test_bimcv | paciente | 0.825 | 0.812 | 0.837 | 8079 | 4039 | 4343 | 2000 | 0.820 | 0.801 | 0.838 | 0.837 | 0.820 | 0.854 |
| sin_espejo | D_MASK | baseline_erm_mascara_final_seed2024_best | 0.050 | pooled | no | test_bimcv | paciente | 0.817 | 0.804 | 0.829 | 8079 | 4039 | 4343 | 2000 | 0.808 | 0.789 | 0.828 | 0.830 | 0.813 | 0.848 |
| sin_espejo | D_MASK | baseline_erm_mascara_final_seed7_best | 0.050 | pooled | no | test_bimcv | paciente | 0.826 | 0.812 | 0.838 | 8079 | 4039 | 4343 | 2000 | 0.822 | 0.802 | 0.841 | 0.836 | 0.818 | 0.853 |
| sin_espejo | D_MASK | baseline_erm_mascara_final_seed999_best | 0.050 | pooled | no | test_bimcv | paciente | 0.829 | 0.815 | 0.841 | 8079 | 4039 | 4343 | 2000 | 0.820 | 0.799 | 0.840 | 0.844 | 0.827 | 0.860 |
| sin_espejo | D_RES | expl_erm_reserva_seed42_best | 0.050 | pooled | no | test_bimcv | paciente | 0.849 | 0.838 | 0.861 | 8079 | 4039 | 4343 | 2000 | 0.859 | 0.842 | 0.875 | 0.842 | 0.824 | 0.859 |
| sin_espejo | D_RES | expl_erm_reserva_seed123_best | 0.050 | pooled | no | test_bimcv | paciente | 0.844 | 0.832 | 0.856 | 8079 | 4039 | 4343 | 2000 | 0.854 | 0.837 | 0.872 | 0.836 | 0.817 | 0.854 |
| sin_espejo | D_RES | expl_erm_reserva20_seed42_best | 0.050 | pooled | no | test_bimcv | paciente | 0.847 | 0.836 | 0.859 | 8079 | 4039 | 4343 | 2000 | 0.859 | 0.842 | 0.876 | 0.842 | 0.824 | 0.858 |
| sin_espejo | D_RES | expl_erm_reserva20_seed123_best | 0.050 | pooled | no | test_bimcv | paciente | 0.850 | 0.838 | 0.862 | 8079 | 4039 | 4343 | 2000 | 0.856 | 0.839 | 0.873 | 0.843 | 0.825 | 0.859 |


## e3_detectores_resumen

| variante | detector | n_troncos | auc_media | auc_min | auc_max | ic95_inf_min | ic95_sup_max |
|---|---|---|---|---|---|---|---|
| base | D_IN | 1 | 0.821 | 0.821 | 0.821 | 0.808 | 0.834 |
| base | D_ERM | 5 | 0.853 | 0.846 | 0.858 | 0.833 | 0.870 |
| base | D_MASK | 5 | 0.829 | 0.821 | 0.833 | 0.808 | 0.846 |
| base | D_RES | 4 | 0.850 | 0.847 | 0.853 | 0.835 | 0.864 |
| sin_espejo | D_IN | 1 | 0.819 | 0.819 | 0.819 | 0.806 | 0.833 |
| sin_espejo | D_ERM | 5 | 0.850 | 0.841 | 0.856 | 0.828 | 0.868 |
| sin_espejo | D_MASK | 5 | 0.824 | 0.817 | 0.829 | 0.804 | 0.841 |
| sin_espejo | D_RES | 4 | 0.848 | 0.844 | 0.850 | 0.832 | 0.862 |


## e3_erm_desde_logit

| metodo | semilla | auc_prediccion_test_con_espejo | auc_logit_con_espejo | auc_logit_sin_espejo | max_abs_dp_reconstruccion | media_abs_dp_reconstruccion | sin_menos_con_logit |
|---|---|---|---|---|---|---|---|
| erm | 42 | 0.808 | 0.807 | 0.809 | 0.028 | 0.001 | 0.002 |
| erm | 123 | 0.809 | 0.809 | 0.804 | 0.016 | 0.001 | -0.005 |
| erm | 2024 | 0.813 | 0.813 | 0.811 | 0.010 | 0.001 | -0.002 |
| erm | 7 | 0.795 | 0.795 | 0.799 | 0.010 | 0.001 | 0.004 |
| erm | 999 | 0.819 | 0.819 | 0.816 | 0.043 | 0.001 | -0.003 |
| erm_mascara | 42 | 0.797 | 0.797 | 0.795 | 0.027 | 0.001 | -0.003 |
| erm_mascara | 123 | 0.775 | 0.774 | 0.771 | 0.025 | 0.001 | -0.003 |
| erm_mascara | 2024 | 0.760 | 0.759 | 0.757 | 0.024 | 0.001 | -0.002 |
| erm_mascara | 7 | 0.778 | 0.778 | 0.775 | 0.019 | 0.002 | -0.003 |
| erm_mascara | 999 | 0.787 | 0.786 | 0.787 | 0.017 | 0.001 | 0.001 |


## e3_auc_metodos

| metodo | auc_sin_espejo | ic95_inf | ic95_sup | auc_con_espejo | sin_menos_con |
|---|---|---|---|---|---|
| erm | 0.808 | 0.795 | 0.821 | 0.809 | -0.001 |
| control_fuente | 0.755 | 0.741 | 0.769 | 0.758 | -0.003 |
| groupdro | 0.773 | 0.758 | 0.787 | 0.773 | -0.000 |
| dann | 0.780 | 0.767 | 0.792 | 0.782 | -0.002 |
| coral | 0.782 | 0.768 | 0.795 | 0.784 | -0.002 |
| mixstyle | 0.778 | 0.764 | 0.791 | 0.781 | -0.003 |
| fourier | 0.791 | 0.776 | 0.805 | 0.793 | -0.002 |
| lisa | 0.776 | 0.763 | 0.789 | 0.779 | -0.003 |
| dfa_fuente | 0.785 | 0.770 | 0.798 | 0.786 | -0.001 |
| dfr | 0.849 | 0.838 | 0.860 | 0.850 | -0.001 |
| erm_mascara | 0.777 | 0.763 | 0.791 | 0.779 | -0.002 |
| erm_cuotas | 0.787 | 0.772 | 0.800 | 0.789 | -0.002 |


## e3_familias

| referencia | variante | metodo | n_semillas | semillas | auc_referencia | auc_metodo | delta | ic95_inf | ic95_sup | p_superioridad | comparable | n | p_ni_0.02 | p_ni_0.01 | p_holm_0.02 | no_inferior_0.02 | p_holm_0.01 | no_inferior_0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_IN | sin_espejo | erm | 5 | 42,123,2024,7,999 | 0.819 | 0.808 | 0.011 | -0.003 | 0.025 | 0.065 | sí | 8079 | 0.000 | 0.001 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | control_fuente | 5 | 42,123,2024,7,999 | 0.819 | 0.755 | 0.064 | 0.049 | 0.081 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | groupdro | 5 | 42,123,2024,7,999 | 0.819 | 0.773 | 0.046 | 0.031 | 0.062 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | dann | 5 | 42,123,2024,7,999 | 0.819 | 0.780 | 0.039 | 0.024 | 0.054 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | coral | 5 | 42,123,2024,7,999 | 0.819 | 0.782 | 0.037 | 0.022 | 0.054 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | mixstyle | 5 | 42,123,2024,7,999 | 0.819 | 0.778 | 0.041 | 0.027 | 0.056 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | fourier | 5 | 42,123,2024,7,999 | 0.819 | 0.791 | 0.028 | 0.012 | 0.043 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | lisa | 5 | 42,123,2024,7,999 | 0.819 | 0.776 | 0.043 | 0.029 | 0.057 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | dfa_fuente | 5 | 42,123,2024,7,999 | 0.819 | 0.785 | 0.034 | 0.018 | 0.050 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | dfr | 5 | 42,123,2024,7,999 | 0.819 | 0.849 | -0.030 | -0.042 | -0.018 | 1.000 | sí | 8079 | 0.955 | 1.000 | 0.955 | no | 1.000 | no |
| D_IN | sin_espejo | erm_mascara | 5 | 42,123,2024,7,999 | 0.819 | 0.777 | 0.042 | 0.027 | 0.057 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_IN | sin_espejo | erm_cuotas | 5 | 42,123,2024,7,999 | 0.819 | 0.787 | 0.033 | 0.017 | 0.048 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | erm | 5 | 42,123,2024,7,999 | 0.850 | 0.808 | 0.043 | 0.029 | 0.056 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | control_fuente | 5 | 42,123,2024,7,999 | 0.850 | 0.755 | 0.095 | 0.080 | 0.110 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | groupdro | 5 | 42,123,2024,7,999 | 0.850 | 0.773 | 0.078 | 0.063 | 0.092 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | dann | 5 | 42,123,2024,7,999 | 0.850 | 0.780 | 0.071 | 0.056 | 0.084 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | coral | 5 | 42,123,2024,7,999 | 0.850 | 0.782 | 0.069 | 0.053 | 0.084 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | mixstyle | 5 | 42,123,2024,7,999 | 0.850 | 0.778 | 0.073 | 0.058 | 0.087 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | fourier | 5 | 42,123,2024,7,999 | 0.850 | 0.791 | 0.059 | 0.044 | 0.074 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | lisa | 5 | 42,123,2024,7,999 | 0.850 | 0.776 | 0.074 | 0.061 | 0.088 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | dfa_fuente | 5 | 42,123,2024,7,999 | 0.850 | 0.785 | 0.066 | 0.051 | 0.081 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | dfr | 5 | 42,123,2024,7,999 | 0.850 | 0.849 | 0.001 | -0.009 | 0.011 | 0.378 | sí | 8079 | 0.000 | 0.013 | 0.006 | sí | 0.013 | sí |
| D_ERM | sin_espejo | erm_mascara | 5 | 42,123,2024,7,999 | 0.850 | 0.777 | 0.073 | 0.059 | 0.088 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |
| D_ERM | sin_espejo | erm_cuotas | 5 | 42,123,2024,7,999 | 0.850 | 0.787 | 0.064 | 0.048 | 0.078 | 0.000 | sí | 8079 | 0.000 | 0.000 | 0.006 | sí | 0.006 | sí |

