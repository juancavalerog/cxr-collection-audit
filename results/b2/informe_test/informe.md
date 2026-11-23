# C2-R, réplica en tuberculosis (B2)

Generado por `python scripts/eval/informe_b2.py` (2026-11-20 03:39:12 UTC). B = 2000, semilla 2026; remuestreo jerárquico de semillas e imágenes. Advertencia de potencia de V2.5: Montgomery tiene 58 positivas y 80 negativas. 

## AUC media por semilla

| conjunto | metodo | auc_media_semillas | dt_semillas | ic95_inf | ic95_sup | ic95_inf_pacientes | ic95_sup_pacientes | n_semillas | semillas |
|---|---|---|---|---|---|---|---|---|---|
| test_shenzhen | control_fuente | 0.906 | 0.012 | 0.883 | 0.928 | 0.884 | 0.925 | 5 | 0 1 2 3 4 |
| test_shenzhen | coral | 0.887 | 0.009 | 0.862 | 0.909 | 0.863 | 0.909 | 5 | 0 1 2 3 4 |
| test_shenzhen | dann | 0.924 | 0.013 | 0.901 | 0.944 | 0.904 | 0.941 | 5 | 0 1 2 3 4 |
| test_shenzhen | dfa_fuente | 0.915 | 0.010 | 0.893 | 0.934 | 0.895 | 0.933 | 5 | 0 1 2 3 4 |
| test_shenzhen | dfr | 0.907 | 0.021 | 0.880 | 0.934 | 0.886 | 0.926 | 5 | 0 1 2 3 4 |
| test_shenzhen | erm | 0.895 | 0.014 | 0.869 | 0.918 | 0.872 | 0.915 | 5 | 0 1 2 3 4 |
| test_shenzhen | erm_cuotas | 0.881 | 0.010 | 0.855 | 0.904 | 0.856 | 0.903 | 5 | 0 1 2 3 4 |
| test_shenzhen | erm_mascara | 0.848 | 0.048 | 0.801 | 0.895 | 0.823 | 0.872 | 5 | 0 1 2 3 4 |
| test_shenzhen | fourier | 0.873 | 0.022 | 0.841 | 0.903 | 0.848 | 0.897 | 5 | 0 1 2 3 4 |
| test_shenzhen | groupdro | 0.894 | 0.022 | 0.863 | 0.921 | 0.872 | 0.914 | 5 | 0 1 2 3 4 |
| test_shenzhen | lisa | 0.864 | 0.021 | 0.832 | 0.891 | 0.838 | 0.887 | 5 | 0 1 2 3 4 |
| test_shenzhen | mixstyle | 0.888 | 0.013 | 0.862 | 0.911 | 0.864 | 0.909 | 5 | 0 1 2 3 4 |
| test_montgomery | control_fuente | 0.789 | 0.028 | 0.709 | 0.865 | 0.713 | 0.861 | 5 | 0 1 2 3 4 |
| test_montgomery | coral | 0.822 | 0.033 | 0.747 | 0.887 | 0.748 | 0.883 | 5 | 0 1 2 3 4 |
| test_montgomery | dann | 0.762 | 0.042 | 0.675 | 0.843 | 0.683 | 0.840 | 5 | 0 1 2 3 4 |
| test_montgomery | dfa_fuente | 0.753 | 0.049 | 0.663 | 0.835 | 0.671 | 0.829 | 5 | 0 1 2 3 4 |
| test_montgomery | dfr | 0.827 | 0.042 | 0.751 | 0.894 | 0.758 | 0.888 | 5 | 0 1 2 3 4 |
| test_montgomery | erm | 0.785 | 0.021 | 0.702 | 0.859 | 0.702 | 0.859 | 5 | 0 1 2 3 4 |
| test_montgomery | erm_cuotas | 0.790 | 0.028 | 0.707 | 0.862 | 0.709 | 0.859 | 5 | 0 1 2 3 4 |
| test_montgomery | erm_mascara | 0.862 | 0.029 | 0.795 | 0.928 | 0.798 | 0.922 | 5 | 0 1 2 3 4 |
| test_montgomery | fourier | 0.754 | 0.035 | 0.666 | 0.838 | 0.672 | 0.831 | 5 | 0 1 2 3 4 |
| test_montgomery | groupdro | 0.774 | 0.033 | 0.687 | 0.853 | 0.690 | 0.848 | 5 | 0 1 2 3 4 |
| test_montgomery | lisa | 0.757 | 0.068 | 0.661 | 0.851 | 0.678 | 0.830 | 5 | 0 1 2 3 4 |
| test_montgomery | mixstyle | 0.783 | 0.044 | 0.703 | 0.859 | 0.707 | 0.849 | 5 | 0 1 2 3 4 |

## Diferencia frente a ERM (F-C2R por fuente, Holm al 0.025 unilateral)

| conjunto | metodo | medida | delta | ic95_inf | ic95_sup | p_unilateral | ic95_inf_pacientes | ic95_sup_pacientes | p_unilateral_pacientes | ic95_inf_doble | ic95_sup_doble | de_semillas_sigma_S | de_pacientes_V_P | semillas_a_favor | n_semillas | t_media | t_ic95_inf | t_ic95_sup | t_p_unilateral | t_gl | familia | papel | p_holm | rechaza_holm | n_familia | confirmatoria | coherente_con_c2_rho099 | coherente_con_c2_area |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_shenzhen | control_fuente | delta_auc_rho000 | 0.011 | -0.014 | 0.036 | 0.172 | -0.006 | 0.030 | 0.098 | -0.013 | 0.037 | 0.019 | 0.009 | 3 | 5 | 0.011 | -0.014 | 0.037 | 0.145 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | None | None |
| test_shenzhen | coral | delta_auc_rho000 | -0.008 | -0.023 | 0.008 | 0.829 | -0.020 | 0.004 | 0.900 | -0.025 | 0.009 | 0.012 | 0.006 | 3 | 5 | -0.008 | -0.025 | 0.009 | 0.865 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | sí | no |
| test_shenzhen | dann | delta_auc_rho000 | 0.029 | 0.017 | 0.042 | 0.000 | 0.020 | 0.039 | 0.000 | 0.015 | 0.044 | 0.009 | 0.005 | 5 | 5 | 0.029 | 0.015 | 0.043 | 0.003 | 4 | F-C2R-shenzhen | secundaria | 0.005 | sí | 11 | no | no | sí |
| test_shenzhen | dfa_fuente | delta_auc_rho000 | 0.020 | -0.000 | 0.039 | 0.029 | 0.010 | 0.030 | 0.000 | -0.001 | 0.041 | 0.020 | 0.005 | 4 | 5 | 0.020 | -0.007 | 0.047 | 0.054 | 4 | F-C2R-shenzhen | secundaria | 0.265 | no | 11 | no | no | no |
| test_shenzhen | dfr | delta_auc_rho000 | 0.012 | 0.001 | 0.024 | 0.012 | 0.005 | 0.020 | 0.000 | -0.002 | 0.026 | 0.009 | 0.004 | 4 | 5 | 0.012 | -0.003 | 0.028 | 0.043 | 4 | F-C2R-shenzhen | secundaria | 0.120 | no | 11 | no | sí | sí |
| test_shenzhen | erm_cuotas | delta_auc_rho000 | -0.014 | -0.029 | 0.002 | 0.962 | -0.025 | -0.003 | 0.994 | -0.030 | 0.003 | 0.012 | 0.006 | 1 | 5 | -0.014 | -0.032 | 0.004 | 0.952 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | sí | no |
| test_shenzhen | erm_mascara | delta_auc_rho000 | -0.047 | -0.085 | -0.007 | 0.993 | -0.071 | -0.021 | 1.000 | -0.089 | -0.010 | 0.036 | 0.013 | 1 | 5 | -0.047 | -0.093 | -0.001 | 0.976 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | sí | sí |
| test_shenzhen | fourier | delta_auc_rho000 | -0.021 | -0.045 | 0.001 | 0.971 | -0.039 | -0.005 | 0.994 | -0.045 | 0.001 | 0.018 | 0.009 | 1 | 5 | -0.021 | -0.045 | 0.003 | 0.966 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | no | sí |
| test_shenzhen | groupdro | delta_auc_rho000 | -0.001 | -0.015 | 0.013 | 0.537 | -0.007 | 0.005 | 0.625 | -0.017 | 0.015 | 0.015 | 0.003 | 2 | 5 | -0.001 | -0.022 | 0.020 | 0.552 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | no | no |
| test_shenzhen | lisa | delta_auc_rho000 | -0.031 | -0.056 | -0.009 | 0.999 | -0.043 | -0.019 | 1.000 | -0.057 | -0.009 | 0.024 | 0.006 | 0 | 5 | -0.031 | -0.062 | 0.000 | 0.975 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | sí | no |
| test_shenzhen | mixstyle | delta_auc_rho000 | -0.007 | -0.025 | 0.010 | 0.781 | -0.017 | 0.003 | 0.911 | -0.026 | 0.011 | 0.016 | 0.005 | 3 | 5 | -0.007 | -0.029 | 0.015 | 0.791 | 4 | F-C2R-shenzhen | secundaria | 1.000 | no | 11 | no | no | no |
| test_montgomery | control_fuente | delta_auc_rho000 | 0.004 | -0.052 | 0.061 | 0.449 | -0.050 | 0.060 | 0.443 | -0.060 | 0.065 | 0.014 | 0.028 | 3 | 5 | 0.004 | -0.032 | 0.039 | 0.392 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | None | None |
| test_montgomery | coral | delta_auc_rho000 | 0.037 | -0.015 | 0.089 | 0.081 | -0.001 | 0.077 | 0.031 | -0.016 | 0.094 | 0.042 | 0.020 | 4 | 5 | 0.037 | -0.023 | 0.096 | 0.080 | 4 | F-C2R-montgomery | secundaria | 0.733 | no | 11 | no | no | sí |
| test_montgomery | dann | delta_auc_rho000 | -0.023 | -0.070 | 0.024 | 0.827 | -0.061 | 0.018 | 0.862 | -0.086 | 0.030 | 0.030 | 0.021 | 1 | 5 | -0.023 | -0.076 | 0.030 | 0.850 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | sí | no |
| test_montgomery | dfa_fuente | delta_auc_rho000 | -0.032 | -0.087 | 0.022 | 0.871 | -0.063 | 0.001 | 0.969 | -0.098 | 0.031 | 0.051 | 0.017 | 2 | 5 | -0.032 | -0.107 | 0.043 | 0.846 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | sí | sí |
| test_montgomery | dfr | delta_auc_rho000 | 0.042 | -0.002 | 0.085 | 0.029 | 0.016 | 0.073 | 0.001 | -0.012 | 0.093 | 0.035 | 0.015 | 4 | 5 | 0.042 | -0.012 | 0.096 | 0.049 | 4 | F-C2R-montgomery | secundaria | 0.290 | no | 11 | no | sí | sí |
| test_montgomery | erm_cuotas | delta_auc_rho000 | 0.005 | -0.040 | 0.049 | 0.409 | -0.022 | 0.035 | 0.354 | -0.045 | 0.052 | 0.040 | 0.015 | 3 | 5 | 0.005 | -0.053 | 0.063 | 0.410 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | no | sí |
| test_montgomery | erm_mascara | delta_auc_rho000 | 0.077 | 0.029 | 0.131 | 0.000 | 0.029 | 0.131 | 0.000 | 0.020 | 0.137 | 0.014 | 0.025 | 5 | 5 | 0.077 | 0.040 | 0.113 | 0.002 | 4 | F-C2R-montgomery | secundaria | 0.005 | sí | 11 | no | no | no |
| test_montgomery | fourier | delta_auc_rho000 | -0.031 | -0.086 | 0.025 | 0.871 | -0.076 | 0.015 | 0.914 | -0.090 | 0.029 | 0.036 | 0.023 | 1 | 5 | -0.031 | -0.086 | 0.024 | 0.904 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | no | sí |
| test_montgomery | groupdro | delta_auc_rho000 | -0.011 | -0.047 | 0.029 | 0.713 | -0.033 | 0.012 | 0.824 | -0.056 | 0.034 | 0.037 | 0.012 | 2 | 5 | -0.011 | -0.066 | 0.044 | 0.698 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | no | no |
| test_montgomery | lisa | delta_auc_rho000 | -0.028 | -0.088 | 0.027 | 0.820 | -0.060 | 0.007 | 0.946 | -0.100 | 0.032 | 0.053 | 0.017 | 2 | 5 | -0.028 | -0.106 | 0.051 | 0.807 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | sí | no |
| test_montgomery | mixstyle | delta_auc_rho000 | -0.002 | -0.043 | 0.042 | 0.527 | -0.030 | 0.027 | 0.549 | -0.050 | 0.050 | 0.037 | 0.014 | 2 | 5 | -0.002 | -0.058 | 0.054 | 0.537 | 4 | F-C2R-montgomery | secundaria | 1.000 | no | 11 | no | no | no |

## Selección de época (C3 en B2, descriptivo)

| conjunto | metodo | delta_sel | ic95_inf | ic95_sup | semillas_a_favor_bestdc | n_semillas | t_media | t_ic95_inf | t_ic95_sup | t_p_unilateral | t_gl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_shenzhen | control_fuente | 0.003 | -0.006 | 0.012 | 3 | 5 | 0.003 | -0.007 | 0.013 | 0.223 | 4 |
| test_shenzhen | coral | 0.000 | 0.000 | 0.000 | 0 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 4 |
| test_shenzhen | dann | 0.000 | 0.000 | 0.000 | 0 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 4 |
| test_shenzhen | dfa_fuente | -0.000 | -0.002 | 0.002 | 1 | 5 | -0.000 | -0.002 | 0.001 | 0.657 | 4 |
| test_shenzhen | erm | 0.000 | 0.000 | 0.000 | 0 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 4 |
| test_shenzhen | erm_cuotas | -0.007 | -0.016 | 0.001 | 0 | 5 | -0.007 | -0.018 | 0.005 | 0.905 | 4 |
| test_shenzhen | erm_mascara | 0.004 | -0.002 | 0.012 | 1 | 5 | 0.004 | -0.007 | 0.014 | 0.187 | 4 |
| test_shenzhen | fourier | 0.009 | -0.003 | 0.028 | 1 | 5 | 0.009 | -0.015 | 0.033 | 0.187 | 4 |
| test_shenzhen | groupdro | -0.002 | -0.006 | 0.002 | 0 | 5 | -0.002 | -0.008 | 0.004 | 0.819 | 4 |
| test_shenzhen | lisa | 0.006 | -0.003 | 0.019 | 1 | 5 | 0.006 | -0.010 | 0.022 | 0.187 | 4 |
| test_shenzhen | mixstyle | 0.000 | -0.001 | 0.001 | 1 | 5 | 0.000 | -0.001 | 0.001 | 0.187 | 4 |
| test_montgomery | control_fuente | -0.028 | -0.050 | -0.007 | 0 | 5 | -0.028 | -0.056 | 0.000 | 0.975 | 4 |
| test_montgomery | coral | 0.000 | 0.000 | 0.000 | 0 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 4 |
| test_montgomery | dann | 0.000 | 0.000 | 0.000 | 0 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 4 |
| test_montgomery | dfa_fuente | -0.006 | -0.052 | 0.035 | 1 | 5 | -0.006 | -0.070 | 0.059 | 0.590 | 4 |
| test_montgomery | erm | 0.000 | 0.000 | 0.000 | 0 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 4 |
| test_montgomery | erm_cuotas | -0.002 | -0.013 | 0.009 | 1 | 5 | -0.002 | -0.018 | 0.014 | 0.632 | 4 |
| test_montgomery | erm_mascara | -0.008 | -0.026 | 0.005 | 0 | 5 | -0.008 | -0.030 | 0.014 | 0.813 | 4 |
| test_montgomery | fourier | 0.012 | -0.005 | 0.038 | 1 | 5 | 0.012 | -0.021 | 0.044 | 0.187 | 4 |
| test_montgomery | groupdro | -0.011 | -0.029 | 0.006 | 0 | 5 | -0.011 | -0.035 | 0.014 | 0.858 | 4 |
| test_montgomery | lisa | 0.009 | -0.006 | 0.027 | 1 | 5 | 0.009 | -0.015 | 0.032 | 0.187 | 4 |
| test_montgomery | mixstyle | -0.004 | -0.012 | 0.002 | 0 | 5 | -0.004 | -0.014 | 0.007 | 0.813 | 4 |

