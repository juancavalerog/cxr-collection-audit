# Auditoría de COVIDx CXR-4 en validación

Generado por `python scripts/eval/auditoria_covidx.py` (2026-11-14 03:29:49 UTC). B = 2000, semilla 2026, remuestreo por paciente. No se ha leído ningún conjunto test. Plan: docs/08_evaluacion.md, versión 2, Section V2.3. Cifras de validación con dos semillas: son el ensayo del análisis registrado, no su resultado.

## 1. Detector de colección sin información de clase

| tag | origen_ajuste | n_ajuste | n_antigua | n_nueva | auc_val_robust | ic95_inf | ic95_sup | auc_AP | auc_PA | auc_solo_proyeccion_AP | auc_cabeza_mismo_tronco | delta_vs_cabeza | delta_ic95_inf | delta_ic95_sup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_erm_seed123_best | muestra de positivos de bimcv de entrenamiento | 6000 | 768 | 5232 | 0.877 | 0.858 | 0.895 | 0.865 | 0.890 | 0.550 | 0.881 | -0.003 | -0.026 | 0.019 |
| baseline_erm_seed42_best | muestra de positivos de bimcv de entrenamiento | 6000 | 768 | 5232 | 0.892 | 0.875 | 0.908 | 0.900 | 0.886 | 0.550 | 0.872 | 0.020 | -0.005 | 0.048 |
| expl_erm_reserva20_seed123_best | positivos reservados (no vistos por el tronco) | 2601 | 1301 | 1300 | 0.902 | 0.885 | 0.917 | 0.887 | 0.903 | 0.550 | 0.818 | 0.083 | 0.055 | 0.111 |
| expl_erm_reserva20_seed42_best | positivos reservados (no vistos por el tronco) | 2601 | 1301 | 1300 | 0.904 | 0.888 | 0.920 | 0.907 | 0.890 | 0.550 | 0.837 | 0.067 | 0.037 | 0.101 |
| expl_erm_reserva_seed123_best | positivos reservados (no vistos por el tronco) | 2601 | 1301 | 1300 | 0.921 | 0.907 | 0.933 | 0.933 | 0.895 | 0.550 | 0.642 | 0.279 | 0.240 | 0.319 |
| expl_erm_reserva_seed42_best | positivos reservados (no vistos por el tronco) | 2601 | 1301 | 1300 | 0.907 | 0.892 | 0.922 | 0.904 | 0.902 | 0.550 | 0.643 | 0.264 | 0.223 | 0.305 |

Troncos con reserva frente a la cabeza de ERM de la misma semilla:

| tag | comparador | auc1 | auc2 | delta | ic95_inf | ic95_sup |
|---|---|---|---|---|---|---|
| expl_erm_reserva20_seed123_best | baseline_erm_seed123_best (cabeza ERM, sin TTA) | 0.902 | 0.881 | 0.021 | 0.001 | 0.043 |
| expl_erm_reserva20_seed42_best | baseline_erm_seed42_best (cabeza ERM, sin TTA) | 0.904 | 0.872 | 0.032 | 0.006 | 0.060 |
| expl_erm_reserva_seed123_best | baseline_erm_seed123_best (cabeza ERM, sin TTA) | 0.921 | 0.881 | 0.040 | 0.021 | 0.060 |
| expl_erm_reserva_seed42_best | baseline_erm_seed42_best (cabeza ERM, sin TTA) | 0.907 | 0.872 | 0.035 | 0.011 | 0.061 |

## 2. Reproducción de la Tabla E4 de docs/07_piloto.md (tolerancia 0.005)

| cifra | registrada | reproducida | diferencia | cumple |
|---|---|---|---|---|
| expl_erm_reserva_seed42_best, val_robust | 0.907 | 0.907 | 0.000 | sí |
| expl_erm_reserva_seed42_best, AP | 0.904 | 0.904 | -0.000 | sí |
| expl_erm_reserva_seed42_best, PA | 0.902 | 0.902 | -0.000 | sí |
| baseline_erm_seed42_best, val_robust | 0.892 | 0.892 | 0.000 | sí |
| baseline_erm_seed42_best, AP | 0.900 | 0.900 | 0.000 | sí |
| baseline_erm_seed42_best, PA | 0.886 | 0.886 | 0.000 | sí |
| baseline_erm_seed123_best, val_robust | 0.877 | 0.877 | 0.000 | sí |
| baseline_erm_seed123_best, AP | 0.865 | 0.865 | -0.000 | sí |
| baseline_erm_seed123_best, PA | 0.890 | 0.890 | -0.000 | sí |
| expl_erm_reserva_seed42_best, cv_neg_vs_antiguos | 0.930 | 0.930 | 0.000 | sí |
| expl_erm_reserva_seed42_best, cv_neg_vs_nuevos | 0.736 | 0.736 | 0.000 | sí |
| expl_erm_reserva_seed42_best, cv_antiguos_vs_nuevos | 0.929 | 0.929 | 0.000 | sí |
| expl_erm_reserva_seed42_best, transfer_antiguos_a_nuevos | 0.620 | 0.620 | 0.000 | sí |
| expl_erm_reserva_seed42_best, transfer_nuevos_a_antiguos | 0.629 | 0.629 | 0.000 | sí |
| expl_erm_reserva_seed42_best, lectura_nuevos_en_val_robust | 0.635 | 0.635 | 0.000 | sí |

Resultado: se reproducen todas las cifras.

## 3. Contrastes dentro del conjunto reservado y transferencia

| tag | cv_neg_vs_antiguos | cv_neg_vs_nuevos | cv_antiguos_vs_nuevos | transfer_antiguos_a_nuevos_ic95_inf | transfer_antiguos_a_nuevos_ic95_sup | transfer_nuevos_a_antiguos_ic95_inf | transfer_nuevos_a_antiguos_ic95_sup | lectura_nuevos_en_val_robust_ic95_inf | lectura_nuevos_en_val_robust_ic95_sup | transfer_antiguos_a_nuevos | transfer_nuevos_a_antiguos | lectura_nuevos_en_val_robust | lectura_antiguos_en_val_robust | n_res_neg | n_res_pos_antigua | n_res_pos_nueva | n_neg_ajuste | n_neg_retenidos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expl_erm_reserva20_seed123_best | 0.935 | 0.732 | 0.931 | 0.556 | 0.701 | 0.635 | 0.751 | 0.673 | 0.739 | 0.632 | 0.694 | 0.706 | 0.908 | 132 | 1301 | 1300 | 66 | 66 |
| expl_erm_reserva20_seed42_best | 0.925 | 0.736 | 0.932 | 0.588 | 0.731 | 0.710 | 0.814 | 0.741 | 0.799 | 0.662 | 0.763 | 0.771 | 0.918 | 132 | 1301 | 1300 | 66 | 66 |
| expl_erm_reserva_seed123_best | 0.937 | 0.751 | 0.929 | 0.597 | 0.671 | 0.684 | 0.755 | 0.687 | 0.755 | 0.634 | 0.719 | 0.722 | 0.934 | 647 | 1301 | 1300 | 308 | 339 |
| expl_erm_reserva_seed42_best | 0.930 | 0.736 | 0.929 | 0.585 | 0.657 | 0.594 | 0.665 | 0.600 | 0.671 | 0.620 | 0.629 | 0.635 | 0.919 | 647 | 1301 | 1300 | 308 | 339 |


## 4. Detector (tronco ERM de cada semilla) frente a los métodos con etiquetas

Media por semilla de AUC(detector) menos AUC(método), emparejada por semilla y por imágenes. No inferioridad con margen 0.02: H1 es delta > -0.02; Holm al 0.025 unilateral dentro de cada conjunto. DFR solo es comparable en val_robust_sel. El detector no usa TTA; los métodos sí.

| conjunto | metodo | auc_detector_media | auc_metodo_media | auc_metodo_s42 | auc_metodo_s123 | delta | ic95_inf | ic95_sup | p_no_inferioridad | no_inferior_ic | n | comparable | p_holm_no_inferioridad | no_inferior_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val_robust | erm | 0.885 | 0.882 | 0.879 | 0.885 | 0.003 | -0.018 | 0.025 | 0.014 | sí | 3407 | sí | 0.015 | sí |
| val_robust | control_fuente | 0.885 | 0.852 | 0.848 | 0.855 | 0.033 | 0.006 | 0.062 | 0.000 | sí | 3407 | sí | 0.005 | sí |
| val_robust | groupdro | 0.885 | 0.854 | 0.839 | 0.869 | 0.031 | 0.007 | 0.056 | 0.000 | sí | 3407 | sí | 0.005 | sí |
| val_robust | dann | 0.885 | 0.857 | 0.856 | 0.857 | 0.028 | 0.002 | 0.054 | 0.000 | sí | 3407 | sí | 0.005 | sí |
| val_robust | coral | 0.885 | 0.875 | 0.887 | 0.864 | 0.010 | -0.013 | 0.034 | 0.003 | sí | 3407 | sí | 0.014 | sí |
| val_robust | mixstyle | 0.885 | 0.859 | 0.854 | 0.863 | 0.026 | 0.002 | 0.051 | 0.000 | sí | 3407 | sí | 0.005 | sí |
| val_robust | fourier | 0.885 | 0.872 | 0.867 | 0.876 | 0.013 | -0.013 | 0.040 | 0.005 | sí | 3407 | sí | 0.015 | sí |
| val_robust | lisa | 0.885 | 0.840 | 0.841 | 0.839 | 0.044 | 0.016 | 0.075 | 0.000 | sí | 3407 | sí | 0.005 | sí |
| val_robust | dfa_fuente | 0.885 | 0.873 | 0.871 | 0.874 | 0.012 | -0.011 | 0.037 | 0.005 | sí | 3407 | sí | 0.015 | sí |
| val_robust | erm_mascara | 0.885 | 0.842 | 0.832 | 0.851 | 0.043 | 0.018 | 0.070 | 0.000 | sí | 3407 | sí | 0.005 | sí |
| val_robust | erm_cuotas | 0.885 | 0.849 | 0.858 | 0.840 | 0.035 | 0.011 | 0.061 | 0.000 | sí | 3407 | sí | 0.005 | sí |
| val_robust_sel | erm | 0.900 | 0.885 | 0.883 | 0.887 | 0.015 | -0.012 | 0.043 | 0.007 | sí | 1615 | sí | 0.021 | sí |
| val_robust_sel | control_fuente | 0.900 | 0.844 | 0.845 | 0.844 | 0.056 | 0.023 | 0.091 | 0.000 | sí | 1615 | sí | 0.006 | sí |
| val_robust_sel | groupdro | 0.900 | 0.848 | 0.829 | 0.866 | 0.052 | 0.019 | 0.087 | 0.000 | sí | 1615 | sí | 0.006 | sí |
| val_robust_sel | dann | 0.900 | 0.853 | 0.850 | 0.857 | 0.047 | 0.011 | 0.084 | 0.001 | sí | 1615 | sí | 0.006 | sí |
| val_robust_sel | coral | 0.900 | 0.880 | 0.891 | 0.869 | 0.020 | -0.011 | 0.051 | 0.007 | sí | 1615 | sí | 0.021 | sí |
| val_robust_sel | mixstyle | 0.900 | 0.854 | 0.857 | 0.851 | 0.046 | 0.012 | 0.080 | 0.000 | sí | 1615 | sí | 0.006 | sí |
| val_robust_sel | fourier | 0.900 | 0.873 | 0.868 | 0.878 | 0.027 | -0.008 | 0.061 | 0.001 | sí | 1615 | sí | 0.007 | sí |
| val_robust_sel | lisa | 0.900 | 0.839 | 0.840 | 0.838 | 0.061 | 0.023 | 0.102 | 0.000 | sí | 1615 | sí | 0.006 | sí |
| val_robust_sel | dfa_fuente | 0.900 | 0.877 | 0.878 | 0.876 | 0.023 | -0.009 | 0.054 | 0.004 | sí | 1615 | sí | 0.016 | sí |
| val_robust_sel | dfr | 0.900 | 0.924 | 0.922 | 0.927 | -0.024 | -0.046 | -0.003 | 0.657 | no | 1615 | sí | 0.657 | no |
| val_robust_sel | erm_mascara | 0.900 | 0.841 | 0.831 | 0.851 | 0.059 | 0.023 | 0.093 | 0.000 | sí | 1615 | sí | 0.006 | sí |
| val_robust_sel | erm_cuotas | 0.900 | 0.850 | 0.856 | 0.845 | 0.050 | 0.014 | 0.085 | 0.000 | sí | 1615 | sí | 0.006 | sí |


## 5. AUC por colección en la validación completa (media de las dos semillas)

Cada columna es una pareja negativos / positivos con al menos 20 imágenes por lado; los IC95 por semilla están en `por_coleccion_validacion.csv`. En DFR se excluye la mitad val_robust_fit de bimcv.

| metodo | actmed / bimcv_antigua | actmed / ricord | actmed / sirm | actmed / stonybrook | bimcv / bimcv_antigua | bimcv / ricord | bimcv / sirm | bimcv / stonybrook | cohen / bimcv_antigua | cohen / ricord | cohen / sirm | cohen / stonybrook | rsna / bimcv_antigua | rsna / ricord | rsna / sirm | rsna / stonybrook |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_fuente | 0.842 | 0.899 | 0.760 | 0.889 | 0.852 | 0.909 | 0.779 | 0.900 | 0.706 | 0.795 | 0.612 | 0.754 | 0.785 | 0.864 | 0.692 | 0.843 |
| coral | 0.855 | 0.866 | 0.611 | 0.945 | 0.875 | 0.874 | 0.678 | 0.948 | 0.875 | 0.880 | 0.711 | 0.933 | 0.994 | 0.993 | 0.971 | 0.996 |
| dann | 0.887 | 0.858 | 0.664 | 0.881 | 0.857 | 0.828 | 0.616 | 0.851 | 0.900 | 0.881 | 0.718 | 0.899 | 0.998 | 0.997 | 0.990 | 0.998 |
| detector | 0.782 | 0.377 | 0.291 | 0.296 | 0.885 | 0.609 | 0.478 | 0.478 | 0.865 | 0.599 | 0.486 | 0.484 | 0.918 | 0.721 | 0.597 | 0.594 |
| dfa_fuente | 0.898 | 0.939 | 0.760 | 0.979 | 0.873 | 0.914 | 0.749 | 0.975 | 0.859 | 0.905 | 0.746 | 0.972 | 0.990 | 0.991 | 0.962 | 0.997 |
| dfr | 0.890 | 0.871 | 0.658 | 0.761 | 0.924 | 0.902 | 0.729 | 0.825 | 0.902 | 0.881 | 0.710 | 0.791 | 0.982 | 0.968 | 0.897 | 0.969 |
| erm | 0.874 | 0.909 | 0.644 | 0.963 | 0.882 | 0.920 | 0.663 | 0.969 | 0.874 | 0.908 | 0.692 | 0.963 | 0.996 | 0.994 | 0.977 | 0.998 |
| erm_cuotas | 0.796 | 0.898 | 0.742 | 0.933 | 0.849 | 0.925 | 0.778 | 0.949 | 0.846 | 0.915 | 0.791 | 0.941 | 0.991 | 0.996 | 0.966 | 0.997 |
| erm_mascara | 0.780 | 0.747 | 0.515 | 0.902 | 0.842 | 0.800 | 0.585 | 0.919 | 0.820 | 0.789 | 0.600 | 0.915 | 0.992 | 0.991 | 0.945 | 0.990 |
| fourier | 0.913 | 0.936 | 0.661 | 0.971 | 0.872 | 0.908 | 0.630 | 0.964 | 0.878 | 0.902 | 0.696 | 0.961 | 0.998 | 0.995 | 0.985 | 0.998 |
| groupdro | 0.878 | 0.960 | 0.775 | 0.968 | 0.854 | 0.950 | 0.770 | 0.961 | 0.807 | 0.918 | 0.726 | 0.940 | 0.975 | 0.991 | 0.938 | 0.994 |
| lisa | 0.897 | 0.966 | 0.907 | 0.992 | 0.840 | 0.934 | 0.864 | 0.987 | 0.838 | 0.920 | 0.861 | 0.984 | 0.994 | 0.995 | 0.991 | 0.999 |
| mixstyle | 0.886 | 0.907 | 0.672 | 0.941 | 0.859 | 0.882 | 0.629 | 0.931 | 0.900 | 0.912 | 0.733 | 0.948 | 0.998 | 0.996 | 0.985 | 0.999 |

