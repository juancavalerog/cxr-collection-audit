# Curva dosis-respuesta de b3a, orientación A, conjunto test_desconf, estado best

Generado por `python scripts/eval/informe_b3.py` (2026-11-20 03:32:01 UTC). B = 2000, semilla 2026; remuestreo jerárquico de semillas y pacientes (enmienda E1 de docs/08); cifra principal: media por semilla. 

## Curva (AUC media por semilla, IC95 jerárquico y de pacientes)

| metodo | rho | auc_media_semillas | dt_semillas | ic95_inf | ic95_sup | ic95_inf_pacientes | ic95_sup_pacientes | n_semillas | semillas |
|---|---|---|---|---|---|---|---|---|---|
| coral | 0.500 | 0.935 | 0.001 | 0.924 | 0.944 | 0.924 | 0.944 | 5 | 0 1 2 3 4 |
| coral | 0.600 | 0.934 | 0.003 | 0.923 | 0.944 | 0.924 | 0.944 | 5 | 0 1 2 3 4 |
| coral | 0.700 | 0.923 | 0.001 | 0.912 | 0.934 | 0.912 | 0.934 | 5 | 0 1 2 3 4 |
| coral | 0.800 | 0.905 | 0.005 | 0.892 | 0.917 | 0.892 | 0.917 | 5 | 0 1 2 3 4 |
| coral | 0.900 | 0.828 | 0.015 | 0.809 | 0.847 | 0.811 | 0.844 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| coral | 0.950 | 0.755 | 0.023 | 0.731 | 0.779 | 0.735 | 0.774 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| coral | 0.990 | 0.673 | 0.010 | 0.650 | 0.696 | 0.652 | 0.696 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| coral | 1.000 | 0.534 | 0.010 | 0.508 | 0.559 | 0.510 | 0.558 | 5 | 0 1 2 3 4 |
| dann | 0.500 | 0.932 | 0.003 | 0.921 | 0.942 | 0.921 | 0.942 | 5 | 0 1 2 3 4 |
| dann | 0.600 | 0.932 | 0.001 | 0.922 | 0.942 | 0.922 | 0.942 | 5 | 0 1 2 3 4 |
| dann | 0.700 | 0.916 | 0.002 | 0.904 | 0.928 | 0.904 | 0.927 | 5 | 0 1 2 3 4 |
| dann | 0.800 | 0.909 | 0.006 | 0.896 | 0.922 | 0.897 | 0.922 | 5 | 0 1 2 3 4 |
| dann | 0.900 | 0.853 | 0.021 | 0.833 | 0.872 | 0.837 | 0.868 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dann | 0.950 | 0.749 | 0.036 | 0.721 | 0.780 | 0.730 | 0.768 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dann | 0.990 | 0.622 | 0.050 | 0.594 | 0.651 | 0.599 | 0.645 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dann | 1.000 | 0.530 | 0.010 | 0.505 | 0.556 | 0.506 | 0.554 | 5 | 0 1 2 3 4 |
| dfa_fuente | 0.500 | 0.937 | 0.003 | 0.927 | 0.947 | 0.927 | 0.947 | 5 | 0 1 2 3 4 |
| dfa_fuente | 0.600 | 0.933 | 0.001 | 0.923 | 0.943 | 0.923 | 0.943 | 5 | 0 1 2 3 4 |
| dfa_fuente | 0.700 | 0.917 | 0.006 | 0.904 | 0.929 | 0.905 | 0.929 | 5 | 0 1 2 3 4 |
| dfa_fuente | 0.800 | 0.898 | 0.007 | 0.884 | 0.912 | 0.885 | 0.912 | 5 | 0 1 2 3 4 |
| dfa_fuente | 0.900 | 0.829 | 0.017 | 0.810 | 0.848 | 0.812 | 0.846 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dfa_fuente | 0.950 | 0.750 | 0.033 | 0.725 | 0.778 | 0.731 | 0.771 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dfa_fuente | 0.990 | 0.659 | 0.039 | 0.627 | 0.691 | 0.637 | 0.682 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dfa_fuente | 1.000 | 0.551 | 0.006 | 0.526 | 0.575 | 0.527 | 0.575 | 5 | 0 1 2 3 4 |
| dfr | 0.500 | 0.932 | 0.003 | 0.921 | 0.942 | 0.921 | 0.942 | 5 | 0 1 2 3 4 |
| dfr | 0.600 | 0.931 | 0.003 | 0.920 | 0.941 | 0.920 | 0.941 | 5 | 0 1 2 3 4 |
| dfr | 0.700 | 0.922 | 0.006 | 0.909 | 0.934 | 0.911 | 0.933 | 5 | 0 1 2 3 4 |
| dfr | 0.800 | 0.910 | 0.009 | 0.895 | 0.923 | 0.897 | 0.922 | 5 | 0 1 2 3 4 |
| dfr | 0.900 | 0.850 | 0.038 | 0.821 | 0.876 | 0.834 | 0.865 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dfr | 0.950 | 0.785 | 0.034 | 0.757 | 0.812 | 0.768 | 0.803 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| dfr | 0.990 | 0.736 | 0.062 | 0.696 | 0.778 | 0.723 | 0.750 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm | 0.500 | 0.934 | 0.001 | 0.924 | 0.944 | 0.924 | 0.944 | 5 | 0 1 2 3 4 |
| erm | 0.600 | 0.930 | 0.003 | 0.919 | 0.941 | 0.919 | 0.940 | 5 | 0 1 2 3 4 |
| erm | 0.700 | 0.917 | 0.004 | 0.905 | 0.928 | 0.905 | 0.928 | 5 | 0 1 2 3 4 |
| erm | 0.800 | 0.892 | 0.014 | 0.874 | 0.910 | 0.879 | 0.905 | 5 | 0 1 2 3 4 |
| erm | 0.900 | 0.834 | 0.013 | 0.816 | 0.853 | 0.818 | 0.851 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm | 0.950 | 0.768 | 0.022 | 0.746 | 0.792 | 0.750 | 0.788 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm | 0.990 | 0.688 | 0.015 | 0.665 | 0.711 | 0.667 | 0.710 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm | 1.000 | 0.534 | 0.011 | 0.508 | 0.560 | 0.509 | 0.558 | 5 | 0 1 2 3 4 |
| erm_cuotas | 0.500 | 0.933 | 0.003 | 0.923 | 0.943 | 0.923 | 0.943 | 5 | 0 1 2 3 4 |
| erm_cuotas | 0.600 | 0.931 | 0.003 | 0.920 | 0.941 | 0.920 | 0.941 | 5 | 0 1 2 3 4 |
| erm_cuotas | 0.700 | 0.920 | 0.005 | 0.907 | 0.931 | 0.908 | 0.931 | 5 | 0 1 2 3 4 |
| erm_cuotas | 0.800 | 0.901 | 0.010 | 0.886 | 0.916 | 0.888 | 0.914 | 5 | 0 1 2 3 4 |
| erm_cuotas | 0.900 | 0.856 | 0.018 | 0.838 | 0.875 | 0.841 | 0.871 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm_cuotas | 0.950 | 0.783 | 0.017 | 0.762 | 0.804 | 0.765 | 0.802 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm_cuotas | 0.990 | 0.687 | 0.015 | 0.665 | 0.710 | 0.666 | 0.709 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm_cuotas | 1.000 | 0.547 | 0.008 | 0.522 | 0.572 | 0.522 | 0.571 | 5 | 0 1 2 3 4 |
| erm_mascara | 0.500 | 0.915 | 0.003 | 0.903 | 0.926 | 0.903 | 0.926 | 5 | 0 1 2 3 4 |
| erm_mascara | 0.600 | 0.910 | 0.002 | 0.898 | 0.922 | 0.898 | 0.922 | 5 | 0 1 2 3 4 |
| erm_mascara | 0.700 | 0.894 | 0.006 | 0.880 | 0.908 | 0.881 | 0.908 | 5 | 0 1 2 3 4 |
| erm_mascara | 0.800 | 0.872 | 0.009 | 0.855 | 0.888 | 0.857 | 0.887 | 5 | 0 1 2 3 4 |
| erm_mascara | 0.900 | 0.807 | 0.030 | 0.782 | 0.832 | 0.790 | 0.825 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm_mascara | 0.950 | 0.746 | 0.022 | 0.722 | 0.769 | 0.726 | 0.766 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm_mascara | 0.990 | 0.673 | 0.014 | 0.649 | 0.698 | 0.651 | 0.695 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| erm_mascara | 1.000 | 0.594 | 0.017 | 0.566 | 0.623 | 0.569 | 0.618 | 5 | 0 1 2 3 4 |
| fourier | 0.500 | 0.935 | 0.004 | 0.924 | 0.945 | 0.924 | 0.944 | 5 | 0 1 2 3 4 |
| fourier | 0.600 | 0.929 | 0.005 | 0.917 | 0.940 | 0.918 | 0.939 | 5 | 0 1 2 3 4 |
| fourier | 0.700 | 0.915 | 0.003 | 0.902 | 0.927 | 0.902 | 0.926 | 5 | 0 1 2 3 4 |
| fourier | 0.800 | 0.889 | 0.012 | 0.872 | 0.906 | 0.875 | 0.902 | 5 | 0 1 2 3 4 |
| fourier | 0.900 | 0.817 | 0.012 | 0.798 | 0.836 | 0.799 | 0.834 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| fourier | 0.950 | 0.758 | 0.013 | 0.737 | 0.779 | 0.738 | 0.777 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| fourier | 0.990 | 0.691 | 0.015 | 0.668 | 0.714 | 0.670 | 0.713 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| fourier | 1.000 | 0.571 | 0.021 | 0.542 | 0.601 | 0.548 | 0.595 | 5 | 0 1 2 3 4 |
| groupdro | 0.500 | 0.932 | 0.001 | 0.922 | 0.942 | 0.922 | 0.942 | 5 | 0 1 2 3 4 |
| groupdro | 0.600 | 0.932 | 0.003 | 0.921 | 0.942 | 0.921 | 0.942 | 5 | 0 1 2 3 4 |
| groupdro | 0.700 | 0.925 | 0.003 | 0.914 | 0.936 | 0.914 | 0.936 | 5 | 0 1 2 3 4 |
| groupdro | 0.800 | 0.907 | 0.012 | 0.891 | 0.923 | 0.895 | 0.919 | 5 | 0 1 2 3 4 |
| groupdro | 0.900 | 0.862 | 0.015 | 0.846 | 0.878 | 0.847 | 0.876 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| groupdro | 0.950 | 0.820 | 0.021 | 0.799 | 0.840 | 0.803 | 0.836 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| groupdro | 0.990 | 0.697 | 0.013 | 0.674 | 0.721 | 0.676 | 0.719 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| groupdro | 1.000 | 0.532 | 0.005 | 0.507 | 0.557 | 0.508 | 0.556 | 5 | 0 1 2 3 4 |
| lisa | 0.500 | 0.936 | 0.002 | 0.925 | 0.945 | 0.925 | 0.945 | 5 | 0 1 2 3 4 |
| lisa | 0.600 | 0.936 | 0.003 | 0.926 | 0.946 | 0.926 | 0.946 | 5 | 0 1 2 3 4 |
| lisa | 0.700 | 0.929 | 0.005 | 0.918 | 0.940 | 0.918 | 0.939 | 5 | 0 1 2 3 4 |
| lisa | 0.800 | 0.908 | 0.004 | 0.895 | 0.919 | 0.895 | 0.919 | 5 | 0 1 2 3 4 |
| lisa | 0.900 | 0.866 | 0.028 | 0.843 | 0.887 | 0.851 | 0.880 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| lisa | 0.950 | 0.800 | 0.020 | 0.781 | 0.821 | 0.783 | 0.818 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| lisa | 0.990 | 0.684 | 0.014 | 0.661 | 0.708 | 0.662 | 0.706 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| lisa | 1.000 | 0.534 | 0.011 | 0.508 | 0.560 | 0.509 | 0.558 | 5 | 0 1 2 3 4 |
| mixstyle | 0.500 | 0.934 | 0.002 | 0.924 | 0.944 | 0.924 | 0.944 | 5 | 0 1 2 3 4 |
| mixstyle | 0.600 | 0.931 | 0.003 | 0.921 | 0.941 | 0.921 | 0.941 | 5 | 0 1 2 3 4 |
| mixstyle | 0.700 | 0.920 | 0.005 | 0.907 | 0.931 | 0.908 | 0.931 | 5 | 0 1 2 3 4 |
| mixstyle | 0.800 | 0.896 | 0.006 | 0.882 | 0.909 | 0.883 | 0.909 | 5 | 0 1 2 3 4 |
| mixstyle | 0.900 | 0.836 | 0.016 | 0.817 | 0.857 | 0.820 | 0.853 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| mixstyle | 0.950 | 0.772 | 0.014 | 0.752 | 0.792 | 0.753 | 0.790 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| mixstyle | 0.990 | 0.688 | 0.007 | 0.667 | 0.711 | 0.667 | 0.710 | 10 | 0 1 2 3 4 5 6 7 8 9 |
| mixstyle | 1.000 | 0.579 | 0.012 | 0.553 | 0.604 | 0.555 | 0.603 | 5 | 0 1 2 3 4 |

## Medidas resumen

| metodo | medida | estimacion | ic95_inf | ic95_sup | ic95_inf_pacientes | ic95_sup_pacientes |
|---|---|---|---|---|---|---|
| coral | auc_rho100 | 0.534 | 0.508 | 0.559 | 0.510 | 0.558 |
| coral | auc_rho099 | 0.673 | 0.650 | 0.696 | 0.652 | 0.696 |
| coral | auc_rho050 | 0.935 | 0.924 | 0.944 | 0.924 | 0.944 |
| coral | caida_050_100 | 0.401 | 0.374 | 0.428 | 0.374 | 0.427 |
| coral | area_dosis | 0.882 | 0.871 | 0.894 | 0.871 | 0.894 |
| dann | auc_rho100 | 0.530 | 0.505 | 0.556 | 0.506 | 0.554 |
| dann | auc_rho099 | 0.622 | 0.594 | 0.651 | 0.599 | 0.645 |
| dann | auc_rho050 | 0.932 | 0.921 | 0.942 | 0.921 | 0.942 |
| dann | caida_050_100 | 0.402 | 0.375 | 0.429 | 0.376 | 0.428 |
| dann | area_dosis | 0.883 | 0.871 | 0.895 | 0.871 | 0.895 |
| dfa_fuente | auc_rho100 | 0.551 | 0.526 | 0.575 | 0.527 | 0.575 |
| dfa_fuente | auc_rho099 | 0.659 | 0.627 | 0.691 | 0.637 | 0.682 |
| dfa_fuente | auc_rho050 | 0.937 | 0.927 | 0.947 | 0.927 | 0.947 |
| dfa_fuente | caida_050_100 | 0.387 | 0.361 | 0.414 | 0.361 | 0.413 |
| dfa_fuente | area_dosis | 0.879 | 0.867 | 0.892 | 0.867 | 0.891 |
| dfr | auc_rho099 | 0.736 | 0.696 | 0.778 | 0.723 | 0.750 |
| dfr | auc_rho050 | 0.932 | 0.921 | 0.942 | 0.921 | 0.942 |
| dfr | area_dosis | 0.891 | 0.878 | 0.903 | 0.880 | 0.903 |
| erm | auc_rho100 | 0.534 | 0.508 | 0.560 | 0.509 | 0.558 |
| erm | auc_rho099 | 0.688 | 0.665 | 0.711 | 0.667 | 0.710 |
| erm | auc_rho050 | 0.934 | 0.924 | 0.944 | 0.924 | 0.944 |
| erm | caida_050_100 | 0.401 | 0.372 | 0.427 | 0.374 | 0.426 |
| erm | area_dosis | 0.881 | 0.868 | 0.893 | 0.868 | 0.893 |
| erm_cuotas | auc_rho100 | 0.547 | 0.522 | 0.572 | 0.522 | 0.571 |
| erm_cuotas | auc_rho099 | 0.687 | 0.665 | 0.710 | 0.666 | 0.709 |
| erm_cuotas | auc_rho050 | 0.933 | 0.923 | 0.943 | 0.923 | 0.943 |
| erm_cuotas | caida_050_100 | 0.386 | 0.359 | 0.413 | 0.360 | 0.412 |
| erm_cuotas | area_dosis | 0.888 | 0.876 | 0.900 | 0.876 | 0.899 |
| erm_mascara | auc_rho100 | 0.594 | 0.566 | 0.623 | 0.569 | 0.618 |
| erm_mascara | auc_rho099 | 0.673 | 0.649 | 0.698 | 0.651 | 0.695 |
| erm_mascara | auc_rho050 | 0.915 | 0.903 | 0.926 | 0.903 | 0.926 |
| erm_mascara | caida_050_100 | 0.321 | 0.292 | 0.349 | 0.297 | 0.346 |
| erm_mascara | area_dosis | 0.859 | 0.845 | 0.873 | 0.846 | 0.873 |
| fourier | auc_rho100 | 0.571 | 0.542 | 0.601 | 0.548 | 0.595 |
| fourier | auc_rho099 | 0.691 | 0.668 | 0.714 | 0.670 | 0.713 |
| fourier | auc_rho050 | 0.935 | 0.924 | 0.945 | 0.924 | 0.944 |
| fourier | caida_050_100 | 0.363 | 0.331 | 0.393 | 0.338 | 0.387 |
| fourier | area_dosis | 0.876 | 0.863 | 0.889 | 0.863 | 0.888 |
| groupdro | auc_rho100 | 0.532 | 0.507 | 0.557 | 0.508 | 0.556 |
| groupdro | auc_rho099 | 0.697 | 0.674 | 0.721 | 0.676 | 0.719 |
| groupdro | auc_rho050 | 0.932 | 0.922 | 0.942 | 0.922 | 0.942 |
| groupdro | caida_050_100 | 0.400 | 0.373 | 0.426 | 0.374 | 0.426 |
| groupdro | area_dosis | 0.895 | 0.883 | 0.906 | 0.883 | 0.906 |
| lisa | auc_rho100 | 0.534 | 0.508 | 0.560 | 0.509 | 0.558 |
| lisa | auc_rho099 | 0.684 | 0.661 | 0.708 | 0.662 | 0.706 |
| lisa | auc_rho050 | 0.936 | 0.925 | 0.945 | 0.925 | 0.945 |
| lisa | caida_050_100 | 0.402 | 0.374 | 0.429 | 0.376 | 0.428 |
| lisa | area_dosis | 0.895 | 0.883 | 0.907 | 0.884 | 0.906 |
| mixstyle | auc_rho100 | 0.579 | 0.553 | 0.604 | 0.555 | 0.603 |
| mixstyle | auc_rho099 | 0.688 | 0.667 | 0.711 | 0.667 | 0.710 |
| mixstyle | auc_rho050 | 0.934 | 0.924 | 0.944 | 0.924 | 0.944 |
| mixstyle | caida_050_100 | 0.355 | 0.330 | 0.382 | 0.331 | 0.380 |
| mixstyle | area_dosis | 0.883 | 0.871 | 0.895 | 0.871 | 0.895 |

## Familias registradas frente a ERM (Holm al 0.025 unilateral; solo F-C2-area es confirmatoria)

| metodo | medida | familia | papel | delta | ic95_inf | ic95_sup | p_unilateral | p_holm | rechaza_holm | n_familia | n_semillas | semillas_a_favor | ic95_inf_pacientes | ic95_sup_pacientes | t_p_unilateral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coral | delta_auc_rho099 | F-C2-099 | secundaria | -0.015 | -0.024 | -0.004 | 0.994 | 1.000 | no | 10 | 10 | 1 | -0.018 | -0.012 | 0.984 |
| coral | delta_auc_rho100 | F-C2-100 | secundaria | 0.000 | -0.016 | 0.018 | 0.513 | 1.000 | no | 9 | 5 | 2 | -0.007 | 0.007 | 0.498 |
| coral | delta_area_dosis | F-C2-area | confirmatoria | 0.002 | -0.001 | 0.005 | 0.098 | 0.442 | no | 10 | 5 | 3 | 0.000 | 0.003 | 0.433 |
| coral | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.000 | -0.002 | 0.003 | 0.000 | 0.005 | sí | 10 | 5 | 3 | -0.002 | 0.003 | 0.000 |
| dann | delta_auc_rho099 | F-C2-099 | secundaria | -0.066 | -0.086 | -0.045 | 1.000 | 1.000 | no | 10 | 10 | 0 | -0.072 | -0.060 | 0.999 |
| dann | delta_auc_rho100 | F-C2-100 | secundaria | -0.004 | -0.022 | 0.014 | 0.650 | 1.000 | no | 9 | 5 | 2 | -0.010 | 0.002 | 0.657 |
| dann | delta_area_dosis | F-C2-area | confirmatoria | 0.002 | -0.001 | 0.005 | 0.088 | 0.442 | no | 10 | 5 | 3 | 0.000 | 0.004 | 0.100 |
| dann | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.002 | -0.004 | 0.001 | 0.000 | 0.005 | sí | 10 | 5 | 1 | -0.004 | 0.001 | 0.000 |
| dfa_fuente | delta_auc_rho099 | F-C2-099 | secundaria | -0.029 | -0.053 | -0.007 | 0.998 | 1.000 | no | 10 | 10 | 2 | -0.033 | -0.025 | 0.967 |
| dfa_fuente | delta_auc_rho100 | F-C2-100 | secundaria | 0.017 | 0.010 | 0.025 | 0.000 | 0.004 | sí | 9 | 5 | 5 | 0.012 | 0.023 | 0.003 |
| dfa_fuente | delta_area_dosis | F-C2-area | confirmatoria | -0.001 | -0.005 | 0.003 | 0.735 | 1.000 | no | 10 | 5 | 2 | -0.003 | 0.000 | 0.746 |
| dfa_fuente | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.003 | 0.000 | 0.007 | 0.000 | 0.005 | sí | 10 | 5 | 4 | 0.001 | 0.006 | 0.000 |
| dfr | delta_auc_rho099 | F-C2-099 | secundaria | 0.048 | 0.009 | 0.087 | 0.007 | 0.070 | no | 10 | 10 | 8 | 0.035 | 0.063 | 0.015 |
| dfr | delta_area_dosis | F-C2-area | confirmatoria | 0.010 | 0.005 | 0.016 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.008 | 0.012 | 0.014 |
| dfr | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.003 | -0.005 | 0.001 | 0.000 | 0.005 | sí | 10 | 5 | 1 | -0.005 | -0.000 | 0.004 |
| erm_cuotas | delta_auc_rho099 | F-C2-099 | secundaria | -0.001 | -0.013 | 0.014 | 0.571 | 1.000 | no | 10 | 10 | 4 | -0.004 | 0.002 | 0.542 |
| erm_cuotas | delta_auc_rho100 | F-C2-100 | secundaria | 0.014 | 0.000 | 0.030 | 0.025 | 0.125 | no | 9 | 5 | 4 | 0.009 | 0.018 | 0.072 |
| erm_cuotas | delta_area_dosis | F-C2-area | confirmatoria | 0.007 | 0.005 | 0.010 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.006 | 0.009 | 0.001 |
| erm_cuotas | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.001 | -0.004 | 0.003 | 0.000 | 0.005 | sí | 10 | 5 | 2 | -0.004 | 0.003 | 0.000 |
| erm_mascara | delta_auc_rho099 | F-C2-099 | secundaria | -0.015 | -0.027 | -0.003 | 0.993 | 1.000 | no | 10 | 10 | 2 | -0.020 | -0.010 | 0.985 |
| erm_mascara | delta_auc_rho100 | F-C2-100 | secundaria | 0.060 | 0.036 | 0.083 | 0.000 | 0.004 | sí | 9 | 5 | 5 | 0.049 | 0.072 | 0.003 |
| erm_mascara | delta_area_dosis | F-C2-area | confirmatoria | -0.022 | -0.029 | -0.014 | 1.000 | 1.000 | no | 10 | 5 | 0 | -0.028 | -0.015 | 1.000 |
| erm_mascara | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.020 | -0.026 | -0.013 | 0.996 | 0.996 | no | 10 | 5 | 0 | -0.026 | -0.013 | 0.999 |
| fourier | delta_auc_rho099 | F-C2-099 | secundaria | 0.003 | -0.009 | 0.018 | 0.378 | 1.000 | no | 10 | 10 | 5 | -0.000 | 0.006 | 0.360 |
| fourier | delta_auc_rho100 | F-C2-100 | secundaria | 0.038 | 0.014 | 0.064 | 0.001 | 0.009 | sí | 9 | 5 | 5 | 0.030 | 0.045 | 0.020 |
| fourier | delta_area_dosis | F-C2-area | confirmatoria | -0.005 | -0.008 | -0.002 | 1.000 | 1.000 | no | 10 | 5 | 0 | -0.007 | -0.003 | 0.997 |
| fourier | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.000 | -0.004 | 0.004 | 0.000 | 0.005 | sí | 10 | 5 | 3 | -0.002 | 0.002 | 0.003 |
| groupdro | delta_auc_rho099 | F-C2-099 | secundaria | 0.009 | -0.004 | 0.025 | 0.119 | 1.000 | no | 10 | 10 | 6 | 0.006 | 0.012 | 0.139 |
| groupdro | delta_auc_rho100 | F-C2-100 | secundaria | -0.001 | -0.007 | 0.005 | 0.665 | 1.000 | no | 9 | 5 | 2 | -0.004 | 0.001 | 0.646 |
| groupdro | delta_area_dosis | F-C2-area | confirmatoria | 0.014 | 0.010 | 0.019 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.012 | 0.017 | 0.002 |
| groupdro | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.002 | -0.004 | 0.000 | 0.000 | 0.005 | sí | 10 | 5 | 0 | -0.004 | 0.000 | 0.000 |
| lisa | delta_auc_rho099 | F-C2-099 | secundaria | -0.004 | -0.019 | 0.012 | 0.703 | 1.000 | no | 10 | 10 | 4 | -0.007 | -0.001 | 0.693 |
| lisa | delta_auc_rho100 | F-C2-100 | secundaria | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | no | 9 | 5 | 0 | 0.000 | 0.000 | 1.000 |
| lisa | delta_area_dosis | F-C2-area | confirmatoria | 0.015 | 0.009 | 0.020 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.011 | 0.018 | 0.002 |
| lisa | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.001 | -0.002 | 0.005 | 0.000 | 0.005 | sí | 10 | 5 | 4 | -0.002 | 0.004 | 0.000 |
| mixstyle | delta_auc_rho099 | F-C2-099 | secundaria | 0.000 | -0.010 | 0.012 | 0.511 | 1.000 | no | 10 | 10 | 4 | -0.002 | 0.002 | 0.496 |
| mixstyle | delta_auc_rho100 | F-C2-100 | secundaria | 0.045 | 0.030 | 0.064 | 0.000 | 0.004 | sí | 9 | 5 | 5 | 0.038 | 0.053 | 0.003 |
| mixstyle | delta_area_dosis | F-C2-area | confirmatoria | 0.002 | -0.000 | 0.005 | 0.037 | 0.225 | no | 10 | 5 | 3 | 0.001 | 0.003 | 0.150 |
| mixstyle | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.000 | -0.002 | 0.003 | 0.000 | 0.005 | sí | 10 | 5 | 1 | -0.002 | 0.002 | 0.001 |

## Todas las comparaciones con ERM

| metodo | medida | familia | papel | delta | ic95_inf | ic95_sup | p_unilateral | p_holm | rechaza_holm | n_familia | n_semillas | semillas_a_favor | ic95_inf_pacientes | ic95_sup_pacientes | t_p_unilateral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coral | delta_auc_rho050 |  | descriptiva | 0.000 | -0.002 | 0.003 | 0.343 |  | no | 0 | 5 | 3 | -0.002 | 0.003 | 0.319 |
| coral | delta_auc_rho060 |  | descriptiva | 0.004 | 0.000 | 0.008 | 0.022 |  | no | 0 | 5 | 4 | 0.002 | 0.007 | 0.049 |
| coral | delta_auc_rho070 |  | descriptiva | 0.006 | 0.003 | 0.009 | 0.000 |  | no | 0 | 5 | 5 | 0.003 | 0.009 | 0.004 |
| coral | delta_auc_rho080 |  | descriptiva | 0.012 | -0.001 | 0.028 | 0.033 |  | no | 0 | 5 | 4 | 0.009 | 0.016 | 0.078 |
| coral | delta_auc_rho090 |  | descriptiva | -0.006 | -0.013 | 0.001 | 0.944 |  | no | 0 | 10 | 2 | -0.010 | -0.003 | 0.923 |
| coral | delta_auc_rho095 |  | descriptiva | -0.013 | -0.025 | -0.001 | 0.983 |  | no | 0 | 10 | 3 | -0.017 | -0.010 | 0.970 |
| coral | delta_auc_rho099 | F-C2-099 | secundaria | -0.015 | -0.024 | -0.004 | 0.994 | 1.000 | no | 10 | 10 | 1 | -0.018 | -0.012 | 0.984 |
| coral | delta_auc_rho100 | F-C2-100 | secundaria | 0.000 | -0.016 | 0.018 | 0.513 | 1.000 | no | 9 | 5 | 2 | -0.007 | 0.007 | 0.498 |
| coral | delta_area_dosis | F-C2-area | confirmatoria | 0.002 | -0.001 | 0.005 | 0.098 | 0.442 | no | 10 | 5 | 3 | 0.000 | 0.003 | 0.433 |
| coral | delta_caida_050_100 |  | descriptiva | 0.000 | -0.017 | 0.017 | 0.529 |  | no | 0 | 5 | 2 | -0.007 | 0.008 | 0.519 |
| coral | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.000 | -0.002 | 0.003 | 0.000 | 0.005 | sí | 10 | 5 | 3 | -0.002 | 0.003 | 0.000 |
| dann | delta_auc_rho050 |  | descriptiva | -0.002 | -0.004 | 0.001 | 0.942 |  | no | 0 | 5 | 1 | -0.004 | 0.001 | 0.959 |
| dann | delta_auc_rho060 |  | descriptiva | 0.003 | -0.000 | 0.005 | 0.048 |  | no | 0 | 5 | 5 | -0.000 | 0.005 | 0.040 |
| dann | delta_auc_rho070 |  | descriptiva | -0.001 | -0.006 | 0.004 | 0.658 |  | no | 0 | 5 | 2 | -0.004 | 0.002 | 0.648 |
| dann | delta_auc_rho080 |  | descriptiva | 0.017 | 0.004 | 0.032 | 0.004 |  | no | 0 | 5 | 4 | 0.014 | 0.021 | 0.039 |
| dann | delta_auc_rho090 |  | descriptiva | 0.019 | 0.007 | 0.030 | 0.001 |  | no | 0 | 10 | 7 | 0.016 | 0.022 | 0.013 |
| dann | delta_auc_rho095 |  | descriptiva | -0.019 | -0.044 | 0.007 | 0.922 |  | no | 0 | 10 | 4 | -0.023 | -0.015 | 0.911 |
| dann | delta_auc_rho099 | F-C2-099 | secundaria | -0.066 | -0.086 | -0.045 | 1.000 | 1.000 | no | 10 | 10 | 0 | -0.072 | -0.060 | 0.999 |
| dann | delta_auc_rho100 | F-C2-100 | secundaria | -0.004 | -0.022 | 0.014 | 0.650 | 1.000 | no | 9 | 5 | 2 | -0.010 | 0.002 | 0.657 |
| dann | delta_area_dosis | F-C2-area | confirmatoria | 0.002 | -0.001 | 0.005 | 0.088 | 0.442 | no | 10 | 5 | 3 | 0.000 | 0.004 | 0.100 |
| dann | delta_caida_050_100 |  | descriptiva | 0.002 | -0.016 | 0.020 | 0.571 |  | no | 0 | 5 | 2 | -0.004 | 0.008 | 0.577 |
| dann | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.002 | -0.004 | 0.001 | 0.000 | 0.005 | sí | 10 | 5 | 1 | -0.004 | 0.001 | 0.000 |
| dfa_fuente | delta_auc_rho050 |  | descriptiva | 0.003 | 0.000 | 0.007 | 0.018 |  | no | 0 | 5 | 4 | 0.001 | 0.006 | 0.037 |
| dfa_fuente | delta_auc_rho060 |  | descriptiva | 0.003 | 0.000 | 0.007 | 0.023 |  | no | 0 | 5 | 5 | 0.001 | 0.006 | 0.051 |
| dfa_fuente | delta_auc_rho070 |  | descriptiva | -0.000 | -0.004 | 0.004 | 0.495 |  | no | 0 | 5 | 2 | -0.003 | 0.003 | 0.511 |
| dfa_fuente | delta_auc_rho080 |  | descriptiva | 0.006 | -0.007 | 0.018 | 0.176 |  | no | 0 | 5 | 3 | 0.003 | 0.009 | 0.200 |
| dfa_fuente | delta_auc_rho090 |  | descriptiva | -0.005 | -0.017 | 0.005 | 0.806 |  | no | 0 | 10 | 4 | -0.009 | -0.001 | 0.786 |
| dfa_fuente | delta_auc_rho095 |  | descriptiva | -0.018 | -0.039 | 0.006 | 0.923 |  | no | 0 | 10 | 3 | -0.021 | -0.014 | 0.887 |
| dfa_fuente | delta_auc_rho099 | F-C2-099 | secundaria | -0.029 | -0.053 | -0.007 | 0.998 | 1.000 | no | 10 | 10 | 2 | -0.033 | -0.025 | 0.967 |
| dfa_fuente | delta_auc_rho100 | F-C2-100 | secundaria | 0.017 | 0.010 | 0.025 | 0.000 | 0.004 | sí | 9 | 5 | 5 | 0.012 | 0.023 | 0.003 |
| dfa_fuente | delta_area_dosis | F-C2-area | confirmatoria | -0.001 | -0.005 | 0.003 | 0.735 | 1.000 | no | 10 | 5 | 2 | -0.003 | 0.000 | 0.746 |
| dfa_fuente | delta_caida_050_100 |  | descriptiva | -0.014 | -0.021 | -0.007 | 0.000 |  | no | 0 | 5 | 5 | -0.020 | -0.007 | 0.002 |
| dfa_fuente | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.003 | 0.000 | 0.007 | 0.000 | 0.005 | sí | 10 | 5 | 4 | 0.001 | 0.006 | 0.000 |
| dfr | delta_auc_rho050 |  | descriptiva | -0.003 | -0.005 | 0.001 | 0.945 |  | no | 0 | 5 | 1 | -0.005 | -0.000 | 0.919 |
| dfr | delta_auc_rho060 |  | descriptiva | 0.001 | -0.001 | 0.003 | 0.222 |  | no | 0 | 5 | 3 | -0.001 | 0.003 | 0.213 |
| dfr | delta_auc_rho070 |  | descriptiva | 0.005 | -0.000 | 0.010 | 0.029 |  | no | 0 | 5 | 4 | 0.002 | 0.008 | 0.045 |
| dfr | delta_auc_rho080 |  | descriptiva | 0.017 | 0.004 | 0.036 | 0.001 |  | no | 0 | 5 | 5 | 0.012 | 0.022 | 0.054 |
| dfr | delta_auc_rho090 |  | descriptiva | 0.016 | -0.010 | 0.039 | 0.127 |  | no | 0 | 10 | 6 | 0.013 | 0.018 | 0.125 |
| dfr | delta_auc_rho095 |  | descriptiva | 0.017 | -0.006 | 0.039 | 0.078 |  | no | 0 | 10 | 7 | 0.012 | 0.021 | 0.093 |
| dfr | delta_auc_rho099 | F-C2-099 | secundaria | 0.048 | 0.009 | 0.087 | 0.007 | 0.070 | no | 10 | 10 | 8 | 0.035 | 0.063 | 0.015 |
| dfr | delta_area_dosis | F-C2-area | confirmatoria | 0.010 | 0.005 | 0.016 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.008 | 0.012 | 0.014 |
| dfr | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.003 | -0.005 | 0.001 | 0.000 | 0.005 | sí | 10 | 5 | 1 | -0.005 | -0.000 | 0.004 |
| erm_cuotas | delta_auc_rho050 |  | descriptiva | -0.001 | -0.004 | 0.003 | 0.688 |  | no | 0 | 5 | 2 | -0.004 | 0.003 | 0.827 |
| erm_cuotas | delta_auc_rho060 |  | descriptiva | 0.001 | -0.002 | 0.004 | 0.249 |  | no | 0 | 5 | 4 | -0.002 | 0.004 | 0.203 |
| erm_cuotas | delta_auc_rho070 |  | descriptiva | 0.003 | -0.003 | 0.009 | 0.213 |  | no | 0 | 5 | 3 | -0.001 | 0.006 | 0.197 |
| erm_cuotas | delta_auc_rho080 |  | descriptiva | 0.009 | -0.008 | 0.028 | 0.160 |  | no | 0 | 5 | 3 | 0.006 | 0.012 | 0.191 |
| erm_cuotas | delta_auc_rho090 |  | descriptiva | 0.022 | 0.012 | 0.032 | 0.000 |  | no | 0 | 10 | 9 | 0.018 | 0.026 | 0.001 |
| erm_cuotas | delta_auc_rho095 |  | descriptiva | 0.015 | -0.001 | 0.031 | 0.032 |  | no | 0 | 10 | 7 | 0.011 | 0.019 | 0.051 |
| erm_cuotas | delta_auc_rho099 | F-C2-099 | secundaria | -0.001 | -0.013 | 0.014 | 0.571 | 1.000 | no | 10 | 10 | 4 | -0.004 | 0.002 | 0.542 |
| erm_cuotas | delta_auc_rho100 | F-C2-100 | secundaria | 0.014 | 0.000 | 0.030 | 0.025 | 0.125 | no | 9 | 5 | 4 | 0.009 | 0.018 | 0.072 |
| erm_cuotas | delta_area_dosis | F-C2-area | confirmatoria | 0.007 | 0.005 | 0.010 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.006 | 0.009 | 0.001 |
| erm_cuotas | delta_caida_050_100 |  | descriptiva | -0.015 | -0.031 | -0.001 | 0.020 |  | no | 0 | 5 | 4 | -0.020 | -0.009 | 0.077 |
| erm_cuotas | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.001 | -0.004 | 0.003 | 0.000 | 0.005 | sí | 10 | 5 | 2 | -0.004 | 0.003 | 0.000 |
| erm_mascara | delta_auc_rho050 |  | descriptiva | -0.020 | -0.026 | -0.013 | 1.000 |  | no | 0 | 5 | 0 | -0.026 | -0.013 | 1.000 |
| erm_mascara | delta_auc_rho060 |  | descriptiva | -0.020 | -0.027 | -0.012 | 1.000 |  | no | 0 | 5 | 0 | -0.027 | -0.013 | 1.000 |
| erm_mascara | delta_auc_rho070 |  | descriptiva | -0.023 | -0.031 | -0.014 | 1.000 |  | no | 0 | 5 | 0 | -0.030 | -0.015 | 1.000 |
| erm_mascara | delta_auc_rho080 |  | descriptiva | -0.021 | -0.036 | -0.004 | 0.994 |  | no | 0 | 5 | 1 | -0.027 | -0.013 | 0.976 |
| erm_mascara | delta_auc_rho090 |  | descriptiva | -0.027 | -0.051 | -0.002 | 0.984 |  | no | 0 | 10 | 3 | -0.034 | -0.019 | 0.972 |
| erm_mascara | delta_auc_rho095 |  | descriptiva | -0.022 | -0.044 | -0.000 | 0.975 |  | no | 0 | 10 | 1 | -0.029 | -0.015 | 0.965 |
| erm_mascara | delta_auc_rho099 | F-C2-099 | secundaria | -0.015 | -0.027 | -0.003 | 0.993 | 1.000 | no | 10 | 10 | 2 | -0.020 | -0.010 | 0.985 |
| erm_mascara | delta_auc_rho100 | F-C2-100 | secundaria | 0.060 | 0.036 | 0.083 | 0.000 | 0.004 | sí | 9 | 5 | 5 | 0.049 | 0.072 | 0.003 |
| erm_mascara | delta_area_dosis | F-C2-area | confirmatoria | -0.022 | -0.029 | -0.014 | 1.000 | 1.000 | no | 10 | 5 | 0 | -0.028 | -0.015 | 1.000 |
| erm_mascara | delta_caida_050_100 |  | descriptiva | -0.080 | -0.103 | -0.054 | 0.000 |  | no | 0 | 5 | 5 | -0.092 | -0.068 | 0.001 |
| erm_mascara | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.020 | -0.026 | -0.013 | 0.996 | 0.996 | no | 10 | 5 | 0 | -0.026 | -0.013 | 0.999 |
| fourier | delta_auc_rho050 |  | descriptiva | 0.000 | -0.004 | 0.004 | 0.397 |  | no | 0 | 5 | 3 | -0.002 | 0.002 | 0.426 |
| fourier | delta_auc_rho060 |  | descriptiva | -0.001 | -0.007 | 0.006 | 0.622 |  | no | 0 | 5 | 2 | -0.003 | 0.001 | 0.612 |
| fourier | delta_auc_rho070 |  | descriptiva | -0.002 | -0.007 | 0.002 | 0.835 |  | no | 0 | 5 | 2 | -0.005 | 0.000 | 0.819 |
| fourier | delta_auc_rho080 |  | descriptiva | -0.004 | -0.022 | 0.014 | 0.641 |  | no | 0 | 5 | 3 | -0.008 | 0.001 | 0.651 |
| fourier | delta_auc_rho090 |  | descriptiva | -0.017 | -0.025 | -0.011 | 1.000 |  | no | 0 | 10 | 1 | -0.021 | -0.014 | 1.000 |
| fourier | delta_auc_rho095 |  | descriptiva | -0.011 | -0.027 | 0.005 | 0.893 |  | no | 0 | 10 | 4 | -0.014 | -0.008 | 0.885 |
| fourier | delta_auc_rho099 | F-C2-099 | secundaria | 0.003 | -0.009 | 0.018 | 0.378 | 1.000 | no | 10 | 10 | 5 | -0.000 | 0.006 | 0.360 |
| fourier | delta_auc_rho100 | F-C2-100 | secundaria | 0.038 | 0.014 | 0.064 | 0.001 | 0.009 | sí | 9 | 5 | 5 | 0.030 | 0.045 | 0.020 |
| fourier | delta_area_dosis | F-C2-area | confirmatoria | -0.005 | -0.008 | -0.002 | 1.000 | 1.000 | no | 10 | 5 | 0 | -0.007 | -0.003 | 0.997 |
| fourier | delta_caida_050_100 |  | descriptiva | -0.037 | -0.064 | -0.012 | 0.002 |  | no | 0 | 5 | 5 | -0.045 | -0.030 | 0.023 |
| fourier | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.000 | -0.004 | 0.004 | 0.000 | 0.005 | sí | 10 | 5 | 3 | -0.002 | 0.002 | 0.003 |
| groupdro | delta_auc_rho050 |  | descriptiva | -0.002 | -0.004 | 0.000 | 0.962 |  | no | 0 | 5 | 0 | -0.004 | 0.000 | 0.999 |
| groupdro | delta_auc_rho060 |  | descriptiva | 0.002 | -0.002 | 0.005 | 0.164 |  | no | 0 | 5 | 4 | -0.000 | 0.004 | 0.176 |
| groupdro | delta_auc_rho070 |  | descriptiva | 0.008 | 0.003 | 0.013 | 0.001 |  | no | 0 | 5 | 4 | 0.005 | 0.011 | 0.013 |
| groupdro | delta_auc_rho080 |  | descriptiva | 0.015 | -0.006 | 0.038 | 0.097 |  | no | 0 | 5 | 3 | 0.010 | 0.020 | 0.125 |
| groupdro | delta_auc_rho090 |  | descriptiva | 0.028 | 0.015 | 0.040 | 0.000 |  | no | 0 | 10 | 9 | 0.023 | 0.032 | 0.002 |
| groupdro | delta_auc_rho095 |  | descriptiva | 0.052 | 0.039 | 0.063 | 0.000 |  | no | 0 | 10 | 10 | 0.045 | 0.058 | 0.000 |
| groupdro | delta_auc_rho099 | F-C2-099 | secundaria | 0.009 | -0.004 | 0.025 | 0.119 | 1.000 | no | 10 | 10 | 6 | 0.006 | 0.012 | 0.139 |
| groupdro | delta_auc_rho100 | F-C2-100 | secundaria | -0.001 | -0.007 | 0.005 | 0.665 | 1.000 | no | 9 | 5 | 2 | -0.004 | 0.001 | 0.646 |
| groupdro | delta_area_dosis | F-C2-area | confirmatoria | 0.014 | 0.010 | 0.019 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.012 | 0.017 | 0.002 |
| groupdro | delta_caida_050_100 |  | descriptiva | -0.001 | -0.007 | 0.005 | 0.419 |  | no | 0 | 5 | 3 | -0.004 | 0.002 | 0.408 |
| groupdro | coste_rho050_no_inferioridad | F-C2-coste | secundaria | -0.002 | -0.004 | 0.000 | 0.000 | 0.005 | sí | 10 | 5 | 0 | -0.004 | 0.000 | 0.000 |
| lisa | delta_auc_rho050 |  | descriptiva | 0.001 | -0.002 | 0.005 | 0.170 |  | no | 0 | 5 | 4 | -0.002 | 0.004 | 0.130 |
| lisa | delta_auc_rho060 |  | descriptiva | 0.006 | 0.002 | 0.010 | 0.000 |  | no | 0 | 5 | 5 | 0.003 | 0.010 | 0.006 |
| lisa | delta_auc_rho070 |  | descriptiva | 0.012 | 0.008 | 0.017 | 0.000 |  | no | 0 | 5 | 5 | 0.008 | 0.017 | 0.001 |
| lisa | delta_auc_rho080 |  | descriptiva | 0.015 | 0.001 | 0.031 | 0.020 |  | no | 0 | 5 | 4 | 0.010 | 0.020 | 0.057 |
| lisa | delta_auc_rho090 |  | descriptiva | 0.032 | 0.012 | 0.049 | 0.001 |  | no | 0 | 10 | 9 | 0.026 | 0.038 | 0.004 |
| lisa | delta_auc_rho095 |  | descriptiva | 0.032 | 0.016 | 0.048 | 0.000 |  | no | 0 | 10 | 8 | 0.026 | 0.038 | 0.003 |
| lisa | delta_auc_rho099 | F-C2-099 | secundaria | -0.004 | -0.019 | 0.012 | 0.703 | 1.000 | no | 10 | 10 | 4 | -0.007 | -0.001 | 0.693 |
| lisa | delta_auc_rho100 | F-C2-100 | secundaria | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | no | 9 | 5 | 0 | 0.000 | 0.000 | 1.000 |
| lisa | delta_area_dosis | F-C2-area | confirmatoria | 0.015 | 0.009 | 0.020 | 0.000 | 0.005 | sí | 10 | 5 | 5 | 0.011 | 0.018 | 0.002 |
| lisa | delta_caida_050_100 |  | descriptiva | 0.001 | -0.002 | 0.005 | 0.831 |  | no | 0 | 5 | 1 | -0.002 | 0.004 | 0.870 |
| lisa | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.001 | -0.002 | 0.005 | 0.000 | 0.005 | sí | 10 | 5 | 4 | -0.002 | 0.004 | 0.000 |
| mixstyle | delta_auc_rho050 |  | descriptiva | 0.000 | -0.002 | 0.003 | 0.463 |  | no | 0 | 5 | 1 | -0.002 | 0.002 | 0.460 |
| mixstyle | delta_auc_rho060 |  | descriptiva | 0.002 | -0.002 | 0.006 | 0.273 |  | no | 0 | 5 | 4 | -0.000 | 0.003 | 0.258 |
| mixstyle | delta_auc_rho070 |  | descriptiva | 0.003 | -0.003 | 0.008 | 0.159 |  | no | 0 | 5 | 4 | -0.000 | 0.006 | 0.172 |
| mixstyle | delta_auc_rho080 |  | descriptiva | 0.003 | -0.007 | 0.015 | 0.253 |  | no | 0 | 5 | 3 | 0.000 | 0.007 | 0.282 |
| mixstyle | delta_auc_rho090 |  | descriptiva | 0.002 | -0.003 | 0.008 | 0.221 |  | no | 0 | 10 | 6 | 0.000 | 0.004 | 0.224 |
| mixstyle | delta_auc_rho095 |  | descriptiva | 0.003 | -0.015 | 0.023 | 0.362 |  | no | 0 | 10 | 6 | 0.001 | 0.006 | 0.369 |
| mixstyle | delta_auc_rho099 | F-C2-099 | secundaria | 0.000 | -0.010 | 0.012 | 0.511 | 1.000 | no | 10 | 10 | 4 | -0.002 | 0.002 | 0.496 |
| mixstyle | delta_auc_rho100 | F-C2-100 | secundaria | 0.045 | 0.030 | 0.064 | 0.000 | 0.004 | sí | 9 | 5 | 5 | 0.038 | 0.053 | 0.003 |
| mixstyle | delta_area_dosis | F-C2-area | confirmatoria | 0.002 | -0.000 | 0.005 | 0.037 | 0.225 | no | 10 | 5 | 3 | 0.001 | 0.003 | 0.150 |
| mixstyle | delta_caida_050_100 |  | descriptiva | -0.045 | -0.062 | -0.030 | 0.000 |  | no | 0 | 5 | 5 | -0.053 | -0.038 | 0.002 |
| mixstyle | coste_rho050_no_inferioridad | F-C2-coste | secundaria | 0.000 | -0.002 | 0.003 | 0.000 | 0.005 | sí | 10 | 5 | 1 | -0.002 | 0.002 | 0.001 |

## Ficheros ausentes

- /workspace/paper_new/results/train/b3a_A_dfr_rho100/seed0/predictions/dfr_best_test_desconf.csv
- /workspace/paper_new/results/train/b3a_A_dfr_rho100/seed1/predictions/dfr_best_test_desconf.csv
- /workspace/paper_new/results/train/b3a_A_dfr_rho100/seed2/predictions/dfr_best_test_desconf.csv
- /workspace/paper_new/results/train/b3a_A_dfr_rho100/seed3/predictions/dfr_best_test_desconf.csv
- /workspace/paper_new/results/train/b3a_A_dfr_rho100/seed4/predictions/dfr_best_test_desconf.csv
