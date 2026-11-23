# Puerta 6: piloto de B3a

Veredicto: **informativo**. 

Criterio: caída de ERM >= 0.05 entre rho 0.5 y 1.0 en val_desconf, IC95 > 0 y positiva en cada semilla.

Caída media por semilla: 0.397 (IC95 de 0.356 a 0.440); por semilla: 0: 0.392, 1: 0.402.

## Descriptivo (no interviene en el veredicto)

| metodo | rho | auc_media_semillas | dt_semillas | ic95_inf | ic95_sup | n_semillas | semillas |
|---|---|---|---|---|---|---|---|
| dfr | 0.500 | 0.926 | 0.000 | 0.909 | 0.943 | 2 | 0 1 |
| dfr | 0.900 | 0.864 | 0.013 | 0.839 | 0.887 | 2 | 0 1 |
| erm | 0.500 | 0.926 | 0.000 | 0.907 | 0.943 | 2 | 0 1 |
| erm | 0.900 | 0.828 | 0.006 | 0.800 | 0.854 | 2 | 0 1 |
| erm | 1.000 | 0.529 | 0.007 | 0.489 | 0.567 | 2 | 0 1 |
| groupdro | 0.500 | 0.928 | 0.001 | 0.910 | 0.945 | 2 | 0 1 |
| groupdro | 0.900 | 0.842 | 0.019 | 0.815 | 0.867 | 2 | 0 1 |
| groupdro | 1.000 | 0.529 | 0.004 | 0.488 | 0.567 | 2 | 0 1 |

| metodo | medida | rho | delta | ic95_inf | ic95_sup | p_unilateral | semillas_a_favor | n_semillas | margen | no_inferior |
|---|---|---|---|---|---|---|---|---|---|---|
| dfr | delta_auc_rho050 | 0.500 | 0.001 | -0.003 | 0.005 | 0.360 | 2.000 | 2 |  |  |
| dfr | delta_auc_rho090 | 0.900 | 0.035 | 0.025 | 0.045 | 0.000 | 2.000 | 2 |  |  |
| dfr | coste_rho050_no_inferioridad | 0.500 | 0.001 | -0.003 | 0.005 | 0.000 | 2.000 | 2 | 0.010 | sí |
| groupdro | delta_auc_rho050 | 0.500 | 0.003 | -0.001 | 0.006 | 0.068 | 2.000 | 2 |  |  |
| groupdro | delta_auc_rho090 | 0.900 | 0.013 | 0.004 | 0.023 | 0.002 | 2.000 | 2 |  |  |
| groupdro | delta_auc_rho100 | 1.000 | -0.000 | -0.003 | 0.002 | 0.526 | 1.000 | 2 |  |  |
| groupdro | delta_area_dosis |  | 0.008 | 0.003 | 0.013 | 0.000 |  | 2 |  |  |
| groupdro | delta_caida_050_100 |  | 0.003 | -0.001 | 0.007 | 0.908 |  | 2 |  |  |
| groupdro | coste_rho050_no_inferioridad | 0.500 | 0.003 | -0.001 | 0.006 | 0.000 | 2.000 | 2 | 0.010 | sí |

