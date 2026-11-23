# c1b_jerarquico

Generado por `python scripts/sensibilidad/intervalos.py c1b-jerarquico` (2026-11-20 18:22:46 UTC). B = 2000, semilla de pacientes 2026, semilla de semillas 20261117. Código posterior a la congelación para análisis registrados antes de ella (docs/17_intervalos_alternativos.md). Congelación: db617b620c2b9a6c7ef588898982006f0d4731457bd63ae0f6d367c10f6e24dc.

- Control: delta e IC95 de pacientes frente a results/c1/test/f_c1b.csv; diferencia máxima 9.71e-17.
- Las columnas li_jerarquico_sobre_menos_* son solo una lectura de sensibilidad: la prueba registrada es la de f_c1b.csv (no inferioridad con Holm sobre doce comparaciones).

## c1b_jerarquico

| metodo | n_semillas | semillas | auc_referencia | auc_metodo | delta | ic95_inf_jerarquico | ic95_sup_jerarquico | ic95_inf_pacientes | ic95_sup_pacientes | de_semillas_sigma_S | de_pacientes_V_P | de_entre_semillas_s | lambda | li_jerarquico_sobre_menos_0.02 | no_inferior_0.02_publicado | li_jerarquico_sobre_menos_0.01 | no_inferior_0.01_publicado | delta_publicado | ic95_inf_publicado | ic95_sup_publicado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| erm | 5 | 42,123,2024,7,999 | 0.8530 | 0.8086 | 0.0444 | 0.0299 | 0.0595 | 0.0309 | 0.0574 | 0.0076 | 0.0067 | 0.0086 | 0.9954 | sí | sí | sí | sí | 0.0444 | 0.0309 | 0.0574 |
| control_fuente | 5 | 42,123,2024,7,999 | 0.8530 | 0.7582 | 0.0949 | 0.0661 | 0.1211 | 0.0803 | 0.1096 | 0.0261 | 0.0076 | 0.0265 | 1.1018 | sí | sí | sí | sí | 0.0949 | 0.0803 | 0.1096 |
| groupdro | 5 | 42,123,2024,7,999 | 0.8530 | 0.7731 | 0.0799 | 0.0631 | 0.0978 | 0.0653 | 0.0950 | 0.0104 | 0.0076 | 0.0112 | 1.0417 | sí | sí | sí | sí | 0.0799 | 0.0653 | 0.0950 |
| dann | 5 | 42,123,2024,7,999 | 0.8530 | 0.7821 | 0.0709 | 0.0491 | 0.0926 | 0.0567 | 0.0843 | 0.0194 | 0.0069 | 0.0199 | 1.0872 | sí | sí | sí | sí | 0.0709 | 0.0567 | 0.0843 |
| coral | 5 | 42,123,2024,7,999 | 0.8530 | 0.7836 | 0.0694 | 0.0474 | 0.0901 | 0.0545 | 0.0844 | 0.0182 | 0.0076 | 0.0188 | 1.0871 | sí | sí | sí | sí | 0.0694 | 0.0545 | 0.0844 |
| mixstyle | 5 | 42,123,2024,7,999 | 0.8530 | 0.7810 | 0.0720 | 0.0528 | 0.0900 | 0.0577 | 0.0858 | 0.0144 | 0.0072 | 0.0149 | 1.0757 | sí | sí | sí | sí | 0.0720 | 0.0577 | 0.0858 |
| fourier | 5 | 42,123,2024,7,999 | 0.8530 | 0.7931 | 0.0600 | 0.0422 | 0.0781 | 0.0449 | 0.0748 | 0.0107 | 0.0076 | 0.0112 | 1.0687 | sí | sí | sí | sí | 0.0600 | 0.0449 | 0.0748 |
| lisa | 5 | 42,123,2024,7,999 | 0.8530 | 0.7789 | 0.0741 | 0.0429 | 0.1089 | 0.0605 | 0.0880 | 0.0350 | 0.0070 | 0.0353 | 1.1094 | sí | sí | sí | sí | 0.0741 | 0.0605 | 0.0880 |
| dfa_fuente | 5 | 42,123,2024,7,999 | 0.8530 | 0.7857 | 0.0673 | 0.0363 | 0.0982 | 0.0523 | 0.0822 | 0.0309 | 0.0078 | 0.0312 | 1.1084 | sí | sí | sí | sí | 0.0673 | 0.0523 | 0.0822 |
| dfr | 5 | 42,123,2024,7,999 | 0.8530 | 0.8502 | 0.0028 | -0.0089 | 0.0139 | -0.0071 | 0.0126 | 0.0069 | 0.0050 | 0.0076 | 1.0113 | sí | sí | sí | sí | 0.0028 | -0.0071 | 0.0126 |
| erm_mascara | 5 | 42,123,2024,7,999 | 0.8530 | 0.7792 | 0.0738 | 0.0558 | 0.0925 | 0.0592 | 0.0890 | 0.0123 | 0.0074 | 0.0131 | 1.0479 | sí | sí | sí | sí | 0.0738 | 0.0592 | 0.0890 |
| erm_cuotas | 5 | 42,123,2024,7,999 | 0.8530 | 0.7886 | 0.0644 | 0.0467 | 0.0816 | 0.0492 | 0.0792 | 0.0105 | 0.0077 | 0.0113 | 1.0404 | sí | sí | sí | sí | 0.0644 | 0.0492 | 0.0792 |


## c1b_jerarquico_control

| tabla | columna | filas | max_abs_dif | reproduce |
|---|---|---|---|---|
| f_c1b.csv | delta | 12 | 0.0000 | sí |
| f_c1b.csv | ic95_inf | 12 | 0.0000 | sí |
| f_c1b.csv | ic95_sup | 12 | 0.0000 | sí |
| f_c1b.csv | auc_referencia | 12 | 0.0000 | sí |
| f_c1b.csv | auc_metodo | 12 | 0.0000 | sí |


## c1b_jerarquico_resumen

| margen | metodos | li_jerarquico_sobre_menos_margen | no_inferior_publicado_holm |
|---|---|---|---|
| 0.0200 | 12 | 12 | 12 |
| 0.0100 | 12 | 12 | 12 |

