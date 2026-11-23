# S-B, estabilidad de la búsqueda

S-B (docs/08, enmienda E8, punto 5). Validación solamente; descriptivo. La regla registrada elige el ensayo con la máxima early_stopping.best (empate al menor). dif_auc_s<S> es la AUC en val_desconf (estado best, promediado de espejo) del ensayo elegido con la semilla 1 menos la del elegido con la semilla 0, en las corridas de la semilla S; IC95 por pacientes. Las configuraciones congeladas siguen siendo las de la semilla 0.

## estabilidad

| metodo | rho | ensayo_semilla0 | ensayo_semilla1 | coincide | spearman_metricas_seleccion | n_ensayos_comunes | metrica_constante | auc_val_desconf_s0_ensayo_s0 | auc_val_desconf_s0_ensayo_s1 | dif_auc_s0 | ic95_inf_s0 | ic95_sup_s0 | auc_val_desconf_s1_ensayo_s0 | auc_val_desconf_s1_ensayo_s1 | dif_auc_s1 | ic95_inf_s1 | ic95_sup_s1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| erm | 0.9000 | 5 | 3 | no | -0.0714 | 8 | no | 0.8487 | 0.8195 | -0.0292 | -0.0403 | -0.0180 | 0.8151 | 0.8335 | 0.0184 | 0.0071 | 0.0296 |
| erm | 0.9900 | 6 | 3 | no | 0.1190 | 8 | no | 0.6789 | 0.6993 | 0.0204 | 0.0102 | 0.0314 | 0.6773 | 0.6658 | -0.0115 | -0.0210 | -0.0024 |
| lisa | 0.9000 | 3 | 7 | no | 0.8095 | 8 | no | 0.8755 | 0.9043 | 0.0288 | 0.0178 | 0.0398 | 0.8752 | 0.8967 | 0.0215 | 0.0099 | 0.0329 |
| lisa | 0.9900 | 0 | 1 | no | -0.6905 | 8 | no | 0.7010 | 0.6523 | -0.0487 | -0.0612 | -0.0367 | 0.6873 | 0.6659 | -0.0214 | -0.0332 | -0.0093 |

