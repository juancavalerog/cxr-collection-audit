# Validación por simulación del término de semillas

Estructura del benchmark de derrame (ocho niveles, cinco semillas en todos y cinco más en rho 0.9, 0.95 y 0.99), 400 pacientes por réplica, B = 2000, nivel 0.025 unilateral. Escenarios: realista: magnitudes observadas; efecto de semilla con correlación 0.6 entre niveles contiguos; realista_corr_alta: como el realista con correlación 0.9 del efecto de semilla entre niveles; interaccion_grande: interacción tan grande como el efecto de semilla y sin correlación entre niveles (estructura no proporcional); sin_semillas: sin efecto de semilla (solo pacientes e interacción).

| escenario | estimador | nivel | replicas | razon_varianza | cobertura_ic95 | cobertura_ee_mc | error_tipo_I_0.025 | tipo_I_ee_mc |
|---|---|---|---|---|---|---|---|---|
| realista | original | 0.80 | 5000 | 0.9954 | 0.8816 | 0.0046 | 0.0628 | 0.0034 |
| realista | original | 0.90 | 5000 | 0.8971 | 0.9034 | 0.0042 | 0.0518 | 0.0031 |
| realista | original | 0.95 | 5000 | 0.9027 | 0.8958 | 0.0043 | 0.0494 | 0.0031 |
| realista | original | 0.99 | 5000 | 0.8981 | 0.8994 | 0.0043 | 0.0462 | 0.0030 |
| realista | original | 1.00 | 5000 | 1.0016 | 0.8876 | 0.0045 | 0.0544 | 0.0032 |
| realista | original | área 0.5 a 0.99 | 5000 | 0.9373 | 0.9028 | 0.0042 | 0.0488 | 0.0030 |
| realista | corregido | 0.80 | 5000 | 0.9954 | 0.8816 | 0.0046 | 0.0628 | 0.0034 |
| realista | corregido | 0.90 | 5000 | 1.0055 | 0.9212 | 0.0038 | 0.0422 | 0.0028 |
| realista | corregido | 0.95 | 5000 | 1.0102 | 0.9148 | 0.0039 | 0.0390 | 0.0027 |
| realista | corregido | 0.99 | 5000 | 1.0039 | 0.9166 | 0.0039 | 0.0374 | 0.0027 |
| realista | corregido | 1.00 | 5000 | 1.0016 | 0.8876 | 0.0045 | 0.0544 | 0.0032 |
| realista | corregido | área 0.5 a 0.99 | 5000 | 0.9832 | 0.9144 | 0.0040 | 0.0444 | 0.0029 |
| realista | sin_estratos | 0.80 | 5000 | 1.1270 | 0.8988 | 0.0043 | 0.0540 | 0.0032 |
| realista | sin_estratos | 0.90 | 5000 | 1.0055 | 0.9226 | 0.0038 | 0.0408 | 0.0028 |
| realista | sin_estratos | 0.95 | 5000 | 1.0093 | 0.9162 | 0.0039 | 0.0390 | 0.0027 |
| realista | sin_estratos | 0.99 | 5000 | 1.0034 | 0.9146 | 0.0040 | 0.0380 | 0.0027 |
| realista | sin_estratos | 1.00 | 5000 | 1.1286 | 0.9014 | 0.0042 | 0.0490 | 0.0031 |
| realista | sin_estratos | área 0.5 a 0.99 | 5000 | 1.0169 | 0.9190 | 0.0039 | 0.0420 | 0.0028 |
| realista | pacientes | 0.80 | 5000 | 0.0826 | 0.4278 | 0.0070 | 0.2872 | 0.0064 |
| realista | pacientes | 0.90 | 5000 | 0.0716 | 0.4124 | 0.0070 | 0.2852 | 0.0064 |
| realista | pacientes | 0.95 | 5000 | 0.0425 | 0.3224 | 0.0066 | 0.3318 | 0.0067 |
| realista | pacientes | 0.99 | 5000 | 0.0378 | 0.3050 | 0.0065 | 0.3508 | 0.0067 |
| realista | pacientes | 1.00 | 5000 | 0.1209 | 0.5020 | 0.0071 | 0.2464 | 0.0061 |
| realista | pacientes | área 0.5 a 0.99 | 5000 | 0.1444 | 0.5490 | 0.0070 | 0.2264 | 0.0059 |
| realista_corr_alta | original | 0.80 | 5000 | 0.9923 | 0.8774 | 0.0046 | 0.0644 | 0.0035 |
| realista_corr_alta | original | 0.90 | 5000 | 0.8998 | 0.8974 | 0.0043 | 0.0514 | 0.0031 |
| realista_corr_alta | original | 0.95 | 5000 | 0.8950 | 0.9004 | 0.0042 | 0.0524 | 0.0032 |
| realista_corr_alta | original | 0.99 | 5000 | 0.8935 | 0.9008 | 0.0042 | 0.0522 | 0.0031 |
| realista_corr_alta | original | 1.00 | 5000 | 0.9993 | 0.8922 | 0.0044 | 0.0570 | 0.0033 |
| realista_corr_alta | original | área 0.5 a 0.99 | 5000 | 0.9019 | 0.8862 | 0.0045 | 0.0580 | 0.0033 |
| realista_corr_alta | corregido | 0.80 | 5000 | 0.9923 | 0.8774 | 0.0046 | 0.0644 | 0.0035 |
| realista_corr_alta | corregido | 0.90 | 5000 | 1.0009 | 0.9176 | 0.0039 | 0.0422 | 0.0028 |
| realista_corr_alta | corregido | 0.95 | 5000 | 0.9995 | 0.9202 | 0.0038 | 0.0434 | 0.0029 |
| realista_corr_alta | corregido | 0.99 | 5000 | 1.0006 | 0.9174 | 0.0039 | 0.0444 | 0.0029 |
| realista_corr_alta | corregido | 1.00 | 5000 | 0.9993 | 0.8922 | 0.0044 | 0.0570 | 0.0033 |
| realista_corr_alta | corregido | área 0.5 a 0.99 | 5000 | 0.9478 | 0.8982 | 0.0043 | 0.0518 | 0.0031 |
| realista_corr_alta | sin_estratos | 0.80 | 5000 | 1.1241 | 0.8938 | 0.0044 | 0.0554 | 0.0032 |
| realista_corr_alta | sin_estratos | 0.90 | 5000 | 1.0024 | 0.9170 | 0.0039 | 0.0422 | 0.0028 |
| realista_corr_alta | sin_estratos | 0.95 | 5000 | 1.0004 | 0.9216 | 0.0038 | 0.0428 | 0.0029 |
| realista_corr_alta | sin_estratos | 0.99 | 5000 | 1.0017 | 0.9190 | 0.0039 | 0.0434 | 0.0029 |
| realista_corr_alta | sin_estratos | 1.00 | 5000 | 1.1259 | 0.9064 | 0.0041 | 0.0502 | 0.0031 |
| realista_corr_alta | sin_estratos | área 0.5 a 0.99 | 5000 | 0.9698 | 0.9036 | 0.0042 | 0.0506 | 0.0031 |
| realista_corr_alta | pacientes | 0.80 | 5000 | 0.0825 | 0.4276 | 0.0070 | 0.2842 | 0.0064 |
| realista_corr_alta | pacientes | 0.90 | 5000 | 0.0718 | 0.3992 | 0.0069 | 0.2956 | 0.0065 |
| realista_corr_alta | pacientes | 0.95 | 5000 | 0.0425 | 0.3136 | 0.0066 | 0.3404 | 0.0067 |
| realista_corr_alta | pacientes | 0.99 | 5000 | 0.0378 | 0.3044 | 0.0065 | 0.3396 | 0.0067 |
| realista_corr_alta | pacientes | 1.00 | 5000 | 0.1208 | 0.5036 | 0.0071 | 0.2460 | 0.0061 |
| realista_corr_alta | pacientes | área 0.5 a 0.99 | 5000 | 0.1021 | 0.4690 | 0.0071 | 0.2638 | 0.0062 |
| interaccion_grande | original | 0.80 | 5000 | 1.0477 | 0.9162 | 0.0039 | 0.0384 | 0.0027 |
| interaccion_grande | original | 0.90 | 5000 | 0.9565 | 0.9278 | 0.0037 | 0.0356 | 0.0026 |
| interaccion_grande | original | 0.95 | 5000 | 0.9545 | 0.9234 | 0.0038 | 0.0374 | 0.0027 |
| interaccion_grande | original | 0.99 | 5000 | 0.9691 | 0.9224 | 0.0038 | 0.0364 | 0.0026 |
| interaccion_grande | original | 1.00 | 5000 | 1.0411 | 0.9140 | 0.0040 | 0.0422 | 0.0028 |
| interaccion_grande | original | área 0.5 a 0.99 | 5000 | 0.7305 | 0.8724 | 0.0047 | 0.0618 | 0.0034 |
| interaccion_grande | corregido | 0.80 | 5000 | 1.0477 | 0.9162 | 0.0039 | 0.0384 | 0.0027 |
| interaccion_grande | corregido | 0.90 | 5000 | 1.0143 | 0.9308 | 0.0036 | 0.0340 | 0.0026 |
| interaccion_grande | corregido | 0.95 | 5000 | 1.0090 | 0.9282 | 0.0037 | 0.0342 | 0.0026 |
| interaccion_grande | corregido | 0.99 | 5000 | 1.0265 | 0.9268 | 0.0037 | 0.0350 | 0.0026 |
| interaccion_grande | corregido | 1.00 | 5000 | 1.0411 | 0.9140 | 0.0040 | 0.0422 | 0.0028 |
| interaccion_grande | corregido | área 0.5 a 0.99 | 5000 | 0.7550 | 0.8784 | 0.0046 | 0.0590 | 0.0033 |
| interaccion_grande | sin_estratos | 0.80 | 5000 | 1.1238 | 0.9188 | 0.0039 | 0.0374 | 0.0027 |
| interaccion_grande | sin_estratos | 0.90 | 5000 | 1.0136 | 0.9298 | 0.0036 | 0.0348 | 0.0026 |
| interaccion_grande | sin_estratos | 0.95 | 5000 | 1.0085 | 0.9280 | 0.0037 | 0.0340 | 0.0026 |
| interaccion_grande | sin_estratos | 0.99 | 5000 | 1.0256 | 0.9280 | 0.0037 | 0.0340 | 0.0026 |
| interaccion_grande | sin_estratos | 1.00 | 5000 | 1.1161 | 0.9168 | 0.0039 | 0.0406 | 0.0028 |
| interaccion_grande | sin_estratos | área 0.5 a 0.99 | 5000 | 0.7745 | 0.8806 | 0.0046 | 0.0580 | 0.0033 |
| interaccion_grande | pacientes | 0.80 | 5000 | 0.5107 | 0.8408 | 0.0052 | 0.0790 | 0.0038 |
| interaccion_grande | pacientes | 0.90 | 5000 | 0.5096 | 0.8452 | 0.0051 | 0.0760 | 0.0037 |
| interaccion_grande | pacientes | 0.95 | 5000 | 0.5049 | 0.8352 | 0.0052 | 0.0768 | 0.0038 |
| interaccion_grande | pacientes | 0.99 | 5000 | 0.5030 | 0.8366 | 0.0052 | 0.0782 | 0.0038 |
| interaccion_grande | pacientes | 1.00 | 5000 | 0.5208 | 0.8382 | 0.0052 | 0.0776 | 0.0038 |
| interaccion_grande | pacientes | área 0.5 a 0.99 | 5000 | 0.2714 | 0.7010 | 0.0065 | 0.1502 | 0.0051 |
| sin_semillas | original | 0.80 | 5000 | 1.1287 | 0.9554 | 0.0029 | 0.0228 | 0.0021 |
| sin_semillas | original | 0.90 | 5000 | 1.0662 | 0.9514 | 0.0030 | 0.0240 | 0.0022 |
| sin_semillas | original | 0.95 | 5000 | 1.0636 | 0.9538 | 0.0030 | 0.0226 | 0.0021 |
| sin_semillas | original | 0.99 | 5000 | 1.0778 | 0.9518 | 0.0030 | 0.0250 | 0.0022 |
| sin_semillas | original | 1.00 | 5000 | 1.0815 | 0.9516 | 0.0030 | 0.0248 | 0.0022 |
| sin_semillas | original | área 0.5 a 0.99 | 5000 | 1.0723 | 0.9546 | 0.0029 | 0.0228 | 0.0021 |
| sin_semillas | corregido | 0.80 | 5000 | 1.1287 | 0.9554 | 0.0029 | 0.0228 | 0.0021 |
| sin_semillas | corregido | 0.90 | 5000 | 1.0743 | 0.9516 | 0.0030 | 0.0238 | 0.0022 |
| sin_semillas | corregido | 0.95 | 5000 | 1.0716 | 0.9548 | 0.0029 | 0.0222 | 0.0021 |
| sin_semillas | corregido | 0.99 | 5000 | 1.0882 | 0.9526 | 0.0030 | 0.0244 | 0.0022 |
| sin_semillas | corregido | 1.00 | 5000 | 1.0815 | 0.9516 | 0.0030 | 0.0248 | 0.0022 |
| sin_semillas | corregido | área 0.5 a 0.99 | 5000 | 1.0735 | 0.9548 | 0.0029 | 0.0224 | 0.0021 |
| sin_semillas | sin_estratos | 0.80 | 5000 | 1.1469 | 0.9564 | 0.0029 | 0.0226 | 0.0021 |
| sin_semillas | sin_estratos | 0.90 | 5000 | 1.0744 | 0.9518 | 0.0030 | 0.0236 | 0.0021 |
| sin_semillas | sin_estratos | 0.95 | 5000 | 1.0712 | 0.9536 | 0.0030 | 0.0228 | 0.0021 |
| sin_semillas | sin_estratos | 0.99 | 5000 | 1.0875 | 0.9528 | 0.0030 | 0.0244 | 0.0022 |
| sin_semillas | sin_estratos | 1.00 | 5000 | 1.0938 | 0.9528 | 0.0030 | 0.0238 | 0.0022 |
| sin_semillas | sin_estratos | área 0.5 a 0.99 | 5000 | 1.0816 | 0.9552 | 0.0029 | 0.0224 | 0.0021 |
| sin_semillas | pacientes | 0.80 | 5000 | 0.9976 | 0.9486 | 0.0031 | 0.0260 | 0.0023 |
| sin_semillas | pacientes | 0.90 | 5000 | 0.9978 | 0.9452 | 0.0032 | 0.0268 | 0.0023 |
| sin_semillas | pacientes | 0.95 | 5000 | 0.9973 | 0.9458 | 0.0032 | 0.0260 | 0.0023 |
| sin_semillas | pacientes | 0.99 | 5000 | 0.9956 | 0.9454 | 0.0032 | 0.0282 | 0.0023 |
| sin_semillas | pacientes | 1.00 | 5000 | 0.9954 | 0.9446 | 0.0032 | 0.0296 | 0.0024 |
| sin_semillas | pacientes | área 0.5 a 0.99 | 5000 | 0.9975 | 0.9450 | 0.0032 | 0.0272 | 0.0023 |

