"""Diccionario español -> inglés del material suplementario (manuscript/en/GLOSARIO.md).

Lo usa scripts/suplemento/tablas.py con --idioma en. Solo texto: leyendas (con sus marcadores {0}, {1}... y
[[t_...]], que se conservan), cabeceras, notas y celdas de texto. Ninguna cifra de resultados está aquí: las que
aparecen en las leyendas son constantes del protocolo que ya figuran literalmente en la leyenda española (2 000
remuestreos, nivel de 0.025, número de ecuación o de tabla...), y la verificación inglesa comprueba que cada celda,
leyenda y nota inglesa contiene las mismas cifras que la española.
"""
from __future__ import annotations

import re

# Cadenas que no cambian en inglés (nombres propios, símbolos, versiones, etiquetas de nivel)
IDENTIDAD = {"", "η", "λ", "p", "α", "ERM", "GPU", " / ", "NIH", "DFR", "DANN", "LISA", "CORAL", "GroupDRO",
             "MixStyle", "CheXpert", "timm", "SciPy", "NumPy", "cuDNN", "pandas", "OpenCV", "Python", "Pillow",
             "PyTorch", "PyWavelets", "torchvision", "scikit-learn", "ResNet-50", "1/P(y | s)", "COVIDx CXR-4",
             "Benchmark", "Tuberculosis", "Shenzhen: p", "Montgomery: p", "p, δ = 0.02", "p, δ = 0.01", "DFR (a)",
             "$\\rho$"}

EXACTO = {
    # ---- cabeceras, celdas y rótulos (en orden de aparición)
    "# Material suplementario": "# Supplementary material",
    "(a) Para las reservas de entrenamiento y de validación solo constan las fracciones.":
        "(a) For the training and validation reserves, only the fractions are available.",
    "Conjunto": "Set",
    "Fuente": "Source",
    "Clase": "Class",
    "Imágenes": "Images",
    "Vistas por el segmentador": "Seen by the segmentation network",
    "Fracción de imágenes": "Fraction of images",
    "Fracción de imágenes de pacientes con alguna imagen vista":
        "Fraction of images from patients with at least one seen image",
    "Derrame pleural": "Pleural effusion",
    "reserva de entrenamiento": "training reserve",
    "negativa": "negative",
    "positiva": "positive",
    "reserva de validación": "validation reserve",
    "reserva de test": "test reserve",
    "test desconfundido": "deconfounded test set",
    "Cardiomegalia": "Cardiomegaly",
    "sin desglose (a)": "no breakdown (a)",
    "Método": "Method",
    "Hiperparámetro": "Hyperparameter",
    "Valor por defecto": "Default value",
    "Espacio de búsqueda": "Search space",
    "Todos": "All",
    "tasa de la segunda fase": "second-stage learning rate",
    "solo la tasa común": "common learning rate only",
    "Control de fuente": "Source control",
    "η, tasa de actualización de los pesos de grupo": "η, update rate of the group weights",
    "λ, peso del gradiente invertido": "λ, weight of the reversed gradient",
    "anchura del discriminador": "discriminator width",
    "capas del discriminador": "discriminator layers",
    "abandono del discriminador": "discriminator dropout",
    "multiplicador de la tasa del discriminador": "multiplier of the discriminator learning rate",
    "peso de la penalización del gradiente": "weight of the gradient penalty",
    "peso de la pérdida de alineación": "weight of the alignment loss",
    "probabilidad de aplicar la mezcla": "probability of applying the mixing",
    "parámetro de la distribución beta": "parameter of the beta distribution",
    "Mezcla de amplitud de Fourier": "Fourier amplitude mixing",
    "η, cota del coeficiente de mezcla de amplitud": "η, bound of the amplitude mixing coefficient",
    "lado de la región del espectro que se mezcla": "side of the spectral region that is mixed",
    "peso de la consistencia": "consistency weight",
    "probabilidad del modo dentro de la clase": "probability of the within-class mode",
    "DFA con fuente": "Source-aware DFA",
    "peso de la pérdida de sesgo": "weight of the bias loss",
    "peso de la pérdida de sesgo tras el intercambio": "weight of the bias loss after the swap",
    "peso de las pérdidas tras el intercambio": "weight of the losses after the swap",
    "época de la segunda fase en que empieza el intercambio": "second-stage epoch at which the swap starts",
    "pesos por celda de la pérdida de contenido": "per-cell weights of the content loss",
    "C de la regresión L1 (rejilla, elegido en el subconjunto de selección)":
        "C of the L1 regression (grid, chosen on the selection subset)",
    "no se busca": "not searched",
    "submuestras equilibradas del reajuste": "balanced subsamples of the last-layer refit",
    "ERM con máscara": "Masked ERM",
    "ERM con cuotas": "Quota ERM",
    "ninguno": "none",
    "COVIDx: ensayo": "COVIDx: trial",
    "COVIDx: tasa": "COVIDx: learning rate",
    "COVIDx: parámetros": "COVIDx: parameters",
    "COVIDx: AUC de selección": "COVIDx: selection AUC",
    "Tuberculosis: ensayo": "Tuberculosis: trial",
    "Tuberculosis: tasa": "Tuberculosis: learning rate",
    "Tuberculosis: parámetros": "Tuberculosis: parameters",
    "Tuberculosis: AUC de selección": "Tuberculosis: selection AUC",
    "solo la tasa": "learning rate only",
    "anchura": "width",
    "capas": "layers",
    "abandono": "dropout",
    "tasa del discriminador": "discriminator learning rate",
    "penalización": "penalty",
    "valores por defecto": "default values",
    "peso": "weight",
    "lado": "side",
    "consistencia": "consistency",
    "dentro de la clase": "within class",
    "sesgo": "bias",
    "sesgo tras intercambio": "bias after swap",
    "intercambio": "swap",
    "época de inicio": "start epoch",
    "pesos": "weights",
    "Ensayo": "Trial",
    "Tasa": "Learning rate",
    "Parámetros": "Parameters",
    "AUC de selección": "Selection AUC",
    "Subconjunto de selección de la parte BIMCV de la validación":
        "Selection subset of the BIMCV part of the validation set",
    "Parte BIMCV de la validación": "BIMCV part of the validation set",
    "Validación limpia completa": "Full clean validation set",
    "no comparable (a)": "not comparable (a)",
    "Familia": "Family",
    "Comparación": "Comparison",
    "Estimación": "Estimate",
    "IC95, factor original": "95% CI, original factor",
    "IC95, factor corregido": "95% CI, corrected factor",
    "p de Holm, factor original": "Holm-adjusted p, original factor",
    "p de Holm, factor corregido": "Holm-adjusted p, corrected factor",
    "Cambia la decisión": "Decision changes",
    "Área frente a ERM": "Area vs. ERM",
    "AUC con $\\rho = 0.99$ frente a ERM": "AUC at $\\rho = 0.99$ vs. ERM",
    "Pérdida media por selección de época": "Mean epoch-selection loss",
    "Escenario": "Scenario",
    "Semillas": "Seeds",
    "Razón de varianzas, original": "Variance ratio, original",
    "Razón de varianzas, corregido": "Variance ratio, corrected",
    "Cobertura del IC95, original": "95% CI coverage, original",
    "Cobertura del IC95, corregido": "95% CI coverage, corrected",
    "Error de tipo I, original": "Type I error, original",
    "Error de tipo I, corregido": "Type I error, corrected",
    "Realista": "Realistic",
    "área de 0.5 a 0.99": "area from 0.5 to 0.99",
    "Correlación alta entre niveles": "High correlation between levels",
    "Interacción grande, no proporcional": "Large, non-proportional interaction",
    "Sin efecto de semilla": "No seed effect",
    "5 y 10": "5 and 10",
    "Elemento": "Item",
    "Valor": "Value",
    "Sistema": "System",
    "CUDA de PyTorch": "PyTorch CUDA",
    "Pesos de ConvNeXt-Tiny": "ConvNeXt-Tiny weights",
    "Paquetes listados en el contenedor": "Packages listed in the container",
    "Ficheros con huella SHA256 en la congelación del protocolo":
        "Files with a SHA-256 hash in the protocol freeze",
    "Fecha de la congelación (UTC)": "Freeze date (UTC)",
    "Fase": "Stage",
    "Corridas": "Runs",
    "Terminadas con código 0": "Finished with exit code 0",
    "Horas de GPU": "GPU hours",
    "búsqueda de hiperparámetros (c)": "hyperparameter search (c)",
    "corridas finales, cinco semillas": "final runs, five seeds",
    "repetición de las corridas de la semilla 42 (b)": "repetition of the seed-42 runs (b)",
    "piloto": "pilot",
    "búsqueda de hiperparámetros": "hyperparameter search",
    "corridas finales, cinco semillas (a)": "final runs, five seeds (a)",
    "semillas adicionales en 0.9, 0.95 y 0.99": "additional seeds at 0.9, 0.95 and 0.99",
    "repetición de la búsqueda con la semilla 1": "repetition of the search with seed 1",
    "repetición de la búsqueda con evaluación del test": "repetition of the search with test-set evaluation",
    "segundo tronco (ResNet-50)": "second backbone (ResNet-50)",
    "segunda etiqueta": "second label",
    "AUC del método": "AUC of the method",
    "Diferencia (IC95)": "Difference (95% CI)",
    "p de Holm, δ = 0.02": "Holm-adjusted p, δ = 0.02",
    "No inferior, δ = 0.02": "Non-inferior, δ = 0.02",
    "p de Holm, δ = 0.01": "Holm-adjusted p, δ = 0.01",
    "No inferior, δ = 0.01": "Non-inferior, δ = 0.01",
    "p de diferencia mayor que cero": "p for a difference greater than zero",
    "IC95 de semillas y pacientes": "Seed-and-patient 95% CI",
    "Referencia sin información de clase": "Class-blind reference",
    "AUC (IC95)": "AUC (95% CI)",
    "AUC en AP (IC95)": "AUC in AP (95% CI)",
    "AUC en PA (IC95)": "AUC in PA (95% CI)",
    "Positivos de ajuste, cinco cifras / seis cifras": "Fitting positives, five-digit / six-digit",
    "Referencia principal (red de ImageNet)": "Primary reference (ImageNet network)",
    "AP: AUC del método": "AP: AUC of the method",
    "AP: diferencia (IC95)": "AP: difference (95% CI)",
    "PA: AUC del método": "PA: AUC of the method",
    "PA: diferencia (IC95)": "PA: difference (95% CI)",
    "Referencia principal (AUC del detector)": "Primary reference (detector AUC)",
    "no aplica": "not applicable",
    "Negativos de BIMCV frente a positivos de BIMCV": "BIMCV negatives vs. BIMCV positives",
    "Negativos de BIMCV frente a positivos de RICORD": "BIMCV negatives vs. RICORD positives",
    "Negativos de RSNA frente a positivos de BIMCV": "RSNA negatives vs. BIMCV positives",
    "Negativos de RSNA frente a positivos de RICORD": "RSNA negatives vs. RICORD positives",
    "Referencia secundaria (troncos de ERM)": "Secondary reference (ERM backbones)",
    "Tronco": "Backbone",
    "Validación cruzada: negativos frente a positivos de cinco cifras":
        "Cross-validation: negatives vs. five-digit positives",
    "Validación cruzada: negativos frente a positivos de seis cifras":
        "Cross-validation: negatives vs. six-digit positives",
    "Validación cruzada: positivos de cinco cifras frente a seis cifras":
        "Cross-validation: five-digit vs. six-digit positives",
    "Lectura de cinco cifras aplicada a seis cifras (IC95)":
        "Five-digit-collection probe applied to six-digit collection (95% CI)",
    "Lectura de seis cifras aplicada a cinco cifras (IC95)":
        "Six-digit-collection probe applied to five-digit collection (95% CI)",
    "Test: lectura de cinco cifras": "Test: five-digit-collection probe",
    "Test: lectura de seis cifras (IC95)": "Test: six-digit-collection probe (95% CI)",
    "Reservadas: negativas / cinco cifras / seis cifras": "Held out: negatives / five-digit / six-digit",
    "Análisis": "Analysis",
    "Diferencia con ERM (IC95)": "Difference from ERM (95% CI)",
    "Clasificador de solo adquisición, ajustado con BIMCV de entrenamiento":
        "Acquisition-only classifier, fitted on BIMCV training data",
    "Clasificador de solo adquisición, validación cruzada en el conjunto evaluado (usa la clase)":
        "Acquisition-only classifier, cross-validated on the evaluated set (uses the class labels)",
    "Detector de colección sobre los troncos de ERM con máscara":
        "Collection detector on the masked ERM backbones",
    "AUC interna": "Internal AUC",
    "AUC en COVIDGR (IC95)": "AUC on COVIDGR (95% CI)",
    "Caída (IC95)": "Drop (95% CI)",
    "Sensibilidad": "Sensitivity",
    "Especificidad": "Specificity",
    "Radiografía normal, PCR positiva": "Normal radiograph, positive PCR",
    "Leve": "Mild",
    "Moderada": "Moderate",
    "Grave": "Severe",
    "Medida": "Measure",
    "Diferencia frente a ERM (IC95)": "Difference vs. ERM (95% CI)",
    "p unilateral": "One-sided p",
    "Papel": "Role",
    "p de Holm": "Holm-adjusted p",
    "Rechaza con Holm": "Rejected after Holm correction",
    "Área (ecuación (9))": "Area (Eq. (9))",
    "No inferioridad, $\\rho = 0.5$, δ = 0.01": "Non-inferiority, $\\rho = 0.5$, δ = 0.01",
    "Caída entre $\\rho = 0.5$ y 1.0": "Drop from $\\rho = 0.5$ to 1.0",
    "confirmatoria": "confirmatory",
    "descriptiva": "descriptive",
    "secundaria": "secondary",
    "Shenzhen: AUC (IC95)": "Shenzhen: AUC (95% CI)",
    "Shenzhen: diferencia con ERM (IC95)": "Shenzhen: difference from ERM (95% CI)",
    "Shenzhen: p de Holm": "Shenzhen: Holm-adjusted p",
    "Montgomery: AUC (IC95)": "Montgomery: AUC (95% CI)",
    "Montgomery: diferencia con ERM (IC95)": "Montgomery: difference from ERM (95% CI)",
    "Montgomery: p de Holm": "Montgomery: Holm-adjusted p",
    "referencia": "reference",
    "Media de los once métodos": "Mean of the eleven methods",
    "Área de $\\rho = 0.5$ a 0.99 (IC95)": "Area from $\\rho = 0.5$ to 0.99 (95% CI)",
    "Área de $\\rho = 0.9$ a 0.99 (IC95)": "Area from $\\rho = 0.9$ to 0.99 (95% CI)",
    "Pérdida media por selección de época, once métodos": "Mean epoch-selection loss, eleven methods",
    "Ensayo de la regla": "Rule trial",
    "Ensayo de la validación desconfundida": "Deconfounded-validation trial",
    "Coinciden": "Same trial",
    "AUC, regla": "AUC, rule trial",
    "AUC, media de los ensayos": "AUC, mean of the trials",
    "AUC, ensayo de la validación desconfundida": "AUC, deconfounded-validation trial",
    "G (IC95)": "G (95% CI)",
    "L (IC95)": "L (95% CI)",
    "Spearman entre la métrica de selección y la AUC": "Spearman between the selection metric and the AUC",
    "Ensayo elegido, semilla 0": "Selected trial, seed 0",
    "Ensayo elegido, semilla 1": "Selected trial, seed 1",
    "Spearman entre las métricas de selección": "Spearman between the selection metrics",
    "Cambio de AUC, corridas de la semilla 0 (IC95)": "Change in AUC, seed-0 runs (95% CI)",
    "Cambio de AUC, corridas de la semilla 1 (IC95)": "Change in AUC, seed-1 runs (95% CI)",
    "(a) El cálculo de la selección de época en tuberculosis no incluye DFR.":
        "(a) The epoch-selection calculation in tuberculosis does not include DFR.",
    "Shenzhen: pérdida (IC95)": "Shenzhen: loss (95% CI)",
    "Shenzhen: semillas a favor del oráculo": "Shenzhen: seeds favoring the oracle",
    "Shenzhen: p, prueba t": "Shenzhen: p, t test",
    "Montgomery: pérdida (IC95)": "Montgomery: loss (95% CI)",
    "Montgomery: semillas a favor del oráculo": "Montgomery: seeds favoring the oracle",
    "Montgomery: p, prueba t": "Montgomery: p, t test",
    "no calculado (a)": "not computed (a)",
    " de ": " of ",
    "ERM, caída entre $\\rho = 0.5$ y 1.0": "ERM, drop from $\\rho = 0.5$ to 1.0",
    "sensibilidad": "sensitivity",
    "(a) La red de ImageNet de la referencia principal no tiene penúltima capa entrenada: su cabeza no se "
    "entrenó.":
        "(a) The ImageNet network of the primary reference has no trained penultimate layer, because its head "
        "was not trained.",
    "Variante": "Variant",
    "Referencia": "Reference",
    "AUC de la referencia": "AUC of the reference",
    "Referencia menos ERM (IC95)": "Reference minus ERM (95% CI)",
    "Referencia menos DFR (IC95)": "Reference minus DFR (95% CI)",
    "Análisis principal": "Main analysis",
    "Principal": "Primary",
    "Secundaria": "Secondary",
    "Detector con $C = 0.01$": "Detector with $C = 0.01$",
    "Detector con $C = 0.3$": "Detector with $C = 0.3$",
    "Detector sobre la penúltima capa": "Detector on the penultimate layer",
    "no aplica (a)": "not applicable (a)",
    "Métodos con la media móvil exponencial de los pesos":
        "Methods with the exponential moving average (EMA) of the weights",
    "Métodos en la última época": "Methods at the last epoch",
    "Conjunto de las cinco semillas de cada método": "Five-seed ensemble of each method",
    "no aplica (b)": "not applicable (b)",
    "Remuestreo por imagen": "Image-level bootstrap",
    "Test limpio completo": "Full clean test set",
    "Sin promediado de espejo": "Without horizontal-flip averaging",
    "Diferencia en el área": "Difference in the area",
    "Principal: jerárquico (IC95)": "Main: hierarchical (95% CI)",
    "Pacientes con semillas fijas (IC95)": "Patient-level interval with fixed seeds (95% CI)",
    "Remuestreo doble (IC95)": "Double bootstrap (95% CI)",
    "Prueba t entre semillas: media (IC95)": "t test across seeds: mean (95% CI)",
    "p, jerárquico": "p, hierarchical",
    "p, pacientes": "p, patient-level",
    "p, doble": "p, double bootstrap",
    "p, prueba t": "p, t test",
    "p de Holm, jerárquico": "Holm-adjusted p, hierarchical",
    "p de Holm, pacientes": "Holm-adjusted p, patient-level",
    "p de Holm, doble": "Holm-adjusted p, double bootstrap",
    "p de Holm, prueba t": "Holm-adjusted p, t test",
    "p: jerárquico / pacientes / doble": "p: hierarchical / patient-level / double bootstrap",
    "p de Holm: jerárquico / pacientes / doble / prueba t":
        "Holm-adjusted p: hierarchical / patient-level / double bootstrap / t test",
    "Diferencia": "Difference",
    "IC95 de pacientes": "Patient-level 95% CI",
    "Desviación de semillas": "Seed standard deviation",
    "Desviación de pacientes": "Patient standard deviation",
}

# ---- leyendas y notas (en orden de aparición; se conservan los marcadores y las remisiones)
LEYENDAS = {
    "Este material reúne las tablas que completan las del artículo: las familias de contraste completas, con sus "
    "valores p sin corregir y corregidos, los análisis descriptivos y de sensibilidad preespecificados, los "
    "cálculos complementarios hechos después de abrir los tests y los datos de reproducibilidad. Todas las "
    "cifras proceden de los ficheros de resultados del estudio, con el formato de la sección de Resultados; las "
    "Figuras S1 a S8 se describen en sus propias leyendas. Los intervalos jerárquicos de semillas y pacientes "
    "usan el factor corregido de la ecuación (7); la [[t_impacto]] compara, en las familias de contraste, los "
    "intervalos y valores p con el factor original y con el corregido.":
        "This material contains the tables that complement those in the article: the complete test families with "
        "their uncorrected and corrected p values, the prespecified descriptive and sensitivity analyses, the "
        "post hoc supplementary calculations made after the test sets were opened, and the reproducibility data. "
        "All values come from the result files of the study and follow the format of the Results section. Figs. "
        "S1 to S8 are described in their own captions. The hierarchical seed-and-patient intervals use the "
        "corrected calibration factor of Eq. (7), and [[t_impacto]] compares the intervals and p values of the "
        "test families under the original and the corrected factor.",

    "Solapamiento entre el conjunto de entrenamiento del segmentador de pulmón (Section 2.3) y los benchmarks "
    "controlados, por fuente y clase: reservas de pacientes del benchmark de derrame pleural y tests "
    "desconfundidos de derrame y de cardiomegalia, que son los mismos en todos los niveles y semillas. Vistas "
    "por el segmentador: imágenes que forman parte de su conjunto de entrenamiento. La última columna es la "
    "fracción de imágenes cuyo paciente tiene alguna imagen en ese conjunto. En el benchmark de derrame, la "
    "calidad media de las pseudomáscaras apenas difiere entre las imágenes vistas y las no vistas: en las ocho "
    "combinaciones de fuente, clase y reserva (la de test y, juntas, las de entrenamiento y validación), la "
    "media de las vistas supera a la de las no vistas en {0} a {1}, con medias de {2} a {3} en las no vistas y "
    "de {4} a {5} en las vistas. No se reentrenó el segmentador sin esas imágenes, de modo que su efecto sobre "
    "los resultados no está medido. Descriptivo.":
        "Overlap between the training set of the lung segmentation network (Section 2.3) and the controlled "
        "benchmarks, by source and class: patient reserves of the pleural effusion benchmark and deconfounded "
        "test sets for effusion and for cardiomegaly, which are the same at all levels and seeds. Seen by the "
        "segmentation network: images that belong to its training set. The last column is the fraction of images "
        "whose patient has at least one image in that set. In the effusion benchmark, the mean quality of the "
        "pseudo-masks hardly differs between seen and unseen images. Across the eight combinations of source, "
        "class and reserve (the test reserve and, pooled, the training and validation reserves), the mean of the "
        "seen images exceeds that of the unseen images by {0} to {1}, with means of {2} to {3} for the unseen "
        "images and {4} to {5} for the seen images. The segmentation network was not retrained without these "
        "images, so their effect on the results has not been measured. Descriptive.",

    "Espacio de búsqueda de hiperparámetros, común a los tres benchmarks (Section 2.5): {0} configuraciones por "
    "método, la primera con los valores por defecto y las demás muestreadas de forma determinista a partir del "
    "método y del número de ensayo; la tasa de la segunda fase se muestrea en todos los métodos. Entre llaves, "
    "elección uniforme entre los valores. El valor por defecto de la tasa es el del ensayo 0 de la búsqueda de "
    "COVIDx; los demás, y los espacios, son los atributos declarados en el código de cada método.":
        "Hyperparameter search space, common to the three benchmarks (Section 2.5): {0} configurations per "
        "method, the first with the default values and the rest sampled deterministically from the method and "
        "the trial number. The second-stage learning rate is sampled for every method. Braces denote a uniform "
        "choice among the listed values. The default learning rate is the one used in trial 0 of the COVIDx "
        "search; the other defaults and the search spaces are the attributes declared in the code of each method.",

    "Configuración elegida y congelada en COVIDx CXR-4 (búsqueda con la semilla 42, selección con el subconjunto de "
    "selección de la parte BIMCV de la validación) y en tuberculosis (semilla 0, selección con la validación). "
    "Ensayo 0: valores por defecto. Tasa: la de la segunda fase. Parámetros: valores de los hiperparámetros "
    "buscados del método ([[t_espacio]]). AUC de selección: la métrica de la regla preespecificada, sin "
    "promediado de espejo, en el conjunto de selección; en DFR, la de su etapa ERM, que es la que usa la regla "
    "para elegir el ensayo; la regularización del reajuste se elige después con el mismo subconjunto de "
    "selección. Solo validación.":
        "Configuration selected and frozen in COVIDx CXR-4 (search with seed 42, selection on the selection subset "
        "of the BIMCV part of the validation set) and in tuberculosis (seed 0, selection on the validation set). "
        "Trial 0: default values. Learning rate: the second-stage learning rate. Parameters: values of the "
        "searched hyperparameters of each method ([[t_espacio]]). Selection AUC: the metric of the prespecified "
        "rule, without horizontal-flip averaging, on the selection set. For DFR it is the AUC of its ERM stage, "
        "which is what the rule uses to choose the trial, and the regularization of the last-layer refit is "
        "chosen afterwards on the same selection subset. Validation only.",

    "Configuración elegida y congelada en cada nivel del benchmark de derrame pleural (búsqueda con la semilla "
    "0, selección con la validación de la misma $\\rho$; la orientación inversa usa la de su nivel). Con $\\rho = "
    "1$ la validación se satura y los ocho ensayos empatan; la regla, con empate al ensayo menor, elige el "
    "ensayo 0 salvo en DFA con fuente (Section 2.5). En DFR, la AUC de selección es la de su etapa ERM, que es "
    "la que usa la regla para elegir el ensayo; la regularización del reajuste se elige después con el subconjunto de "
    "selección de la validación de la misma $\\rho$. Con $\\rho = 1$ la configuración de DFR es la de su etapa "
    "ERM; el reajuste no está definido. Solo validación.":
        "Configuration selected and frozen at each level of the pleural effusion benchmark (search with seed 0, "
        "selection with same-$\\rho$ validation, and the reverse orientation uses the configuration of the "
        "corresponding level). At $\\rho = 1$ the validation saturates and the eight trials tie. The rule, which "
        "breaks ties in favor of the lowest-numbered trial, selects trial 0 except for source-aware DFA (Section "
        "2.5). For DFR, the selection AUC is that of its ERM stage, which is what the rule uses to choose the "
        "trial; the regularization of the last-layer refit is chosen afterwards on the selection subset of the "
        "same-$\\rho$ validation set. At $\\rho = 1$ the DFR configuration is that of its ERM stage and the "
        "last-layer refit is not defined. Validation only.",

    "AUC en validación de las {0} corridas finales de COVIDx CXR-4 (cinco semillas por método), con el estado de "
    "la mejor época: media ± desviación típica entre semillas, y mínimo y máximo entre paréntesis. Solo "
    "validación, sin valor confirmatorio.":
        "Validation AUC of the {0} final COVIDx CXR-4 runs (five seeds per method) with the best-epoch "
        "checkpoint: mean ± standard deviation across seeds, with minimum and maximum in parentheses. Validation "
        "only, with no confirmatory value.",

    "(a) La parte BIMCV de la validación, incluida también en la validación limpia completa, contiene el subconjunto "
    "con el que DFR reajusta su última capa; su AUC en esos conjuntos no es comparable con la de los demás "
    "métodos.":
        "(a) The BIMCV part of the validation set, which is also included in the full clean validation set, "
        "contains the subset that DFR uses to refit its last layer, so its AUC on these sets is not comparable "
        "with that of the other methods.",

    "(b) En DFR, el subconjunto de selección es el que usa la regla para elegir el ensayo con la AUC de su etapa ERM "
    "y, después, la regularización del reajuste.":
        "(b) For DFR, the selection subset is the set on which the rule chooses the trial, using the AUC of its "
        "ERM stage, and then the regularization of the last-layer refit.",

    "Efecto de la corrección del factor del término de semillas (ecuación (7)) sobre las familias de contraste "
    "del benchmark de derrame pleural con la selección principal. El plan preespecificado calibraba ese término "
    "como si las semillas se remuestrearan todas juntas, mientras que el procedimiento las remuestrea por "
    "estratos según los niveles en que existen; en los tres niveles con diez semillas, la varianza del término "
    "quedaba así por debajo de la declarada. El factor se corrigió después de abrir los tests, sin cambiar "
    "estimandos, familias, márgenes, remuestreos ni semillas, y las estimaciones puntuales no cambian. De las "
    "{0} filas de resultados revisadas en las tablas que usan ese término, {1} cambian su intervalo; el número "
    "de filas con una decisión de contraste distinta es {2}. Holm dentro de cada familia al nivel unilateral de "
    "0.025 (área y $\\rho = 0.99$: diez comparaciones; pérdida media: tres niveles). Las demás tablas de este "
    "material dan solo los intervalos con el factor corregido.":
        "Effect of correcting the calibration factor of the seed term (Eq. (7)) on the test families of the "
        "pleural effusion benchmark under the primary selection. The prespecified plan calibrated this term as if "
        "all seeds were resampled together, whereas the procedure resamples them in strata defined by the levels "
        "at which they exist. At the three levels with ten seeds, the variance of the term was therefore smaller "
        "than declared. The factor was corrected after the test sets were opened, without changing estimands, "
        "families, margins, resampling schemes or seeds, and the point estimates do not change. Of the {0} "
        "result rows reviewed in the tables that use this term, {1} change their interval and {2} have a "
        "different test decision. Holm correction within each family at the one-sided 0.025 level (ten "
        "comparisons for the area and for $\\rho = 0.99$, three levels for the mean loss). All other tables in "
        "this material report only the intervals with the corrected factor.",

    "Validación por simulación del factor del término de semillas con el factor original y con el corregido "
    "(ecuación (7)), bajo la hipótesis nula. Estructura simulada: ocho niveles de $\\rho$ con la presencia real "
    "de semillas (cinco en todos y cinco más en $\\rho$ de 0.9, 0.95 y 0.99), diferencia por semilla con efecto "
    "de semilla correlacionado entre niveles contiguos, efecto común de pacientes e interacción, y 400 pacientes "
    "por conjunto simulado, remuestreados con los mismos pesos en todos los niveles y semillas; magnitudes "
    "tomadas de las medianas por nivel observadas en las comparaciones con ERM. A cada conjunto simulado se le "
    "aplican las funciones de remuestreo del análisis (B = 2 000) y la prueba unilateral de la ecuación (4); {0} "
    "conjuntos simulados por escenario. Escenarios: realista (correlación 0.6 del efecto de semilla entre "
    "niveles contiguos), correlación alta (0.9), interacción tan grande como el efecto de semilla y sin "
    "correlación entre niveles, y sin efecto de semilla. Razón de varianzas: media de la varianza del remuestreo "
    "dividida por la varianza verdadera; cobertura del IC95 de percentiles; error de tipo I: proporción de "
    "conjuntos simulados con p unilateral menor o igual que 0.025. En el escenario realista, la razón de "
    "varianzas de los niveles de diez semillas pasa de {5} a {6} con el factor original a {7} a {8} con el "
    "corregido. La corrección no basta para alcanzar el nivel nominal: aun con el factor corregido, en el "
    "escenario realista el error de tipo I es de {1} a {2} en los niveles de diez semillas y de {3} y {4} en "
    "$\\rho$ de 0.8 y 1.0, con cinco, porque el remuestreo trata como conocida una varianza de semillas estimada "
    "con pocos grados de libertad.":
        "Simulation check of the calibration factor of the seed term, original and corrected (Eq. (7)), under "
        "the null hypothesis. Simulated structure: eight levels of $\\rho$ with the actual availability of seeds "
        "(five at every level and five more at $\\rho$ of 0.9, 0.95 and 0.99), a per-seed difference with a seed "
        "effect correlated between adjacent levels, a common patient effect and an interaction, and 400 patients "
        "per simulated data set, resampled with the same weights at all levels and seeds. Magnitudes were taken "
        "from the per-level medians observed in the comparisons with ERM. The resampling functions of the "
        "analysis (B = 2000) and the one-sided test of Eq. (4) were applied to each simulated data set, with {0} "
        "simulated data sets per scenario. Scenarios: realistic (correlation of 0.6 for the seed effect between "
        "adjacent levels), high correlation (0.9), an interaction as large as the seed effect with no "
        "correlation between levels, and no seed effect. Variance ratio: mean bootstrap variance divided by the "
        "true variance. Coverage refers to the percentile 95% CI, and the type I error is the proportion of "
        "simulated data sets with a one-sided p less than or equal to 0.025. In the realistic scenario, the "
        "variance ratio at the ten-seed levels ranges from {5} to {6} with the original factor and from {7} to "
        "{8} with the corrected one. The correction is not enough to reach the nominal level. Even with the "
        "corrected factor, the type I error in the realistic scenario is {1} to {2} at the ten-seed levels, and "
        "{3} and {4} at $\\rho$ of 0.8 and 1.0, which have five seeds, because the bootstrap treats as known a "
        "seed variance estimated with few degrees of freedom.",

    "Entorno de software del contenedor en el que se entrenaron y evaluaron todos los modelos, según la "
    "descripción del entorno guardada con la congelación del protocolo (Section 2.1 y Section 2.8). La lista "
    "completa de paquetes con sus versiones forma parte del material de reproducibilidad.":
        "Software environment of the container in which all models were trained and evaluated, as given in the "
        "environment description saved with the protocol freeze (Sections 2.1 and 2.8). The full list of "
        "packages and their versions is part of the reproducibility material.",

    "Recuento de corridas de entrenamiento según los ficheros de control de las cadenas de ejecución. Horas de "
    "GPU: suma de las duraciones anotadas de las corridas, como en la Section 2.8. No se incluyen los pasos de "
    "evaluación, extracción de características e informes.":
        "Number of training runs according to the control files of the execution chains. GPU hours: sum of the "
        "recorded run durations, as in Section 2.8. Evaluation, feature extraction and reporting steps are not "
        "included.",

    "(a) Las corridas del control de fuente terminan sin entrenar con un código distinto de 0, porque el método "
    "no es aplicable en este benchmark (Section 2.4).":
        "(a) The source control runs end without training with an exit code other than 0, because the method is "
        "not applicable in this benchmark (Section 2.4).",

    "(b) Las diez corridas de la semilla 42 que habían empezado antes de un cambio en la política de "
    "almacenamiento se repitieron desde el principio para conservar los estados de mejor época, última época y "
    "media móvil; en esos diez métodos, la semilla 42 de las corridas finales es la de estas repeticiones.":
        "(b) The ten seed-42 runs that had started before a change in the storage policy were rerun from the "
        "start to keep the best-epoch, last-epoch and moving-average checkpoints. For those ten methods, the "
        "seed-42 final runs are these repetitions.",

    "(c) Las 96 configuraciones de la búsqueda incluyen los 12 ensayos con los valores por defecto, que proceden "
    "de las corridas piloto de la fase anterior y no se cuentan aquí.":
        "(c) The 96 configurations of the search include the 12 default-value trials, which come from the pilot "
        "runs of the previous stage and are not counted here.",

    "Familia confirmatoria de la auditoría (Tabla 4, primera fila): no inferioridad de la referencia principal "
    "(AUC {0}) frente a cada uno de los 12 métodos en la parte BIMCV del test ({1} imágenes), con el margen "
    "preespecificado δ = 0.02 y, como sensibilidad, δ = 0.01. Diferencia: AUC de la referencia menos la del "
    "método, media sobre las cinco semillas (ecuación (3)), con IC95 de percentiles de 2 000 remuestreos de "
    "pacientes. Valores p unilaterales de la ecuación (4) sin corregir y con la corrección de Holm sobre las 12 "
    "comparaciones de cada margen, al nivel de 0.025. La última columna es el valor p unilateral, sin corregir y "
    "descriptivo, de que la diferencia sea mayor que cero.":
        "Confirmatory family of the audit (Table 4, first row): non-inferiority of the primary reference (AUC "
        "{0}) against each of the 12 methods on the BIMCV part of the test set ({1} images), with the "
        "prespecified margin δ = 0.02 and, as a sensitivity analysis, δ = 0.01. Difference: AUC of the reference "
        "minus that of the method, averaged over the five seeds (Eq. (3)), with a percentile 95% CI from 2000 "
        "patient-level bootstrap resamples. One-sided p values from Eq. (4), uncorrected and after Holm "
        "correction over the 12 comparisons for each margin, at the 0.025 level. The last column gives the "
        "uncorrected, descriptive one-sided p value for a difference greater than zero.",

    "Familia secundaria de la auditoría (Tabla 4, segunda fila): no inferioridad de la referencia secundaria, el "
    "detector de colección sobre los troncos finales de ERM (AUC media {0}), frente a cada método, con la "
    "diferencia emparejada por semilla (la referencia de la semilla s frente al método de la semilla s). Mismo "
    "conjunto, remuestreo, márgenes y corrección que la tabla anterior. La cuarta columna repite la diferencia "
    "con el IC95 jerárquico de semillas y pacientes de las ecuaciones (5) a (7), previsto en el plan congelado "
    "para esta familia y calculado después de abrir los tests con las funciones congeladas; es una lectura de "
    "sensibilidad y no sustituye a la prueba con Holm.":
        "Secondary family of the audit (Table 4, second row): non-inferiority of the secondary reference, the "
        "collection detector on the final ERM backbones (mean AUC {0}), against each method, with the difference "
        "paired by seed (the reference of seed s against the method of seed s). The set, bootstrap, margins and "
        "correction are those of the previous table. The fourth column repeats the difference with the "
        "hierarchical seed-and-patient 95% CI of Eqs. (5) to (7), which the frozen plan specifies for this "
        "family and which was computed after the test sets were opened, using the frozen functions. It is a "
        "sensitivity analysis and does not replace the test with Holm correction.",

    "Referencias sin información de clase, tronco por tronco, en la parte BIMCV del test ({0} imágenes de {1} "
    "pacientes; {2} AP y {3} PA entre las que tienen la proyección en el nombre). Detector de colección: "
    "regresión logística L2 ($C = {4}$, pesos de clase equilibrados) sobre las 768 características agrupadas del "
    "tronco, con diana 1 para la colección de cinco cifras, ajustada solo con positivos y aplicada con "
    "promediado de espejo (Section 2.6). En los troncos con conjunto reservado se ajusta con los positivos "
    "reservados, que el tronco no vio. IC95 de percentiles con 2 000 remuestreos de pacientes. La Tabla 5 y la "
    "Section 3.2 del artículo dan la media y el recorrido de cada grupo de troncos.":
        "Class-blind references, backbone by backbone, on the BIMCV part of the test set ({0} images from {1} "
        "patients; {2} AP and {3} PA among the images whose file name records the projection). Collection "
        "detector: L2 logistic regression ($C = {4}$, balanced class weights) on the 768 pooled features of the "
        "backbone, with target 1 for the five-digit collection, fitted on positives only and applied with "
        "horizontal-flip averaging (Section 2.6). For backbones with a held-out set, the detector is fitted on "
        "the held-out positives, which the backbone did not see. Percentile 95% CI from 2000 patient-level "
        "bootstrap resamples. Table 5 and Section 3.2 of the article give the mean and range of each group of "
        "backbones.",

    "Estratos de proyección en la parte BIMCV del test. La proyección consta en el nombre de {0} de las {1} "
    "imágenes ({2} AP, {3} de ellas negativas; {4} PA, {5} negativas), y el análisis se limita a ellas. "
    "Diferencia: AUC de la referencia principal menos la del método dentro del estrato, media sobre cinco "
    "semillas, con IC95 de 2 000 remuestreos de pacientes. Descriptivo, sin familia de contraste. Por sí sola, "
    "la proyección (AP = 1) separa las clases con una AUC de {6}; son AP el {7} % de las negativas y el {8} % de "
    "las positivas.":
        "Projection strata in the BIMCV part of the test set. The projection is recorded in the file name of {0} "
        "of the {1} images ({2} AP, {3} of them negative, and {4} PA, {5} of them negative), and the analysis is "
        "restricted to these images. Difference: AUC of the primary reference minus that of the method within "
        "the stratum, averaged over five seeds, with 95% CI from 2000 patient-level bootstrap resamples. "
        "Descriptive, with no test family. On its own, the projection (AP = 1) separates the classes with an AUC "
        "of {6}. AP images make up {7}% of the negatives and {8}% of the positives.",

    "AUC en el test limpio según la colección de origen de los negativos y de los positivos. Los positivos de "
    "BIMCV del test son todos de la colección de cinco cifras. Tamaños (negativas / positivas): {0} / {1}, {2} / "
    "{3}, {4} / {5} y {6} / {7}, en el orden de las columnas. Media de cinco semillas en los métodos y en la "
    "referencia secundaria; IC95 de 2 000 remuestreos de pacientes. Descriptivo.":
        "AUC on the clean test set by collection of origin of the negatives and of the positives. All BIMCV "
        "positives in the test set belong to the five-digit collection. Sizes (negative / positive), in column "
        "order: {0} / {1}, {2} / {3}, {4} / {5} and {6} / {7}. Mean of five seeds for the methods and for the "
        "secondary reference, with 95% CI from 2000 patient-level bootstrap resamples. Descriptive.",

    "Lecturas lineales entre colecciones de positivos sobre los cuatro troncos de ERM entrenados sin un conjunto "
    "reservado de BIMCV (Section 2.6). Regresión logística L2 sobre las características agrupadas, "
    "estandarizadas con el conjunto reservado. Validación cruzada: AUC fuera de pliegue con cinco pliegues "
    "agrupados por paciente dentro del conjunto reservado; la tercera columna enfrenta dos colecciones de la "
    "misma clase. Transferencia: lectura ajustada con una mitad de los negativos reservados (partición por "
    "paciente) y los positivos de una colección, y evaluada con la otra mitad de los negativos frente a los "
    "positivos de la otra colección. Test: las dos lecturas aplicadas a la parte BIMCV del test, cuyos positivos "
    "son todos de la colección de cinco cifras. IC95 de 2 000 remuestreos de pacientes; las columnas sin "
    "intervalo no lo tienen calculado. Descriptivo.":
        "Linear probes across collections of positives on the four ERM backbones whose training excluded a "
        "held-out set of BIMCV images (Section 2.6). L2 logistic regression on the pooled features, standardized "
        "with the held-out set. Cross-validation: out-of-fold AUC with five patient-grouped folds within the "
        "held-out set. The third column contrasts two collections of the same class. Transfer: probe fitted on "
        "one half of the held-out negatives (patient-level split) and the positives of one collection, then "
        "evaluated on the other half of the negatives against the positives of the other collection. Test: the "
        "two probes applied to the BIMCV part of the test set, whose positives all belong to the five-digit "
        "collection. 95% CI from 2000 patient-level bootstrap resamples; columns without an interval have none "
        "computed. Descriptive.",

    "Origen de la señal en la parte BIMCV del test ({0} imágenes). Clasificador de solo adquisición: regresión "
    "logística sobre 14 estadísticos de sub-bandas de Haar fuera de la región torácica (Section 2.6); la segunda "
    "fila usa la clase del test y no es una referencia sin información de clase. Diferencia con ERM: AUC de la "
    "fila menos la de ERM ({1}, media de cinco semillas). ERM con máscara y el detector enmascarado son medias "
    "de cinco semillas. IC95 de 2 000 remuestreos de pacientes. Descriptivo.":
        "Origin of the signal in the BIMCV part of the test set ({0} images). Acquisition-only classifier: "
        "logistic regression on 14 Haar sub-band statistics from outside the thoracic region (Section 2.6); the "
        "second row uses the class labels of the test set and is not a class-blind reference. Difference from "
        "ERM: AUC of the row minus that of ERM ({1}, mean of five seeds). The masked ERM and masked detector "
        "rows are means of five seeds. 95% CI from 2000 patient-level bootstrap resamples. Descriptive.",

    "Cohorte externa COVIDGR ({0} imágenes, una por unidad de remuestreo, sin identificador de paciente). AUC "
    "interna: parte BIMCV del test ({1} imágenes). Caída: ecuación (8), con los dos términos remuestreados por "
    "separado (2 000 remuestreos de pacientes en la parte BIMCV y de imágenes en COVIDGR). Sensibilidad y "
    "especificidad al umbral de Youden de la parte BIMCV de la validación de cada semilla, medias sobre las "
    "cinco semillas (en las referencias, al umbral de su propia función de decisión); sin intervalo calculado. "
    "Ningún modelo se ajustó, seleccionó ni recalibró con COVIDGR. Descriptivo.":
        "External cohort COVIDGR ({0} images, each one a resampling unit, with no patient identifier). Internal "
        "AUC: BIMCV part of the test set ({1} images). Drop: Eq. (8), with the two terms resampled separately "
        "(2000 resamples, of patients in the BIMCV part and of images in COVIDGR). Sensitivity and specificity "
        "at the Youden threshold of the BIMCV part of the validation set of each seed, averaged over the five "
        "seeds (for the references, at the threshold of their own decision function), with no interval computed. "
        "No model was fitted, selected or recalibrated on COVIDGR. Descriptive.",

    "Sensibilidad en COVIDGR por gravedad radiológica de las positivas, al umbral de Youden de la validación de "
    "cada semilla (media de cinco semillas). Positivas por grupo, en el orden de las columnas: {0}, {1}, {2} y "
    "{3}; en total {4}, porque una positiva de COVIDGR no tiene gravedad anotada (su fila de metadatos no casa "
    "con ningún fichero). Descriptivo, sin intervalos.":
        "Sensitivity on COVIDGR by radiological severity of the positive images, at the Youden threshold of the "
        "validation set of each seed (mean of five seeds). Positives per group, in column order: {0}, {1}, {2} "
        "and {3}. The total is {4} because one COVIDGR positive has no severity annotation (its metadata row "
        "does not match any file). Descriptive, with no intervals.",

    "AUC en el test desconfundido del benchmark de derrame pleural por nivel de confusión, con el estado elegido "
    "con la validación de la misma $\\rho$: media por semilla (ecuación (2)) con IC95 jerárquico de semillas y "
    "pacientes (ecuaciones (5) a (7); 2 000 remuestreos). {0} semillas por nivel y {1} en $\\rho$ de 0.9, 0.95 y "
    "0.99. Es la Tabla 6 del artículo con sus intervalos. DFR no está definido con $\\rho = 1$ (Section 2.4).":
        "AUC on the deconfounded test set of the pleural effusion benchmark by confounding level, with the "
        "checkpoint selected by same-$\\rho$ validation: mean over seeds (Eq. (2)) with hierarchical "
        "seed-and-patient 95% CI (Eqs. (5) to (7), 2000 resamples). There are {0} seeds per level and {1} at "
        "$\\rho$ of 0.9, 0.95 and 0.99. This is Table 6 of the article with its intervals. DFR is not defined at "
        "$\\rho = 1$ (Section 2.4).",

    "Comparación completa con ERM en el test desconfundido del benchmark de derrame pleural (estado elegido con "
    "la validación de la misma $\\rho$). Diferencia: método menos ERM, media por semilla, con IC95 jerárquico de "
    "semillas y pacientes (ecuaciones (5) a (7); 2 000 remuestreos). p unilateral del mismo remuestreo, sin "
    "corregir: de que la diferencia sea mayor que cero en el área y en cada nivel, y de la ecuación (4) con δ = "
    "0.01 en la no inferioridad. Holm dentro de cada familia de la Tabla 4 al nivel unilateral de 0.025: área "
    "({0} comparaciones, confirmatoria), $\\rho = 0.99$ ({1}), $\\rho = 1.0$ ({2}, sin DFR) y no inferioridad en "
    "$\\rho = 0.5$ ({3}), secundarias. Los niveles de 0.6 a 0.95 y la diferencia en la caída son descriptivos. "
    "DFR no está definido con $\\rho = 1$, y su cifra con $\\rho = 0.99$ se marcó de antemano como inestable "
    "(Section 2.4). Cuatro decimales en el área.":
        "Full comparison with ERM on the deconfounded test set of the pleural effusion benchmark (checkpoint "
        "selected by same-$\\rho$ validation). Difference: method minus ERM, mean over seeds, with hierarchical "
        "seed-and-patient 95% CI (Eqs. (5) to (7), 2000 resamples). The one-sided p values come from the same "
        "bootstrap and are uncorrected. They test a difference greater than zero in the area and at each level, "
        "and use Eq. (4) with δ = 0.01 for non-inferiority. Holm correction within each family of Table 4 at the "
        "one-sided 0.025 level: area ({0} comparisons, confirmatory), and $\\rho = 0.99$ ({1}), $\\rho = 1.0$ "
        "({2}, without DFR) and non-inferiority at $\\rho = 0.5$ ({3}), all secondary. The levels from 0.6 to "
        "0.95 and the difference in the drop are descriptive. DFR is not defined at $\\rho = 1$, and its estimate "
        "at $\\rho = 0.99$ was flagged in advance as unstable (Section 2.4). The area is given to four decimals.",

    "AUC en el test invertido del benchmark de derrame pleural (prevalencia $1 - \\rho$ en NIH y $\\rho$ en "
    "CheXpert), con los mismos modelos que la tabla de la curva. Media por semilla con IC95 jerárquico de "
    "semillas y pacientes. Descriptivo.":
        "AUC on the flipped test set of the pleural effusion benchmark (prevalence $1 - \\rho$ in NIH and $\\rho$ "
        "in CheXpert), with the same models as the dose-response curve table. Mean over seeds with hierarchical "
        "seed-and-patient 95% CI. Descriptive.",

    "AUC dentro de cada fuente del test desconfundido del benchmark de derrame pleural (cada fuente aporta la "
    "mitad del test, equilibrada en clase). Media por semilla con IC95 jerárquico de semillas y pacientes. "
    "Descriptivo.":
        "AUC within each source of the deconfounded test set of the pleural effusion benchmark (each source "
        "contributes half of the test set, balanced by class). Mean over seeds with hierarchical "
        "seed-and-patient 95% CI. Descriptive.",

    "Brecha de especificidad en el test desconfundido del benchmark de derrame pleural: especificidad en "
    "CheXpert menos especificidad en NIH, al umbral de Youden de la validación de la misma $\\rho$ de cada "
    "semilla (en DFR, de su subconjunto de selección). Un valor positivo indica que el modelo clasifica como positivos "
    "más negativos de la fuente asociada a la clase positiva. Media por semilla con IC95 jerárquico. Descriptivo.":
        "Specificity gap on the deconfounded test set of the pleural effusion benchmark: specificity in CheXpert "
        "minus specificity in NIH, at the Youden threshold of the same-$\\rho$ validation set of each seed (for "
        "DFR, of its selection subset). A positive value indicates that the model classifies more negatives from "
        "the source associated with the positive class as positive. Mean over seeds with hierarchical 95% CI. "
        "Descriptive.",

    "Orientación inversa del benchmark de derrame pleural, con CheXpert como fuente positiva ({0} semillas: {1}; "
    "configuración de cada nivel tomada de la orientación principal). AUC media por semilla en el test "
    "desconfundido con IC95 jerárquico. Caída de ERM entre $\\rho = 0.5$ y 1.0: {2} en la orientación principal y "
    "{3} en la inversa; diferencia, {4}, frente al umbral preespecificado de {5}. Descriptivo. DFR no está "
    "definido con $\\rho = 1$.":
        "Reverse orientation of the pleural effusion benchmark, with CheXpert as the positive source ({0} seeds: "
        "{1}; configuration of each level taken from the primary orientation). Mean AUC over seeds on the "
        "deconfounded test set with hierarchical 95% CI. Drop of ERM from $\\rho = 0.5$ to 1.0: {2} in the "
        "primary orientation and {3} in the reverse one, a difference of {4} against the prespecified threshold "
        "of {5}. Descriptive. DFR is not defined at $\\rho = 1$.",

    "Evaluación en tuberculosis: AUC por cohorte de test (Shenzhen y Montgomery) y diferencia frente a ERM, "
    "media de {0} semillas. La unidad de remuestreo es la imagen; IC95 jerárquico de semillas e imágenes. p "
    "unilateral sin corregir, de que la diferencia sea mayor que cero, y p de Holm dentro de cada cohorte ({1} "
    "comparaciones, familias secundarias de la Tabla 4, nivel unilateral de 0.025).":
        "Tuberculosis evaluation: AUC by test cohort (Shenzhen and Montgomery) and difference relative to ERM, "
        "mean of {0} seeds. The resampling unit is the image, with a hierarchical 95% CI over seeds and images. "
        "Uncorrected one-sided p for a difference greater than zero, and Holm-adjusted p within each cohort ({1} "
        "comparisons, secondary families of Table 4, one-sided 0.025 level).",

    "Pérdida por la selección de época (ecuación (10)) por método y nivel en el test desconfundido del benchmark "
    "de derrame pleural: AUC con el estado que elige la validación desconfundida menos AUC con el que elige la "
    "validación de la misma $\\rho$, media por semilla con IC95 jerárquico. En DFR se usan los estados de su "
    "etapa ERM, de modo que está definido también con $\\rho = 1$. La última fila es el estimando de la Tabla 8 "
    "(cuatro decimales), confirmatorio en $\\rho$ de 0.9, 0.95 y 0.99 (Holm sobre tres); el resto es descriptivo.":
        "Epoch-selection loss (Eq. (10)) by method and level on the deconfounded test set of the pleural "
        "effusion benchmark: AUC with the checkpoint selected by deconfounded validation minus AUC with the "
        "checkpoint selected by same-$\\rho$ validation, mean over seeds with hierarchical 95% CI. For DFR the "
        "checkpoints of its ERM stage are used, so the loss is also defined at $\\rho = 1$. The last row is the "
        "estimand of Table 8 (four decimals), confirmatory at $\\rho$ of 0.9, 0.95 and 0.99 (Holm correction over "
        "three levels), and the rest is descriptive.",

    "Pérdida por la selección de época en la escala del área, en el test desconfundido del benchmark de derrame "
    "pleural. Primera fila: la pérdida media de los once métodos en cada nivel (ecuación (10)), integrada con "
    "los mismos pesos de trapecio que el área de la ecuación (9) entre $\\rho = 0.5$ y 0.99 y, en la tercera "
    "columna, solo entre 0.9 y 0.99, normalizada por la longitud de ese tramo. Filas siguientes: la diferencia "
    "de área de cada método con ERM en los mismos dos tramos, para compararla en la misma escala. IC95 "
    "jerárquico de semillas y pacientes con el factor corregido (ecuaciones (5) a (7); 2 000 remuestreos). Es un "
    "cálculo complementario hecho después de abrir los tests, descriptivo y sin contraste. Cuatro decimales.":
        "Epoch-selection loss on the scale of the area under the dose-response curve, on the deconfounded test "
        "set of the pleural effusion benchmark. First row: the mean loss of the eleven methods at each level "
        "(Eq. (10)), integrated with the same trapezoidal weights as the area of Eq. (9) between $\\rho = 0.5$ "
        "and 0.99 and, in the third column, only between 0.9 and 0.99, normalized by the length of that "
        "interval. Remaining rows: the area difference between each method and ERM over the same two intervals, "
        "for comparison on the same scale. Hierarchical seed-and-patient 95% CI with the corrected calibration "
        "factor (Eqs. (5) to (7), 2000 resamples). This is a post hoc supplementary calculation made after the "
        "test sets were opened, descriptive and without hypothesis testing. Four decimals.",

    "Selección de hiperparámetros, repetida tras la congelación con la semilla 0 y {0} ensayos por método y "
    "nivel, con evaluación del test desconfundido del benchmark de derrame pleural (ecuación (11)). Ensayos "
    "numerados de 0 a 7; el 0 lleva los valores por defecto. G: AUC del ensayo de la regla preespecificada menos "
    "la media de los ocho; L: AUC del ensayo que elegiría la validación desconfundida menos la del ensayo de la "
    "regla. Con una semilla, los IC95 remuestrean solo pacientes (2 000 remuestreos) y no recogen la variación "
    "entre semillas. Correlación de Spearman entre los ocho ensayos. Descriptivo.":
        "Hyperparameter selection, repeated after the freeze with seed 0 and {0} trials per method and level, "
        "with evaluation on the deconfounded test set of the pleural effusion benchmark (Eq. (11)). Trials are "
        "numbered from 0 to 7, and trial 0 uses the default values. G: AUC of the trial chosen by the "
        "prespecified rule minus the mean of the eight trials. L: AUC of the trial that deconfounded validation "
        "would choose minus that of the rule trial. With a single seed, the 95% CIs resample patients only (2000 "
        "resamples) and do not capture variation across seeds. Spearman correlation across the eight trials. "
        "Descriptive.",

    "Estabilidad de la búsqueda de hiperparámetros en el benchmark de derrame pleural: la búsqueda de {0} "
    "ensayos se repitió antes de la congelación con la semilla 1 (manifiesto e inicialización), sin cambiar las "
    "configuraciones. Spearman: correlación entre las métricas de selección de los ensayos en las dos búsquedas. "
    "Cambio de AUC: AUC en la validación desconfundida del ensayo que elige la semilla 1 menos la del que elige "
    "la semilla 0, medida con las corridas de cada semilla; IC95 de 2 000 remuestreos de pacientes. Solo "
    "validación; descriptivo.":
        "Stability of the hyperparameter search in the pleural effusion benchmark: the {0}-trial search was "
        "repeated before the freeze with seed 1 (manifest and initialization), without changing the "
        "configurations. Spearman: correlation between the selection metrics of the trials in the two searches. "
        "Change in AUC: AUC on the deconfounded validation set of the trial selected with seed 1 minus that of "
        "the trial selected with seed 0, measured with the runs of each seed, with 95% CI from 2000 "
        "patient-level bootstrap resamples. Validation only and descriptive.",

    "Pérdida por la selección de época en tuberculosis (ecuación (10)): AUC en cada cohorte de test con el "
    "estado que elige la parte de TBX11K de la validación (estado de oráculo) menos la AUC con el estado que "
    "elige la validación (selección principal), media de {0} semillas con IC95 jerárquico de semillas e imágenes "
    "(2 000 remuestreos). Semillas a favor del oráculo: semillas en que el estado de oráculo da mayor AUC. "
    "Prueba t pareada unilateral entre semillas, de que la pérdida sea mayor que cero ({1} grados de libertad). "
    "Un cero indica que las dos validaciones eligen la misma época en todas las semillas. Descriptivo, sin "
    "familia de contraste.":
        "Epoch-selection loss in tuberculosis (Eq. (10)): AUC in each test cohort with the checkpoint selected "
        "by the TBX11K part of the validation set (oracle checkpoint) minus the AUC with the checkpoint selected "
        "by the validation set (primary selection), mean of {0} seeds with hierarchical 95% CI over seeds and "
        "images (2000 resamples). Seeds favoring the oracle: seeds in which the oracle checkpoint gives a higher "
        "AUC. One-sided paired t test across seeds for a loss greater than zero ({1} degrees of freedom). A zero "
        "means that the two validation sets select the same epoch in every seed. Descriptive, with no test "
        "family.",

    "Segundo tronco (ResNet-50, en el benchmark de derrame pleural) y segunda etiqueta (cardiomegalia, con las "
    "mismas reservas de pacientes y tamaños multiplicados por 0.25): AUC en el test desconfundido con {0} "
    "semillas y los valores por defecto, media por semilla con IC95 jerárquico. DFR no está definido con $\\rho = "
    "1$. Análisis de sensibilidad sin contraste (Section 2.7).":
        "Second backbone (ResNet-50, in the pleural effusion benchmark) and second label (cardiomegaly, with the "
        "same patient reserves and sizes multiplied by 0.25): AUC on the deconfounded test set with {0} seeds "
        "and the default values, mean over seeds with hierarchical 95% CI. DFR is not defined at $\\rho = 1$. "
        "Sensitivity analysis without hypothesis testing (Section 2.7).",

    "Segundo tronco y segunda etiqueta: caída de ERM entre $\\rho = 0.5$ y 1.0 y diferencias de cada método "
    "frente a ERM en el test desconfundido (media por semilla, IC95 jerárquico). Tres semillas y valores por "
    "defecto; las diferencias no se contrastan y el plan congelado solo prevé la caída de ERM y el signo de las "
    "diferencias en $\\rho$ de 0.9 y 0.99 (Section 2.7). El área no se calcula porque la curva solo tiene cuatro "
    "niveles. La Tabla 9 del artículo resume la concordancia con el análisis principal.":
        "Second backbone and second label: drop of ERM from $\\rho = 0.5$ to 1.0 and difference of each method "
        "relative to ERM on the deconfounded test set (mean over seeds, hierarchical 95% CI). Three seeds and "
        "default values. The differences are not tested, and the frozen plan specifies only the drop of ERM and "
        "the sign of the differences at $\\rho$ of 0.9 and 0.99 (Section 2.7). The area is not computed because "
        "the curve has only four levels. Table 9 of the article summarizes the agreement with the main analysis.",

    "Selección de oráculo (análisis de sensibilidad preespecificado, Section 2.7): AUC en el test desconfundido "
    "del benchmark de derrame pleural con el estado de la época que elige la validación desconfundida. Media por "
    "semilla con IC95 jerárquico; mismas semillas que con la selección principal. No es un resultado principal: "
    "usa una validación que en la práctica no existe. La celda de DFR con $\\rho = 1.0$ se explica en la nota (a) "
    "de la [[t_oraculo_comp]].":
        "Oracle selection (prespecified sensitivity analysis, Section 2.7): AUC on the deconfounded test set of "
        "the pleural effusion benchmark with the checkpoint of the epoch selected by deconfounded validation. "
        "Mean over seeds with hierarchical 95% CI, using the same seeds as the primary selection. This is not a "
        "main result, because it relies on a validation set that does not exist in practice. The DFR cell at "
        "$\\rho = 1.0$ is explained in note (a) of [[t_oraculo_comp]].",

    "Comparación con ERM con la selección de oráculo en el test desconfundido del benchmark de derrame pleural. "
    "Misma estructura, remuestreo y corrección que la comparación con la selección principal "
    "([[t_comparacion]]); la corrección de Holm se aplica a las mismas familias (área: {0} comparaciones; $\\rho "
    "= 0.99$: {1}; $\\rho = 1.0$: {2}; no inferioridad: {3}), pero todas son de sensibilidad: el protocolo no "
    "admite la selección de oráculo como resultado principal ni como familia confirmatoria.":
        "Comparison with ERM under oracle selection on the deconfounded test set of the pleural effusion "
        "benchmark. The structure, bootstrap and correction are those of the comparison under the primary selection "
        "([[t_comparacion]]). Holm correction is applied to the same families ({0} comparisons for the area, {1} "
        "at $\\rho = 0.99$, {2} at $\\rho = 1.0$ and {3} for non-inferiority), but all four are sensitivity "
        "analyses, since the protocol does not admit oracle selection as a main result or as a confirmatory "
        "family.",

    "(a) Con la selección de oráculo, DFR parte del estado de la época que elige una mitad de la validación "
    "desconfundida y reajusta la última capa con la otra mitad, eligiendo la regularización en la primera. Esa "
    "validación tiene las cuatro celdas clase por fuente en todos los niveles, de modo que este DFR está "
    "definido también con $\\rho = 1.0$, a diferencia del DFR principal, que reajusta con la validación de la "
    "misma $\\rho$, sin celdas minoritarias en ese nivel (Section 2.4). Usa etiquetas desconfundidas que los "
    "demás métodos no ven en el entrenamiento; por eso su familia de $\\rho = 1.0$ tiene una comparación más que "
    "la principal.":
        "(a) Under oracle selection, DFR starts from the checkpoint of the epoch selected by one half of the "
        "deconfounded validation set and refits the last layer on the other half, choosing the regularization on "
        "the first half. That validation set contains the four class-by-source cells at every level, so this DFR "
        "is also defined at $\\rho = 1.0$. The main DFR, in contrast, refits on the same-$\\rho$ validation set, "
        "which has no minority cells at that level (Section 2.4). The oracle DFR uses deconfounded labels that "
        "the other methods do not see during training, which is why its $\\rho = 1.0$ family has one more "
        "comparison than the main one.",

    "Sensibilidades preespecificadas de la auditoría de COVIDx CXR-4 (Section 2.7). Referencia principal: "
    "detector de colección sobre la red de ImageNet; referencia secundaria: el mismo detector sobre los troncos "
    "finales de ERM, emparejados por semilla con cada método (su AUC es la media de los cinco troncos). Para "
    "cada variante se da el número de los {0} métodos frente a los que la referencia es no inferior tras la "
    "corrección de Holm al nivel unilateral de 0.025, con el margen preespecificado (δ = 0.02) y con δ = 0.01, y "
    "la diferencia de AUC entre la referencia y ERM y entre la referencia y DFR (media sobre las cinco semillas "
    "del método, ecuación (3)), con IC95 de percentiles. Cada variante forma, para cada referencia, una familia "
    "de {0} comparaciones con la estructura de las dos primeras filas de la Tabla 4. Conjunto: parte BIMCV del "
    "test ({1} imágenes de {2} pacientes), salvo en el test limpio completo ({3} imágenes). Unidad de "
    "remuestreo: el paciente, con 2 000 remuestreos (semilla 2026), salvo en la variante de remuestreo por "
    "imagen. En la variante del conjunto, cada método está representado por la media de las probabilidades de "
    "sus cinco semillas. La variante de remuestreo por paciente coincide con el análisis principal y no se "
    "repite.":
        "Prespecified sensitivity analyses of the COVIDx CXR-4 audit (Section 2.7). Primary reference: "
        "collection detector on the ImageNet network. Secondary reference: the same detector on the final ERM "
        "backbones, paired by seed with each method (its AUC is the mean over the five backbones). For each "
        "variant, the table gives the number of the {0} methods to which the reference is non-inferior after "
        "Holm correction at the one-sided 0.025 level, with the prespecified margin (δ = 0.02) and with δ = "
        "0.01, and the AUC difference between the reference and ERM and between the reference and DFR (mean over "
        "the five seeds of the method, Eq. (3)), with percentile 95% CIs. For each reference, every variant "
        "forms a family of {0} comparisons with the structure of the first two rows of Table 4. Set: BIMCV part "
        "of the test set ({1} images from {2} patients), except for the full clean test set ({3} images). "
        "Resampling unit: the patient, with 2000 resamples (seed 2026), except in the image-level bootstrap "
        "variant. In the ensemble variant, each method is represented by the mean of the probabilities of its "
        "five seeds. The patient-level bootstrap variant coincides with the main analysis and is not repeated.",

    "(b) El protocolo no prevé un conjunto de la referencia secundaria, cuyas puntuaciones son funciones de "
    "decisión de cinco detectores con escalas distintas.":
        "(b) The protocol does not provide for an ensemble of the secondary reference, whose scores are decision "
        "functions of five detectors with different scales.",

    "Intervalos alternativos preespecificados (Section 2.7) para la diferencia en el área de la ecuación (9) "
    "frente a ERM, la familia confirmatoria del benchmark de derrame pleural. Principal: remuestreo jerárquico "
    "de las ecuaciones (5) a (7). Pacientes con semillas fijas: 2 000 remuestreos de pacientes con los modelos "
    "fijos. Remuestreo doble: semillas y pacientes a la vez, sin la corrección de las ecuaciones (5) a (7), de "
    "modo que cuenta dos veces la interacción. Prueba t pareada entre las diferencias por semilla, con {0} "
    "grados de libertad; usa solo las semillas presentes en todos los niveles ({1}), por lo que su media puede "
    "diferir de la estimación principal, que usa todas las semillas de cada nivel. Valores p unilaterales de que "
    "la diferencia sea mayor que cero, sin corregir y con la corrección de Holm sobre las {2} comparaciones de "
    "la familia al nivel unilateral de 0.025, calculada por separado con cada intervalo. El plan no preveía "
    "valor p para el remuestreo doble; el suyo se calcula con la misma regla que el del intervalo principal, "
    "solo como lectura. Los valores p con los intervalos alternativos se calcularon después de abrir los tests. "
    "Cuatro decimales en las diferencias.":
        "Prespecified alternative intervals (Section 2.7) for the difference in the area of Eq. (9) relative to "
        "ERM, the confirmatory family of the pleural effusion benchmark. Main: hierarchical bootstrap of Eqs. "
        "(5) to (7). Patient-level interval with fixed seeds: 2000 patient-level resamples with the models held "
        "fixed. Double bootstrap: seeds and patients resampled together, without the correction of Eqs. (5) to "
        "(7), so that the interaction is counted twice. The t test across seeds is a paired test on the per-seed "
        "differences with {0} degrees of freedom. It uses only the seeds present at all levels ({1}), so its "
        "mean can differ from the main estimate, which uses all seeds at each level. One-sided p values for a "
        "difference greater than zero, uncorrected and after Holm correction over the {2} comparisons of the "
        "family at the one-sided 0.025 level, applied separately for each interval. The plan did not specify a p "
        "value for the double bootstrap; this one is computed with the same rule as for the main interval and is "
        "reported for information only. The p values for the alternative intervals were computed after the test "
        "sets were opened. Differences are given to four decimals.",

    "Con Holm al nivel unilateral de 0.025 superan a ERM {0} métodos con el intervalo principal, {1} con el de "
    "pacientes con semillas fijas, {2} con el remuestreo doble y {3} con la prueba t entre semillas. Con el "
    "intervalo de pacientes, el p de Holm de DANN es {4}, justo por debajo del umbral.":
        "After Holm correction at the one-sided 0.025 level, {0} methods exceed ERM with the main interval, {1} "
        "with the patient-level interval with fixed seeds, {2} with the double bootstrap and {3} with the t test "
        "across seeds. With the patient-level interval, the Holm-adjusted p of DANN is {4}, just below the "
        "threshold.",

    "Intervalos alternativos de la pérdida por la selección de época (ecuación (10)) en los tres niveles de la "
    "familia confirmatoria. Por método, el intervalo principal es el jerárquico y la prueba t pareada usa las "
    "{0} semillas del nivel ({1} grados de libertad). La media de los once métodos lleva cuatro decimales, como "
    "en la Tabla 8. Pacientes con semillas fijas: 2 000 remuestreos de pacientes con los modelos fijos. "
    "Remuestreo doble: semillas y pacientes a la vez, sin la corrección de las ecuaciones (5) a (7). Estos dos "
    "intervalos y la prueba t de la media se calcularon después de abrir los tests con las funciones congeladas "
    "del análisis principal, por estar previstos en el plan congelado (Section 2.7). En la media de los once "
    "métodos, las dos últimas columnas dan los valores p unilaterales de que la pérdida sea mayor que cero con "
    "cada intervalo y los de Holm sobre los tres niveles, al nivel unilateral de 0.025; el valor p del "
    "remuestreo doble, que el plan no preveía, se da solo como lectura. Por método no se calculan.":
        "Alternative intervals for the epoch-selection loss (Eq. (10)) at the three levels of the confirmatory "
        "family. For each method, the main interval is the hierarchical one, and the paired t test uses the {0} "
        "seeds of the level ({1} degrees of freedom). The mean of the eleven methods is given to four decimals, "
        "as in Table 8. Patient-level interval with fixed seeds: 2000 patient-level resamples with the models "
        "held fixed. Double bootstrap: seeds and patients resampled together, without the correction of Eqs. (5) "
        "to (7). These two intervals and the t test of the mean were computed after the test sets were opened, "
        "with the frozen functions of the main analysis, because the frozen plan specifies them (Section 2.7). "
        "For the mean of the eleven methods, the last two columns give the one-sided p values for a loss greater "
        "than zero with each interval and the Holm-adjusted values over the three levels, at the one-sided 0.025 "
        "level. The p value of the double bootstrap, which the plan did not specify, is reported for information "
        "only. These p values are not computed for individual methods.",

    "Familia confirmatoria de la auditoría de COVIDx CXR-4 ([[t_fc1]]) con el intervalo de semillas y pacientes: "
    "referencia principal, fija, frente a cada uno de los {0} métodos con sus cinco semillas en la parte BIMCV "
    "del test, con la AUC de la referencia repetida en cada semilla. Diferencia: AUC de la referencia menos la "
    "del método, media sobre las semillas (ecuación (3)). IC95 de pacientes: el del análisis confirmatorio. IC95 "
    "de semillas y pacientes: remuestreo jerárquico de las ecuaciones (5) a (7), con 2 000 remuestreos; con un "
    "solo nivel y un solo estrato, el factor corregido y el original coinciden. Desviación de semillas: raíz de "
    "la varianza del efecto de semilla de la ecuación (5), descontada la interacción; desviación de pacientes: "
    "raíz de la varianza por pacientes de la media. Valores p de Holm de no inferioridad (ecuación (4)) con el "
    "intervalo de semillas y pacientes, sobre las {0} comparaciones de cada margen al nivel unilateral de 0.025. "
    "Con ese intervalo la referencia es no inferior frente a {1} métodos con δ = 0.02 y frente a {2} con δ = "
    "0.01. Es un cálculo complementario hecho después de abrir los tests; la prueba confirmatoria sigue siendo "
    "la del intervalo de pacientes.":
        "Confirmatory family of the COVIDx CXR-4 audit ([[t_fc1]]) with the seed-and-patient interval: the fixed "
        "primary reference against each of the {0} methods with its five seeds on the BIMCV part of the test "
        "set, with the AUC of the reference repeated for every seed. Difference: AUC of the reference minus that "
        "of the method, averaged over seeds (Eq. (3)). Patient-level 95% CI: the one from the confirmatory "
        "analysis. Seed-and-patient 95% CI: hierarchical bootstrap of Eqs. (5) to (7) with 2000 resamples. With "
        "a single level and a single stratum, the corrected and the original calibration factors coincide. Seed "
        "standard deviation: square root of the variance of the seed effect in Eq. (5), after removing the "
        "interaction. Patient standard deviation: square root of the patient-level variance of the mean. "
        "Holm-adjusted non-inferiority p values (Eq. (4)) with the seed-and-patient interval, over the {0} "
        "comparisons for each margin at the one-sided 0.025 level. With this interval the reference is "
        "non-inferior to {1} methods with δ = 0.02 and to {2} with δ = 0.01. This is a post hoc supplementary "
        "calculation made after the test sets were opened, and the confirmatory test remains the one based on "
        "the patient-level interval.",
}
EXACTO.update(LEYENDAS)
EXACTO.update({k: k for k in IDENTIDAD})


def _t(s):
    r = traducir(s)
    if r is None:
        raise KeyError(s)
    return r


REGLAS = [
    (r"\$\\rho = [0-9.]+\$", lambda m: m.group(0)),
    (r"[0-9.]+", lambda m: m.group(0)),
    (r"AUC, \$\\rho = [0-9.]+\$", lambda m: m.group(0)),
    (r"(.+) frente a ERM", lambda m: _t(m.group(1)) + " vs. ERM"),
    (r"Referencia secundaria, tronco de ERM, semilla (\d+)", lambda m: "Secondary reference, ERM backbone, seed " + m.group(1)),
    (r"Tronco de ERM con máscara, semilla (\d+)", lambda m: "Masked ERM backbone, seed " + m.group(1)),
    (r"Tronco con conjunto reservado \(todos los negativos\), semilla (\d+)",
     lambda m: "Backbone with held-out set (all negatives), seed " + m.group(1)),
    (r"Tronco con conjunto reservado \(20 % de los pacientes negativos\), semilla (\d+)",
     lambda m: "Backbone with held-out set (20% of the negative patients), seed " + m.group(1)),
    (r"20 % de los pacientes negativos reservados, semilla (\d+)",
     lambda m: "20% of the negative patients held out, seed " + m.group(1)),
    (r"Todos los negativos reservados, semilla (\d+)", lambda m: "All negatives held out, seed " + m.group(1)),
]


def traducir(s):
    if s in EXACTO:
        return EXACTO[s]
    for rx, f in REGLAS:
        m = re.fullmatch(rx, s)
        if m:
            try:
                return f(m)
            except KeyError:
                return None
    return None
