#!/usr/bin/env python3
"""Material suplementario del artículo: tablas suplementarias desde los resultados ya calculados.

    python3 scripts/suplemento/tablas.py              # escribe results/suplemento/tabla_SNN.csv, trazas.json
                                                      # y manuscript/es/07_suplemento.md
    python3 scripts/suplemento/tablas.py --verificar  # relee cada tabla escrita y la coteja con su fuente
    python3 scripts/suplemento/tablas.py --idioma en  # escribe solo manuscript/en/07_supplementary_material.md
    python3 scripts/suplemento/tablas.py --idioma en --verificar

Idioma. Por defecto (--idioma es) el programa hace exactamente lo de siempre. Con --idioma en construye las mismas
tablas desde las mismas fuentes y traduce leyendas, cabeceras, notas y celdas de texto con el diccionario de
scripts/suplemento/traduccion_en.py (manuscript/en/GLOSARIO.md); las cifras no se teclean. Solo escribe el Markdown
inglés: los CSV y trazas de results/suplemento son los del español. La verificación inglesa coteja además cada cifra
de cada celda, leyenda y nota con la de la versión española.

Qué lee. Solo resultados ya calculados: CSV y JSON de results/ (tablas de los informes previstos en el plan congelado,
hiperparámetros congelados, ficheros de control de las cadenas y del entorno) y los CSV intermedios de
results/suplemento_fuentes/ (scripts/suplemento/solape_segmentador.py).

Intervalos corregidos. El factor del término de semillas de los intervalos jerárquicos (ecuación (7) del manuscrito)
se corrigió después de abrir los tests, y las tablas recalculadas están en results/correccion_ec6/ con la misma ruta
relativa, el mismo nombre y las mismas columnas que las originales. Toda lectura pasa por Fuentes.ruta(): si existe la
versión corregida de un fichero, se lee esa, y la traza de cada celda apunta a ella. Para el espacio de búsqueda lee como texto, con `ast`, los
atributos literales `defaults` y `search` de src/baselines/methods/*.py y `COMMON_SEARCH` de
src/baselines/registry.py, sin importar ni ejecutar código del proyecto. No carga predicciones ni características
de ningún conjunto y no calcula estadísticos: selecciona, redondea y, en casos declarados en la celda, resta, cuenta
o suma (recuento de corridas, positivas con gravedad, porcentajes de proyección AP).

Verificación (--verificar). (1) Reconstruye todo el documento con lecturas nuevas de las fuentes y exige que cada
CSV de results/suplemento y el Markdown coincidan carácter a carácter con lo escrito. (2) Para cada celda con cifra,
comprueba que el filtro de su traza selecciona exactamente una fila de la fuente (o ninguna en las celdas "no
aplica") y que la cifra impresa difiere del valor crudo menos de media unidad de su último decimal. (3) Comprueba que
el Markdown no contiene guiones largos, símbolos de sección ni identificadores internos del proyecto.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "results"
OUT = RES / "suplemento"
MD = RAIZ / "manuscript" / "es" / "07_suplemento.md"
CORR = "correccion_ec6/"           # tablas recalculadas con el factor corregido (misma ruta relativa)
MENOS = "−"
IDIOMA = "es"                      # "en" con --idioma en
MD_EN = RAIZ / "manuscript" / "en" / "07_supplementary_material.md"
_FALTAN = None                     # si es un conjunto, T() anota en él las cadenas sin traducción en vez de fallar


def T(s):
    """Traducción al inglés de una cadena de texto (identidad en español)."""
    if IDIOMA == "es" or not isinstance(s, str):
        return s
    import traduccion_en as TR
    r = TR.traducir(s)
    if r is None:
        if _FALTAN is not None:
            _FALTAN.add(s)
            return s
        raise KeyError("sin traducción al inglés: %r" % s)
    return r


def A():
    """Conector de los intervalos: "a" en español, "to" en inglés."""
    return " a " if IDIOMA == "es" else " to "

MET = ["erm", "control_fuente", "groupdro", "dann", "coral", "mixstyle", "fourier", "lisa", "dfa_fuente", "dfr",
       "erm_mascara", "erm_cuotas"]
MET_B3 = [m for m in MET if m != "control_fuente"]
MET_B3_SIN_ERM = [m for m in MET_B3 if m != "erm"]
NOMBRE = {"erm": "ERM", "control_fuente": "Control de fuente", "groupdro": "GroupDRO", "dann": "DANN",
          "coral": "CORAL", "mixstyle": "MixStyle", "fourier": "Mezcla de amplitud de Fourier", "lisa": "LISA",
          "dfa_fuente": "DFA con fuente", "dfr": "DFR", "erm_mascara": "ERM con máscara",
          "erm_cuotas": "ERM con cuotas", "D_IN": "Referencia principal (red de ImageNet)",
          "D_ERM": "Referencia secundaria (troncos de ERM)"}
RHOS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
ETQ_RHO = {0.5: "0.5", 0.6: "0.6", 0.7: "0.7", 0.8: "0.8", 0.9: "0.9", 0.95: "0.95", 0.99: "0.99", 1.0: "1.0"}
INTERNOS = ["B1", "B2", "B3a", "B3c", "D_IN", "D_ERM", "D_MASK", "D_RES", "bestdc", "dfr_best", "val_robust",
            "val_desconf", "test_desconf", "test_bimcv", "F-C1", "F-C2", "F-C3", "S-T", "S-E", "S-B", "Puerta",
            "plan B", "sub" "agente", "super" "visor", "docs/", "results/"]
# Vocabulario retirado el 21-11-2026 (solo en la versión española).
RETIRADAS_ES = [r"registr", r"[Rr]éplica", r"\bantigu", r"\bnuev[oa]s?\b"]
INTERNOS_RE = [r"\bC[123]\b", r"\bC1-", r"\bE[1-8]\b"]


def med(r):
    return "rho%03d" % int(round(r * 100))


# ===================================================================================== lectura de fuentes
class Fuentes:
    """Caché de lecturas; en la verificación se crea una nueva para releer todo de disco."""

    def __init__(self):
        self.c = {}

    @staticmethod
    def ruta(rel):
        """Ruta relativa a results/ que se lee de verdad: la versión corregida si existe, la original si no."""
        if not isinstance(rel, str) or rel.startswith(CORR):
            return rel
        return CORR + rel if (RES / CORR / rel).is_file() else rel

    def csv(self, rel):
        rel = self.ruta(rel)
        if rel not in self.c:
            self.c[rel] = pd.read_csv(RES / rel)
        return self.c[rel]

    def json(self, rel):
        rel = self.ruta(rel)
        if rel not in self.c:
            self.c[rel] = json.loads((RES / rel).read_text(encoding="utf-8"))
        return self.c[rel]

    def ast_attrs(self, rel):
        """Atributos literales de las clases (name, defaults, search) y asignaciones de módulo, leídos con ast."""
        key = "ast:" + rel
        if key not in self.c:
            t = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
            mod, clases = {}, {}
            for n in t.body:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    try:
                        mod[n.targets[0].id] = ast.literal_eval(n.value)
                    except Exception:
                        pass
                if isinstance(n, ast.ClassDef):
                    at = {"_bases": [ast.unparse(b) for b in n.bases]}
                    for s in n.body:
                        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                            try:
                                at[s.targets[0].id] = ast.literal_eval(s.value)
                            except Exception:
                                pass
                    clases[n.name] = at
            self.c[key] = {"mod": mod, "clases": clases}
        return self.c[key]

    def metodos_registrados(self):
        """name -> {'defaults', 'search'} con la herencia simple entre clases del registro."""
        if "_met" in self.c:
            return self.c["_met"]
        clases = {}
        for f in sorted(glob.glob(str(RAIZ / "src/baselines/methods/*.py"))):
            rel = str(Path(f).relative_to(RAIZ))
            clases.update(self.ast_attrs(rel)["clases"])

        def attr(cn, a):
            c = clases.get(cn)
            if c is None:
                return None
            if a in c:
                return c[a]
            for b in c["_bases"]:
                v = attr(b, a)
                if v is not None:
                    return v
            return None

        out = {}
        for cn, c in clases.items():
            if "name" in c:
                out[c["name"]] = {"defaults": attr(cn, "defaults") or {}, "search": attr(cn, "search") or {}}
        self.c["_met"] = out
        return out


def filtrar(df, f):
    m = pd.Series(True, index=df.index)
    for k, v in f.items():
        if isinstance(v, float):
            m &= (pd.to_numeric(df[k], errors="coerce") - v).abs() < 1e-9
        else:
            m &= df[k].astype(str) == str(v)
    return df[m]


def una(F, src, f):
    d = filtrar(F.csv(src), f)
    if len(d) != 1:
        raise ValueError("%s %s: %d filas (se esperaba una)" % (src, f, len(d)))
    return d.iloc[0]


def existe(F, src, f, col=None):
    d = filtrar(F.csv(src), f)
    if not len(d):
        return False
    if col is not None and pd.isna(d.iloc[0][col]):
        return False
    return True


# ===================================================================================== formatos
def num(x, d=3):
    t = "%.*f" % (d, float(x))
    if float(t) == 0:
        t = t.lstrip("-")              # un valor que redondea a cero se imprime sin signo
    return t.replace("-", MENOS)


def ent(x):
    if IDIOMA == "en":                   # glosario: 8079, 37,123
        n = int(round(float(x)))
        return str(n) if abs(n) < 10000 else "{:,}".format(n)
    return "{:,}".format(int(round(float(x)))).replace(",", " ")


def pval(p):
    p = float(p)
    if p < 0.0001:
        return "< 0.0001"
    return "%.3f" % p if p >= 0.005 else "%.4f" % p


def sino(v):
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    if IDIOMA == "en":
        return "yes" if bool(v) else "no"
    return "sí" if bool(v) else "no"


def lr(x):
    x = float(x)
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    if abs(m - round(m)) < 1e-9:
        return "$%d \\times 10^{%d}$" % (round(m), e)
    return "$%.2f \\times 10^{%d}$" % (m, e)


def g3(v):
    if isinstance(v, bool):
        return sino(v)
    if isinstance(v, (int,)) or (isinstance(v, float) and abs(v - round(v)) < 1e-12):
        return str(int(round(v)))
    if isinstance(v, float):
        return ("%.3g" % v).replace("-", MENOS)
    return str(v)


def fmt(v, f):
    if f == "d3":
        return num(v, 3)
    if f == "d4":
        return num(v, 4)
    if f == "d2":
        return num(v, 2)
    if f == "d5":
        return num(v, 5)
    if f == "p":
        return pval(v)
    if f == "int":
        return ent(v)
    if f == "bool":
        return sino(v)
    if f == "lr":
        return lr(v)
    if f == "pct":
        return "%d" % round(100 * float(v))
    if f == "txt":
        return T(str(v))
    if f == "lista":
        x = str(v).split()
        return x[0] if len(x) == 1 else ", ".join(x[:-1]) + (" y " if IDIOMA == "es" else " and ") + x[-1]
    raise ValueError(f)


DEC = {"d2": 2, "d3": 3, "d4": 4, "d5": 5, "int": 0}


# ===================================================================================== celdas
def V(src, f, c, fm="d3"):
    return {"k": "v", "src": src, "f": f, "c": c, "fm": fm}


def IC(src, f, c, d=3):
    return {"k": "ic", "src": src, "f": f, "c": list(c), "d": d}


def IV(src, f, c, d=3):
    """Intervalo sin estimación: "inf a sup"."""
    return {"k": "iv", "src": src, "f": f, "c": list(c), "d": d}


def PS(src, f, cols, sep=" / "):
    """Varios valores p de una misma fila, separados por una barra."""
    return {"k": "ps", "src": src, "f": f, "c": list(cols), "sep": sep}


def NA(src, f, c=None, t="no aplica"):
    return {"k": "na", "src": src, "f": f, "c": c, "t": t}


def FR(src, f, a, b, sep=" de "):
    return {"k": "fr", "src": src, "f": f, "c": [a, b], "sep": sep}


def TRI(src, f, cols, sep=" / "):
    return {"k": "fr", "src": src, "f": f, "c": list(cols), "sep": sep}


def DIF(src, f, a, b, d=3):
    return {"k": "dif", "src": src, "f": f, "c": [a, b], "d": d}


def EQ(src, f, a, b):
    return {"k": "eq", "src": src, "f": f, "c": [a, b]}


def AGG(nombre, *args):
    return {"k": "agg", "n": nombre, "a": list(args)}


def CELDA_V_O_NA(F, src, f, c, fm="d3", t="no aplica"):
    return V(src, f, c, fm) if existe(F, src, f, c) else NA(src, f, c, t)


def IC_O_NA(F, src, f, c, d=3, t="no aplica"):
    return IC(src, f, c, d) if existe(F, src, f, c[0]) else NA(src, f, c[0], t)


# ---- agregados declarados (cuentas, sumas y valores de JSON)
def _cadenas(F, patron):
    pasos = {}
    for fp in sorted(glob.glob(str(RES / "cadenas" / patron))):
        j = F.json(str(Path(fp).relative_to(RES)))
        for k, v in j["pasos"].items():
            pasos[k] = v
    return pasos


def agg(F, n, a):
    if n == "cad_n":
        return ent(len(_cadenas(F, a[0])))
    if n == "cad_ok":
        return ent(sum(1 for v in _cadenas(F, a[0]).values() if v.get("codigo") == 0))
    if n == "cad_h":
        return num(sum(float(v.get("segundos") or 0) for v in _cadenas(F, a[0]).values()) / 3600.0, 1)
    if n == "suma":
        src, f, c = a
        return ent(filtrar(F.csv(src), f)[c].sum())
    if n == "nfilas":
        src, f = a
        return ent(len(filtrar(F.csv(src), f)))
    if n == "n_hasta":                    # filas con la columna menor o igual que un umbral (p.ej. Holm al 0.025)
        src, f, c, umbral = a
        d = filtrar(F.csv(src), f)
        return ent(int((pd.to_numeric(d[c]) <= umbral + 1e-12).sum()))
    if n == "ext":                        # mínimo o máximo de una columna, con su valor crudo para la verificación
        src, f, c, cual, d = a
        x = pd.to_numeric(filtrar(F.csv(src), f)[c])
        v = float(x.min() if cual == "min" else x.max())
        return num(v, d), [(v, d)]
    if n == "js":
        src, ruta, fm = a
        v = F.json(src)
        for k in ruta:
            v = v[k]
        if fm == "len":
            return ent(len(v))
        if fm == "lr":
            return lr(v)
        if fm == "int":
            return ent(v)
        if fm == "d3":
            return num(v, 3)
        return str(v)
    if n == "paquete":
        src, nombre = a
        for p in F.json(src)["paquetes"]:
            k, _, ver = p.partition("==")
            if k.lower() == nombre.lower():
                return ver
        raise KeyError(nombre)
    if n == "icstr":                      # IC guardado como texto "lo hi" en una sola columna
        src, f, ce, cs, d = a
        r = una(F, src, f)
        lo, hi = [float(x) for x in str(r[cs]).split()]
        return ("%s (%s" + A() + "%s)") % (num(r[ce], d), num(lo, d), num(hi, d))
    if n == "hp":                         # hiperparámetros buscados de un JSON congelado
        src, metodo = a
        return hp_texto(F, F.json(src), metodo)
    if n == "hp_lr":
        return lr(F.json(a[0])["lr_B"])
    if n == "hp_ensayo":
        return str(int(F.json(a[0])["trial"]))
    if n == "hp_sel":
        return num(F.json(a[0])["metrica_seleccion"], 3)
    if n == "espacio":
        metodo, clave = a
        return espacio_texto(F.metodos_registrados()[metodo]["search"][clave])
    if n == "defecto":
        metodo, clave = a
        return g3v(F.metodos_registrados()[metodo]["defaults"][clave])
    if n == "espacio_comun":
        spec = F.ast_attrs("src/baselines/registry.py")["mod"]["COMMON_SEARCH"]["lr_B"]
        return espacio_texto(spec)
    if n == "n_ensayos":
        return str(F.ast_attrs("src/baselines/registry.py")["mod"]["N_TRIALS"])
    if n == "rejilla_dfr":
        return ", ".join(g3(x) for x in F.metodos_registrados()["dfr"]["defaults"]["C_grid"])
    if n == "reps_dfr":
        return str(F.metodos_registrados()["dfr"]["defaults"]["n_reps"])
    raise ValueError(n)


NOMBRE_HP = {
    ("groupdro", "eta"): "η, tasa de actualización de los pesos de grupo",
    ("dann", "lambda"): "λ, peso del gradiente invertido",
    ("dann", "mlp_width"): "anchura del discriminador",
    ("dann", "mlp_depth"): "capas del discriminador",
    ("dann", "mlp_dropout"): "abandono del discriminador",
    ("dann", "disc_lr_mult"): "multiplicador de la tasa del discriminador",
    ("dann", "grad_penalty"): "peso de la penalización del gradiente",
    ("coral", "mmd_gamma"): "peso de la pérdida de alineación",
    ("mixstyle", "p"): "probabilidad de aplicar la mezcla",
    ("mixstyle", "alpha"): "parámetro de la distribución beta",
    ("fourier", "eta"): "η, cota del coeficiente de mezcla de amplitud",
    ("fourier", "ratio"): "lado de la región del espectro que se mezcla",
    ("fourier", "cons_weight"): "peso de la consistencia",
    ("lisa", "mix_alpha"): "parámetro de la distribución beta",
    ("lisa", "mix_ratio"): "probabilidad del modo dentro de la clase",
    ("dfa_fuente", "lambda_dis_align"): "peso de la pérdida de sesgo",
    ("dfa_fuente", "lambda_swap_align"): "peso de la pérdida de sesgo tras el intercambio",
    ("dfa_fuente", "lambda_swap"): "peso de las pérdidas tras el intercambio",
    ("dfa_fuente", "swap_start_epoch"): "época de la segunda fase en que empieza el intercambio",
    ("dfa_fuente", "peso"): "pesos por celda de la pérdida de contenido",
}
CORTO_HP = {
    ("groupdro", "eta"): "η", ("dann", "lambda"): "λ", ("dann", "mlp_width"): "anchura",
    ("dann", "mlp_depth"): "capas", ("dann", "mlp_dropout"): "abandono", ("dann", "disc_lr_mult"): "tasa del discriminador",
    ("dann", "grad_penalty"): "penalización", ("coral", "mmd_gamma"): "peso", ("mixstyle", "p"): "p",
    ("mixstyle", "alpha"): "α", ("fourier", "eta"): "η", ("fourier", "ratio"): "lado", ("fourier", "cons_weight"): "consistencia",
    ("lisa", "mix_alpha"): "α", ("lisa", "mix_ratio"): "dentro de la clase", ("dfa_fuente", "lambda_dis_align"): "sesgo",
    ("dfa_fuente", "lambda_swap_align"): "sesgo tras intercambio", ("dfa_fuente", "lambda_swap"): "intercambio",
    ("dfa_fuente", "swap_start_epoch"): "época de inicio", ("dfa_fuente", "peso"): "pesos",
}
VALOR_HP = {"inverso_p_y_dado_s": "1/P(y | s)", "ninguno": "ninguno"}


def g3v(v):
    return T(VALOR_HP.get(v, g3(v))) if isinstance(v, str) else g3(v)


def espacio_texto(spec):
    kind = spec[0]
    if kind == "log10":
        return "$10^{U(%s,\\,%s)}$" % (g3(spec[1]).replace(MENOS, "-"), g3(spec[2]).replace(MENOS, "-"))
    if kind == "uniform":
        return "U(%s, %s)" % (g3(spec[1]), g3(spec[2]))
    if kind == "choice":
        return "{" + ", ".join(g3v(x) for x in spec[1]) + "}"
    if kind == "int":
        return ("entero de %s a %s" if IDIOMA == "es" else "integer from %s to %s") % (spec[1], spec[2])
    raise ValueError(spec)


def hp_texto(F, j, metodo):
    if int(j.get("trial", -1)) == 0:
        return T("valores por defecto")
    search = F.metodos_registrados()[metodo]["search"]
    if not search:
        return T("solo la tasa")
    p = j.get("params") or {}
    partes = []
    for k in search:
        if k in p:
            partes.append("%s %s" % (T(CORTO_HP.get((metodo, k), k)), g3v(p[k])))
    return "; ".join(partes) if partes else T("valores por defecto")


# ---- resolución
def resolver(F, s):
    """Texto de la celda y lista de (valor crudo, decimales) para la comprobación de redondeo."""
    if isinstance(s, str):
        return s, []
    k = s["k"]
    if k == "v":
        r = una(F, s["src"], s["f"])
        v = r[s["c"]]
        if pd.isna(v):
            raise ValueError("valor vacío en %s %s %s" % (s["src"], s["f"], s["c"]))
        t = fmt(v, s["fm"])
        d = DEC.get(s["fm"])
        if s["fm"] == "p":
            d = 3 if float(v) >= 0.005 else 4
            if float(v) < 0.0001:
                return t, []
        return t, ([(float(v), d)] if d is not None else [])
    if k == "ic":
        r = una(F, s["src"], s["f"])
        vals = [r[c] for c in s["c"]]
        if any(pd.isna(x) for x in vals):
            raise ValueError("IC vacío en %s %s" % (s["src"], s["f"]))
        d = s["d"]
        return ("%s (%s" + A() + "%s)") % tuple(num(x, d) for x in vals), [(float(x), d) for x in vals]
    if k == "iv":
        r = una(F, s["src"], s["f"])
        vals = [r[c] for c in s["c"]]
        if any(pd.isna(x) for x in vals):
            raise ValueError("intervalo vacío en %s %s" % (s["src"], s["f"]))
        d = s["d"]
        return ("%s" + A() + "%s") % tuple(num(x, d) for x in vals), [(float(x), d) for x in vals]
    if k == "ps":
        r = una(F, s["src"], s["f"])
        txt, cr = [], []
        for c in s["c"]:
            v = float(r[c])
            txt.append(pval(v))
            if v >= 0.0001:
                cr.append((v, 3 if v >= 0.005 else 4))
            else:
                cr.append(None)
        if any(x is None for x in cr):          # "< 0.0001" no lleva cifra que cotejar en su posición
            return s["sep"].join(txt), []
        return s["sep"].join(txt), cr
    if k == "na":
        d = filtrar(F.csv(s["src"]), s["f"])
        if len(d) and (s["c"] is None or not pd.isna(d.iloc[0][s["c"]])):
            raise ValueError("celda 'no aplica' con valor en la fuente: %s %s" % (s["src"], s["f"]))
        return T(s["t"]), []
    if k == "fr":
        r = una(F, s["src"], s["f"])
        return T(s["sep"]).join(ent(r[c]) for c in s["c"]), [(float(r[c]), 0) for c in s["c"]]
    if k == "dif":
        r = una(F, s["src"], s["f"])
        v = float(r[s["c"][0]]) - float(r[s["c"][1]])
        return num(v, s["d"]), [(v, s["d"])]
    if k == "eq":
        r = una(F, s["src"], s["f"])
        return sino(str(r[s["c"][0]]) == str(r[s["c"][1]])), []
    if k == "agg":
        r = agg(F, s["n"], s["a"])
        return r if isinstance(r, tuple) else (r, [])
    raise ValueError(k)


def redirigir(spec):
    """Copia de la celda con sus fuentes apuntando a la versión que se lee de verdad (corregida si existe)."""
    if not isinstance(spec, dict):
        return spec
    sp = dict(spec)
    if "src" in sp:
        sp["src"] = Fuentes.ruta(sp["src"])
    if sp.get("k") == "agg":
        sp["a"] = [Fuentes.ruta(x) if isinstance(x, str) and (x.endswith(".csv") or x.endswith(".json")) else x
                   for x in sp["a"]]
    return sp


def numeros(texto):
    t = texto.replace(MENOS, "-")
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)
    if IDIOMA == "en":                                  # miles con coma (37,123)
        t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]


# ===================================================================================== tablas
class Tabla:
    def __init__(self, clave, cols, filas, leyenda, args=(), notas=(), nargs=()):
        """nargs: celdas cuyas cifras se insertan en las notas ({0}, {1}... de las notas, numeradas aparte)."""
        self.clave, self.cols, self.filas, self.leyenda, self.args, self.notas = clave, cols, filas, leyenda, list(args), list(notas)
        self.nargs = list(nargs)


S28 = "sensibilidad_v28/"
C1T = "c1/test/"
B3B = "b3/test_A_best/"
B3D = "b3/test_A_bestdc/"


def t_sens_auditoria(F):
    src = S28 + "resumen_familias.csv"
    var = [("base", "Análisis principal"), ("C0.01", "Detector con $C = 0.01$"), ("C0.3", "Detector con $C = 0.3$"),
           ("penult", "Detector sobre la penúltima capa"),
           ("ema", "Métodos con la media móvil exponencial de los pesos"), ("last", "Métodos en la última época"),
           ("conjunto", "Conjunto de las cinco semillas de cada método"), ("imagen", "Remuestreo por imagen"),
           ("test_completo", "Test limpio completo"), ("sin_espejo", "Sin promediado de espejo")]
    motivo = {("penult", "D_IN"): "no aplica (a)", ("conjunto", "D_ERM"): "no aplica (b)"}
    filas = []
    for v, lab in var:
        for ref in ("D_IN", "D_ERM"):
            f = {"variante": v, "referencia": ref}
            nom = "Principal" if ref == "D_IN" else "Secundaria"
            if existe(F, src, f):
                filas.append([lab, nom, V(src, f, "auc_referencia"),
                              FR(src, f, "no_inferior_0.02", "n_comparaciones"),
                              FR(src, f, "no_inferior_0.01", "n_comparaciones"),
                              IC(src, f, ["delta_erm", "ic95_inf_erm", "ic95_sup_erm"]),
                              IC(src, f, ["delta_dfr", "ic95_inf_dfr", "ic95_sup_dfr"])])
            else:
                t = motivo.get((v, ref), "no aplica")
                filas.append([lab, nom, NA(src, f, None, t), t, t, t, t])
    cols = ["Variante", "Referencia", "AUC de la referencia", "No inferior, δ = 0.02", "No inferior, δ = 0.01",
            "Referencia menos ERM (IC95)", "Referencia menos DFR (IC95)"]
    ley = ("Sensibilidades preespecificadas de la auditoría de COVIDx CXR-4 (Section 2.7). Referencia principal: detector "
           "de colección sobre la red de ImageNet; referencia secundaria: el mismo detector sobre los troncos finales "
           "de ERM, emparejados por semilla con cada método (su AUC es la media de los cinco troncos). Para cada "
           "variante se da el número de los {0} métodos frente a los que la referencia es no inferior tras la "
           "corrección de Holm al nivel unilateral de 0.025, con el margen preespecificado (δ = 0.02) y con δ = 0.01, y la "
           "diferencia de AUC entre la referencia y ERM y entre la referencia y DFR (media sobre las cinco semillas "
           "del método, ecuación (3)), con IC95 de percentiles. Cada variante forma, para cada referencia, una familia "
           "de {0} comparaciones con la estructura de las dos primeras filas de la Tabla 4. Conjunto: parte BIMCV del "
           "test ({1} imágenes de {2} pacientes), salvo en el test limpio completo ({3} imágenes). Unidad de "
           "remuestreo: el paciente, con 2 000 remuestreos (semilla 2026), salvo en la variante de remuestreo por "
           "imagen. En la variante del conjunto, cada método está representado por la media de las probabilidades de "
           "sus cinco semillas. La variante de remuestreo por paciente coincide con el análisis principal y no se "
           "repite.")
    args = [V(src, {"variante": "base", "referencia": "D_IN"}, "n_comparaciones", "int"),
            V(C1T + "f_c1.csv", {"metodo": "erm"}, "n", "int"),
            V(C1T + "detectores.csv", {"detector": "D_IN"}, "n_pacientes", "int"),
            AGG("suma", S28 + "s7_composicion.csv", {}, "n")]
    notas = ["(a) La red de ImageNet de la referencia principal no tiene penúltima capa entrenada: su cabeza no se "
             "entrenó.",
             "(b) El protocolo no prevé un conjunto de la referencia secundaria, cuyas puntuaciones son funciones de "
             "decisión de cinco detectores con escalas distintas."]
    return Tabla("sensibilidades de la auditoría", cols, filas, ley, args, notas)


def _familia_c1(F, src, quien):
    filas = []
    for m in MET:
        f = {"metodo": m}
        filas.append([NOMBRE[m], V(src, f, "auc_metodo"), IC(src, f, ["delta", "ic95_inf", "ic95_sup"]),
                      V(src, f, "p_ni_0.02", "p"), V(src, f, "p_holm_0.02", "p"), V(src, f, "no_inferior_0.02", "bool"),
                      V(src, f, "p_ni_0.01", "p"), V(src, f, "p_holm_0.01", "p"), V(src, f, "no_inferior_0.01", "bool"),
                      V(src, f, "p_superioridad", "p")])
    cols = ["Método", "AUC del método", "Diferencia (IC95)", "p, δ = 0.02", "p de Holm, δ = 0.02",
            "No inferior, δ = 0.02", "p, δ = 0.01", "p de Holm, δ = 0.01", "No inferior, δ = 0.01",
            "p de diferencia mayor que cero"]
    return cols, filas


def t_fc1(F):
    src = C1T + "f_c1.csv"
    cols, filas = _familia_c1(F, src, "principal")
    ley = ("Familia confirmatoria de la auditoría (Tabla 4, primera fila): no inferioridad de la referencia principal "
           "(AUC {0}) frente a cada uno de los 12 métodos en la parte BIMCV del test ({1} imágenes), con el margen "
           "preespecificado δ = 0.02 y, como sensibilidad, δ = 0.01. Diferencia: AUC de la referencia menos la del método, "
           "media sobre las cinco semillas (ecuación (3)), con IC95 de percentiles de 2 000 remuestreos de pacientes. "
           "Valores p unilaterales de la ecuación (4) sin corregir y con la corrección de Holm sobre las 12 "
           "comparaciones de cada margen, al nivel de 0.025. La última columna es el valor p unilateral, sin corregir "
           "y descriptivo, de que la diferencia sea mayor que cero.")
    return Tabla("familia confirmatoria de la auditoría", cols, filas, ley,
                 [V(src, {"metodo": "erm"}, "auc_referencia"), V(src, {"metodo": "erm"}, "n", "int")])


def t_fc1b(F):
    src = C1T + "f_c1b.csv"
    cols, filas = _familia_c1(F, src, "secundaria")
    jer = "sensibilidad_intervalos/c1b_jerarquico.csv"
    cols = cols[:3] + ["IC95 de semillas y pacientes"] + cols[3:]
    for m, fila in zip(MET, filas):
        fila.insert(3, IC(jer, {"metodo": m}, ["delta", "ic95_inf_jerarquico", "ic95_sup_jerarquico"]))
    ley = ("Familia secundaria de la auditoría (Tabla 4, segunda fila): no inferioridad de la referencia secundaria, "
           "el detector de colección sobre los troncos finales de ERM (AUC media {0}), frente a cada método, con la "
           "diferencia emparejada por semilla (la referencia de la semilla s frente al método de la semilla s). "
           "Mismo conjunto, remuestreo, márgenes y corrección que la tabla anterior. La cuarta columna repite la "
           "diferencia con el IC95 jerárquico de semillas y pacientes de las ecuaciones (5) a (7), previsto en el "
           "plan congelado para esta familia y calculado después de abrir los tests con las funciones congeladas; es una lectura de "
           "sensibilidad y no sustituye a la prueba con Holm.")
    return Tabla("familia de la referencia secundaria", cols, filas, ley, [V(src, {"metodo": "erm"}, "auc_referencia")])


def t_detectores(F):
    src = C1T + "detectores.csv"
    d = F.csv(src)
    filas = []
    for _, r in d.iterrows():
        tag, det = r["tag"], r["detector"]
        sem = re.search(r"seed(\d+)", str(tag))
        s = sem.group(1) if sem else ""
        if det == "D_IN":
            lab = "Referencia principal (red de ImageNet)"
        elif det == "D_ERM":
            lab = "Referencia secundaria, tronco de ERM, semilla %s" % s
        elif det == "D_MASK":
            lab = "Tronco de ERM con máscara, semilla %s" % s
        elif det == "D_RES":
            lab = ("Tronco con conjunto reservado (20 %% de los pacientes negativos), semilla %s" % s
                   if "reserva20" in str(tag) else "Tronco con conjunto reservado (todos los negativos), semilla %s" % s)
        else:
            raise ValueError(det)
        f = {"tag": tag}
        filas.append([lab, IC(src, f, ["auc", "ic95_inf", "ic95_sup"]), IC(src, f, ["auc_AP", "ic95_inf_AP", "ic95_sup_AP"]),
                      IC(src, f, ["auc_PA", "ic95_inf_PA", "ic95_sup_PA"]), TRI(src, f, ["n_antigua", "n_nueva"])])
    cols = ["Referencia sin información de clase", "AUC (IC95)", "AUC en AP (IC95)", "AUC en PA (IC95)",
            "Positivos de ajuste, cinco cifras / seis cifras"]
    ley = ("Referencias sin información de clase, tronco por tronco, en la parte BIMCV del test ({0} imágenes de {1} "
           "pacientes; {2} AP y {3} PA entre las que tienen la proyección en el nombre). Detector de colección: "
           "regresión logística L2 ($C = {4}$, pesos de clase equilibrados) sobre las 768 características agrupadas "
           "del tronco, con diana 1 para la colección de cinco cifras, ajustada solo con positivos y aplicada con promediado "
           "de espejo (Section 2.6). En los troncos con conjunto reservado se ajusta con los positivos reservados, que "
           "el tronco no vio. IC95 de percentiles con 2 000 remuestreos de pacientes. La Tabla 5 y la Section 3.2 del "
           "artículo dan la media y el recorrido de cada grupo de troncos.")
    f0 = {"detector": "D_IN"}
    return Tabla("referencias sin información de clase", cols, filas, ley,
                 [V(src, f0, "n", "int"), V(src, f0, "n_pacientes", "int"), V(src, f0, "n_AP", "int"),
                  V(src, f0, "n_PA", "int"), V(src, f0, "C", "d2")])


def t_estratos(F):
    src = C1T + "c1_vii_estratos.csv"
    fa, fp = {"estrato": "AP", "comparacion": "detector"}, {"estrato": "PA", "comparacion": "detector"}
    filas = [["Referencia principal (AUC del detector)", IC(src, fa, ["auc", "ic95_inf", "ic95_sup"]), "no aplica",
              IC(src, fp, ["auc", "ic95_inf", "ic95_sup"]), "no aplica"]]
    for m in MET:
        ga = {"estrato": "AP", "comparacion": "detector menos %s" % m}
        gp = {"estrato": "PA", "comparacion": "detector menos %s" % m}
        filas.append([NOMBRE[m], V(src, ga, "auc"), IC(src, ga, ["delta", "ic95_inf", "ic95_sup"]),
                      V(src, gp, "auc"), IC(src, gp, ["delta", "ic95_inf", "ic95_sup"])])
    cols = ["Método", "AP: AUC del método", "AP: diferencia (IC95)", "PA: AUC del método", "PA: diferencia (IC95)"]
    fs = {"estrato": "todos"}
    ley = ("Estratos de proyección en la parte BIMCV del test. La proyección consta en el nombre de {0} de las {1} "
           "imágenes ({2} AP, {3} de ellas negativas; {4} PA, {5} negativas), y el análisis se limita a ellas. "
           "Diferencia: AUC de la referencia principal menos la del método dentro del estrato, media sobre cinco "
           "semillas, con IC95 de 2 000 remuestreos de pacientes. Descriptivo, sin familia de contraste. Por sí sola, "
           "la proyección (AP = 1) separa las clases con una AUC de {6}; son AP el {7} % de las negativas y el {8} % "
           "de las positivas.")
    args = [V(src, fs, "n", "int"), V(C1T + "f_c1.csv", {"metodo": "erm"}, "n", "int"), V(src, fa, "n", "int"),
            V(src, fa, "n_neg", "int"), V(src, fp, "n", "int"), V(src, fp, "n_neg", "int"), V(src, fs, "auc"),
            V(src, fs, "frac_AP_neg", "pct"), V(src, fs, "frac_AP_pos", "pct")]
    return Tabla("estratos AP y PA", cols, filas, ley, args)


def t_coleccion(F):
    src = C1T + "c1_iii_por_coleccion.csv"
    combos = [("bimcv", "bimcv_antigua"), ("bimcv", "ricord"), ("rsna", "bimcv_antigua"), ("rsna", "ricord")]
    filas = []
    for m in MET + ["D_IN", "D_ERM"]:
        filas.append([NOMBRE[m]] + [IC(src, {"metodo": m, "negativos": a, "positivos": b}, ["auc", "ic95_inf", "ic95_sup"])
                                    for a, b in combos])
    cols = ["Método", "Negativos de BIMCV frente a positivos de BIMCV", "Negativos de BIMCV frente a positivos de RICORD",
            "Negativos de RSNA frente a positivos de BIMCV", "Negativos de RSNA frente a positivos de RICORD"]
    ley = ("AUC en el test limpio según la colección de origen de los negativos y de los positivos. Los positivos de "
           "BIMCV del test son todos de la colección de cinco cifras. Tamaños (negativas / positivas): {0} / {1}, {2} / {3}, "
           "{4} / {5} y {6} / {7}, en el orden de las columnas. Media de cinco semillas en los métodos y en la "
           "referencia secundaria; IC95 de 2 000 remuestreos de pacientes. Descriptivo.")
    args = []
    for a, b in combos:
        f = {"metodo": "erm", "negativos": a, "positivos": b}
        args += [V(src, f, "n_neg", "int"), V(src, f, "n_pos", "int")]
    return Tabla("AUC por colección de origen", cols, filas, ley, args)


def t_lecturas(F):
    src = C1T + "c1_ii_reservado.csv"
    d = F.csv(src)
    filas = []
    for _, r in d.iterrows():
        tag = r["tag"]
        s = re.search(r"seed(\d+)", tag).group(1)
        lab = ("20 %% de los pacientes negativos reservados, semilla %s" % s if "reserva20" in tag
               else "Todos los negativos reservados, semilla %s" % s)
        f = {"tag": tag}
        filas.append([lab, V(src, f, "cv_neg_vs_antiguos"), V(src, f, "cv_neg_vs_nuevos"), V(src, f, "cv_antiguos_vs_nuevos"),
                      IC(src, f, ["transfer_antiguos_a_nuevos", "transfer_antiguos_a_nuevos_ic95_inf", "transfer_antiguos_a_nuevos_ic95_sup"]),
                      IC(src, f, ["transfer_nuevos_a_antiguos", "transfer_nuevos_a_antiguos_ic95_inf", "transfer_nuevos_a_antiguos_ic95_sup"]),
                      V(src, f, "lectura_antiguos_en_test_bimcv"),
                      IC(src, f, ["lectura_nuevos_en_test_bimcv", "lectura_nuevos_en_test_bimcv_ic95_inf", "lectura_nuevos_en_test_bimcv_ic95_sup"]),
                      TRI(src, f, ["n_res_neg", "n_res_pos_antigua", "n_res_pos_nueva"])])
    cols = ["Tronco", "Validación cruzada: negativos frente a positivos de cinco cifras",
            "Validación cruzada: negativos frente a positivos de seis cifras",
            "Validación cruzada: positivos de cinco cifras frente a seis cifras",
            "Lectura de cinco cifras aplicada a seis cifras (IC95)", "Lectura de seis cifras aplicada a cinco cifras (IC95)",
            "Test: lectura de cinco cifras", "Test: lectura de seis cifras (IC95)",
            "Reservadas: negativas / cinco cifras / seis cifras"]
    ley = ("Lecturas lineales entre colecciones de positivos sobre los cuatro troncos de ERM entrenados sin un conjunto "
           "reservado de BIMCV (Section 2.6). Regresión logística L2 sobre las características agrupadas, "
           "estandarizadas con el conjunto reservado. Validación cruzada: AUC fuera de pliegue con cinco pliegues "
           "agrupados por paciente dentro del conjunto reservado; la tercera columna enfrenta dos colecciones de la "
           "misma clase. Transferencia: lectura ajustada con una mitad de los negativos reservados (partición por "
           "paciente) y los positivos de una colección, y evaluada con la otra mitad de los negativos frente a los "
           "positivos de la otra colección. Test: las dos lecturas aplicadas a la parte BIMCV del test, cuyos "
           "positivos son todos de la colección de cinco cifras. IC95 de 2 000 remuestreos de pacientes; las columnas sin "
           "intervalo no lo tienen calculado. Descriptivo.")
    return Tabla("lecturas lineales entre colecciones", cols, filas, ley)


def t_adquisicion(F):
    s4, s5 = C1T + "c1_iv_adquisicion.csv", C1T + "c1_v_mascara.csv"
    fa, fb = {"variante": "a_ajustada_en_entrenamiento"}, {"variante": "b_validacion_cruzada_en_el_conjunto"}
    fm, fd = {"estimando": "a_erm_mascara_menos_erm"}, {"estimando": "b_detector_tronco_enmascarado"}
    filas = [["Clasificador de solo adquisición, ajustado con BIMCV de entrenamiento", IC(s4, fa, ["auc", "ic95_inf", "ic95_sup"]),
              IC(s4, fa, ["delta_vs_erm", "delta_ic95_inf", "delta_ic95_sup"])],
             ["Clasificador de solo adquisición, validación cruzada en el conjunto evaluado (usa la clase)",
              IC(s4, fb, ["auc", "ic95_inf", "ic95_sup"]), IC(s4, fb, ["delta_vs_erm", "delta_ic95_inf", "delta_ic95_sup"])],
             ["ERM con máscara", V(s5, fm, "auc_1"), IC(s5, fm, ["delta", "ic95_inf", "ic95_sup"])],
             ["Detector de colección sobre los troncos de ERM con máscara", IC(s5, fd, ["auc_1", "ic95_inf", "ic95_sup"]),
              NA(s5, fd, "delta")]]
    cols = ["Análisis", "AUC (IC95)", "Diferencia con ERM (IC95)"]
    ley = ("Origen de la señal en la parte BIMCV del test ({0} imágenes). Clasificador de solo adquisición: regresión "
           "logística sobre 14 estadísticos de sub-bandas de Haar fuera de la región torácica (Section 2.6); la "
           "segunda fila usa la clase del test y no es una referencia sin información de clase. Diferencia con ERM: "
           "AUC de la fila menos la de ERM ({1}, media de cinco semillas). ERM con máscara y el detector enmascarado "
           "son medias de cinco semillas. IC95 de 2 000 remuestreos de pacientes. Descriptivo.")
    return Tabla("solo adquisición y máscara", cols, filas, ley, [V(s4, fa, "n", "int"), V(s4, fa, "auc_erm_media")])


def t_covidgr(F):
    src = C1T + "c1_vi_covidgr.csv"
    filas = []
    for m in MET + ["D_IN", "D_ERM"]:
        f = {"metodo": m}
        filas.append([NOMBRE[m], V(src, f, "auc_interna"),
                      IC(src, f, ["auc_externa", "auc_externa_ic95_inf", "auc_externa_ic95_sup"]),
                      IC(src, f, ["caida", "caida_ic95_inf", "caida_ic95_sup"]),
                      V(src, f, "sensibilidad_youden"), V(src, f, "especificidad_youden")])
    cols = ["Método", "AUC interna", "AUC en COVIDGR (IC95)", "Caída (IC95)", "Sensibilidad", "Especificidad"]
    ley = ("Cohorte externa COVIDGR ({0} imágenes, una por unidad de remuestreo, sin identificador de paciente). AUC "
           "interna: parte BIMCV del test ({1} imágenes). Caída: ecuación (8), con los dos términos remuestreados por "
           "separado (2 000 remuestreos de pacientes en la parte BIMCV y de imágenes en COVIDGR). Sensibilidad y "
           "especificidad al umbral de Youden de la parte BIMCV de la validación de cada semilla, medias sobre las "
           "cinco semillas (en las referencias, al umbral de su propia función de decisión); sin intervalo calculado. "
           "Ningún modelo se ajustó, seleccionó ni recalibró con COVIDGR. Descriptivo.")
    return Tabla("cohorte externa", cols, filas, ley, [V(src, {"metodo": "erm"}, "n_externa", "int"),
                                                      V(src, {"metodo": "erm"}, "n_interna", "int")])


def t_gravedad(F):
    src = C1T + "c1_vi_gravedad.csv"
    gr = [("NORMAL-PCR+", "Radiografía normal, PCR positiva"), ("MILD", "Leve"), ("MODERATE", "Moderada"), ("SEVERE", "Grave")]
    filas = [[NOMBRE[m]] + [V(src, {"metodo": m, "gravedad": g}, "sensibilidad") for g, _ in gr] for m in MET + ["D_IN", "D_ERM"]]
    cols = ["Método"] + [lab for _, lab in gr]
    ley = ("Sensibilidad en COVIDGR por gravedad radiológica de las positivas, al umbral de Youden de la validación de "
           "cada semilla (media de cinco semillas). Positivas por grupo, en el orden de las columnas: {0}, {1}, {2} y "
           "{3}; en total {4}, porque una positiva de COVIDGR no tiene gravedad anotada (su fila de metadatos no casa "
           "con ningún fichero). Descriptivo, sin intervalos.")
    args = [V(src, {"metodo": "erm", "gravedad": g}, "n", "int") for g, _ in gr] + [AGG("suma", src, {"metodo": "erm"}, "n")]
    return Tabla("sensibilidad por gravedad", cols, filas, ley, args)


def _curva(F, src, metodos, rhos, d=3):
    filas = []
    for m in metodos:
        fila = [NOMBRE[m]]
        for r in rhos:
            fila.append(IC_O_NA(F, src, {"metodo": m, "rho": r}, ["auc_media_semillas", "ic95_inf", "ic95_sup"], d))
        filas.append(fila)
    return filas


def t_curva(F):
    src = B3B + "curva.csv"
    filas = _curva(F, src, MET_B3, RHOS)
    cols = ["Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in RHOS]
    ley = ("AUC en el test desconfundido del benchmark de derrame pleural por nivel de confusión, con el estado elegido "
           "con la validación de la misma $\\rho$: media por semilla (ecuación (2)) con IC95 jerárquico de semillas y "
           "pacientes (ecuaciones (5) a (7); 2 000 remuestreos). {0} semillas por nivel y {1} en $\\rho$ de 0.9, 0.95 "
           "y 0.99. Es la Tabla 6 del artículo con sus intervalos. DFR no está definido con $\\rho = 1$ (Section 2.4).")
    return Tabla("curva del benchmark de derrame", cols, filas, ley,
                 [V(src, {"metodo": "erm", "rho": 0.5}, "n_semillas", "int"), V(src, {"metodo": "erm", "rho": 0.9}, "n_semillas", "int")])


MEDIDAS_B3 = [("delta_area_dosis", "Área (ecuación (9))", 4)] + \
             [("delta_auc_" + med(r), "AUC, $\\rho = %s$" % ETQ_RHO[r], 3) for r in RHOS] + \
             [("coste_rho050_no_inferioridad", "No inferioridad, $\\rho = 0.5$, δ = 0.01", 3),
              ("delta_caida_050_100", "Caída entre $\\rho = 0.5$ y 1.0", 3)]


def _comparacion(F, src, oraculo=False):
    filas = []
    for medida, lab, d in MEDIDAS_B3:
        for m in MET_B3_SIN_ERM:
            f = {"metodo": m, "medida": medida}
            if not existe(F, src, f, "delta"):
                filas.append([lab, NOMBRE[m], NA(src, f, "delta"), "no aplica", "no aplica", "no aplica", "no aplica"])
                continue
            r = una(F, src, f)
            enfam = not pd.isna(r["familia"])
            filas.append([lab, NOMBRE[m], IC(src, f, ["delta", "ic95_inf", "ic95_sup"], d), V(src, f, "p_unilateral", "p"),
                          ("sensibilidad" if oraculo else V(src, f, "papel", "txt")), V(src, f, "p_holm", "p") if enfam else NA(src, f, "p_holm"),
                          V(src, f, "rechaza_holm", "bool") if enfam else NA(src, f, "p_holm")])
    cols = ["Medida", "Método", "Diferencia frente a ERM (IC95)", "p unilateral", "Papel", "p de Holm", "Rechaza con Holm"]
    return cols, filas


def t_comparacion(F):
    src = B3B + "comparacion_erm.csv"
    cols, filas = _comparacion(F, src)
    ley = ("Comparación completa con ERM en el test desconfundido del benchmark de derrame pleural (estado elegido con "
           "la validación de la misma $\\rho$). Diferencia: método menos ERM, media por semilla, con IC95 jerárquico de "
           "semillas y pacientes (ecuaciones (5) a (7); 2 000 remuestreos). p unilateral del mismo remuestreo, sin corregir: de que la "
           "diferencia sea mayor que cero en el área y en cada nivel, y de la ecuación (4) con δ = 0.01 en la no "
           "inferioridad. Holm dentro de cada familia de la Tabla 4 al nivel unilateral de 0.025: área ({0} "
           "comparaciones, confirmatoria), $\\rho = 0.99$ ({1}), $\\rho = 1.0$ ({2}, sin DFR) y no inferioridad en "
           "$\\rho = 0.5$ ({3}), secundarias. Los niveles de 0.6 a 0.95 y la diferencia en la caída son descriptivos. "
           "DFR no está definido con $\\rho = 1$, y su cifra con $\\rho = 0.99$ se marcó de antemano como inestable "
           "(Section 2.4). Cuatro decimales en el área.")
    a = lambda me: V(src, {"metodo": "lisa", "medida": me}, "n_familia", "int")
    return Tabla("comparación con ERM en el benchmark de derrame", cols, filas, ley,
                 [a("delta_area_dosis"), a("delta_auc_rho099"), a("delta_auc_rho100"), a("coste_rho050_no_inferioridad")])


def _tabla_rhos(F, src, metodos, cols_ic, rhos, extra=None, d=3):
    filas = []
    for m in metodos:
        base = {"metodo": m}
        if extra:
            base.update(extra)
        filas.append([NOMBRE[m]] + [IC_O_NA(F, src, dict(base, rho=r), cols_ic, d) for r in rhos])
    return filas


def t_invertido(F):
    src = "c2c3/invertido.csv"
    filas = _tabla_rhos(F, src, MET_B3, ["auc_invertido", "ic95_inf_invertido", "ic95_sup_invertido"], RHOS)
    cols = ["Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in RHOS]
    ley = ("AUC en el test invertido del benchmark de derrame pleural (prevalencia $1 - \\rho$ en NIH y $\\rho$ en "
           "CheXpert), con los mismos modelos que la tabla de la curva. Media por semilla con IC95 jerárquico de "
           "semillas y pacientes. Descriptivo.")
    return Tabla("test invertido", cols, filas, ley)


def t_fuente(F):
    src = "c2c3/fuente.csv"
    filas = []
    for fu, lab in (("A (NIH)", "NIH"), ("B (CheXpert)", "CheXpert")):
        for m in MET_B3:
            filas.append([lab, NOMBRE[m]] + [IC_O_NA(F, src, {"metodo": m, "fuente": fu, "rho": r},
                                                     ["auc_media_semillas", "ic95_inf", "ic95_sup"]) for r in RHOS])
    cols = ["Fuente", "Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in RHOS]
    ley = ("AUC dentro de cada fuente del test desconfundido del benchmark de derrame pleural (cada fuente aporta la "
           "mitad del test, equilibrada en clase). Media por semilla con IC95 jerárquico de semillas y pacientes. "
           "Descriptivo.")
    return Tabla("AUC por fuente", cols, filas, ley)


def t_brecha(F):
    src = "c2c3/brecha.csv"
    filas = _tabla_rhos(F, src, MET_B3, ["brecha_especificidad", "ic95_inf", "ic95_sup"], RHOS)
    cols = ["Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in RHOS]
    ley = ("Brecha de especificidad en el test desconfundido del benchmark de derrame pleural: especificidad en "
           "CheXpert menos especificidad en NIH, al umbral de Youden de la validación de la misma $\\rho$ de cada "
           "semilla (en DFR, de su subconjunto de selección). Un valor positivo indica que el modelo clasifica como "
           "positivos más negativos de la fuente asociada a la clase positiva. Media por semilla con IC95 jerárquico. "
           "Descriptivo.")
    return Tabla("brecha de especificidad", cols, filas, ley)


def t_orientacion(F):
    src = "c2c3/orientacion_b_curva.csv"
    rh = [0.5, 0.8, 1.0]
    filas = _tabla_rhos(F, src, MET_B3, ["auc_media_semillas", "ic95_inf", "ic95_sup"], rh)
    so = "c2c3/orientacion.csv"
    fo = {"metodo": "erm"}
    cols = ["Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in rh]
    ley = ("Orientación inversa del benchmark de derrame pleural, con CheXpert como fuente positiva ({0} semillas: "
           "{1}; configuración de cada nivel tomada de la orientación principal). AUC media por semilla en "
           "el test desconfundido con IC95 jerárquico. Caída de ERM entre $\\rho = 0.5$ y 1.0: {2} en la orientación "
           "principal y {3} en la inversa; diferencia, {4}, frente al umbral preespecificado de {5}. Descriptivo. DFR no "
           "está definido con $\\rho = 1$.")
    args = [V(src, {"metodo": "erm", "rho": 0.5}, "n_semillas", "int"), V(src, {"metodo": "erm", "rho": 0.5}, "semillas", "lista"),
            V(so, fo, "caida_A"), V(so, fo, "caida_B"), AGG("icstr", so, fo, "diferencia_A_menos_B", "ic95", 3),
            V(so, fo, "umbral", "d2")]
    return Tabla("orientación inversa", cols, filas, ley, args)


def t_oraculo_curva(F):
    src = B3D + "curva.csv"
    filas = _curva(F, src, MET_B3, RHOS)
    cols = ["Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in RHOS]
    ley = ("Selección de oráculo (análisis de sensibilidad preespecificado, Section 2.7): AUC en el test desconfundido del "
           "benchmark de derrame pleural con el estado de la época que elige la validación desconfundida. Media por "
           "semilla con IC95 jerárquico; mismas semillas que con la selección principal. No es un resultado principal: "
           "usa una validación que en la práctica no existe. La celda de DFR con $\\rho = 1.0$ se explica en la nota (a) "
           "de la [[t_oraculo_comp]].")
    return Tabla("curva con la selección de oráculo", cols, filas, ley)


def t_oraculo_comp(F):
    src = B3D + "comparacion_erm.csv"
    cols, filas = _comparacion(F, src, oraculo=True)
    for fl in filas:
        if fl[1] == "DFR" and fl[0] in ("AUC, $\\rho = 1.0$", "Caída entre $\\rho = 0.5$ y 1.0"):
            fl[1] = "DFR (a)"
    ley = ("Comparación con ERM con la selección de oráculo en el test desconfundido del benchmark de derrame pleural. "
           "Misma estructura, remuestreo y corrección que la comparación con la selección principal ([[t_comparacion]]); "
           "la corrección de Holm se aplica a las mismas familias (área: {0} comparaciones; $\\rho = 0.99$: {1}; "
           "$\\rho = 1.0$: {2}; no inferioridad: {3}), pero todas son de sensibilidad: el protocolo no admite la "
           "selección de oráculo como resultado principal ni como familia confirmatoria.")
    a = lambda me: V(src, {"metodo": "lisa", "medida": me}, "n_familia", "int")
    notas = ["(a) Con la selección de oráculo, DFR parte del estado de la época que elige una mitad de la validación "
             "desconfundida y reajusta la última capa con la otra mitad, eligiendo la regularización en la primera. "
             "Esa validación tiene las cuatro celdas clase por fuente en todos los niveles, de modo que este DFR está "
             "definido también con $\\rho = 1.0$, a diferencia del DFR principal, que reajusta con la validación de la "
             "misma $\\rho$, sin celdas minoritarias en ese nivel (Section 2.4). Usa etiquetas desconfundidas que los "
             "demás métodos no ven en el entrenamiento; por eso su familia de $\\rho = 1.0$ tiene una comparación más "
             "que la principal."]
    return Tabla("comparación con ERM con la selección de oráculo", cols, filas, ley,
                 [a("delta_area_dosis"), a("delta_auc_rho099"), a("delta_auc_rho100"), a("coste_rho050_no_inferioridad")], notas)


HOLM_E3 = ["corregido_p_holm", "pacientes_p_holm", "doble_p_holm", "t_p_holm"]
E3A = CORR + "complementos/e3_area_alternativos.csv"
E3S = CORR + "complementos/e3_seleccion_alternativos.csv"


def t_alternativos(F):
    src = B3B + "comparacion_erm.csv"
    e3 = E3A
    filas = []
    for m in MET_B3_SIN_ERM:
        f = {"metodo": m, "medida": "delta_area_dosis"}
        filas.append([NOMBRE[m], V(src, f, "delta", "d4"), IC(src, f, ["delta", "ic95_inf", "ic95_sup"], 4),
                      IC(src, f, ["delta", "ic95_inf_pacientes", "ic95_sup_pacientes"], 4),
                      IC(src, f, ["delta", "ic95_inf_doble", "ic95_sup_doble"], 4),
                      IC(src, f, ["t_media", "t_ic95_inf", "t_ic95_sup"], 4),
                      V(src, f, "p_unilateral", "p"), V(src, f, "p_unilateral_pacientes", "p"),
                      V(e3, {"metodo": m}, "doble_p_unilateral", "p"), V(src, f, "t_p_unilateral", "p")] +
                     [V(e3, {"metodo": m}, c, "p") for c in HOLM_E3])
    cols = ["Método", "Diferencia en el área", "Principal: jerárquico (IC95)", "Pacientes con semillas fijas (IC95)",
            "Remuestreo doble (IC95)", "Prueba t entre semillas: media (IC95)", "p, jerárquico", "p, pacientes",
            "p, doble", "p, prueba t", "p de Holm, jerárquico", "p de Holm, pacientes", "p de Holm, doble",
            "p de Holm, prueba t"]
    f0 = {"metodo": "lisa", "medida": "delta_area_dosis"}
    ley = ("Intervalos alternativos preespecificados (Section 2.7) para la diferencia en el área de la ecuación (9) frente "
           "a ERM, la familia confirmatoria del benchmark de derrame pleural. Principal: remuestreo jerárquico de las "
           "ecuaciones (5) a (7). Pacientes con semillas fijas: 2 000 remuestreos de pacientes con los modelos fijos. "
           "Remuestreo doble: semillas y pacientes a la vez, sin la corrección de las ecuaciones (5) a (7), de modo "
           "que cuenta dos veces la interacción. Prueba t pareada entre las diferencias por semilla, con {0} grados de libertad; usa "
           "solo las semillas presentes en todos los niveles ({1}), por lo que su media puede diferir de la "
           "estimación principal, que usa todas las semillas de cada nivel. Valores p unilaterales de que la "
           "diferencia sea mayor que cero, sin corregir y con la corrección de Holm sobre las {2} comparaciones de la "
           "familia al nivel unilateral de 0.025, calculada por separado con cada intervalo. El plan no preveía valor p "
           "para el remuestreo doble; el suyo se calcula con la misma regla que el del intervalo principal, solo como "
           "lectura. Los valores p con los intervalos alternativos se calcularon después de abrir los tests. Cuatro "
           "decimales en las diferencias.")
    notas = ["Con Holm al nivel unilateral de 0.025 superan a ERM {0} métodos con el intervalo principal, {1} con el de "
             "pacientes con semillas fijas, {2} con el remuestreo doble y {3} con la prueba t entre semillas. Con el "
             "intervalo de pacientes, el p de Holm de DANN es {4}, justo por debajo del umbral."]
    nargs = [AGG("n_hasta", e3, {}, c, 0.025) for c in HOLM_E3] + [V(e3, {"metodo": "dann"}, "pacientes_p_holm", "d5")]
    return Tabla("intervalos alternativos del área", cols, filas, ley,
                 [V(src, f0, "t_gl", "int"), V(src, f0, "n_semillas", "int"), AGG("nfilas", e3, {})], notas, nargs)


def t_alternativos_sel(F):
    src = "c2c3/c3_epoca_por_metodo.csv"
    sm = "c2c3/c3-epoca.csv"
    alt = "sensibilidad_intervalos/c3_epoca_por_metodo_alternativos.csv"
    altm = "sensibilidad_intervalos/c3_epoca_media_alternativos.csv"
    filas = []
    for r in (0.9, 0.95, 0.99):
        for m in MET_B3:
            f = {"metodo": m, "rho": r}
            filas.append([ETQ_RHO[r], NOMBRE[m], IC(src, f, ["delta_sel", "ic95_inf", "ic95_sup"]),
                          IC(src, f, ["t_media", "t_ic95_inf", "t_ic95_sup"]), V(src, f, "t_p_unilateral", "p"),
                          IC(alt, f, ["delta_sel", "ic95_inf_pacientes", "ic95_sup_pacientes"]),
                          IC(alt, f, ["delta_sel", "ic95_inf_doble", "ic95_sup_doble"]), "", ""])
        fr = {"rho": r}
        filas.append([ETQ_RHO[r], "Media de los once métodos", IC(sm, fr, ["delta_sel_media_metodos", "ic95_inf", "ic95_sup"], 4),
                      IC(altm, fr, ["t_media", "t_ic95_inf", "t_ic95_sup"], 4), V(altm, fr, "t_p_unilateral", "p"),
                      IC(altm, fr, ["delta_sel_media_metodos", "ic95_inf_pacientes", "ic95_sup_pacientes"], 4),
                      IC(altm, fr, ["delta_sel_media_metodos", "ic95_inf_doble", "ic95_sup_doble"], 4),
                      PS(E3S, fr, ["corregido_p_unilateral", "pacientes_p_unilateral", "doble_p_unilateral"]),
                      PS(E3S, fr, HOLM_E3)])
    cols = ["$\\rho$", "Método", "Principal: jerárquico (IC95)", "Prueba t entre semillas: media (IC95)", "p, prueba t",
            "Pacientes con semillas fijas (IC95)", "Remuestreo doble (IC95)",
            "p: jerárquico / pacientes / doble", "p de Holm: jerárquico / pacientes / doble / prueba t"]
    ley = ("Intervalos alternativos de la pérdida por la selección de época (ecuación (10)) en los tres niveles de la "
           "familia confirmatoria. Por método, el intervalo principal es el jerárquico y la prueba t pareada usa las "
           "{0} semillas del nivel ({1} grados de libertad). La media de los once métodos lleva cuatro decimales, como "
           "en la Tabla 8. Pacientes con semillas fijas: 2 000 remuestreos de pacientes con los modelos fijos. "
           "Remuestreo doble: semillas y pacientes a la vez, sin la corrección de las ecuaciones (5) a (7). Estos dos "
           "intervalos y la prueba t de la media se calcularon después de abrir los tests con las funciones "
           "congeladas del análisis principal, por estar previstos en el plan congelado (Section 2.7). En la media de "
           "los once métodos, las dos últimas columnas dan los valores p unilaterales de que la pérdida sea mayor que "
           "cero con cada intervalo y los de Holm sobre los tres niveles, al nivel unilateral de 0.025; el valor p del "
           "remuestreo doble, que el plan no preveía, se da solo como lectura. Por método no se calculan.")
    f0 = {"metodo": "erm", "rho": 0.9}
    return Tabla("intervalos alternativos de la selección de época", cols, filas, ley,
                 [V(src, f0, "n_semillas", "int"), V(src, f0, "t_gl", "int")])


def t_tb(F):
    sc, sd = "b2/informe_test/curva.csv", "b2/informe_test/comparacion_erm.csv"
    filas = []
    for m in MET:
        fila = [NOMBRE[m]]
        for conj in ("test_shenzhen", "test_montgomery"):
            fc, fd = {"metodo": m, "conjunto": conj}, {"metodo": m, "conjunto": conj, "medida": "delta_auc_rho000"}
            fila.append(IC(sc, fc, ["auc_media_semillas", "ic95_inf", "ic95_sup"]))
            if m == "erm":
                fila += ["referencia", "no aplica", "no aplica"]
            else:
                fila += [IC(sd, fd, ["delta", "ic95_inf", "ic95_sup"]), V(sd, fd, "p_unilateral", "p"), V(sd, fd, "p_holm", "p")]
        filas.append(fila)
    cols = ["Método", "Shenzhen: AUC (IC95)", "Shenzhen: diferencia con ERM (IC95)", "Shenzhen: p", "Shenzhen: p de Holm",
            "Montgomery: AUC (IC95)", "Montgomery: diferencia con ERM (IC95)", "Montgomery: p", "Montgomery: p de Holm"]
    ley = ("Evaluación en tuberculosis: AUC por cohorte de test (Shenzhen, {0} imágenes; Montgomery, {1}) y diferencia "
           "frente a ERM, media de {2} semillas. La unidad de remuestreo es la imagen; IC95 jerárquico de semillas e "
           "imágenes. p unilateral sin corregir, de que la diferencia sea mayor que cero, y p de Holm dentro de cada "
           "cohorte ({3} comparaciones, familias secundarias de la Tabla 4, nivel unilateral de 0.025).")
    # tamaños de las cohortes desde el informe JSON si los trae; si no, se omiten
    j = F.json("b2/informe_test/informe.json")
    nsz = _buscar_n(j, "test_shenzhen")
    nmg = _buscar_n(j, "test_montgomery")
    if nsz is not None and nmg is not None:
        args = [AGG("js", "b2/informe_test/informe.json", nsz, "int"), AGG("js", "b2/informe_test/informe.json", nmg, "int"),
                V(sc, {"metodo": "erm", "conjunto": "test_shenzhen"}, "n_semillas", "int"),
                V(sd, {"metodo": "lisa", "conjunto": "test_shenzhen", "medida": "delta_auc_rho000"}, "n_familia", "int")]
    else:
        ley = ley.replace("(Shenzhen, {0} imágenes; Montgomery, {1})", "(Shenzhen y Montgomery)")
        ley = ley.replace("{2}", "{0}").replace("{3}", "{1}")
        args = [V(sc, {"metodo": "erm", "conjunto": "test_shenzhen"}, "n_semillas", "int"),
                V(sd, {"metodo": "lisa", "conjunto": "test_shenzhen", "medida": "delta_auc_rho000"}, "n_familia", "int")]
    return Tabla("evaluación en tuberculosis", cols, filas, ley, args)


def _buscar_n(j, conj):
    """Ruta a un recuento de imágenes del conjunto dentro del informe JSON, si existe (no se inventa)."""
    def rec(o, ruta):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == conj and isinstance(v, dict):
                    for kk in ("n", "n_imagenes"):
                        if kk in v and isinstance(v[kk], (int, float)):
                            return ruta + [k, kk]
                r = rec(v, ruta + [k])
                if r:
                    return r
        return None
    return rec(j, [])


def t_sel_epoca(F):
    src, sm = "c2c3/c3_epoca_por_metodo.csv", "c2c3/c3-epoca.csv"
    filas = []
    for m in MET_B3:
        filas.append([NOMBRE[m]] + [IC_O_NA(F, src, {"metodo": m, "rho": r}, ["delta_sel", "ic95_inf", "ic95_sup"]) for r in RHOS])
    filas.append(["Media de los once métodos"] + [IC(sm, {"rho": r}, ["delta_sel_media_metodos", "ic95_inf", "ic95_sup"], 4) for r in RHOS])
    cols = ["Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in RHOS]
    ley = ("Pérdida por la selección de época (ecuación (10)) por método y nivel en el test desconfundido del benchmark "
           "de derrame pleural: AUC con el estado que elige la validación desconfundida menos AUC con el que elige la "
           "validación de la misma $\\rho$, media por semilla con IC95 jerárquico. En DFR se usan los estados de su "
           "etapa ERM, de modo que está definido también con $\\rho = 1$. La última fila es el estimando de la Tabla 8 "
           "(cuatro decimales), confirmatorio en $\\rho$ de 0.9, 0.95 y 0.99 (Holm sobre tres); el resto es "
           "descriptivo.")
    return Tabla("selección de época por método", cols, filas, ley)


def t_hiper(F):
    src = "c2c3/c3-hiper.csv"
    filas = []
    for m in ("erm", "lisa", "groupdro", "dfr"):
        for r in (0.9, 0.95, 0.99):
            f = {"metodo": m, "rho": r}
            filas.append([NOMBRE[m], ETQ_RHO[r], V(src, f, "ensayo_regla", "int"), V(src, f, "ensayo_oraculo", "int"),
                          EQ(src, f, "ensayo_regla", "ensayo_oraculo"), V(src, f, "auc_regla"), V(src, f, "auc_media_ensayos"),
                          V(src, f, "auc_oraculo"), IC(src, f, ["ganancia_sobre_azar", "ic95_inf_G", "ic95_sup_G"]),
                          IC(src, f, ["perdida_frente_oraculo", "ic95_inf_L", "ic95_sup_L"]), V(src, f, "spearman_sel_test")])
    cols = ["Método", "$\\rho$", "Ensayo de la regla", "Ensayo de la validación desconfundida", "Coinciden",
            "AUC, regla", "AUC, media de los ensayos", "AUC, ensayo de la validación desconfundida", "G (IC95)",
            "L (IC95)", "Spearman entre la métrica de selección y la AUC"]
    ley = ("Selección de hiperparámetros, repetida tras la congelación con la semilla 0 y {0} ensayos por método y "
           "nivel, con evaluación del test desconfundido del benchmark de derrame pleural (ecuación (11)). Ensayos "
           "numerados de 0 a 7; el 0 lleva los valores por defecto. G: AUC del ensayo de la regla preespecificada menos la "
           "media de los ocho; L: AUC del ensayo que elegiría la validación desconfundida menos la del ensayo de la "
           "regla. Con una semilla, los IC95 remuestrean solo pacientes (2 000 remuestreos) y no recogen la variación "
           "entre semillas. Correlación de Spearman entre los ocho ensayos. Descriptivo.")
    return Tabla("selección de hiperparámetros", cols, filas, ley, [V(src, {"metodo": "erm", "rho": 0.9}, "n_ensayos", "int")])


def t_estabilidad(F):
    src = "sensibilidad/estabilidad/estabilidad.csv"
    filas = []
    for m in ("erm", "lisa"):
        for r in (0.9, 0.99):
            f = {"metodo": m, "rho": r}
            filas.append([NOMBRE[m], ETQ_RHO[r], V(src, f, "ensayo_semilla0", "int"), V(src, f, "ensayo_semilla1", "int"),
                          V(src, f, "coincide", "bool"), V(src, f, "spearman_metricas_seleccion"),
                          IC(src, f, ["dif_auc_s0", "ic95_inf_s0", "ic95_sup_s0"]),
                          IC(src, f, ["dif_auc_s1", "ic95_inf_s1", "ic95_sup_s1"])])
    cols = ["Método", "$\\rho$", "Ensayo elegido, semilla 0", "Ensayo elegido, semilla 1", "Coinciden",
            "Spearman entre las métricas de selección", "Cambio de AUC, corridas de la semilla 0 (IC95)",
            "Cambio de AUC, corridas de la semilla 1 (IC95)"]
    ley = ("Estabilidad de la búsqueda de hiperparámetros en el benchmark de derrame pleural: la búsqueda de {0} "
           "ensayos se repitió antes de la congelación con la semilla 1 (manifiesto e inicialización), sin cambiar "
           "las configuraciones. Spearman: correlación entre las métricas de selección de los ensayos en las dos "
           "búsquedas. Cambio de AUC: AUC en la validación desconfundida del ensayo que elige la semilla 1 menos la del "
           "que elige la semilla 0, medida con las corridas de cada semilla; IC95 de 2 000 remuestreos de pacientes. "
           "Solo validación; descriptivo.")
    return Tabla("estabilidad de la búsqueda", cols, filas, ley, [V(src, {"metodo": "erm", "rho": 0.9}, "n_ensayos_comunes", "int")])


def t_sens_curva(F):
    filas = []
    for an, d, lab in (("tronco", "sensibilidad/tronco/curva.csv", "ResNet-50"),
                       ("etiqueta", "sensibilidad/etiqueta/curva.csv", "Cardiomegalia")):
        for m in ("erm", "groupdro", "lisa", "dfr"):
            filas.append([lab, NOMBRE[m]] + [IC_O_NA(F, d, {"metodo": m, "rho": r}, ["auc_media_semillas", "ic95_inf", "ic95_sup"])
                                             for r in (0.5, 0.9, 0.99, 1.0)])
    cols = ["Análisis", "Método"] + ["$\\rho = %s$" % ETQ_RHO[r] for r in (0.5, 0.9, 0.99, 1.0)]
    ley = ("Segundo tronco (ResNet-50, en el benchmark de derrame pleural) y segunda etiqueta (cardiomegalia, con las "
           "mismas reservas de pacientes y tamaños multiplicados por 0.25): AUC en el test desconfundido con {0} "
           "semillas y los valores por defecto, media por semilla con IC95 jerárquico. DFR no está definido con "
           "$\\rho = 1$. Análisis de sensibilidad sin contraste (Section 2.7).")
    return Tabla("segundo tronco y segunda etiqueta: AUC", cols, filas, ley,
                 [V("sensibilidad/tronco/curva.csv", {"metodo": "erm", "rho": 0.5}, "n_semillas", "int")])


def t_sens_dif(F):
    filas = []
    for d, rs, lab in (("sensibilidad/tronco/", "sensibilidad/tronco/resumen.csv", "ResNet-50"),
                       ("sensibilidad/etiqueta/", "sensibilidad/etiqueta/resumen.csv", "Cardiomegalia")):
        src = d + "comparacion_erm.csv"
        fila = [lab, "ERM, caída entre $\\rho = 0.5$ y 1.0", IC(rs, {"metodo": "erm", "medida": "caida_050_100"},
                                                                 ["estimacion", "ic95_inf", "ic95_sup"]), "", "", "", ""]
        filas.append(fila)
        for m in ("groupdro", "lisa", "dfr"):
            celdas = []
            for medida in ("delta_auc_rho050", "delta_auc_rho090", "delta_auc_rho099", "delta_auc_rho100",
                           "coste_rho050_no_inferioridad"):
                celdas.append(IC_O_NA(F, src, {"metodo": m, "medida": medida}, ["delta", "ic95_inf", "ic95_sup"]))
            filas.append([lab, NOMBRE[m] + " frente a ERM", ""] + celdas[:4])
    cols = ["Análisis", "Medida", "Caída (IC95)", "$\\rho = 0.5$", "$\\rho = 0.9$", "$\\rho = 0.99$", "$\\rho = 1.0$"]
    ley = ("Segundo tronco y segunda etiqueta: caída de ERM entre $\\rho = 0.5$ y 1.0 y diferencias de cada método "
           "frente a ERM en el test desconfundido (media por semilla, IC95 jerárquico). Tres semillas y valores por "
           "defecto; las diferencias no se contrastan y el plan congelado solo prevé la caída de ERM y el signo de las "
           "diferencias en $\\rho$ de 0.9 y 0.99 (Section 2.7). El área no se calcula porque la curva solo tiene "
           "cuatro niveles. La Tabla 9 del artículo resume la concordancia con el análisis principal.")
    return Tabla("segundo tronco y segunda etiqueta: diferencias con ERM", cols, filas, ley)


def t_espacio(F):
    reg = F.metodos_registrados()
    filas = [["Todos", "tasa de la segunda fase", V("b1/busqueda_todas.csv", {"metodo": "erm", "trial": 0}, "lr_B", "lr"), AGG("espacio_comun")]]
    for m in MET:
        s = reg[m]["search"]
        if m == "dfr":
            filas.append([NOMBRE[m], "C de la regresión L1 (rejilla, elegido en el subconjunto de selección)",
                          AGG("rejilla_dfr"), "no se busca"])
            filas.append([NOMBRE[m], "submuestras equilibradas del reajuste", AGG("reps_dfr"), "no se busca"])
            continue
        if not s:
            filas.append([NOMBRE[m], "solo la tasa común", "", ""])
            continue
        for k in s:
            filas.append([NOMBRE[m], NOMBRE_HP.get((m, k), k), AGG("defecto", m, k), AGG("espacio", m, k)])
    cols = ["Método", "Hiperparámetro", "Valor por defecto", "Espacio de búsqueda"]
    ley = ("Espacio de búsqueda de hiperparámetros, común a los tres benchmarks (Section 2.5): {0} configuraciones por "
           "método, la primera con los valores por defecto y las demás muestreadas de forma determinista a partir del "
           "método y del número de ensayo; la tasa de la segunda fase se muestrea en todos los métodos. Entre llaves, "
           "elección uniforme entre los valores. El valor por defecto de la tasa es el del ensayo 0 de la búsqueda "
           "de COVIDx; los demás, y los espacios, son los atributos declarados en el código de cada método.")
    return Tabla("espacio de búsqueda", cols, filas, ley, [AGG("n_ensayos")])


def t_elegidos(F):
    filas = []
    for m in MET:
        j1, j2 = "b1/congelados/baseline_%s.json" % m, "b2/congelados/b2_%s.json" % m
        filas.append([NOMBRE[m], AGG("hp_ensayo", j1), AGG("hp_lr", j1), AGG("hp", j1, m), AGG("hp_sel", j1),
                      AGG("hp_ensayo", j2), AGG("hp_lr", j2), AGG("hp", j2, m), AGG("hp_sel", j2)])
    cols = ["Método", "COVIDx: ensayo", "COVIDx: tasa", "COVIDx: parámetros", "COVIDx: AUC de selección",
            "Tuberculosis: ensayo", "Tuberculosis: tasa", "Tuberculosis: parámetros", "Tuberculosis: AUC de selección"]
    ley = ("Configuración elegida y congelada en COVIDx CXR-4 (búsqueda con la semilla 42, selección con el subconjunto de "
           "selección de la parte BIMCV de la validación) y en tuberculosis (semilla 0, selección con la validación). "
           "Ensayo 0: valores por defecto. Tasa: la de la segunda fase. Parámetros: valores de los hiperparámetros "
           "buscados del método ([[t_espacio]]). AUC de selección: la métrica de la regla preespecificada, "
           "sin promediado de espejo, en el conjunto de selección; en DFR, la de su etapa ERM, que es la que usa la "
           "regla para elegir el ensayo; la regularización del reajuste se elige después con el mismo subconjunto de "
           "selección. Solo validación.")
    return Tabla("configuraciones elegidas en COVIDx y tuberculosis", cols, filas, ley)


def t_elegidos_b3(F):
    filas = []
    for m in MET_B3:
        for r in RHOS:
            j = "b3a/congelados/b3a_A_%s_%s.json" % (m, med(r))
            if not (RES / j).exists():
                filas.append([NOMBRE[m], ETQ_RHO[r], "sin configuración", "", "", ""])
                continue
            filas.append([NOMBRE[m], ETQ_RHO[r], AGG("hp_ensayo", j), AGG("hp_lr", j), AGG("hp", j, m), AGG("hp_sel", j)])
    cols = ["Método", "$\\rho$", "Ensayo", "Tasa", "Parámetros", "AUC de selección"]
    ley = ("Configuración elegida y congelada en cada nivel del benchmark de derrame pleural (búsqueda con la semilla "
           "0, selección con la validación de la misma $\\rho$; la orientación inversa usa la de su nivel). Con "
           "$\\rho = 1$ la validación se satura y los ocho ensayos empatan; la regla, con empate al ensayo menor, "
           "elige el ensayo 0 salvo en DFA con fuente (Section 2.5). En DFR, la AUC de selección es la de su etapa ERM, "
           "que es la que usa la regla para elegir el ensayo; la regularización del reajuste se elige después con el "
           "subconjunto de selección de la validación de la misma $\\rho$. Con $\\rho = 1$ la configuración de DFR es la de "
           "su etapa ERM; el reajuste no está definido. Solo validación.")
    return Tabla("configuraciones elegidas en el benchmark de derrame", cols, filas, ley)


def t_val_finales(F):
    src = "b1/finales_validacion.csv"
    filas = []
    for m in MET:
        f = {"metodo": m}
        def cel(c):
            if not existe(F, src, f, c + "_media"):
                return NA(src, f, c + "_media", "no comparable (a)")
            return {"k": "agg", "n": "mdt", "a": [src, f, c]}
        fila = [NOMBRE[m], cel("val_robust_sel"), cel("val_robust"), cel("val")]
        if m == "dfr":                    # la validación limpia contiene la mitad de reajuste (enmienda E5)
            fila[1] = {"k": "agg", "n": "mdt_nota", "a": [src, f, "val_robust_sel", " (b)"]}
            fila[3] = "no comparable (a)"
        filas.append(fila)
    cols = ["Método", "Subconjunto de selección de la parte BIMCV de la validación", "Parte BIMCV de la validación",
            "Validación limpia completa"]
    ley = ("AUC en validación de las {0} corridas finales de COVIDx CXR-4 (cinco semillas por método), con el estado "
           "de la mejor época: media ± desviación típica entre semillas, y mínimo y máximo entre paréntesis. Solo "
           "validación, sin valor confirmatorio.")
    notas = ["(a) La parte BIMCV de la validación, incluida también en la validación limpia completa, contiene el subconjunto "
             "con el que DFR reajusta su última capa; su AUC en esos conjuntos no es comparable con la de los demás "
             "métodos.",
             "(b) En DFR, el subconjunto de selección es el que usa la regla para elegir el ensayo con la AUC de su etapa ERM y, "
             "después, la regularización del reajuste."]
    return Tabla("validación de las corridas finales de COVIDx", cols, filas, ley,
                 [AGG("suma", src, {}, "val_n")], notas)


def t_corridas(F):
    grupos = [("b1_busq_cadena_*.json", "COVIDx CXR-4", "búsqueda de hiperparámetros (c)"),
              ("b1_final_cadena_[0-9]*.json", "COVIDx CXR-4", "corridas finales, cinco semillas"),
              ("b1_final_cadena_rep*.json", "COVIDx CXR-4", "repetición de las corridas de la semilla 42 (b)"),
              ("b3a_p6_cadena_*.json", "Derrame pleural", "piloto"),
              ("b3a_busq_cadena_*.json", "Derrame pleural", "búsqueda de hiperparámetros"),
              ("b3a_fin_cadena_*.json", "Derrame pleural", "corridas finales, cinco semillas (a)"),
              ("b3a_s10_cadena_*.json", "Derrame pleural", "semillas adicionales en 0.9, 0.95 y 0.99"),
              ("b3a_estab_cadena_*.json", "Derrame pleural", "repetición de la búsqueda con la semilla 1"),
              ("b3a_c3b_cadena_*.json", "Derrame pleural", "repetición de la búsqueda con evaluación del test"),
              ("st_r50_cadena_*.json", "Derrame pleural", "segundo tronco (ResNet-50)"),
              ("se_card_cadena_*.json", "Cardiomegalia", "segunda etiqueta"),
              ("b2_busq_cadena_*.json", "Tuberculosis", "búsqueda de hiperparámetros"),
              ("b2_fin_cadena_*.json", "Tuberculosis", "corridas finales, cinco semillas")]
    filas = [[b, fase, AGG("cad_n", p), AGG("cad_ok", p), AGG("cad_h", p)] for p, b, fase in grupos]
    cols = ["Benchmark", "Fase", "Corridas", "Terminadas con código 0", "Horas de GPU"]
    ley = ("Recuento de corridas de entrenamiento según los ficheros de control de las cadenas de ejecución. Horas "
           "de GPU: suma de las duraciones anotadas de las corridas, como en la Section 2.8. No se incluyen los pasos "
           "de evaluación, extracción de características e informes.")
    notas = ["(a) Las corridas del control de fuente terminan sin entrenar con un código distinto de 0, porque el "
             "método no es aplicable en este benchmark (Section 2.4).",
             "(b) Las diez corridas de la semilla 42 que habían empezado antes de un cambio en la política de "
             "almacenamiento se repitieron desde el principio para conservar los estados de mejor época, última época "
             "y media móvil; en esos diez métodos, la semilla 42 de las corridas finales es la de estas repeticiones.",
             "(c) Las 96 configuraciones de la búsqueda incluyen los 12 ensayos con los valores por defecto, que "
             "proceden de las corridas piloto de la fase anterior y no se cuentan aquí."]
    return Tabla("recuento de corridas", cols, filas, ley, [], notas)


def t_entorno(F):
    e, c = "gate7/entorno_contenedor.json", "gate7/congelacion.json"
    filas = [["Python", AGG("js", e, ["python"], "txt")],
             ["Sistema", AGG("js", e, ["plataforma"], "txt")],
             ["GPU", AGG("js", e, ["gpus", 0], "txt")],
             ["PyTorch", AGG("js", e, ["version_torch"], "txt")],
             ["CUDA de PyTorch", AGG("js", e, ["torch_cuda"], "txt")],
             ["cuDNN", AGG("paquete", e, "nvidia-cudnn-cu12")],
             ["torchvision", AGG("paquete", e, "torchvision")],
             ["timm", AGG("js", e, ["version_timm"], "txt")],
             ["Pesos de ConvNeXt-Tiny", AGG("js", e, ["timm_convnext_tiny_pesos", "hf_hub_id"], "txt")],
             ["NumPy", AGG("js", e, ["version_numpy"], "txt")],
             ["pandas", AGG("js", e, ["version_pandas"], "txt")],
             ["SciPy", AGG("js", e, ["version_scipy"], "txt")],
             ["scikit-learn", AGG("js", e, ["version_sklearn"], "txt")],
             ["PyWavelets", AGG("paquete", e, "PyWavelets")],
             ["OpenCV", AGG("paquete", e, "opencv-python-headless")],
             ["Pillow", AGG("paquete", e, "pillow")],
             ["Paquetes listados en el contenedor", AGG("js", e, ["paquetes"], "len")],
             ["Ficheros con huella SHA256 en la congelación del protocolo", AGG("js", c, ["ficheros"], "len")],
             ["Fecha de la congelación (UTC)", AGG("js", c, ["fecha_utc"], "txt")]]
    cols = ["Elemento", "Valor"]
    ley = ("Entorno de software del contenedor en el que se entrenaron y evaluaron todos los modelos, según la "
           "descripción del entorno guardada con la congelación del protocolo (Section 2.1 y Section 2.8). La lista completa de "
           "paquetes con sus versiones forma parte del material de reproducibilidad.")
    return Tabla("entorno de software", cols, filas, ley)


# ===================================================================================== tablas nuevas del 21-11-2026
SOLAPE = "suplemento_fuentes/solape_segmentador.csv"
SOLAPE_Q = "suplemento_fuentes/solape_segmentador_calidad.csv"
IMPACTO = CORR + "antes_despues_todas_las_filas.csv"
SIMUL = CORR + "simulacion/simulacion.csv"
E1 = CORR + "complementos/e1_c1_din_jerarquico.csv"
E2P = CORR + "complementos/e2_perdida_seleccion_area.csv"
E2D = CORR + "complementos/e2_diferencias_area_frente_erm.csv"


def t_solape(F):
    src = SOLAPE
    filas = []
    for bench, lab_b, conjuntos in (("derrame", "Derrame pleural", ("reserva de entrenamiento", "reserva de validación",
                                                                     "reserva de test", "test desconfundido")),
                                    ("cardiomegalia", "Cardiomegalia", ("test desconfundido",))):
        for conj in conjuntos:
            for fu in ("NIH", "CheXpert"):
                for cl in ("negativa", "positiva"):
                    f = {"benchmark": bench, "conjunto": conj, "fuente": fu, "clase": cl}
                    filas.append([lab_b, conj, fu, cl, CELDA_V_O_NA(F, src, f, "n_imagenes", "int", "sin desglose (a)"),
                                  CELDA_V_O_NA(F, src, f, "n_vistas", "int", "sin desglose (a)"),
                                  V(src, f, "frac_imagenes"), V(src, f, "frac_pacientes")])
    cols = ["Benchmark", "Conjunto", "Fuente", "Clase", "Imágenes", "Vistas por el segmentador", "Fracción de imágenes",
            "Fracción de imágenes de pacientes con alguna imagen vista"]
    ley = ("Solapamiento entre el conjunto de entrenamiento del segmentador de pulmón (Section 2.3) y los benchmarks "
           "controlados, por fuente y clase: reservas de pacientes del benchmark de derrame pleural y tests "
           "desconfundidos de derrame y de cardiomegalia, que son los mismos en todos los niveles y semillas. "
           "Vistas por el segmentador: imágenes que forman parte de su conjunto de entrenamiento. La última columna es "
           "la fracción de imágenes cuyo paciente tiene alguna imagen en ese conjunto. En el benchmark de derrame, la "
           "calidad media de las pseudomáscaras apenas difiere entre las imágenes vistas y las no vistas: en las ocho "
           "combinaciones de fuente, clase y reserva (la de test y, juntas, las de entrenamiento y validación), la media de las "
           "vistas supera a la de las no vistas en {0} a {1}, con medias de {2} a {3} en las no vistas y de {4} a {5} "
           "en las vistas. No se reentrenó el "
           "segmentador sin esas imágenes, de modo que su efecto sobre los resultados no está medido. Descriptivo.")
    args = [AGG("ext", SOLAPE_Q, {}, "diferencia_vistas_menos_no_vistas", "min", 4),
            AGG("ext", SOLAPE_Q, {}, "diferencia_vistas_menos_no_vistas", "max", 4),
            AGG("ext", SOLAPE_Q, {}, "q_media_no_vistas", "min", 4),
            AGG("ext", SOLAPE_Q, {}, "q_media_no_vistas", "max", 4),
            AGG("ext", SOLAPE_Q, {}, "q_media_vistas", "min", 4),
            AGG("ext", SOLAPE_Q, {}, "q_media_vistas", "max", 4)]
    notas = ["(a) Para las reservas de entrenamiento y de validación solo constan las fracciones."]
    return Tabla("solapamiento del segmentador con los benchmarks", cols, filas, ley, args, notas)


def t_impacto(F):
    src = IMPACTO
    tp, tc = "B3a selección principal, comparación con ERM", "c3-epoca (media de los once métodos)"
    bloques = [("Área frente a ERM", tp, "delta_area_dosis", 4),
               ("AUC con $\\rho = 0.99$ frente a ERM", tp, "delta_auc_rho099", 3)]
    filas = []
    for lab, tabla, medida, d in bloques:
        for m in MET_B3_SIN_ERM:
            f = {"tabla": tabla, "metodo": m, "medida": medida}
            filas.append([lab, NOMBRE[m], V(src, f, "estimacion", "d%d" % d), IV(src, f, ["ic95_inf_antes", "ic95_sup_antes"], d),
                          IV(src, f, ["ic95_inf_despues", "ic95_sup_despues"], d), V(src, f, "p_holm_antes", "p"),
                          V(src, f, "p_holm_despues", "p"), V(src, f, "cambia_decision", "bool")])
    for r in (0.9, 0.95, 0.99):
        f = {"tabla": tc, "metodo": "media de once", "medida": "delta_sel_" + med(r)}
        filas.append(["Pérdida media por selección de época", "$\\rho = %s$" % ETQ_RHO[r], V(src, f, "estimacion", "d4"),
                      IV(src, f, ["ic95_inf_antes", "ic95_sup_antes"], 4), IV(src, f, ["ic95_inf_despues", "ic95_sup_despues"], 4),
                      V(src, f, "p_holm_antes", "p"), V(src, f, "p_holm_despues", "p"), V(src, f, "cambia_decision", "bool")])
    cols = ["Familia", "Comparación", "Estimación", "IC95, factor original", "IC95, factor corregido",
            "p de Holm, factor original", "p de Holm, factor corregido", "Cambia la decisión"]
    ley = ("Efecto de la corrección del factor del término de semillas (ecuación (7)) sobre las familias de contraste "
           "del benchmark de derrame pleural con la selección principal. El plan preespecificado calibraba ese término "
           "como si las semillas se remuestrearan todas juntas, mientras que el procedimiento las remuestrea por "
           "estratos según los niveles en que existen; en los tres niveles con diez semillas, la varianza del término "
           "quedaba así por debajo de la declarada. El factor se corrigió después de abrir los tests, sin cambiar "
           "estimandos, familias, márgenes, remuestreos ni semillas, y las estimaciones puntuales no cambian. De las "
           "{0} filas de resultados revisadas en las tablas que usan ese término, {1} cambian su intervalo; el "
           "número de filas con una decisión de contraste distinta es {2}. Holm dentro de cada familia al nivel unilateral de 0.025 (área y $\\rho = 0.99$: diez "
           "comparaciones; pérdida media: tres niveles). Las demás tablas de este material dan solo los intervalos "
           "con el factor corregido.")
    args = [AGG("nfilas", src, {}), AGG("nfilas", src, {"cambia_intervalo": "True"}),
            AGG("nfilas", src, {"cambia_decision": "True"})]
    return Tabla("impacto de la corrección del factor de semillas", cols, filas, ley, args)


ESCENARIOS = [("realista", "Realista"), ("realista_corr_alta", "Correlación alta entre niveles"),
              ("interaccion_grande", "Interacción grande, no proporcional"), ("sin_semillas", "Sin efecto de semilla")]
NIVELES_SIM = [("0.80", "0.8"), ("0.90", "0.9"), ("0.95", "0.95"), ("0.99", "0.99"), ("1.00", "1.0"),
               ("área 0.5 a 0.99", "área de 0.5 a 0.99")]


def t_simulacion(F):
    src = SIMUL
    filas = []
    for esc, lab in ESCENARIOS:
        for niv, lniv in NIVELES_SIM:
            fo, fc = {"escenario": esc, "estimador": "original", "nivel": niv}, {"escenario": esc, "estimador": "corregido", "nivel": niv}
            filas.append([lab, lniv, CELDA_V_O_NA(F, src, fo, "K", "int", "5 y 10"),
                          V(src, fo, "razon_varianza"), V(src, fc, "razon_varianza"),
                          V(src, fo, "cobertura_ic95"), V(src, fc, "cobertura_ic95"),
                          V(src, fo, "error_tipo_I_0.025"), V(src, fc, "error_tipo_I_0.025")])
    cols = ["Escenario", "$\\rho$", "Semillas", "Razón de varianzas, original", "Razón de varianzas, corregido",
            "Cobertura del IC95, original", "Cobertura del IC95, corregido", "Error de tipo I, original",
            "Error de tipo I, corregido"]
    fr = {"escenario": "realista", "estimador": "corregido"}
    ley = ("Validación por simulación del factor del término de semillas con el factor original y con el corregido "
           "(ecuación (7)), bajo la hipótesis nula. Estructura simulada: ocho niveles de $\\rho$ con la presencia real "
           "de semillas (cinco en todos y cinco más en $\\rho$ de 0.9, 0.95 y 0.99), diferencia por semilla con efecto "
           "de semilla correlacionado entre niveles contiguos, efecto común de pacientes e interacción, y 400 "
           "pacientes por conjunto simulado, remuestreados con los mismos pesos en todos los niveles y semillas; magnitudes "
           "tomadas de las medianas por nivel observadas en las comparaciones con ERM. A cada conjunto simulado se le aplican las "
           "funciones de remuestreo del análisis (B = 2 000) y la prueba unilateral de la ecuación (4); {0} conjuntos "
           "simulados por escenario. Escenarios: realista (correlación 0.6 del efecto de semilla entre niveles contiguos), "
           "correlación alta (0.9), interacción tan grande como el efecto de semilla y sin correlación entre niveles, "
           "y sin efecto de semilla. Razón de varianzas: media de la varianza del remuestreo dividida por la varianza "
           "verdadera; cobertura del IC95 de percentiles; error de tipo I: proporción de conjuntos simulados con p unilateral "
           "menor o igual que 0.025. En el escenario realista, la razón de varianzas de los niveles de diez semillas "
           "pasa de {5} a {6} con el factor original a {7} a {8} con el corregido. La corrección no basta para "
           "alcanzar el nivel nominal: aun con el factor corregido, en el escenario realista el error de tipo I "
           "es de {1} a {2} en los niveles de diez semillas y de {3} y {4} en $\\rho$ de 0.8 y 1.0, con cinco, porque "
           "el remuestreo trata como conocida una varianza de semillas estimada con pocos grados de libertad.")
    args = [V(src, {"escenario": "realista", "estimador": "corregido", "nivel": "0.90"}, "replicas", "int"),
            AGG("ext", src, dict(fr, K=10.0), "error_tipo_I_0.025", "min", 3),
            AGG("ext", src, dict(fr, K=10.0), "error_tipo_I_0.025", "max", 3),
            V(src, dict(fr, nivel="0.80"), "error_tipo_I_0.025"), V(src, dict(fr, nivel="1.00"), "error_tipo_I_0.025"),
            AGG("ext", src, dict(fr, estimador="original", K=10.0), "razon_varianza", "min", 3),
            AGG("ext", src, dict(fr, estimador="original", K=10.0), "razon_varianza", "max", 3),
            AGG("ext", src, dict(fr, K=10.0), "razon_varianza", "min", 3),
            AGG("ext", src, dict(fr, K=10.0), "razon_varianza", "max", 3)]
    return Tabla("validación por simulación", cols, filas, ley, args)


def t_sel_area(F):
    filas = [["Pérdida media por selección de época, once métodos",
              IC(E2P, {"medida": "perdida_seleccion_area_050_099"}, ["estimacion", "corregido_ic95_inf", "corregido_ic95_sup"], 4),
              IC(E2P, {"medida": "perdida_seleccion_area_fuerte_090_099"}, ["estimacion", "corregido_ic95_inf", "corregido_ic95_sup"], 4)]]
    for m in MET_B3_SIN_ERM:
        f = {"metodo": m}
        filas.append([NOMBRE[m] + " frente a ERM", IC(E2D, f, ["delta_area", "corregido_ic95_inf", "corregido_ic95_sup"], 4),
                      IC(E2D, f, ["delta_area_fuerte_090_099", "corregido_fuerte_ic95_inf", "corregido_fuerte_ic95_sup"], 4)])
    cols = ["Medida", "Área de $\\rho = 0.5$ a 0.99 (IC95)", "Área de $\\rho = 0.9$ a 0.99 (IC95)"]
    ley = ("Pérdida por la selección de época en la escala del área, en el test desconfundido del benchmark de derrame "
           "pleural. Primera fila: la pérdida media de los once métodos en cada nivel (ecuación (10)), integrada con "
           "los mismos pesos de trapecio que el área de la ecuación (9) entre $\\rho = 0.5$ y 0.99 y, en la tercera "
           "columna, solo entre 0.9 y 0.99, normalizada por la longitud de ese tramo. Filas siguientes: la diferencia "
           "de área de cada método con ERM en los mismos dos tramos, para compararla en la misma escala. IC95 "
           "jerárquico de semillas y pacientes con el factor corregido (ecuaciones (5) a (7); 2 000 remuestreos). Es "
           "un cálculo complementario hecho después de abrir los tests, descriptivo y sin contraste. Cuatro "
           "decimales.")
    return Tabla("pérdida por selección en la escala del área", cols, filas, ley)


def t_tb_sel(F):
    src = "b2/informe_test/seleccion_epoca.csv"
    filas = []
    for m in MET:
        fila = [NOMBRE[m]]
        for conj in ("test_shenzhen", "test_montgomery"):
            f = {"metodo": m, "conjunto": conj}
            if existe(F, src, f):
                fila += [IC(src, f, ["delta_sel", "ic95_inf", "ic95_sup"]),
                         FR(src, f, "semillas_a_favor_bestdc", "n_semillas"), V(src, f, "t_p_unilateral", "p")]
            else:
                fila += [NA(src, f, None, "no calculado (a)"), "no calculado (a)", "no calculado (a)"]
        filas.append(fila)
    cols = ["Método", "Shenzhen: pérdida (IC95)", "Shenzhen: semillas a favor del oráculo", "Shenzhen: p, prueba t",
            "Montgomery: pérdida (IC95)", "Montgomery: semillas a favor del oráculo", "Montgomery: p, prueba t"]
    f0 = {"metodo": "erm", "conjunto": "test_shenzhen"}
    ley = ("Pérdida por la selección de época en tuberculosis (ecuación (10)): AUC en cada cohorte de test con el "
           "estado que elige la parte de TBX11K de la validación (estado de oráculo) menos la AUC con el estado que "
           "elige la validación (selección principal), media de {0} semillas con IC95 jerárquico de semillas e "
           "imágenes (2 000 remuestreos). Semillas a favor del oráculo: semillas en que el estado de oráculo da mayor "
           "AUC. Prueba t pareada unilateral entre semillas, de que la pérdida sea mayor que cero ({1} grados de "
           "libertad). Un cero indica que las dos validaciones eligen la misma época en todas las semillas. "
           "Descriptivo, sin familia de contraste.")
    notas = ["(a) El cálculo de la selección de época en tuberculosis no incluye DFR."]
    return Tabla("selección de época en tuberculosis con el estado de oráculo", cols, filas, ley,
                 [V(src, f0, "n_semillas", "int"), V(src, f0, "t_gl", "int")], notas)


def t_auditoria_jer(F):
    src = E1
    filas = []
    for m in MET:
        f = {"metodo": m}
        filas.append([NOMBRE[m], V(src, f, "delta"), IV(src, f, ["ic95_inf_pacientes", "ic95_sup_pacientes"]),
                      IV(src, f, ["ic95_inf_jerarquico", "ic95_sup_jerarquico"]), V(src, f, "de_semillas_sigma_S"),
                      V(src, f, "de_pacientes_V_P"), V(src, f, "p_holm_0.02_jerarquico", "p"),
                      V(src, f, "no_inferior_0.02_jerarquico_holm", "bool"), V(src, f, "p_holm_0.01_jerarquico", "p"),
                      V(src, f, "no_inferior_0.01_jerarquico_holm", "bool")])
    cols = ["Método", "Diferencia", "IC95 de pacientes", "IC95 de semillas y pacientes", "Desviación de semillas",
            "Desviación de pacientes", "p de Holm, δ = 0.02", "No inferior, δ = 0.02", "p de Holm, δ = 0.01",
            "No inferior, δ = 0.01"]
    ley = ("Familia confirmatoria de la auditoría de COVIDx CXR-4 ([[t_fc1]]) con el intervalo de semillas y "
           "pacientes: referencia principal, fija, frente a cada uno de los {0} métodos con sus cinco semillas en la "
           "parte BIMCV del test, con la AUC de la referencia repetida en cada semilla. Diferencia: AUC de la "
           "referencia menos la del método, media sobre las semillas (ecuación (3)). IC95 de pacientes: el del "
           "análisis confirmatorio. IC95 de semillas y pacientes: remuestreo jerárquico de las ecuaciones (5) a (7), "
           "con 2 000 remuestreos; con un solo nivel y un solo estrato, el factor corregido y el original coinciden. "
           "Desviación de semillas: raíz de la varianza del efecto de semilla de la ecuación (5), descontada la "
           "interacción; desviación de pacientes: raíz de la varianza por pacientes de la media. Valores p de Holm de "
           "no inferioridad (ecuación (4)) con el intervalo de semillas y pacientes, sobre las {0} comparaciones de "
           "cada margen al nivel unilateral de 0.025. Con ese intervalo la referencia es no inferior frente a {1} "
           "métodos con δ = 0.02 y frente a {2} con δ = 0.01. Es un cálculo complementario hecho después de abrir los "
           "tests; la prueba confirmatoria sigue siendo la del intervalo de pacientes.")
    args = [AGG("nfilas", src, {}), AGG("nfilas", src, {"no_inferior_0.02_jerarquico_holm": "True"}),
            AGG("nfilas", src, {"no_inferior_0.01_jerarquico_holm": "True"})]
    return Tabla("auditoría con el intervalo de semillas y pacientes", cols, filas, ley, args)


# agregado de media ± dt (mín a máx)
_agg_base = agg


def agg(F, n, a):  # noqa: F811
    if n == "mdt_nota":
        src, f, c, sufijo = a
        return agg(F, "mdt", [src, f, c]) + sufijo
    if n == "mdt":
        src, f, c = a
        r = una(F, src, f)
        return ("%s ± %s (%s" + A() + "%s)") % (num(r[c + "_media"]), num(r[c + "_dt"]), num(r[c + "_min"]), num(r[c + "_max"]))
    return _agg_base(F, n, a)


# Orden de numeración (21-11-2026): por primera cita en el manuscrito (Métodos 2.3, 2.5, 2.7 y 2.8, después
# Resultados). Para renumerar basta con reordenar esta lista; las remisiones internas usan [[nombre_de_la_función]].
ORDEN = [t_solape, t_espacio, t_elegidos, t_elegidos_b3, t_val_finales, t_impacto, t_simulacion, t_entorno, t_corridas,
         t_fc1, t_fc1b, t_detectores, t_estratos, t_coleccion, t_lecturas, t_adquisicion, t_covidgr, t_gravedad,
         t_curva, t_comparacion, t_invertido, t_fuente, t_brecha, t_orientacion, t_tb, t_sel_epoca, t_sel_area,
         t_hiper, t_estabilidad, t_tb_sel, t_sens_curva, t_sens_dif, t_oraculo_curva, t_oraculo_comp,
         t_sens_auditoria, t_alternativos, t_alternativos_sel, t_auditoria_jer]


# ===================================================================================== composición
def construir(F):
    tablas, md = [], []
    TABLA = "Tabla" if IDIOMA == "es" else "Table"
    numero = {fn.__name__: "%s S%d" % (TABLA, i) for i, fn in enumerate(ORDEN, 1)}

    def xref(txt):
        return re.sub(r"\[\[(\w+)\]\]", lambda m: numero[m.group(1)], txt)

    md.append(T("# Material suplementario") + "\n")
    md.append(xref(T("Este material reúne las tablas que completan las del artículo: las familias de contraste "
                     "completas, con sus valores p sin corregir y corregidos, los análisis descriptivos y de "
                     "sensibilidad preespecificados, los cálculos complementarios hechos después de abrir los tests y "
                     "los datos de reproducibilidad. Todas las cifras proceden de los ficheros de resultados del "
                     "estudio, con el formato de la sección de Resultados; las Figuras S1 a S8 se describen en sus "
                     "propias leyendas. Los intervalos jerárquicos de semillas y pacientes usan el factor corregido "
                     "de la ecuación (7); la [[t_impacto]] compara, en las familias de contraste, los intervalos y "
                     "valores p con el factor original y con el corregido.")) + "\n")

    for i, fn in enumerate(ORDEN, 1):
        t = fn(F)
        t.leyenda = xref(T(t.leyenda))
        t.notas = [xref(T(x)) for x in t.notas]
        t.cols = [T(c) for c in t.cols]
        t.filas = [[T(c) if isinstance(c, str) else c for c in fila] for fila in t.filas]
        tid = "S%d" % i
        filas_txt, trazas = [], []
        for a, fila in enumerate(t.filas):
            ft = []
            for b, cel in enumerate(fila):
                cel = redirigir(cel)
                txt, crudos = resolver(F, cel)
                ft.append(txt)
                trazas.append({"fila": a, "col": b, "spec": cel, "crudos": crudos})
            if len(ft) != len(t.cols):
                raise ValueError("%s: fila %d con %d celdas para %d columnas" % (tid, a, len(ft), len(t.cols)))
            filas_txt.append(ft)
        largs, ltr, nargs = [], [], []
        for cel in t.args:
            cel = redirigir(cel)
            txt, crudos = resolver(F, cel)
            largs.append(txt)
            ltr.append({"spec": cel, "crudos": crudos, "texto": txt})
        for cel in t.nargs:
            cel = redirigir(cel)
            txt, crudos = resolver(F, cel)
            nargs.append(txt)
            ltr.append({"spec": cel, "crudos": crudos, "texto": txt, "nota": True})
        leyenda = t.leyenda.format(*largs)
        if nargs:
            t.notas = [x.format(*nargs) for x in t.notas]
        df = pd.DataFrame(filas_txt, columns=t.cols)
        tablas.append((tid, t, df, trazas, ltr, leyenda))
        md.append("**%s %s.** %s\n" % (TABLA, tid, leyenda))
        md.append(a_markdown(df))
        if t.notas:
            md.append("\n".join(t.notas) + "\n")
    return tablas, "\n".join(md)


def a_markdown(df):
    def esc(x):
        return str(x).replace("|", "\\|")
    lin = ["| " + " | ".join(esc(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lin.append("| " + " | ".join(esc(x) for x in r.tolist()) + " |")
    return "\n".join(lin) + "\n"


def escribir():
    F = Fuentes()
    tablas, md = construir(F)
    if IDIOMA == "en":                  # solo el Markdown inglés; los CSV y trazas son los del español
        MD_EN.parent.mkdir(parents=True, exist_ok=True)
        MD_EN.write_text(md, encoding="utf-8")
        print("escritas %d tablas en %s" % (len(tablas), MD_EN.relative_to(RAIZ)))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    traz = {}
    for tid, t, df, trazas, ltr, ley in tablas:
        df.to_csv(OUT / ("tabla_S%02d.csv" % int(tid[1:])), index=False)
        traz[tid] = {"titulo": t.clave, "celdas": trazas, "leyenda": ltr}
    (OUT / "trazas.json").write_text(json.dumps(traz, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    MD.write_text(md, encoding="utf-8")
    print("escritas %d tablas en %s y %s" % (len(tablas), OUT.relative_to(RAIZ), MD.relative_to(RAIZ)))
    for tid, t, df, *_ in tablas:
        print("  %-4s %-60s %3d filas" % (tid, t.clave, len(df)))


# Comprobaciones propias de la versión inglesa
RESTOS_ES = [r"[áéíóúñ¿¡]", r"\bIC95\b", r"\bTabla\b", r"\bno aplica\b", r"\bsemillas?\b", r"\bde la\b",
             r"\bdel\b", r"\bfrente a\b", r"\becuaci", r"\bsí\b", r"\d \d{3}\b"]
PROHIBIDAS_EN = [r"\bnovel\b", r"\bcomprehensive\b", r"\brobust\b", r"\bdelve", r"\bshed light\b", r"\bnotably\b",
                 r"\bit is worth noting\b", r"\bcrucial\b", r"\blandscape\b", r"\bleverag", r"\bwe\b"]


def _cifras(texto, idioma):
    """Multiconjunto de los valores absolutos de las cifras de un texto, con los miles de cada idioma."""
    global IDIOMA
    antes = IDIOMA
    IDIOMA = idioma
    try:
        return sorted(round(abs(x), 6) for x in numeros(texto))
    finally:
        IDIOMA = antes


def verificar_en():
    """Verificación de la versión inglesa: Markdown reconstruido, redondeo de cada cifra contra su valor crudo y
    mismas cifras que la versión española en cada celda, cabecera, leyenda y nota."""
    global IDIOMA
    fallos, n_celdas, n_cifras = [], 0, 0
    IDIOMA = "es"
    tablas_es, _ = construir(Fuentes())
    IDIOMA = "en"
    tablas, md = construir(Fuentes())
    escrito = MD_EN.read_text(encoding="utf-8") if MD_EN.exists() else ""
    if escrito != md:
        fallos.append("el Markdown inglés escrito no coincide con el reconstruido desde las fuentes")
    for (tid, t, df, trazas, ltr, ley), (tid_es, t_es, df_es, _, _, ley_es) in zip(tablas, tablas_es):
        if tid != tid_es or df.shape != df_es.shape:
            fallos.append("%s: forma distinta de la versión española" % tid)
            continue
        es_csv = pd.read_csv(OUT / ("tabla_S%02d.csv" % int(tid[1:])), dtype=str, keep_default_na=False)
        if es_csv.shape != df.shape or (es_csv.values != df_es.values).any():
            fallos.append("%s: la versión española reconstruida no coincide con results/suplemento" % tid)
        for c_en, c_es in zip(df.columns, df_es.columns):
            if _cifras(c_en, "en") != _cifras(c_es, "es"):
                fallos.append("%s: cifras distintas en la cabecera '%s' / '%s'" % (tid, c_en, c_es))
        for tr in trazas:
            n_celdas += 1
            txt, txt_es = df.iat[tr["fila"], tr["col"]], df_es.iat[tr["fila"], tr["col"]]
            if _cifras(txt, "en") != _cifras(txt_es, "es"):
                fallos.append("%s (%d, %d): cifras distintas: '%s' / '%s'" % (tid, tr["fila"], tr["col"], txt, txt_es))
            if tr["crudos"]:
                nums = numeros(txt)
                for (v, d), x in zip(tr["crudos"], nums):
                    n_cifras += 1
                    if abs(x - v) > 0.5 * 10 ** (-d) + 1e-9:
                        fallos.append("%s (%d, %d): %s frente a %r" % (tid, tr["fila"], tr["col"], txt, v))
        if _cifras(ley, "en") != _cifras(ley_es, "es"):
            fallos.append("%s: la leyenda inglesa no tiene las mismas cifras que la española" % tid)
        for tr in ltr:
            for (v, d), x in zip(tr["crudos"], numeros(tr["texto"])):
                n_cifras += 1
                if abs(x - v) > 0.5 * 10 ** (-d) + 1e-9:
                    fallos.append("%s, leyenda: %s frente a %r" % (tid, tr["texto"], v))
            if tr["texto"] not in ley + "\n" + "\n".join(t.notas):
                fallos.append("%s, leyenda: falta '%s'" % (tid, tr["texto"]))
        for n_en, n_es in zip(t.notas, t_es.notas):
            if _cifras(n_en, "en") != _cifras(n_es, "es"):
                fallos.append("%s: una nota inglesa no tiene las mismas cifras que la española" % tid)
        if len(t.notas) != len(t_es.notas):
            fallos.append("%s: número de notas distinto" % tid)
        print("%-4s %-60s %3d filas" % (tid, t.clave, len(df)))
    for s, nombre in (("—", "guion largo"), ("§", "símbolo de sección")):
        if s in md:
            fallos.append("el Markdown contiene %s" % nombre)
    for k in INTERNOS:
        if k in md:
            fallos.append("el Markdown contiene el identificador interno '%s'" % k)
    for rx in INTERNOS_RE + RESTOS_ES:
        for m in re.finditer(rx, md):
            fallos.append("el Markdown contiene '%s' (%s)" % (md[max(0, m.start() - 30):m.end() + 30].replace("\n", " "), rx))
    for rx in PROHIBIDAS_EN:
        for m in re.finditer(rx, md, flags=re.I):
            fallos.append("palabra excluida por el glosario: '%s'" % md[max(0, m.start() - 30):m.end() + 30].replace("\n", " "))
    print("celdas comprobadas: %d; cifras cotejadas con su valor crudo: %d" % (n_celdas, n_cifras))
    if fallos:
        print("FALLOS (%d):" % len(fallos))
        for f in fallos:
            print("  " + f)
        return 1
    print("0 fallos: las %d tablas inglesas coinciden con sus fuentes y con la versión española" % len(tablas))
    return 0


def verificar():
    if IDIOMA == "en":
        return verificar_en()
    F = Fuentes()                                   # lecturas nuevas de todas las fuentes
    tablas, md = construir(F)
    fallos, n_celdas, n_cifras = [], 0, 0
    escrito = MD.read_text(encoding="utf-8")
    if escrito != md:
        fallos.append("el Markdown escrito no coincide con el reconstruido desde las fuentes")
    for tid, t, df, trazas, ltr, ley in tablas:
        p = OUT / ("tabla_S%02d.csv" % int(tid[1:]))
        w = pd.read_csv(p, dtype=str, keep_default_na=False)
        if list(w.columns) != list(df.columns) or w.shape != df.shape:
            fallos.append("%s: forma o columnas distintas en %s" % (tid, p.name))
            continue
        dif = (w.values != df.values)
        if dif.any():
            fallos.append("%s: %d celdas del CSV escrito distintas de las fuentes" % (tid, int(dif.sum())))
        for tr in trazas:
            n_celdas += 1
            txt = w.iat[tr["fila"], tr["col"]]
            if tr["crudos"]:
                nums = numeros(txt)
                if len(nums) < len(tr["crudos"]):
                    fallos.append("%s (%d, %d): faltan cifras en '%s'" % (tid, tr["fila"], tr["col"], txt))
                    continue
                for (v, d), x in zip(tr["crudos"], nums):
                    n_cifras += 1
                    if abs(x - v) > 0.5 * 10 ** (-d) + 1e-9:
                        fallos.append("%s (%d, %d): %s frente a %r" % (tid, tr["fila"], tr["col"], txt, v))
        for tr in ltr:
            for (v, d), x in zip(tr["crudos"], numeros(tr["texto"])):
                n_cifras += 1
                if abs(x - v) > 0.5 * 10 ** (-d) + 1e-9:
                    fallos.append("%s, leyenda: %s frente a %r" % (tid, tr["texto"], v))
            if tr["texto"] not in ley + "\n" + "\n".join(t.notas):
                fallos.append("%s, leyenda: falta '%s'" % (tid, tr["texto"]))
        print("%-4s %-60s %3d filas  %s" % (tid, t.clave, len(df), "fuentes: " + ", ".join(sorted(fuentes_de(trazas, ltr)))))
    for s, nombre in (("—", "guion largo"), ("§", "símbolo de sección")):
        if s in escrito:
            fallos.append("el Markdown contiene %s" % nombre)
    for k in INTERNOS:
        if k in escrito:
            fallos.append("el Markdown contiene el identificador interno '%s'" % k)
    for rx in INTERNOS_RE:
        if re.search(rx, escrito):
            fallos.append("el Markdown contiene un identificador interno (%s)" % rx)
    for rx in RETIRADAS_ES:
        for m in re.finditer(rx, escrito):
            fallos.append("vocabulario retirado en la revisión: '%s'" % escrito[max(0, m.start() - 30):m.end() + 30].replace("\n", " "))
    print("celdas comprobadas: %d; cifras cotejadas con su valor crudo: %d" % (n_celdas, n_cifras))
    if fallos:
        print("FALLOS (%d):" % len(fallos))
        for f in fallos:
            print("  " + f)
        return 1
    print("0 fallos: las %d tablas y el Markdown coinciden con sus fuentes" % len(tablas))
    return 0


def fuentes_de(trazas, ltr):
    s = set()
    for tr in list(trazas) + list(ltr):
        sp = tr["spec"]
        if isinstance(sp, dict):
            if "src" in sp:
                s.add(sp["src"])
            elif sp.get("k") == "agg":
                if sp["n"].startswith("cad_"):
                    s.add("cadenas/" + sp["a"][0])
                    continue
                for x in sp["a"]:
                    if isinstance(x, str) and (x.endswith(".csv") or x.endswith(".json")):
                        s.add(x)
                if sp["n"] in ("espacio", "defecto", "espacio_comun", "n_ensayos", "rejilla_dfr", "reps_dfr", "hp"):
                    s.add("src/baselines (ast)")
    return compactar(s)


def compactar(s):
    """Agrupa los JSON de un mismo directorio de configuraciones congeladas o de cadenas."""
    grupos, out = {}, set()
    for x in s:
        d = x.rsplit("/", 1)[0] if "/" in x else ""
        if d.endswith("congelados") or d == "cadenas":
            grupos.setdefault(d, []).append(x)
        else:
            out.add(x)
    for d, xs in grupos.items():
        out.add("%s/*.json (%d %s)" % (d, len(xs), "patrones" if d == "cadenas" else "ficheros"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--idioma", choices=("es", "en"), default="es")
    a = ap.parse_args()
    IDIOMA = a.idioma
    sys.exit(verificar() if a.verificar else (escribir() or 0))
