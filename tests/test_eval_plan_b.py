"""Pruebas del plan B de evaluación (docs/08_evaluacion.md, versión 2): auditoría de COVIDx (src/eval/audit.py,
scripts/eval/auditoria_covidx.py), dosis-respuesta de B3a (src/eval/dosis.py, scripts/eval/informe_b3.py) y piloto de B3a
(scripts/eval/puerta6.py). CPU y datos sintéticos, salvo la reproducción de la Tabla E4, que usa las características
ya extraídas si están en disco. Ejecutar desde la raíz:
    pytest -q tests/test_eval_plan_b.py --basetemp=/tmp/pt_planb
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import audit as A  # noqa: E402
from src.eval import dosis as D  # noqa: E402
from src.eval import stats as S  # noqa: E402


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _datos(n_pat=150, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_pat), rng.integers(1, 4, n_pat)).astype(str)
    y = (rng.random(len(g)) < 0.4).astype(int)
    p = rng.random(len(g)) + 0.6 * y
    if ties:
        p = np.round(p, 1)
    return y, p, g


# ------------------------------------------------------------------ remuestreo rápido
@pytest.mark.parametrize("ties", [True, False])
def test_auc_remuestreo_igual_que_bootstrap_de_stats(ties):
    y, p, g = _datos(ties=ties)
    rapido = D.auc_remuestreo(y, p, g, B=300, seed=2026, bloque=70)[:, 0]
    lento = S.bootstrap_draws(lambda i: [S.auc_roc(y[i], p[i])], g, 300, 2026)[:, 0]
    assert np.allclose(rapido, lento, equal_nan=True, atol=1e-12)


def test_auc_remuestreo_nan_si_falta_una_clase():
    g = np.array(["a", "b", "c"])
    y = np.array([0, 1, 1])
    a = D.auc_remuestreo(y, np.array([0.1, 0.5, 0.9]), g, B=200, seed=1)[:, 0]
    assert np.isnan(a).any() and np.isfinite(a).any()


def test_diferencia_igual_que_prueba_primaria_de_stats():
    y, p1, g = _datos(seed=3)
    rng = np.random.default_rng(9)
    P1 = np.vstack([p1 + rng.normal(0, 0.2, len(y)) for _ in range(2)])
    P2 = np.vstack([p1 + rng.normal(0, 0.4, len(y)) for _ in range(2)])
    r = A.diferencia(y, P1, P2, g, B=400, seed=2026, margen=0.02)
    s = S.paired_auc_test_seedmean(y, P1, P2, g, B=400, seed=2026, margin=0.02)
    assert r["delta"] == pytest.approx(s["delta"])
    assert r["ic95_inf"] == pytest.approx(s["ic95"][0]) and r["ic95_sup"] == pytest.approx(s["ic95"][1])
    assert r["p_superioridad"] == pytest.approx(s["p_unilateral"])
    assert r["p_no_inferioridad"] == pytest.approx(s["no_inferioridad"]["p_unilateral_margen"])
    assert r["no_inferior"] == s["no_inferioridad"]["no_inferior"]


def test_diferencia_replica_una_referencia_unica_frente_a_varias_semillas():
    y, p, g = _datos(seed=4, ties=False)
    P2 = np.vstack([p * 0.5, p ** 2])
    r = A.diferencia(y, p, P2, g, B=100, seed=1)
    assert r["n_semillas"] == 2 and len(r["delta_por_semilla"]) == 2


# ------------------------------------------------------------------ auditoría
def test_coleccion_y_proyeccion_desde_el_nombre():
    ids = ["sub-S03044_ses-E06138_run-1_bp-chest_vp-ap_cr-corrected.png", "sub-S308200_ses-E1_vp-pa_dx.png",
           "912449f8-feda-4eef-89dd-55a2786e77f8.png"]
    assert list(A.coleccion_bimcv(ids)) == ["antigua", "nueva", ""]
    assert list(A.proyeccion(ids)) == ["AP", "PA", ""]


def _meta_sintetica(rng, n=120, con_reserva=False):
    filas = []
    def add(conj, k, label, digits, source="bimcv"):
        for i in range(k):
            sid = str(10 ** (digits - 1) + i + (500 if label == 0 else 0))
            filas.append({"image_id": "sub-S%s_ses-E%d_vp-%s.png" % (sid, i, "ap" if i % 2 else "pa"),
                          "patient_id": "S%s" % sid, "label": label, "source": source, "conjunto": conj})
    add("tr_bimcv_pos", n, 1, 5); add("tr_bimcv_pos", n, 1, 6); add("tr_bimcv_neg", 40, 0, 5)
    add("val", 30, 0, 5); add("val", 60, 1, 5)
    if con_reserva:
        add("res_pos_antigua", 50, 1, 5); add("res_pos_nueva", 50, 1, 6); add("res_neg", 30, 0, 5)
    m = pd.DataFrame(filas)
    m["coleccion"] = A.coleccion_bimcv(m["image_id"])
    m["proyeccion"] = A.proyeccion(m["image_id"])
    m["val_robust"] = (m["conjunto"] == "val") & (m["source"] == "bimcv")
    X = rng.normal(0, 1, (len(m), 16))
    X[:, 0] += 2.0 * ((m["coleccion"] == "antigua") & (m["label"] == 1)).to_numpy()
    return {"pooled": X, "penult": X[:, :8], "logit": X[:, 0]}, m


def test_detector_no_ve_negativos_y_usa_el_reservado_si_existe():
    rng = np.random.default_rng(0)
    f, m = _meta_sintetica(rng)
    s, info = A.detector(f, m)
    assert info["negativos_en_ajuste"] == 0 and info["n_ajuste"] == 240 and info["n_antigua"] == 120
    assert "muestra" in info["origen_ajuste"]
    f2, m2 = _meta_sintetica(rng, con_reserva=True)
    _, info2 = A.detector(f2, m2)
    assert info2["n_ajuste"] == 100 and "reservados" in info2["origen_ajuste"]
    vr = m["val_robust"].to_numpy()
    assert S.auc_roc(m["label"].to_numpy()[vr], s[vr]) > 0.8        # los positivos de val son de la colección antigua


def test_detector_rechaza_un_ajuste_con_negativos():
    rng = np.random.default_rng(1)
    f, m = _meta_sintetica(rng)
    m.loc[m.index[0], "label"] = 0
    with pytest.raises(ValueError):
        A.detector(f, m)


def test_cargar_rechaza_ficheros_con_conjuntos_test(tmp_path):
    rng = np.random.default_rng(2)
    f, m = _meta_sintetica(rng)
    m.loc[m.index[:3], "conjunto"] = "test"
    np.savez(tmp_path / "x.npz", **f)
    m[["image_id", "patient_id", "label", "source", "conjunto"]].to_csv(tmp_path / "x_meta.csv", index=False)
    with pytest.raises(S.TestSetForbidden):
        A.cargar(tmp_path, "x")


def test_tabla_por_coleccion_separa_positivos_de_bimcv_por_coleccion():
    df = pd.DataFrame({"image_id": ["sub-S1%04d_a.png" % i for i in range(30)] + ["sub-S2%05d_a.png" % i for i in range(30)]
                       + ["sub-S3%04d_a.png" % i for i in range(30)],
                       "label": [1] * 60 + [0] * 30, "source": [1] * 90, "patient": ["p%d" % i for i in range(90)]})
    sc = np.r_[np.ones(30), np.zeros(30), np.zeros(30) + 0.5]
    t = A.tabla_por_coleccion(df, sc, B=50, seed=1)
    d = t.set_index("positivos")["auc"]
    assert d["bimcv_antigua"] == pytest.approx(1.0) and d["bimcv_nueva"] == pytest.approx(0.0)


E4_DIR = ROOT / "results/diag/expl_lectura"


@pytest.mark.skipif(not (E4_DIR / "baseline_erm_seed123_best.npz").exists(), reason="características no extraídas")
def test_auditoria_reproduce_la_tabla_E4(tmp_path):
    mod = _load("scripts/eval/auditoria_covidx.py", "auditoria_covidx")
    tags = ["expl_erm_reserva_seed42_best", "baseline_erm_seed42_best", "baseline_erm_seed123_best"]
    code = mod.main(["--tags", *tags, "--B", "50", "--sin-metodos", "--out", str(tmp_path)])
    rep = pd.read_csv(tmp_path / "reproduccion_tabla_E4.csv")
    assert code == 0 and rep["cumple"].all() and len(rep) == 15
    assert (rep["diferencia"].abs() <= 0.005).all()


# ------------------------------------------------------------------ dosis-respuesta
def test_nombres_de_experimento_y_rho():
    assert D.parse_nombre("b3a_A_erm_rho095") == {"bench": "b3a", "orient": "A", "metodo": "erm", "rho": 0.95}
    assert D.parse_nombre("b3a_B_dfa_fuente_rho100")["metodo"] == "dfa_fuente"
    assert D.parse_nombre("baseline_erm") is None


def test_rho_del_nombre_y_de_la_configuracion_deben_coincidir(tmp_path):
    D.arbol_sintetico_b3(tmp_path, "informativo", metodos=("erm",), rhos=(0.5,), semillas=(0,), n_test=10, n_val=10)
    (tmp_path / "b3a_A_erm_rho050").rename(tmp_path / "b3a_A_erm_rho060")
    with pytest.raises(ValueError):
        D.descubrir(tmp_path)


def test_descubrir_y_leer_no_abren_el_test_sin_permiso(tmp_path):
    D.arbol_sintetico_b3(tmp_path, "informativo", metodos=("erm",), rhos=(0.5,), semillas=(0,), n_test=10, n_val=10,
                         estados=("best", "bestdc"))
    with pytest.raises(S.TestSetForbidden):
        D.descubrir(tmp_path, conjuntos=("test_desconf",))
    with pytest.raises(S.TestSetForbidden):
        D.descubrir(tmp_path, conjuntos=("test_invertido",))
    f = tmp_path / "b3a_A_erm_rho050" / "seed0" / "predictions" / "bestdc_test_desconf.csv"
    assert f.exists()
    with pytest.raises(S.TestSetForbidden):
        S.read_predictions(f, forbid_test=True)


def test_area_dosis_de_una_recta_es_su_media():
    r = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    a = 0.9 - 0.2 * (np.array(r) - 0.5)
    assert D.area_dosis(r, a) == pytest.approx(0.9 - 0.2 * 0.25)


def _curva(tmp_path, escenario, metodos=("erm", "groupdro", "dfr"), B=300):
    D.arbol_sintetico_b3(tmp_path, escenario, metodos=metodos, semillas=(0, 1, 2), n_test=120, n_val=40)
    inv = D.descubrir(tmp_path, conjuntos=(D.CONJ_VAL_DESCONF,))
    preds = D.cargar(inv, D.CONJ_VAL_DESCONF)
    c, r, k, info = D.curva(preds, B=B, seed=2026)
    return c, r, D.familias_holm(k), info


def test_escenario_informativo_erm_cae_y_dfr_sin_rho_1(tmp_path):
    c, r, k, info = _curva(tmp_path, "informativo")
    e = c[c.metodo == "erm"].set_index("rho")["auc_media_semillas"]
    assert e[0.5] - e[1.0] > 0.1
    assert 1.0 not in info["niveles_por_metodo"]["dfr"]
    assert not ((k.metodo == "dfr") & (k.medida == "delta_auc_rho100")).any()
    assert np.isfinite(r[(r.metodo == "dfr") & (r.medida == "area_dosis")]["estimacion"]).all()  # área de 0.5 a 0.99


def test_area_no_se_calcula_con_una_curva_incompleta(tmp_path):
    D.arbol_sintetico_b3(tmp_path, "informativo", metodos=("erm", "lisa"), semillas=(0, 1), n_test=40, n_val=20)
    inv = D.descubrir(tmp_path, conjuntos=(D.CONJ_VAL_DESCONF,))
    preds = {k: v for k, v in D.cargar(inv, D.CONJ_VAL_DESCONF).items() if not (k[0] == "lisa" and k[1] == 0.99)}
    _, r, k, _ = D.curva(preds, B=50)
    assert np.isnan(r[(r.metodo == "lisa") & (r.medida == "area_dosis")]["estimacion"]).all()
    assert not ((k.metodo == "lisa") & (k.medida == "delta_area_dosis")).any()


def test_escenario_un_metodo_mejor_se_detecta_con_holm(tmp_path):
    _, _, k, _ = _curva(tmp_path, "uno_mejor")
    fila = k[(k.metodo == "groupdro") & (k.medida == "delta_auc_rho100")].iloc[0]
    # 17-11-2026 (enmienda E5 de docs/08, hallazgo m1): F-C2-100 es secundaria, no confirmatoria
    assert fila["delta"] > 0.05 and fila["rechaza_holm"] and not fila["confirmatoria"] and fila["papel"] == "secundaria"
    area = k[(k.metodo == "groupdro") & (k.medida == "delta_area_dosis")].iloc[0]
    assert area["rechaza_holm"]


def test_escenario_ninguno_no_declara_mejoras(tmp_path):
    _, _, k, _ = _curva(tmp_path, "ninguno")
    conf = k[k.familia.isin(["F-C2-100", "F-C2-099", "F-C2-area"])]
    assert not conf["rechaza_holm"].any()


def test_coste_en_rho_05_es_no_inferioridad_con_margen(tmp_path):
    _, _, k, _ = _curva(tmp_path, "uno_mejor")
    c = k[k.medida == "coste_rho050_no_inferioridad"].set_index("metodo")
    assert set(c.index) == {"groupdro", "dfr"} and (c["margen"] == D.MARGEN_COSTE).all()
    assert (c["familia"] == "F-C2-coste").all()
    for m, f in c.iterrows():
        assert f["no_inferior"] == (f["ic95_inf"] > -D.MARGEN_COSTE)


def test_puerta6_veredictos(tmp_path):
    for esc, ver in (("informativo", "informativo"), ("no_informativo", "no informativo")):
        d = tmp_path / esc
        D.arbol_sintetico_b3(d, esc, metodos=("erm",), rhos=(0.5, 0.9, 1.0), semillas=(0, 1), n_test=60, n_val=150)
        inv = D.descubrir(d, conjuntos=(D.CONJ_VAL_DESCONF,))
        res = D.puerta6(D.cargar(inv, D.CONJ_VAL_DESCONF), B=300)
        assert res["veredicto"] == ver, (esc, res)


def test_puerta6_no_evaluable_con_una_semilla(tmp_path):
    D.arbol_sintetico_b3(tmp_path, "informativo", metodos=("erm",), rhos=(0.5, 1.0), semillas=(0,), n_test=10, n_val=20)
    inv = D.descubrir(tmp_path, conjuntos=(D.CONJ_VAL_DESCONF,))
    assert D.puerta6(D.cargar(inv, D.CONJ_VAL_DESCONF), B=50)["veredicto"] == "no evaluable"


def test_guiones_b3_y_puerta6_de_extremo_a_extremo(tmp_path):
    p6 = _load("scripts/eval/puerta6.py", "puerta6")
    assert p6.main(["--demo", "informativo", "--out", str(tmp_path / "p6"), "--B", "100"]) == 0
    assert p6.main(["--demo", "no_informativo", "--out", str(tmp_path / "p6n"), "--B", "100"]) == 2
    assert (tmp_path / "p6" / "puerta6.md").exists()
    b3 = _load("scripts/eval/informe_b3.py", "informe_b3")
    assert b3.main(["--demo", "uno_mejor", "--out", str(tmp_path / "b3"), "--B", "100"]) == 0
    for f in ("curva.csv", "resumen.csv", "comparacion_erm.csv", "informe.md", "informe.json"):
        assert (tmp_path / "b3" / f).exists()
    with pytest.raises(SystemExit):
        b3.main(["--root", str(tmp_path / "nada"), "--conjunto", "test_desconf", "--permitir-test", "--out", str(tmp_path / "x")])
