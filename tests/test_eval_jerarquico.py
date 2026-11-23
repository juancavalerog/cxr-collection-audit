"""Pruebas del remuestreo jerárquico de semillas y pacientes (src/eval/jerarquico.py, src/eval/c2.py) y del aporte C3
(src/eval/c3.py). Casos sintéticos de varianza conocida; CPU. Ejecutar desde la raíz:
    pytest -q tests/test_eval_jerarquico.py --basetemp=/tmp/pt_jer
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import c2 as C2  # noqa: E402
from src.eval import c3 as C3  # noqa: E402
from src.eval import dosis as D  # noqa: E402
from src.eval import jerarquico as J  # noqa: E402
from src.eval import stats as S  # noqa: E402


def _sin_congelacion(motivo):
    return {"doble": True, "motivo": motivo}


# ----------------------------------------------------------------------------- casos de varianza conocida
def test_varianza_conocida_solo_semillas():
    """Sin variación de pacientes, el término de semillas tiene la varianza entre semillas / K (corregido, con
    denominador K - 1) y el remuestreo doble la poblacional / K."""
    v = np.array([0.01, 0.03, -0.02, 0.05, 0.00])
    B = 40000
    X = np.broadcast_to(v[None, None, :], (B, 1, 5)).copy()
    P = np.ones((1, 5), bool)
    h = J.remuestreo_jerarquico(X, P, seed=1, punt=v[None])[:, 0]
    assert h.mean() == pytest.approx(v.mean(), abs=5e-4)
    assert h.std() == pytest.approx(v.std(ddof=1) / np.sqrt(5), rel=0.03)
    d = J.remuestreo_jerarquico(X, P, seed=1, modo="doble")[:, 0]
    assert d.std() == pytest.approx(v.std(ddof=0) / np.sqrt(5), rel=0.03)
    with pytest.raises(ValueError):
        J.remuestreo_jerarquico(X, P)


def test_la_interaccion_no_se_cuenta_dos_veces():
    """Diferencias por semilla que solo difieren por ruido de imagen (sin efecto de semilla): el corregido coincide
    con el de pacientes y el doble es más ancho."""
    rng = np.random.default_rng(5)
    B, K = 3000, 5
    Xr = rng.normal(0, 0.01, size=(B, 1, K))                # V_1 = 1e-4 por semilla, independiente entre semillas
    punt = rng.normal(0, 0.01, size=(1, K))                   # s² ≈ Var(c): σ_S² ≈ 0
    c = J.componentes(Xr, punt, np.ones((1, K), bool))
    assert c["var_interaccion"][0] == pytest.approx(1e-4, rel=0.15)
    h = J.remuestreo_jerarquico(Xr, np.ones((1, K), bool), punt=punt)[:, 0]
    d = J.remuestreo_jerarquico(Xr, np.ones((1, K), bool), modo="doble")[:, 0]
    hp = J.media_semillas(Xr, np.ones((1, K), bool))[:, 0]
    assert h.std() < 1.5 * hp.std() and d.std() > 1.2 * hp.std()


def test_una_semilla_se_reduce_al_remuestreo_de_pacientes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2, 1))
    h = J.remuestreo_jerarquico(X, np.ones((2, 1), bool), punt=np.zeros((2, 1)))
    assert np.allclose(h, X[:, :, 0])


def test_semillas_iguales_coinciden_con_pacientes():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(300, 3, 1))
    X = np.repeat(base, 5, axis=2)
    P = np.ones((3, 5), bool)
    assert np.allclose(J.remuestreo_jerarquico(X, P, punt=np.zeros((3, 5))), J.media_semillas(X, P))


def _experimento(rng, n=200, K=5, tau=0.25, mu=(1.0, 1.0)):
    """Una réplica completa: n pacientes (una imagen cada uno), K semillas de ERM y de otro método. La puntuación es
    (mu_m + u_{m,s}) * y + z_i + e_{i,s,m}; u es el efecto de la semilla (desviación tau) y z el del paciente."""
    y = (np.arange(2 * n) < n).astype(int)
    z = rng.normal(0, 0.7, 2 * n)
    preds = {}
    for m, mu_m in zip(("erm", "otro"), mu):
        for s in range(K):
            u = rng.normal(0, tau)
            p = (mu_m + u) * y + z + rng.normal(0, 1.0, 2 * n)
            preds[(m, 0.9, s)] = pd.DataFrame({"image_id": ["i%04d" % i for i in range(2 * n)], "label": y,
                                               "source": 0, "patient": ["p%04d" % i for i in range(2 * n)],
                                               "prob": 1 / (1 + np.exp(-p))})
    return preds


def _delta(preds):
    K = len({k[2] for k in preds})
    return np.mean([S.auc_roc(preds[("otro", 0.9, s)].label, preds[("otro", 0.9, s)].prob)
                    - S.auc_roc(preds[("erm", 0.9, s)].label, preds[("erm", 0.9, s)].prob) for s in range(K)])


@pytest.mark.parametrize("tau", [0.0, 0.3])
def test_error_tipico_jerarquico_frente_al_verdadero(tau):
    """Con el error típico verdadero de la media por semilla (Monte Carlo sobre semillas y pacientes nuevos), el
    jerárquico lo recoge (cociente de 0.75 a 1.35) y, si la varianza entre semillas domina, el de pacientes con
    modelos fijos queda muy por debajo."""
    rng = np.random.default_rng(11)
    verdad = np.std([_delta(_experimento(rng, tau=tau)) for _ in range(250)])
    ej, ep = [], []
    for r in range(12):
        preds = _experimento(np.random.default_rng(1000 + r), tau=tau)
        _, _, k, _ = C2.curva_jerarquica(preds, B=300, seed=2026, diseno=(0.9,))
        f = k[k.medida == "delta_auc_rho090"].iloc[0]
        ej.append((f.ic95_sup - f.ic95_inf) / (2 * 1.96))
        ep.append((f.ic95_sup_pacientes - f.ic95_inf_pacientes) / (2 * 1.96))
    rj, rp = np.mean(ej) / verdad, np.mean(ep) / verdad
    assert 0.75 < rj < 1.35, (rj, rp, verdad)
    if tau > 0:
        assert rp < 0.6, (rj, rp)


def test_t_pareada():
    t = J.t_pareada([0.01, 0.02, 0.03])
    assert t["t_media"] == pytest.approx(0.02) and t["t_gl"] == 2 and t["t_p_unilateral"] < 0.05
    assert np.isnan(J.t_pareada([0.1])["t_p_unilateral"])


# ----------------------------------------------------------------------------- C2 sobre árboles sintéticos
def _arbol(tmp, escenario="uno_mejor", semillas=(0, 1, 2), metodos=("erm", "groupdro", "dfr")):
    D.arbol_sintetico_b3(tmp, escenario, metodos=metodos, semillas=semillas, n_test=100, n_val=40,
                         estados=("best", "bestdc"))
    return tmp


def test_curva_jerarquica_y_familias(tmp_path):
    _arbol(tmp_path)
    preds, inv, excl = C2.cargar_b3(tmp_path, "test_desconf", verificador=_sin_congelacion)
    c, r, k, info = C2.curva_jerarquica(preds, B=200)
    k = C2.familias_c2(k)
    assert set(k[k.papel == "confirmatoria"].familia) == {"F-C2-area"}
    assert "F-C2-caida" not in set(k.familia)
    f = k[(k.metodo == "groupdro") & (k.medida == "delta_auc_rho100")].iloc[0]
    assert f.familia == "F-C2-100" and f.papel == "secundaria" and f.rechaza_holm
    # el intervalo jerárquico no es más estrecho que el de pacientes (en promedio)
    a = (k.ic95_sup - k.ic95_inf).mean()
    b = (k.ic95_sup_pacientes - k.ic95_inf_pacientes).mean()
    assert a >= 0.9 * b
    assert not ((k.metodo == "dfr") & (k.medida == "delta_auc_rho100")).any()


def test_cualquier_numero_de_semillas_por_nivel(tmp_path):
    _arbol(tmp_path, semillas=(0, 1, 2), metodos=("erm", "groupdro"))
    D.arbol_sintetico_b3(tmp_path / "extra", "uno_mejor", metodos=("erm", "groupdro"), rhos=(0.9, 0.95, 0.99),
                         semillas=(3, 4), n_test=100, n_val=40)
    import shutil
    for d in (tmp_path / "extra").iterdir():
        for s in d.iterdir():
            shutil.copytree(s, tmp_path / d.name / s.name)
    shutil.rmtree(tmp_path / "extra")
    preds, _, _ = C2.cargar_b3(tmp_path, "test_desconf", verificador=_sin_congelacion)
    c, r, k, info = C2.curva_jerarquica(preds, B=100)
    n = c[c.metodo == "erm"].set_index("rho").n_semillas
    assert n[0.5] == 3 and n[0.9] == 5 and n[0.99] == 5
    assert np.isfinite(k[(k.metodo == "groupdro") & (k.medida == "delta_area_dosis")].delta).all()


def test_leer_un_test_exige_la_congelacion(tmp_path):
    _arbol(tmp_path, metodos=("erm",), semillas=(0,))
    with pytest.raises(Exception) as e:
        C2.cargar_b3(tmp_path, "test_desconf")          # sin doble: no hay congelación del protocolo
    assert "bloqueado" in str(e.value).lower() or "Bloqueado" in type(e.value).__name__
    llamadas = []
    C2.cargar_b3(tmp_path, "test_desconf", verificador=lambda m: llamadas.append(m) or {})
    assert llamadas and "test_desconf" in llamadas[0]
    C2.cargar_b3(tmp_path, "val_desconf", verificador=lambda m: (_ for _ in ()).throw(AssertionError("no debe llamarse")))


def test_dfr_no_comparable_en_validacion(tmp_path):
    _arbol(tmp_path, metodos=("erm", "dfr"), semillas=(0,))
    preds, inv, excl = C2.cargar_b3(tmp_path, "val_desconf")
    assert not any(k[0] == "dfr" for k in preds) and len(excl) and (excl.metodo == "dfr").all()
    assert C2.no_comparable("b3a", "dfr_best", "val_desconf") and C2.no_comparable("b3a", "dfr_bestdc", "val")
    assert C2.no_comparable("b2", "dfr_best", "val_robust") and not C2.no_comparable("b3a", "dfr_best", "test_desconf")
    assert not C2.no_comparable("b3a", "best", "val_desconf")


def test_invertido_fuente_brecha_y_orientacion(tmp_path):
    _arbol(tmp_path / "a", metodos=("erm", "groupdro"), semillas=(0, 1))
    D.arbol_sintetico_b3(tmp_path / "b", "uno_mejor", metodos=("erm",), rhos=(0.5, 0.8, 1.0), semillas=(0, 1),
                         n_test=100, n_val=40, orient="B")
    dc, _, _ = C2.cargar_b3(tmp_path / "a", "test_desconf", verificador=_sin_congelacion)
    iv, _, _ = C2.cargar_b3(tmp_path / "a", "test_invertido", verificador=_sin_congelacion)
    vv, _, _ = C2.cargar_b3(tmp_path / "a", "val")
    t = C2.invertido_frente_desconf(dc, iv, B=100)
    e = t[t.metodo == "erm"].set_index("rho")
    assert e.loc[1.0, "delta_invertido_menos_desconf"] < -0.05 and e.loc[1.0, "ic95_sup"] < 0
    f = C2.auc_por_fuente(dc, B=50)
    assert set(f.fuente) == {"A (NIH)", "B (CheXpert)"}
    b = C2.brecha_especificidad(dc, vv, B=100)
    eb = b[b.metodo == "erm"].set_index("rho")
    assert eb.loc[1.0, "brecha_especificidad"] > eb.loc[0.5, "brecha_especificidad"]
    db, _, _ = C2.cargar_b3(tmp_path / "b", "test_desconf", orient="B", verificador=_sin_congelacion)
    o = C2.orientacion_b(dc, db, B=100)
    assert np.isfinite(o["diferencia_A_menos_B"]) and o["semillas_B"] == [0, 1]


def test_b2_familias_por_fuente(tmp_path):
    C2.arbol_sintetico_b2(tmp_path, semillas=(0, 1, 2))
    pc = {c: C2.cargar_b2(tmp_path, c, verificador=_sin_congelacion)[0] for c in C2.CONJUNTOS_B2}
    curva, comp = C2.informe_b2_tablas(pc, B=200)
    assert set(comp.familia) == {"F-C2R-shenzhen", "F-C2R-montgomery"}
    g = comp[(comp.metodo == "groupdro") & (comp.conjunto == "test_shenzhen")].iloc[0]
    assert g.delta > 0 and g.rechaza_holm
    assert (comp.groupby("familia").n_familia.first() == 3).all()
    with pytest.raises(Exception):
        C2.cargar_b2(tmp_path, "test_shenzhen")


# ----------------------------------------------------------------------------- C3
def test_c3_seleccion_de_epoca(tmp_path):
    import shutil
    _arbol(tmp_path, escenario="informativo", metodos=("erm", "groupdro", "dfr"), semillas=(0, 1, 2))
    for f in tmp_path.glob("b3a_A_dfr_*/seed*/predictions/dfr_*_test_desconf.csv"):   # estados de la etapa ERM
        shutil.copy(f, f.with_name(f.name[4:]))
    pb, pc = C3._estados_epoca(tmp_path, "b3a", "A", "test_desconf", ("erm", "groupdro", "dfr"), _sin_congelacion)
    assert any(k[0] == "dfr" for k in pb)
    pm, pn = C3.seleccion_epoca(pb, pc, B=100)
    assert set(pn[pn.familia == "F-C3"].rho) == {0.9, 0.95, 0.99}
    assert pn.p_holm.notna().sum() == 3 and np.isfinite(pm.delta_sel).all()


def test_c3_spearman_y_regla():
    assert C3.elegir_regla({0: 1.0, 1: 1.0, 2: 0.9}) == 0
    assert C3.elegir_regla({0: 0.5, 1: 0.7, 2: 0.7}) == 1
    rng = np.random.default_rng(0)
    preds, sel = {}, {}
    y = (np.arange(200) < 100).astype(int)
    for q, m in enumerate(("a", "b", "c", "d")):
        for s in range(3):
            p = (0.5 + 0.4 * q) * y + rng.normal(0, 1, 200)
            preds[(m, 0.9, s)] = pd.DataFrame({"image_id": ["i%d" % i for i in range(200)], "label": y, "source": 0,
                                               "patient": ["p%d" % i for i in range(200)], "prob": 1 / (1 + np.exp(-p))})
            sel[(m, 0.9, s)] = 0.9 + 0.01 * q
    t = C3.spearman_metodos(preds, sel, B=100)
    assert t.iloc[0].spearman == pytest.approx(1.0)


def test_c3_hiperparametros_sintetico(tmp_path):
    import json
    rng = np.random.default_rng(0)
    y = (np.arange(100) < 50).astype(int)
    for t in range(8):
        d = tmp_path / ("b3a_A_erm_rho090_t%d" % t) / "seed0"
        (d / "predictions").mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"early_stopping": {"best": 0.99 + 0.001 * (t % 3)}}))
        for conj in ("val_desconf", "test_desconf"):
            p = (0.5 + 0.1 * t) * y + rng.normal(0, 1, 100)
            pd.DataFrame({"image_id": ["%s%d" % (conj, i) for i in range(100)], "label": y, "source": 0,
                          "patient": ["p%s%d" % (conj, i) for i in range(100)],
                          "prob": 1 / (1 + np.exp(-p))}).to_csv(d / "predictions" / ("best_%s.csv" % conj), index=False)
    with pytest.raises(Exception):
        C3.hiperparametros(tmp_path, metodos=("erm",), niveles=(0.9,), B=50)
    r = C3.hiperparametros(tmp_path, metodos=("erm",), niveles=(0.9,), B=50, verificador=_sin_congelacion)
    f = r.iloc[0]
    assert f.n_ensayos == 8 and f.ensayo_regla == 2 and f.perdida_frente_oraculo >= -0.2


# ----------------------------------------------------------------------------- guiones
def _load(rel, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guiones_de_test_se_niegan_sin_congelacion(tmp_path):
    cc = _load("scripts/eval/informe_c2_c3.py", "informe_c2_c3")
    with pytest.raises(SystemExit):
        cc.main(["invertido", "--root", str(tmp_path), "--out", str(tmp_path / "o")])
    b2 = _load("scripts/eval/informe_b2.py", "informe_b2")
    with pytest.raises(SystemExit):
        b2.main(["--test", "--root", str(tmp_path), "--out", str(tmp_path / "o2")])
    b3 = _load("scripts/eval/informe_b3.py", "informe_b3")
    with pytest.raises(SystemExit):
        b3.main(["--root", str(tmp_path), "--conjunto", "test_invertido", "--permitir-test", "--out", str(tmp_path / "o3")])


def test_guiones_demo_de_extremo_a_extremo(tmp_path):
    cc = _load("scripts/eval/informe_c2_c3.py", "informe_c2_c3")
    assert cc.main(["todas", "--demo", "--out", str(tmp_path / "cc"), "--B", "60"]) == 0
    for n in cc.TABLAS:
        assert (tmp_path / "cc" / ("%s.csv" % n)).exists(), n
    b2 = _load("scripts/eval/informe_b2.py", "informe_b2")
    assert b2.main(["--demo", "--out", str(tmp_path / "b2"), "--B", "60"]) == 0
    comp = pd.read_csv(tmp_path / "b2" / "comparacion_erm.csv")
    assert set(comp.familia) == {"F-C2R-shenzhen", "F-C2R-montgomery"}
    pb = _load("scripts/eval/informe_plan_b.py", "informe_plan_b")
    assert pb.main(["--demo", "--out", str(tmp_path / "pb")]) == 0
    txt = (tmp_path / "pb" / "informe.md").read_text(encoding="utf-8")
    for t in ("Tabla 2.", "Tabla 3.", "Tabla 4.", "Tabla 5.", "Tabla S1.", "Tabla S4.", "Tabla S8."):
        assert t in txt


@pytest.mark.skipif(not (ROOT / "results" / "b3a" / "busqueda_todas.csv").exists(), reason="sin resultados de la búsqueda")
def test_tablas_de_busqueda_reproducen_los_ficheros_actuales(tmp_path):
    tb = _load("scripts/eval/tablas_busqueda.py", "tablas_busqueda")
    assert tb.main(["todo", "--comparar", "--out", str(tmp_path)]) == 0


@pytest.mark.skipif(not (ROOT / "results" / "b1" / "congelados" / "elegidos.json").exists(), reason="sin B1")
def test_elegir_b1_reproduce_elegidos_y_escribe_los_doce_json(tmp_path):
    import json
    eb = _load("scripts/eval/elegir_b1.py", "elegir_b1")
    assert eb.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "elegidos.json").read_bytes() == (ROOT / "results" / "b1" / "congelados" / "elegidos.json").read_bytes()
    js = sorted(p.name for p in tmp_path.glob("baseline_*.json"))
    assert len(js) == 12
    for p in tmp_path.glob("baseline_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        assert set(d) == {"method", "params", "lr_B", "trial", "metrica_seleccion", "corrida_busqueda"}
        assert (tmp_path / p.name).read_bytes() == (ROOT / "results" / "b1" / "congelados" / p.name).read_bytes()


def test_dosis_cargar_exige_la_congelacion_al_abrir_un_test(tmp_path):
    _arbol(tmp_path, metodos=("erm",), semillas=(0,))
    inv = D.descubrir(tmp_path, conjuntos=("test_desconf",), permitir_test=True)
    with pytest.raises(Exception) as e:
        D.cargar(inv, "test_desconf", permitir_test=True)
    assert "Bloqueado" in type(e.value).__name__
    llamadas = []
    assert D.cargar(inv, "test_desconf", permitir_test=True, verificador=llamadas.append)
    assert llamadas == ["dosis.cargar test_desconf"]
