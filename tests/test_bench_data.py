"""Pruebas rápidas de los datos del plan B (src/data/bench.py, scripts/data/*): no leen imágenes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from src.data import bench  # noqa: E402
from src.data.paths import PROJECT_ROOT, data_root  # noqa: E402
import bench_common as bc  # noqa: E402

BENCH = ROOT / "data" / "manifests" / "bench"


def _toy():
    rows = []
    for i in range(8):
        rows.append({"dataset": "x", "source": "a" if i < 4 else "b", "image_path": "img/%d.png" % i,
                     "raiz": "datos" if i % 2 else "proyecto", "patient_id": "p%d" % i, "label": i % 2,
                     "split": ["train", "train", "val", "test"][i % 4], "image_id": "i%d" % i,
                     "test_x": int(i % 4 == 3)})
    meta = {"benchmark": "toy", "fuentes": ["b", "a"], "fuentes_mixtas": ["a", "b"], "rho": 0.5, "semilla": 0,
            "subconjuntos": ["test_x"], "seleccion": "val"}
    return pd.DataFrame(rows), meta


def test_contrato_valido_y_load(tmp_path):
    df, meta = _toy()
    p = bench.write_bench(df, meta, tmp_path / "t.csv")
    frames, m = bench.load_bench(p)
    assert set(frames) == {"train", "val", "test", "test_x"}
    assert len(frames["train"]) == 4 and len(frames["test_x"]) == 2
    tr = frames["train"]
    assert (tr.source_idx == tr.source.map({"b": 0, "a": 1})).all()     # orden del JSON, no alfabético
    r = tr.iloc[0]
    base = data_root() if r.raiz == "datos" else PROJECT_ROOT
    assert r.image_path == str(base / r.image_path_rel) and Path(r.image_path).is_absolute()
    f2, _ = bench.load_bench(p, absolute_paths=False)
    assert f2["train"].image_path.iloc[0] == r.image_path_rel
    for c in ("image_id", "patient_id", "label", "source", "source_idx", "raiz", "abs_path"):
        assert c in tr.columns


@pytest.mark.parametrize("roto", ["dup_id", "split", "label", "sub", "fuente", "sel", "una_clase"])
def test_contrato_detecta_errores(roto):
    df, meta = _toy()
    if roto == "dup_id":
        df.loc[1, "image_id"] = "i0"
    elif roto == "split":
        df.loc[0, "split"] = "valid"
    elif roto == "label":
        df.loc[0, "label"] = 2
    elif roto == "sub":
        df = df.drop(columns="test_x")
    elif roto == "fuente":
        df.loc[0, "source"] = "z"
    elif roto == "sel":
        meta["seleccion"] = "otro"
    elif roto == "una_clase":
        df.loc[df.split == "train", "label"] = 1
    assert bench.validate(df, meta)


def test_muestra_ap_disjunta_y_factible():
    pac = pd.DataFrame({"patient_id": ["p%03d" % i for i in range(30)],
                        "has_ap": [True] * 10 + [False] * 10 + [True] * 10,
                        "has_pa": [False] * 10 + [True] * 10 + [True] * 10})
    o = bc.orden_ap_pa(pac, "t", 0)
    s = bc.muestra_ap(o, 30, 0.5)
    assert len({p for p, _ in s}) == 30 and sum(v == "AP" for _, v in s) == 15
    with pytest.raises(ValueError):
        bc.muestra_ap(o, 30, 0.8)                    # 24 AP con solo 20 pacientes con AP
    assert bc.max_factible(o, 0.5) == 30
    a, b = bc.muestra_ap(o, 10, 0.5), bc.muestra_ap(o, 20, 0.5)
    assert set(a) <= set(b)                          # prefijos anidados
    assert bc.muestra_ap(bc.orden_ap_pa(pac, "t", 0), 12, 0.5) == bc.muestra_ap(o, 12, 0.5)


def test_reservas_deterministas_y_estratificadas():
    pac = pd.DataFrame({"patient_id": ["p%04d" % i for i in range(1000)], "clase": [0, 1] * 500})
    r1 = bc.reservas_pacientes(pac, ["clase"], clave="x")
    r2 = bc.reservas_pacientes(pac.sample(frac=1, random_state=3), ["clase"], clave="x").reindex(pac.index)
    assert (r1 == r2).all()
    c = pd.crosstab(r1, pac.clase)
    assert c.loc["train"].tolist() == [350, 350] and c.loc["val"].tolist() == [50, 50]


def _man(b):
    m = sorted((BENCH / b).rglob("*.csv"))
    m = [p for p in m if bench.meta_path(p).exists()]
    if not m:
        pytest.skip("sin manifiestos de %s" % b)
    return m


def test_b3a_tablas_y_test_fijo():
    ms = _man("b3a")
    orig = [p for p in ms if int(p.stem.split("_s")[1]) <= 4]       # los 46 originales
    assert len(orig) == 46 and len(ms) - len(orig) in (0, 15)        # + semillas 5 a 9 (E8, punto 1)
    tds = set()
    for p in [BENCH / "b3a/apos/rho0.50_s0.csv", BENCH / "b3a/apos/rho1.00_s3.csv", BENCH / "b3a/bpos/rho0.80_s1.csv"]:
        d, m = bench.read_csv(p), bench.read_meta(p)
        assert not bench.validate(d, m)
        pa = m["rho"] if m["orientacion"] == "apos" else 1 - m["rho"]
        tr = d[d.split == "train"]
        for s, prev in (("nih", pa), ("chexpert", 1 - pa)):
            t = tr[tr.source == s]
            assert len(t) == 2800 and t.label.sum() == round(2800 * prev)
            assert (d[(d.split == "val") & (d.source == s)].label.sum()) == round(400 * prev)
        assert len(d[d.test_desconf == 1]) == 2000
        tds.add(tuple(sorted(d[d.test_desconf == 1].image_id)))
        assert not set(d[d.split == "train"].patient_id) & set(d[d.split != "train"].patient_id)
    assert len(tds) == 1
    d = bench.read_csv(BENCH / "b3a/apos/rho1.00_s0.csv")
    m = bench.read_meta(BENCH / "b3a/apos/rho1.00_s0.csv")
    assert m["fuentes_mixtas"] == [] and m["fuentes"] == ["nih", "chexpert"]


def test_b2_y_covidgr():
    p = _man("b2")[0]
    d, m = bench.read_csv(p), bench.read_meta(p)
    assert not bench.validate(d, m) and m["fuentes_mixtas"] == ["tbx11k"]
    tr = d[d.split == "train"]
    assert tr[tr.source == "belarus"].label.eq(1).all() and tr[tr.source == "rsna_nih"].label.eq(0).all()
    assert tr[tr.source == "tbx11k"].label.value_counts().to_dict() == {0: 600, 1: 600}
    te = d[d.split == "test"]
    assert set(te.source) == {"shenzhen", "montgomery"} and len(te) == 800
    g = _man("covidgr")[0]
    c = bench.read_csv(g)
    assert len(c) == 852 and c.label.sum() == 426 and c.split.eq("test").all()


def test_colecciones_reproduce_e1():
    p = ROOT / "data" / "manifests" / "covidx_cxr4_colecciones.csv"
    if not p.exists():
        pytest.skip("sin fichero de colecciones")
    import build_colecciones_covidx as bcc
    c = pd.read_csv(p)
    for (col, sp), n in bcc.TABLA_E1.items():
        assert ((c.coleccion == col) & (c.split_clean == sp)).sum() == n
    assert len(c) == 84818


def test_check_bench_falla_con_manifiesto_roto(tmp_path):
    df, meta = _toy()
    meta["benchmark"] = "covidgr"
    bench.write_bench(df, meta, tmp_path / "covidgr" / "x.csv")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/data/check_bench.py"), "--root", str(tmp_path),
                        "--sin-hash", "--sin-ficheros", "--log", str(tmp_path / "log.txt")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "FALLO" in r.stdout
