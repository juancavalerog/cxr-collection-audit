"""Pruebas de los datos de los experimentos adicionales (docs/11_benchmarks_plan_b.md, Section 12): semillas 5 a 9 de
B3a, benchmark B3c de cardiomegalia, trabajos e_* y fusión de cachés. No leen imágenes ni predicciones.

    pytest -q -p no:cacheprovider --basetemp=/tmp/pt_extra tests/test_bench_extra.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "runner"))

from src.data import bench  # noqa: E402

BENCH = ROOT / "data" / "manifests" / "bench"
SHA_REF = ROOT / "logs" / "supervisor" / "sha_manifiestos_bench_20261117.txt"
NUEVOS_B3A = [BENCH / "b3a" / "apos" / ("rho%.2f_s%d.csv" % (r, s)) for r in (0.9, 0.95, 0.99) for s in range(5, 10)]
B3C = [BENCH / "b3c" / "apos" / ("rho%.2f_s%d.csv" % (r, s)) for r in (0.5, 0.9, 0.99, 1.0) for s in range(3)]
SUBS = ("val_desconf", "test_desconf", "test_invertido")


def _req(paths):
    if not all(p.exists() and bench.meta_path(p).exists() for p in paths):
        pytest.skip("faltan manifiestos")


def _cuentas(d):
    out = {}
    sets = {"train": d[d.split == "train"], "val": d[d.split == "val"]}
    for s in SUBS:
        sets[s] = d[d[s].astype(int) == 1]
    for k, x in sets.items():
        out[k] = x.groupby(["source", "label", "view"]).size().to_dict()
    return out


def _sub(d, s):
    x = d[d[s].astype(int) == 1]
    return sorted(zip(x.image_id, x.label.astype(int), x.patient_id, x.source))


def test_originales_intactos():
    if not SHA_REF.exists():
        pytest.skip("sin lista de huellas")
    malos = []
    for ln in SHA_REF.read_text().splitlines():
        h, p = ln.split(None, 1)
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h:
            malos.append(p)
    assert not malos


@pytest.mark.parametrize("grupo", ["b3a", "b3c"])
def test_cargan_con_load_bench(grupo):
    ps = NUEVOS_B3A if grupo == "b3a" else B3C
    _req(ps)
    for p in ps:
        frames, meta = bench.load_bench(p)
        assert meta["benchmark"] == grupo and meta["fuentes"] == ["nih", "chexpert"]
        assert set(frames) == {"train", "val", "test", *SUBS}
        assert meta["seleccion"] == "val" and meta["test_bloqueado"] is True
        assert meta["orientacion"] == "apos"


def test_b3a_semillas_nuevas_como_semilla_0():
    _req(NUEVOS_B3A)
    for p in NUEVOS_B3A:
        rho = p.stem.split("_")[0]
        d0 = bench.read_csv(p.with_name("%s_s0.csv" % rho))
        d = bench.read_csv(p)
        assert _cuentas(d) == _cuentas(d0), p.name                  # mismos recuentos por celda y vista
        for s in SUBS:
            assert _sub(d, s) == _sub(d0, s), (p.name, s)           # mismos conjuntos de evaluación
        assert set(d[d.split == "test"].image_id) == set(d0[d0.split == "test"].image_id)
        assert set(d[d.split == "train"].image_id) != set(d0[d0.split == "train"].image_id)
        m = bench.read_meta(p)
        assert m["semilla"] == int(p.stem.split("_s")[1]) and m["tamanos"]["train_por_fuente"] == 2800


def test_b3a_test_desconf_igual_en_los_61():
    _req(NUEVOS_B3A)
    ref = _sub(bench.read_csv(BENCH / "b3a/apos/rho0.50_s0.csv"), "test_desconf")
    for p in bench.list_manifests(BENCH / "b3a"):
        assert _sub(bench.read_csv(p), "test_desconf") == ref, p.name


def _todas(paths):
    return pd.concat([bench.read_csv(p).assign(_m=p.name) for p in paths], ignore_index=True)


@pytest.mark.parametrize("grupo", ["b3a", "b3c"])
def test_sin_fuga_de_pacientes(grupo):
    ps = bench.list_manifests(BENCH / grupo)
    _req(ps)
    for p in ps:
        d = bench.read_csv(p)
        tr = set(d[d.split == "train"].patient_id)
        va = set(d[d.split == "val"].patient_id) | set(d[d.val_desconf.astype(int) == 1].patient_id)
        te = set(d[d.split == "test"].patient_id) | set(d[d.test_desconf.astype(int) == 1].patient_id) \
            | set(d[d.test_invertido.astype(int) == 1].patient_id)
        assert not (tr & va) and not (tr & te) and not (va & te), p.name
        for k in ("train", "val", "test"):
            assert not d[d.split == k].patient_id.duplicated().any()
    a = _todas(ps)
    assert (a.groupby(["source", "patient_id"]).reserva.nunique() == 1).all()   # reserva fija entre manifiestos
    assert (a[a.split != "fuera"].split == a[a.split != "fuera"].reserva).all()


def test_b3a_y_b3c_comparten_reservas():
    _req(B3C)
    a = _todas(bench.list_manifests(BENCH / "b3a"))[["source", "patient_id", "reserva"]].drop_duplicates()
    c = _todas(B3C)[["source", "patient_id", "reserva"]].drop_duplicates()
    m = a.merge(c, on=["source", "patient_id"], suffixes=("_a", "_c"))
    assert len(m) > 0 and (m.reserva_a == m.reserva_c).all()
    # y con las reservas completas (todos los pacientes de cada benchmark, no solo los usados)
    ra = pd.read_csv(BENCH / "b3a/reservas_pacientes.csv", dtype={"patient_id": str})
    rc = pd.read_csv(BENCH / "b3c/reservas_pacientes.csv", dtype={"patient_id": str})
    m = ra.merge(rc, on=["source", "patient_id"], suffixes=("_a", "_c"))
    assert (m.reserva_a == m.reserva_c).all()
    assert m.heredada.all() and not rc[~rc.heredada].merge(ra, on=["source", "patient_id"]).shape[0]


def test_b3c_diseno():
    _req(B3C)
    ref = None
    for p in B3C:
        d, m = bench.read_csv(p), bench.read_meta(p)
        t = m["tamanos"]
        k = t["factor_k"]
        assert t["train_por_fuente"] == round(2800 * k) and t["val_por_fuente"] == round(400 * k)
        assert t["val_desconf_por_celda"] == round(200 * k) and t["test_desconf_por_celda"] == round(500 * k)
        assert t["test_invertido_por_fuente"] == round(1000 * k)
        rho = m["rho"]
        prev = {"nih": rho, "chexpert": 1 - rho}
        for s in ("nih", "chexpert"):
            for k2, n in (("train", t["train_por_fuente"]), ("val", t["val_por_fuente"])):
                x = d[(d.split == k2) & (d.source == s)]
                assert len(x) == n and int(x.label.sum()) == int(n * prev[s] + 0.5 + 1e-9)
            x = d[(d.test_desconf.astype(int) == 1) & (d.source == s)]
            assert int(x.label.sum()) == int((x.label == 0).sum()) == t["test_desconf_por_celda"]
            x = d[(d.val_desconf.astype(int) == 1) & (d.source == s)]
            assert int(x.label.sum()) == int((x.label == 0).sum()) == t["val_desconf_por_celda"]
            x = d[(d.test_invertido.astype(int) == 1) & (d.source == s)]
            assert len(x) == t["test_invertido_por_fuente"]
        td = _sub(d, "test_desconf")
        ref = ref or td
        assert td == ref
        seed0 = bench.read_csv(p.with_name(p.stem.split("_")[0] + "_s0.csv"))
        assert _cuentas(d) == _cuentas(seed0)
        for s in SUBS:
            assert _sub(d, s) == _sub(seed0, s)


def test_b3c_etiquetas_segun_definicion():
    _req(B3C)
    m = ROOT / "data" / "manifests"
    n = pd.read_csv(m / "nih_cxr14.csv", usecols=["image_path", "y_cardiomegaly", "y_no_finding"])
    c = pd.read_csv(m / "chexpert_small.csv", usecols=["image_path", "patient_id", "view", "ap_pa",
                                                       "y_cardiomegaly", "y_no_finding"], dtype={"patient_id": str})
    d = _todas(B3C).drop_duplicates("image_id")
    dn = d[d.source == "nih"].merge(n, on="image_path")
    assert len(dn) == (d.source == "nih").sum()
    assert (dn[dn.label == 1].y_cardiomegaly == 1).all() and (dn[dn.label == 0].y_no_finding == 1).all()
    assert (dn[dn.label == 0].y_cardiomegaly == 0).all()
    dc = d[d.source == "chexpert"].merge(c.drop(columns=["patient_id", "view"]), on="image_path")
    assert len(dc) == (d.source == "chexpert").sum()
    assert (dc[dc.label == 1].y_cardiomegaly == 1).all()
    neg = dc[dc.label == 0]
    assert (neg.y_no_finding == 1).all() and not neg.y_cardiomegaly.isin([1, -1]).any()
    cf = c[(c.view == "Frontal") & c.ap_pa.isin(["AP", "PA"])]
    inciertos = set(cf.loc[cf.y_cardiomegaly == -1, "patient_id"])
    assert not set(neg.patient_id) & inciertos
    assert (dc.view == dc.ap_pa).all()


def test_b3c_cache_esperada_por_el_entrenador():
    _req(B3C)
    from src.train.bench import cache_auto  # solo lectura: devuelve None si la caché no existe todavía
    m = bench.read_meta(B3C[0])
    c = ROOT / "data" / "cache_m" / ("bench_%s" % m["benchmark"]) / "homogeneo_224"
    job = json.loads((ROOT / "jobs/staged/e_05_cache_m_b3c.json").read_text())
    assert Path(job["comando"][job["comando"].index("--bench") + 1]).name == "b3c"   # -> bench_b3c
    assert cache_auto(m, "homogeneo", 224) in (None, c)


def test_trabajos_e_validos():
    from job_runner import validate_job
    js = sorted((ROOT / "jobs" / "staged").glob("e_*.json"))
    assert len(js) == 7
    for p in js:
        obj = json.loads(p.read_text())
        assert not validate_job(obj, stem=p.stem), p.name
        cmd = " ".join(obj["comando"])
        if "b3a_s5_9" in obj["id"] and "fusionar" not in cmd:
            # los trabajos de construcción de las semillas nuevas no escriben en los directorios de bench_b3a
            assert "bench_b3a/" not in cmd and "bench/b3a" not in cmd, p.name


def test_delta_b3a():
    _req(NUEVOS_B3A)
    p = ROOT / "logs" / "datos_extra" / "b3a_s5_9_imagenes_nuevas.csv"
    delta = pd.read_csv(p, dtype=str)
    orig = set(pd.concat([bench.read_csv(q) for q in bench.list_manifests(BENCH / "b3a") if q not in NUEVOS_B3A]).image_id)
    new = set(pd.concat([bench.read_csv(q) for q in NUEVOS_B3A]).image_id)
    assert set(delta.image_id) == new - orig


# ------------------------------------------------------------------------------------------ fusión de cachés
def _indice(d: Path, col, filas, crlf):
    d.mkdir(parents=True, exist_ok=True)
    nl = "\r\n" if crlf else "\n"
    cab = ["image_id", col, "x", "status"]
    lines = [",".join(cab)]
    for iid, contenido in filas:
        rel = "%s/%s.png" % (iid[:2], iid)
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(contenido)
        lines.append(",".join([iid, rel, "1", "ok"]))
    (d / "index.csv").write_bytes((nl.join(lines) + nl).encode())


def _fusion(tmp, extra=()):
    ids = tmp / "ids.csv"
    ids.write_text("image_id\nzz1\nzz2\n")
    cmd = [sys.executable, str(ROOT / "scripts/data/fusionar_cache_bench.py"), "--ids", str(ids),
           "--cache-origen", str(tmp / "co"), "--cache-destino", str(tmp / "cd"),
           "--mascaras-origen", str(tmp / "mo"), "--mascaras-destino", str(tmp / "md"), "--registro", str(tmp / "r"),
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def _montar(tmp, conflicto=False):
    _indice(tmp / "cd", "rel_path", [("aa%d" % i, b"viejo%d" % i) for i in range(5)], True)
    _indice(tmp / "md", "path", [("aa%d" % i, b"mviejo%d" % i) for i in range(5)], False)
    _indice(tmp / "co", "rel_path", [("zz1", b"nuevo1"), ("zz2", b"nuevo2")], True)
    _indice(tmp / "mo", "path", [("zz1", b"mnuevo1"), ("zz2", b"mnuevo2")], False)
    if conflicto:
        (tmp / "cd" / "zz" / "zz1.png").parent.mkdir(parents=True, exist_ok=True)
        (tmp / "cd" / "zz" / "zz1.png").write_bytes(b"otro")


def test_fusion_anade_sin_tocar_lo_existente(tmp_path):
    _montar(tmp_path)
    antes = {p: p.read_bytes() for p in (tmp_path / "cd").rglob("*") if p.is_file()}
    antes.update({p: p.read_bytes() for p in (tmp_path / "md").rglob("*") if p.is_file()})
    r = _fusion(tmp_path, ["--comprobar"])
    assert r.returncode == 0
    assert all(p.read_bytes() == b for p, b in antes.items())             # --comprobar no escribe
    r = _fusion(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    for p, b in antes.items():
        if p.name == "index.csv":
            assert p.read_bytes().startswith(b) and len(p.read_bytes()) > len(b)
        else:
            assert p.read_bytes() == b
    assert (tmp_path / "cd/zz/zz1.png").read_bytes() == b"nuevo1"
    from src.data.cache import read_index
    assert len(read_index(tmp_path / "cd")) == 7
    r2 = _fusion(tmp_path)                                                # idempotente
    assert r2.returncode == 0 and json.loads(r2.stdout)["cache"]["nuevas"] == 0


def test_fusion_se_niega_ante_conflicto(tmp_path):
    _montar(tmp_path, conflicto=True)
    r = _fusion(tmp_path)
    assert r.returncode == 2 and (tmp_path / "cd/zz/zz1.png").read_bytes() == b"otro"
    assert b"zz1" not in (tmp_path / "cd/index.csv").read_bytes()


def test_fusion_se_niega_con_cabecera_distinta(tmp_path):
    _montar(tmp_path)
    p = tmp_path / "co" / "index.csv"
    p.write_bytes(p.read_bytes().replace(b"image_id,rel_path,x", b"image_id,rel_path,y", 1))
    r = _fusion(tmp_path)
    assert r.returncode == 2 and not (tmp_path / "cd/zz").exists()


def test_check_bench_pasa():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/data/check_bench.py"), "--sin-hash", "--sin-ficheros",
                        "--log", "/tmp/check_bench_pytest_extra.log"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
