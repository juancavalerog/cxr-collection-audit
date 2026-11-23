"""Extracción de características para la auditoría C1 (docs/08_evaluacion.md, V2.3.1 y V2.3.5; docs/14).

Versión de producción de scripts/diag/expl_lectura_features.py. Para cada tronco escribe, por conjunto, las
características agrupadas (`pooled`, 768), las de la imagen en espejo (`pooled_espejo`, para la decisión sobre la TTA
del detector, m9) y, en los troncos entrenados, la penúltima capa (`penult`) y el logit (`logit`, con sus versiones en
espejo). Todo en float32, sin autocast, como la extracción de la exploración (Tabla E4).

Troncos:
- `--din`: D_IN, el ConvNeXt-Tiny con los pesos de ImageNet con los que se inicializan todos los métodos, congelado y
  sin ningún ajuste. Se construye exactamente como lo hace el entrenador al empezar una corrida (`model.init.type:
  timm_imagenet`, es decir `timm.create_model('convnext_tiny', pretrained=True)`; la etiqueta de los pesos y la huella
  del tronco quedan en `din_imagenet_info.json`). Su cabeza está sin entrenar: no se guardan `penult` ni `logit`.
- `--runs DIR ...`: troncos entrenados (`results/train/<experimento>/seed<S>`), con el estado `--estado` (best).

Conjuntos (`--conjuntos`, por defecto `tr_bimcv_pos val`): `tr_bimcv_pos` es el conjunto de ajuste del detector, la
misma muestra de 6 000 positivos de bimcv de entrenamiento que la exploración (semilla 20261113); se añaden `res_*`
si la corrida tiene reserva. `test` (test limpio completo, que contiene la parte bimcv) y `test_covidgr` solo tras
`exigir_congelacion` y se escriben en `results/c1/caracteristicas_test/<tag>__<conjunto>.*`.

    python scripts/eval/extraer_caracteristicas.py --din                               # trabajo c1_extraer_din_val
    python scripts/eval/extraer_caracteristicas.py --runs results/train/baseline_erm_final/seed42
    python scripts/eval/extraer_caracteristicas.py --din --conjuntos test test_covidgr  # tras la congelación del protocolo
    python scripts/eval/extraer_caracteristicas.py --generar-trabajos                  # trabajos de las finales
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import c1_datos as CD  # noqa: E402
from src.eval.desbloqueo import es_bloqueado, exigir_si_bloqueado, sha256_fichero  # noqa: E402

OUT = ROOT / "results" / "c1" / "caracteristicas"
OUT_TEST = ROOT / "results" / "c1" / "caracteristicas_test"
CONF_DIN = ("results/train/baseline_erm_final/seed42", "results/train/baseline_erm/seed42")
EXPLORACION_META = ROOT / "results" / "diag" / "expl_lectura" / "baseline_erm_seed42_best_meta.csv"
TRABAJOS_FINALES = ("erm", "erm_mascara")
SEMILLAS = (42, 123, 2024, 7, 999)


@torch.no_grad()
def extraer_lotes(model, loader, device, cabeza: bool = True, espejo: bool = True, vigilante=None) -> dict:
    """Características de un cargador. `model` es un BaselineNet (o un modelo con .pooled y la cabeza
    fc1-ln-drop-fc2). Devuelve arrays float32 y la lista de image_id en el orden del cargador."""
    from src.baselines.common import penultimate
    net = model.net if hasattr(model, "net") else model
    model.eval()
    out = {k: [] for k in (("pooled", "penult", "logit") if cabeza else ("pooled",))}
    if espejo:
        for k in list(out):
            out[k + "_espejo"] = []
    ids = []
    for bt in loader:
        x = bt["image"].to(device).float()
        for suf, xx in (("", x), ("_espejo", torch.flip(x, dims=[3]))):
            if suf and not espejo:
                continue
            p = net.pooled(xx)
            out["pooled" + suf].append(p.float().cpu().numpy())
            if cabeza:
                h = penultimate(net, p)
                out["penult" + suf].append(h.float().cpu().numpy())
                out["logit" + suf].append(net.fc2(h).reshape(-1).float().cpu().numpy())
        ids += list(map(str, bt["image_id"]))
        if vigilante is not None:
            vigilante.step()
    arr = {k: (np.concatenate(v).astype(np.float32) if v else np.zeros((0,), np.float32)) for k, v in out.items()}
    return {"ids": ids, "arrays": arr}


def _meta_cols(df):
    cols = [c for c in ("image_id", "patient_id", "label", "source") if c in df.columns]
    return df[cols].copy()


def extraer(tr, tag: str, conjuntos, out_dir: Path, out_test: Path, cabeza: bool, espejo: bool, registro,
            info_base: dict, max_n: int = 0, vigilante=None) -> dict:
    """Extrae los conjuntos y escribe <tag>.npz/_meta.csv/_info.json (validación) y <tag>__<conjunto>.* (bloqueados)."""
    import pandas as pd
    t0 = time.time()
    val_sets = [c for c in conjuntos if not es_bloqueado(c)]
    reserva = bool(tr.cfg["data"].get("manifest"))
    if val_sets and reserva and not any(c.startswith("res_") for c in val_sets):
        val_sets += ["res_neg", "res_pos_antigua", "res_pos_nueva"]
    bloq = [c for c in conjuntos if es_bloqueado(c)]
    escritos = {}

    def un_conjunto(nombre):
        df = CD.marco(tr, nombre, registro)
        if not len(df):
            return None
        if max_n:
            df = df.iloc[:int(max_n)].reset_index(drop=True)
        r = extraer_lotes(tr.model, CD.cargador(tr, df, nombre), tr.device, cabeza, espejo, vigilante)
        if r["ids"] != df["image_id"].astype(str).tolist():
            raise SystemExit("orden de imágenes inesperado en %s" % nombre)
        m = _meta_cols(df)
        m["conjunto"] = nombre
        print("[%s] %s: %d imágenes" % (tag, nombre, len(df)), flush=True)
        return m, r["arrays"]

    grupos = []
    if val_sets:
        partes = [x for x in (un_conjunto(c) for c in val_sets) if x is not None]
        grupos.append((out_dir, tag, partes))
    for c in bloq:
        x = un_conjunto(c)
        if x is not None:
            grupos.append((out_test, "%s__%s" % (tag, c), [x]))
    for d, nombre, partes in grupos:
        meta = pd.concat([p[0] for p in partes], ignore_index=True)
        arrs = {k: np.concatenate([p[1][k] for p in partes]) for k in partes[0][1]}
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (".%s.tmp.npz" % nombre)
        np.savez_compressed(tmp, **arrs)
        tmp.replace(d / ("%s.npz" % nombre))
        meta.to_csv(d / ("%s_meta.csv" % nombre), index=False)
        info = dict(info_base)
        info.update({"tag": tag, "fichero": nombre, "conjuntos": {c: int((meta["conjunto"] == c).sum())
                                                                 for c in meta["conjunto"].unique()},
                     "arrays": {k: list(v.shape) for k, v in arrs.items()}, "espejo": bool(espejo),
                     "cabeza_entrenada": bool(cabeza), "precision": "float32 sin autocast",
                     "dispositivo": str(tr.device), "max_n": int(max_n),
                     "semilla_muestra_ajuste": CD.SEMILLA_MUESTRA, "n_ajuste_pedido": CD.N_AJUSTE,
                     "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "orden": "python " + " ".join([Path(sys.argv[0]).as_posix()] + sys.argv[1:]),
                     "sha256_codigo": {CD.rel(p): sha256_fichero(p) for p in (Path(__file__), Path(CD.__file__))},
                     "congelacion": registro.get("sha256_congelacion") if registro else None,
                     "segundos": round(time.time() - t0, 1)})
        CD.escribir_atomico(d / ("%s_info.json" % nombre), json.dumps(info, indent=1, ensure_ascii=False))
        escritos[nombre] = str(d / ("%s.npz" % nombre))
        if "tr_bimcv_pos" in set(meta["conjunto"]) and d == out_dir:
            aj = meta[meta["conjunto"] == "tr_bimcv_pos"][["image_id", "patient_id"]]
            lp = d / "lista_ajuste_detector.csv"
            if reserva or max_n:          # la reserva cambia el entrenamiento y, con él, la muestra
                aj.to_csv(d / ("lista_ajuste_%s.csv" % tag), index=False)
            elif not lp.exists():
                aj.to_csv(lp, index=False)
            elif pd.read_csv(lp, dtype=str)["image_id"].tolist() != aj["image_id"].astype(str).tolist():
                raise SystemExit("el conjunto de ajuste de %s no coincide con %s" % (tag, lp))
    return escritos


def comprobar_ajuste_exploracion(lista: Path) -> dict:
    """¿Es la lista de ajuste la misma muestra que usó la exploración (Tabla E4)?"""
    import pandas as pd
    if not lista.exists() or not EXPLORACION_META.exists():
        return {"comprobado": False}
    a = pd.read_csv(lista, dtype=str)["image_id"].tolist()
    m = pd.read_csv(EXPLORACION_META, dtype=str)
    b = m.loc[m["conjunto"] == "tr_bimcv_pos", "image_id"].tolist()
    return {"comprobado": True, "coincide": a == b, "n": len(a), "n_exploracion": len(b)}


def generar_trabajos(staged: Path, metodos, semillas, incluir_sin_terminar=False, din=True) -> list:
    sys.path.insert(0, str(ROOT / "runner"))
    from job_runner import validate_job
    hechos = []
    specs = []
    if din:
        specs.append({"id": "c1_extraer_din_val",
                      "comando": ["python", "scripts/eval/extraer_caracteristicas.py", "--din", "--workers", "8"],
                      "descripcion": "Auditoría C1 (M3 del revisor): características de D_IN (ConvNeXt-Tiny de ImageNet, "
                                     "sin ajuste) en el conjunto de ajuste del detector y la validación completa. Sin test.",
                      "notas": "unos 14 000 pases con espejo en fp32: pocos minutos de GPU; menos de 0.1 GB; salida en "
                               "results/c1/caracteristicas/din_imagenet*; después: python scripts/eval/auditoria_covidx.py --din"})
    for m in metodos:
        for s in semillas:
            run = ROOT / "results" / "train" / ("baseline_%s_final" % m) / ("seed%d" % s)
            terminado = (run / "states" / "best.pt").exists() and (run / "summary.json").exists()
            if not terminado and not incluir_sin_terminar:
                continue
            specs.append({"id": "c1_extraer_%s_final_s%d" % (m, s),
                          "comando": ["python", "scripts/eval/extraer_caracteristicas.py", "--runs", CD.rel(run),
                                      "--estado", "best", "--workers", "8"],
                          "descripcion": "Auditoría C1: características del tronco final %s, semilla %d, estado best, en "
                                         "el conjunto de ajuste del detector y la validación completa. Sin test." % (m, s),
                          "notas": "pocos minutos de GPU; menos de 0.1 GB; salida en results/c1/caracteristicas/"})
    staged.mkdir(parents=True, exist_ok=True)
    for sp in specs:
        sp.update({"gpu": "any", "prioridad": 5, "tiempo_max_min": 60, "origen": "auditoría C1",
                   "creado": time.strftime("%Y-%m-%dT%H:%M:%S"), "reencolar_si_termico": True})
        sp = {k: sp[k] for k in ("id", "comando", "gpu", "prioridad", "tiempo_max_min", "descripcion", "creado",
                                 "origen", "reencolar_si_termico", "notas")}
        err = validate_job(sp, sp["id"])
        if err:
            raise SystemExit("trabajo %s no válido: %s" % (sp["id"], err))
        p = staged / ("%s.json" % sp["id"])
        p.write_text(json.dumps(sp, indent=1, ensure_ascii=False), encoding="utf-8")
        hechos.append(str(p))
    return hechos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--din", action="store_true")
    ap.add_argument("--config-din", default=None, help="corrida cuya configuración se usa para D_IN (por defecto, ERM final 42 o piloto)")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--estado", default="best")
    ap.add_argument("--conjuntos", nargs="+", default=["tr_bimcv_pos", "val"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--out-test", default=str(OUT_TEST))
    ap.add_argument("--max-n", type=int, default=0, help="solo pruebas: primeras N imágenes de cada conjunto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--sin-espejo", action="store_true")
    ap.add_argument("--generar-trabajos", action="store_true")
    ap.add_argument("--staged", default=str(ROOT / "jobs" / "staged"))
    ap.add_argument("--incluir-sin-terminar", action="store_true")
    a = ap.parse_args(argv)
    if a.generar_trabajos:
        for p in generar_trabajos(Path(a.staged), TRABAJOS_FINALES, SEMILLAS, a.incluir_sin_terminar, din=a.din):
            print("trabajo: %s" % p)
        return 0
    registro = exigir_si_bloqueado(a.conjuntos, "extraer_caracteristicas: %s" % ("D_IN" if a.din else a.runs))
    out, out_test = Path(a.out), Path(a.out_test)
    vig = CD.Vigilante()
    try:
        if a.din:
            ref = Path(a.config_din) if a.config_din else next((ROOT / c for c in CONF_DIN if (ROOT / c / "config_resolved.json").exists()), None)
            if ref is None:
                raise SystemExit("no hay configuración de referencia para D_IN")
            cfg = CD.leer_config(ref)
            init = (cfg["model"].get("init") or {}).get("type")
            if init != "timm_imagenet" and cfg["model"]["name"] != "tiny":
                raise SystemExit("la configuración de referencia no parte de ImageNet (%s)" % init)
            tr = CD.construir_trainer(cfg, int(cfg.get("seed", 0)), a.workers, preentrenado=True, dispositivo=a.dispositivo)
            net = tr.model.net
            pc = getattr(getattr(net, "backbone", None), "pretrained_cfg", None) or {}
            info = {"tronco": "D_IN", "configuracion_referencia": CD.rel(ref / "config_resolved.json"),
                    "init": cfg["model"].get("init"), "backbone": cfg["model"].get("backbone"),
                    "pesos_timm": {k: pc.get(k) for k in ("architecture", "tag", "hf_hub_id", "url", "license")},
                    "huella_tronco_sha256": CD.huella_tronco(net) if hasattr(net, "backbone") else None,
                    "estado": None, "nota": "tronco de ImageNet sin ajuste; la cabeza no está entrenada y no se guarda"}
            r = extraer(tr, "din_imagenet", a.conjuntos, out, out_test, False, not a.sin_espejo, registro, info, a.max_n, vig)
            print(json.dumps({"escritos": r, "ajuste_igual_exploracion": comprobar_ajuste_exploracion(out / "lista_ajuste_detector.csv")},
                             ensure_ascii=False))
        for run in map(Path, a.runs):
            st = run / "states" / ("%s.pt" % a.estado)
            tr = CD.trainer_de_corrida(run, a.workers, a.dispositivo)
            CD.cargar_estado(tr, st)
            tag = "%s_%s_%s" % (run.parent.name, run.name, a.estado)
            info = {"tronco": tag, "corrida": CD.rel(run), "estado": a.estado, "sha256_estado": sha256_fichero(st),
                    "sha256_config": sha256_fichero(run / "config_resolved.json"),
                    "metodo": (tr.cfg.get("baseline") or {}).get("method")}
            r = extraer(tr, tag, a.conjuntos, out, out_test, True, not a.sin_espejo, registro, info, a.max_n, vig)
            print(json.dumps({"escritos": r}, ensure_ascii=False))
    except CD.Parada as e:
        print("parada: %s; sin resultados parciales" % e, flush=True)
        return CD.EXIT_PARADA
    finally:
        vig.cerrar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
