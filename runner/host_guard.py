#!/usr/bin/env python3
"""Vigilante del anfitrión de paper_new: CPU, discos, memoria y lecturas adicionales de las GPU.

Complementa a gpu_guard.py, que sigue siendo el único que detiene los trabajos. Este proceso solo lee sensores,
los registra y, si la CPU o un disco alcanzan su umbral de corte en varias lecturas seguidas, crea
jobs/THERMAL_STOP: gpu_guard.py detiene entonces los trabajos del proyecto y el ejecutor no lanza ninguno más.
Como en el caso de las GPU, THERMAL_STOP solo lo retira el usuario (runner/clear_thermal_stop.sh).

Qué mide (lo que exista en el servidor; nada es obligatorio):
- temperatura de la CPU: /sys/class/hwmon (coretemp, k10temp, zenpower...) y, si no hay, /sys/class/thermal;
- temperatura de los discos: hwmon nvme y drivetemp;
- otros sensores de temperatura de hwmon (placa, ACPI) y ventiladores, solo como registro;
- contadores de estrangulamiento térmico de la CPU (thermal_throttle);
- memoria disponible e intercambio (/proc/meminfo), carga (/proc/loadavg) y disco libre del proyecto;
- de las GPU, con nvidia-smi: temperatura de la memoria, ventilador y motivos de reducción de reloj.

Umbrales (grados Celsius): CPU aviso 75 y corte 85 (el servidor declara 89 como máximo y 97 como crítico);
memoria de las GPU aviso 85 y corte 90; discos aviso 65 y corte 75. Si el sensor declara su propio valor crítico,
el corte baja a ese valor menos 5 cuando resulta menor. El corte exige N lecturas seguidas (por defecto 3) para no
reaccionar a un pico de un solo instante. Los umbrales pueden bajarse sin reiniciar con runner/host_guard_config.json
(claves cpu_warn, cpu_stop, disk_warn, disk_stop, gpu_mem_warn, gpu_mem_stop), que se relee en cada ciclo; nunca pueden
subir por encima de los topes CPU 90, memoria de GPU 90 y discos 75.

Reducción de reloj de las GPU (clocks_throttle_reasons de nvidia-smi): solo se avisa de los motivos térmicos o de
hardware (0x8, 0x20, 0x40, 0x80). El límite de potencia por software (0x4) es normal a plena carga y solo se registra.

Uso, en el servidor (lo arranca runner/start_host_guard.sh en una ventana de tmux):
    python3 runner/host_guard.py                       # bucle
    python3 runner/host_guard.py --print-sensors       # una lectura detallada y termina
    python3 runner/host_guard.py --check-below 70      # código 0 si la CPU está por debajo de 70 °C
Pruebas: --sysfs-root y --proc-root apuntan a árboles falsos; --no-smi evita nvidia-smi.
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Paths, Logger, now_iso, atomic_write_text, atomic_write_json, append_line, thermal_event  # noqa: E402

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPU_CHIPS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal", "k8temp", "via_cputemp")
DISK_CHIPS = ("nvme", "drivetemp")
CPU_ZONES = ("x86_pkg_temp", "cpu-thermal", "cpu_thermal", "soc_thermal")
CSV_HEADER = ("timestamp,epoch,cpu_max_c,cpu_sensor,cpu_n_sensores,disco_max_c,disco_sensor,otros_max_c,otros_sensor,"
              "vent_min_rpm,throttle_total,mem_disp_pct,mem_disp_gib,swap_usado_mib,carga_1m,n_cpu,disco_libre_gb,"
              "gpu_mem_temp_max_c,gpu_vent_max_pct,gpu_reduccion_reloj,estado")
SMI_EXTRA = ["index", "temperature.memory", "fan.speed", "clocks_throttle_reasons.active"]
TOPES = {"cpu_stop": 90.0, "disk_stop": 75.0, "gpu_mem_stop": 90.0}
MOTIVOS_RELOJ = {0x1: "inactiva", 0x2: "relojes fijados por la aplicación", 0x4: "límite de potencia por software",
                 0x8: "reducción por hardware", 0x10: "sincronización", 0x20: "reducción térmica por software",
                 0x40: "reducción térmica por hardware", 0x80: "freno de potencia por hardware", 0x100: "reloj de pantalla"}
MOTIVOS_GRAVES = 0x8 | 0x20 | 0x40 | 0x80


def motivos_reloj(valor):
    """(texto, grave) a partir del valor hexadecimal de clocks_throttle_reasons.active."""
    try:
        v = int(str(valor), 16)
    except (TypeError, ValueError):
        return "", False
    nombres = [n for b, n in sorted(MOTIVOS_RELOJ.items()) if v & b and b != 0x1]
    return "+".join(nombres), bool(v & MOTIVOS_GRAVES)


def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _milli(path):
    s = _read(path)
    try:
        return float(s) / 1000.0
    except (TypeError, ValueError):
        return None


def leer_hwmon(sysfs):
    """Lista de sensores de temperatura y de ventiladores de /sys/class/hwmon."""
    temps, fans = [], []
    for d in sorted(glob.glob(os.path.join(sysfs, "class", "hwmon", "hwmon*"))):
        chip = _read(os.path.join(d, "name")) or os.path.basename(d)
        clase = "cpu" if chip in CPU_CHIPS else "disco" if chip in DISK_CHIPS else "otro"
        for f in sorted(glob.glob(os.path.join(d, "temp*_input"))):
            v = _milli(f)
            if v is None or v < -30 or v > 150:      # lecturas imposibles: sensor no conectado
                continue
            base = f[:-len("_input")]
            temps.append({"clase": clase, "chip": chip, "hwmon": os.path.basename(d),
                          "etiqueta": _read(base + "_label") or os.path.basename(base), "temp": v,
                          "max": _milli(base + "_max"), "crit": _milli(base + "_crit")})
        for f in sorted(glob.glob(os.path.join(d, "fan*_input"))):
            s = _read(f)
            try:
                fans.append({"chip": chip, "etiqueta": _read(f[:-len("_input")] + "_label") or os.path.basename(f)[:-6],
                             "rpm": float(s)})
            except (TypeError, ValueError):
                pass
    return temps, fans


def leer_zonas(sysfs):
    out = []
    for d in sorted(glob.glob(os.path.join(sysfs, "class", "thermal", "thermal_zone*"))):
        tipo = _read(os.path.join(d, "type")) or os.path.basename(d)
        v = _milli(os.path.join(d, "temp"))
        if v is None or v < -30 or v > 150:
            continue
        out.append({"clase": "cpu" if tipo in CPU_ZONES else "otro", "chip": "thermal_zone", "hwmon": os.path.basename(d),
                    "etiqueta": tipo, "temp": v, "max": None, "crit": None})
    return out


def leer_throttle(sysfs):
    """Suma de los contadores de estrangulamiento térmico por paquete y por núcleo (None si no existen)."""
    tot, visto = 0, False
    paquetes = set()
    for d in glob.glob(os.path.join(sysfs, "devices", "system", "cpu", "cpu[0-9]*")):
        pkg = _read(os.path.join(d, "topology", "physical_package_id"))
        c = _read(os.path.join(d, "thermal_throttle", "core_throttle_count"))
        p = _read(os.path.join(d, "thermal_throttle", "package_throttle_count"))
        try:
            if c is not None:
                tot += int(c); visto = True
            if p is not None and pkg not in paquetes:
                paquetes.add(pkg); tot += int(p); visto = True
        except ValueError:
            pass
    return tot if visto else None


def leer_memoria(proc):
    info = {}
    txt = _read(os.path.join(proc, "meminfo")) or ""
    for line in txt.splitlines():
        k, _, rest = line.partition(":")
        try:
            info[k.strip()] = float(rest.split()[0])          # kB
        except (IndexError, ValueError):
            pass
    tot, disp = info.get("MemTotal"), info.get("MemAvailable")
    swap = None
    if "SwapTotal" in info and "SwapFree" in info:
        swap = (info["SwapTotal"] - info["SwapFree"]) / 1024.0
    carga = None
    la = _read(os.path.join(proc, "loadavg"))
    try:
        carga = float(la.split()[0])
    except (AttributeError, IndexError, ValueError):
        pass
    return {"mem_disp_pct": 100.0 * disp / tot if tot and disp is not None else None,
            "mem_disp_gib": disp / 1048576.0 if disp is not None else None, "swap_usado_mib": swap, "carga_1m": carga}


def leer_smi_extra(timeout=15):
    """Lecturas adicionales de las GPU; cada campo puede no estar disponible. Devuelve None si nvidia-smi no responde."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(SMI_EXTRA), "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.strip().splitlines():
        c = [x.strip() for x in line.split(",")]
        if len(c) != len(SMI_EXTRA):
            continue

        def num(s):
            try:
                return float(s)
            except ValueError:
                return None
        txt, grave = motivos_reloj(c[3]) if c[3] and not c[3].startswith("[") and c[3] != "N/A" else ("", False)
        out.append({"index": c[0], "mem_temp": num(c[1]), "vent_pct": num(c[2]), "reduccion": txt, "reduccion_grave": grave})
    return out


class HostGuard:
    def __init__(self, paths, sysfs="/sys", proc="/proc", interval=10.0, cpu_warn=75.0, cpu_stop=85.0, disk_warn=65.0,
                 disk_stop=75.0, consecutivas=3, use_smi=True, log=None, mem_warn_pct=5.0, disk_free_warn_gb=10.0,
                 gpu_mem_warn=85.0, gpu_mem_stop=90.0, smi_reader=None):
        self.p, self.sysfs, self.proc = paths, sysfs, proc
        self.interval = max(1.0, min(float(interval), 60.0))
        self.base = {"cpu_warn": float(cpu_warn), "cpu_stop": min(float(cpu_stop), TOPES["cpu_stop"]),
                     "disk_warn": float(disk_warn), "disk_stop": min(float(disk_stop), TOPES["disk_stop"]),
                     "gpu_mem_warn": float(gpu_mem_warn), "gpu_mem_stop": min(float(gpu_mem_stop), TOPES["gpu_mem_stop"])}
        # el registro debe existir antes de aplicar_config, que lo usa (fallo del 18-11-2026 al arrancar con el fichero ya creado)
        self.log = log or (lambda m, level="INFO": print("%s [%s] host: %s" % (now_iso(), level, m), flush=True))
        self.config = os.path.join(paths.root, "runner", "host_guard_config.json")
        self.config_visto = None
        self.smi_reader = smi_reader or leer_smi_extra
        self.lim = {}
        self.aplicar_config()
        self.n = max(1, int(consecutivas))
        self.use_smi = use_smi
        self.mem_warn_pct, self.disk_free_warn_gb = mem_warn_pct, disk_free_warn_gb
        self.seguidas = {"cpu": 0, "disco": 0, "gpu_mem": 0}
        self.avisado = {}
        self.csv = os.path.join(paths.logs, "host_monitor.csv")
        self.hb = os.path.join(paths.state, "host_heartbeat.json")
        self.csv_day = None
        self.throttle0 = None
        self.tuvo_cpu = False
        self.sin_cpu = 0

    def aplicar_config(self):
        """Relee runner/host_guard_config.json: solo puede bajar los cortes por debajo de los topes."""
        v = dict(self.base)
        try:
            with open(self.config, encoding="utf-8") as f:
                txt = f.read()
            cfg = json.loads(txt)
            for k in v:
                if k in cfg:
                    x = float(cfg[k])
                    if x == x and 20.0 <= x <= 120.0:
                        v[k] = min(x, TOPES[k]) if k in TOPES else x
        except FileNotFoundError:
            txt = None
        except (OSError, ValueError, TypeError) as e:
            txt = "error"
            if self.config_visto != "error":
                self.log("host_guard_config.json no se pudo leer (%s): se usan los valores por defecto" % e, "ERROR")
        if txt != self.config_visto and txt not in (None, "error"):
            self.log("umbrales de host_guard_config.json aplicados: %s" % v)
        self.config_visto = txt
        self.lim = {"cpu": (v["cpu_warn"], v["cpu_stop"]), "disco": (v["disk_warn"], v["disk_stop"]),
                    "gpu_mem": (v["gpu_mem_warn"], v["gpu_mem_stop"])}

    # ---------------------------------------------------------------- lectura
    def leer(self):
        temps, fans = leer_hwmon(self.sysfs)
        if not any(t["clase"] == "cpu" for t in temps):
            temps += [z for z in leer_zonas(self.sysfs) if z["clase"] == "cpu"]
        r = {"temps": temps, "fans": fans, "throttle": leer_throttle(self.sysfs), "n_cpu": os.cpu_count()}
        r.update(leer_memoria(self.proc))
        try:
            r["disco_libre_gb"] = shutil.disk_usage(self.p.root).free / 1e9
        except OSError:
            r["disco_libre_gb"] = None
        r["gpus"] = self.smi_reader() if self.use_smi else None
        return r

    def corte(self, clase, sensores):
        """Umbral de corte efectivo: el configurado o, si es menor, el crítico del sensor menos 5 °C."""
        stop = self.lim[clase][1]
        crits = [s["crit"] for s in sensores if s.get("crit") and 40.0 < s["crit"] < 130.0]
        if crits:
            stop = min(stop, min(crits) - 5.0)
        return stop

    @staticmethod
    def _max(sensores):
        if not sensores:
            return None, ""
        s = max(sensores, key=lambda x: x["temp"])
        return s["temp"], "%s/%s" % (s["chip"], s["etiqueta"])

    def evento(self, clave, activo, level, msg):
        """Registra un aviso una sola vez mientras dure la condición, y su final."""
        if activo and not self.avisado.get(clave):
            self.avisado[clave] = True
            thermal_event(self.p, level, msg)
            self.log(msg, "WARN")
        elif not activo and self.avisado.get(clave):
            self.avisado[clave] = False
            thermal_event(self.p, "FIN_AVISO", "anfitrión: termina la condición '%s'" % clave)
            self.log("termina la condición '%s'" % clave)

    def trigger_stop(self, motivo, lectura, umbral):
        if os.path.exists(self.p.thermal_stop):
            return
        text = ("THERMAL_STOP creado por host_guard.py\nfecha: %s\nmotivo: %s\nlectura: %s\numbral de corte: %.1f °C\n"
                "Para liberar la cola: bash runner/clear_thermal_stop.sh (exige las GPU < 65 °C y la CPU < 70 °C)\n"
                % (now_iso(), motivo, lectura, umbral))
        try:
            atomic_write_text(self.p.thermal_stop, text)
        except OSError as e:
            self.log("no se pudo crear THERMAL_STOP: %s" % e, "ERROR")
        thermal_event(self.p, "PARADA", "anfitrión: %s | %s | umbral %.1f °C | se crea jobs/THERMAL_STOP; gpu_guard.py detiene "
                      "los trabajos del proyecto" % (motivo, lectura, umbral))
        self.log("PARADA: %s | %s" % (motivo, lectura), "CRIT")

    def cycle(self, now=None):
        now = time.time() if now is None else now
        self.aplicar_config()
        r = self.leer()
        estado = "ok"
        resumen = {}
        for clase in ("cpu", "disco"):
            ss = [t for t in r["temps"] if t["clase"] == clase]
            mx, nombre = self._max(ss)
            resumen[clase] = (mx, nombre, len(ss))
            if mx is None:
                self.seguidas[clase] = 0
                continue
            warn, stop = self.lim[clase][0], self.corte(clase, ss)
            warn = min(warn, stop - 5.0)
            self.seguidas[clase] = self.seguidas[clase] + 1 if mx >= stop else 0
            self.evento("temp_" + clase, mx >= warn, "AVISO", "anfitrión: %s a %.1f °C (%s) >= aviso %.1f °C; corte a %.1f °C"
                        % (clase.upper() if clase == "cpu" else "disco", mx, nombre, warn, stop))
            if mx >= warn:
                estado = "aviso"
            if self.seguidas[clase] >= self.n:
                self.trigger_stop("temperatura de %s >= %.1f °C en %d lecturas seguidas" % ("la CPU" if clase == "cpu" else "un disco",
                                                                                         stop, self.seguidas[clase]),
                                  "%s = %.1f °C" % (nombre, mx), stop)
        # sensores de CPU perdidos tras haber existido: solo se registra (gpu_guard sigue vigilando las GPU)
        if resumen["cpu"][0] is not None:
            self.tuvo_cpu, self.sin_cpu = True, 0
        elif self.tuvo_cpu:
            self.sin_cpu += 1
        self.evento("sensor_cpu", self.tuvo_cpu and self.sin_cpu >= 3, "AVISO", "anfitrión: se perdió la lectura de temperatura de la CPU")
        # estrangulamiento térmico, memoria y disco: solo aviso
        thr = r["throttle"]
        if thr is not None and self.throttle0 is None:
            self.throttle0 = thr
        self.evento("throttle", thr is not None and self.throttle0 is not None and thr > self.throttle0, "AVISO",
                    "anfitrión: la CPU ha reducido su frecuencia por temperatura (contador de estrangulamiento %s, al arrancar %s)"
                    % (thr, self.throttle0))
        self.evento("memoria", r["mem_disp_pct"] is not None and r["mem_disp_pct"] < self.mem_warn_pct, "AVISO",
                    "anfitrión: memoria disponible %.1f %% (< %.1f %%)" % (r["mem_disp_pct"] or 0, self.mem_warn_pct))
        self.evento("disco_libre", r["disco_libre_gb"] is not None and r["disco_libre_gb"] < self.disk_free_warn_gb, "AVISO",
                    "anfitrión: quedan %.1f GB libres en el disco del proyecto (< %.1f GB)" % (r["disco_libre_gb"] or 0, self.disk_free_warn_gb))
        gp = r["gpus"] or []
        red = ";".join("GPU%s:%s" % (g["index"], g["reduccion"]) for g in gp if g["reduccion"])
        grave = ";".join("GPU%s:%s" % (g["index"], g["reduccion"]) for g in gp if g.get("reduccion_grave"))
        self.evento("gpu_reloj", bool(grave), "AVISO", "anfitrión: alguna GPU reduce su reloj por temperatura o por hardware (%s)" % grave)
        # temperatura de la memoria de las GPU: aviso y corte propios
        mt = [(g["mem_temp"], "GPU%s" % g["index"]) for g in gp if g["mem_temp"] is not None]
        if mt:
            mx, nombre = max(mt)
            warn, stop = self.lim["gpu_mem"]
            warn = min(warn, stop - 3.0)
            self.seguidas["gpu_mem"] = self.seguidas["gpu_mem"] + 1 if mx >= stop else 0
            self.evento("temp_gpu_mem", mx >= warn, "AVISO", "anfitrión: memoria de %s a %.1f °C >= aviso %.1f °C; corte a %.1f °C"
                        % (nombre, mx, warn, stop))
            if mx >= warn and estado == "ok":
                estado = "aviso"
            if self.seguidas["gpu_mem"] >= self.n:
                self.trigger_stop("temperatura de la memoria de una GPU >= %.1f °C en %d lecturas seguidas" % (stop, self.seguidas["gpu_mem"]),
                                  "memoria de %s = %.1f °C" % (nombre, mx), stop)
        else:
            self.seguidas["gpu_mem"] = 0
        if os.path.exists(self.p.thermal_stop):
            estado = "parada"
        otros = [t for t in r["temps"] if t["clase"] == "otro"]
        omx, onom = self._max(otros)
        fila = {"cpu_max_c": resumen["cpu"][0], "cpu_sensor": resumen["cpu"][1], "cpu_n_sensores": resumen["cpu"][2],
                "disco_max_c": resumen["disco"][0], "disco_sensor": resumen["disco"][1], "otros_max_c": omx, "otros_sensor": onom,
                "vent_min_rpm": min((f["rpm"] for f in r["fans"]), default=None), "throttle_total": thr,
                "mem_disp_pct": r["mem_disp_pct"], "mem_disp_gib": r["mem_disp_gib"], "swap_usado_mib": r["swap_usado_mib"],
                "carga_1m": r["carga_1m"], "n_cpu": r["n_cpu"], "disco_libre_gb": r["disco_libre_gb"],
                "gpu_mem_temp_max_c": max((g["mem_temp"] for g in gp if g["mem_temp"] is not None), default=None),
                "gpu_vent_max_pct": max((g["vent_pct"] for g in gp if g["vent_pct"] is not None), default=None),
                "gpu_reduccion_reloj": red, "estado": estado}
        self.write_csv(fila, now)
        if self.hb is None:
            return estado, fila, r
        try:
            atomic_write_json(self.hb, dict(fila, ts=now, iso=now_iso(now), pid=os.getpid(), interval_s=self.interval,
                                            cpu_aviso_c=self.lim["cpu"][0], cpu_corte_c=self.lim["cpu"][1],
                                            disco_aviso_c=self.lim["disco"][0], disco_corte_c=self.lim["disco"][1],
                                            gpu_mem_aviso_c=self.lim["gpu_mem"][0], gpu_mem_corte_c=self.lim["gpu_mem"][1],
                                            lecturas_seguidas_para_corte=self.n))
        except OSError as e:
            self.log("no se pudo escribir el latido: %s" % e, "ERROR")
        return estado, fila, r

    def write_csv(self, fila, now):
        try:
            day = time.strftime("%Y%m%d", time.localtime(now))
            if self.csv_day is None:
                self.csv_day = day
            if day != self.csv_day and os.path.exists(self.csv):
                os.replace(self.csv, os.path.join(self.p.logs, "host_monitor_%s.csv" % self.csv_day))
                self.csv_day = day
            new = not os.path.exists(self.csv)
            cols = CSV_HEADER.split(",")[2:]
            vals = [now_iso(now), "%.3f" % now]
            for c in cols:
                v = fila.get(c)
                vals.append("" if v is None else ("%.1f" % v if isinstance(v, float) else str(v).replace(",", ";")))
            with open(self.csv, "a", encoding="utf-8") as f:
                if new:
                    f.write(CSV_HEADER + "\n")
                f.write(",".join(vals) + "\n")
        except OSError as e:
            self.log("no se pudo escribir host_monitor.csv: %s" % e, "ERROR")


def imprimir_sensores(g):
    estado, fila, r = g.cycle()
    print("Sensores de temperatura (%d):" % len(r["temps"]))
    for t in r["temps"]:
        print("  [%s] %s %s %s: %.1f °C (max %s, crit %s)" % (t["clase"], t["hwmon"], t["chip"], t["etiqueta"], t["temp"], t["max"], t["crit"]))
    print("Ventiladores (%d): %s" % (len(r["fans"]), ", ".join("%s/%s=%d rpm" % (f["chip"], f["etiqueta"], f["rpm"]) for f in r["fans"]) or "ninguno visible"))
    print("Contador de estrangulamiento térmico de la CPU: %s" % r["throttle"])
    print("GPU (nvidia-smi, lecturas adicionales): %s" % (r["gpus"] if r["gpus"] is not None else "no disponible"))
    print("Resumen: %s" % json.dumps(fila, ensure_ascii=False))
    for clase in ("cpu", "disco"):
        ss = [t for t in r["temps"] if t["clase"] == clase]
        if ss:
            print("Umbrales de %s: aviso %.1f °C, corte %.1f °C (%d lecturas seguidas)" % (clase, min(g.lim[clase][0], g.corte(clase, ss) - 5), g.corte(clase, ss), g.n))
        else:
            print("Sin sensores de %s visibles" % clase)
    print("Umbrales de la memoria de las GPU: aviso %.1f °C, corte %.1f °C (%d lecturas seguidas)" % (g.lim["gpu_mem"][0], g.lim["gpu_mem"][1], g.n))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vigilante del anfitrión de paper_new (CPU, discos, memoria)")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--cpu-warn", type=float, default=75.0)
    ap.add_argument("--cpu-stop", type=float, default=85.0)
    ap.add_argument("--gpu-mem-warn", type=float, default=85.0)
    ap.add_argument("--gpu-mem-stop", type=float, default=90.0)
    ap.add_argument("--disk-warn", type=float, default=65.0)
    ap.add_argument("--disk-stop", type=float, default=75.0)
    ap.add_argument("--consecutivas", type=int, default=3, help="lecturas seguidas en el umbral de corte para detener")
    ap.add_argument("--sysfs-root", default="/sys")
    ap.add_argument("--proc-root", default="/proc")
    ap.add_argument("--no-smi", action="store_true")
    ap.add_argument("--print-sensors", action="store_true")
    ap.add_argument("--check-below", type=float, default=None, help="código 0 si la CPU está por debajo de este valor o no hay sensor; 3 si no")
    ap.add_argument("--ciclos", type=int, default=0, help="número de ciclos y termina (pruebas); 0 = sin fin")
    a = ap.parse_args(argv)
    paths = Paths(a.root)
    if a.check_below is not None:
        temps, _ = leer_hwmon(a.sysfs_root)
        cpu = [t for t in temps if t["clase"] == "cpu"] or [z for z in leer_zonas(a.sysfs_root) if z["clase"] == "cpu"]
        if not cpu:
            print("CPU: sin sensor de temperatura visible (no se comprueba)")
            return 0
        mx = max(t["temp"] for t in cpu)
        print("CPU: máxima %.1f °C en %d sensores" % (mx, len(cpu)))
        return 0 if mx < a.check_below else 3
    os.makedirs(paths.state, exist_ok=True)
    os.makedirs(paths.logs, exist_ok=True)
    if a.print_sensors:
        g = HostGuard(paths, a.sysfs_root, a.proc_root, a.interval, a.cpu_warn, a.cpu_stop, a.disk_warn, a.disk_stop,
                      a.consecutivas, not a.no_smi, gpu_mem_warn=a.gpu_mem_warn, gpu_mem_stop=a.gpu_mem_stop)
        g.write_csv = lambda fila, now: None          # una lectura suelta no escribe en el registro
        g.hb = None
        g.evento = lambda *x, **k: None                # ni eventos ni paradas: solo muestra
        g.trigger_stop = lambda *x, **k: None
        return imprimir_sensores(g)
    log = Logger(os.path.join(paths.logs, "host_guard.log"), "anfitrión")
    g = HostGuard(paths, a.sysfs_root, a.proc_root, a.interval, a.cpu_warn, a.cpu_stop, a.disk_warn, a.disk_stop,
                  a.consecutivas, not a.no_smi, log, gpu_mem_warn=a.gpu_mem_warn, gpu_mem_stop=a.gpu_mem_stop)
    stopping = {"flag": False}
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stopping.__setitem__("flag", True))
    _, fila, r = g.cycle()
    log("en marcha: PID %d, intervalo %.0f s; CPU: %d sensores (aviso %.0f, corte %.0f °C); discos: %d sensores (aviso %.0f, corte %.0f °C); "
        "memoria de las GPU (aviso %.0f, corte %.0f °C); corte tras %d lecturas seguidas; primera lectura: CPU %s °C, disco %s °C, "
        "memoria de GPU %s °C"
        % (os.getpid(), g.interval, fila["cpu_n_sensores"], g.lim["cpu"][0], g.lim["cpu"][1],
           len([t for t in r["temps"] if t["clase"] == "disco"]), g.lim["disco"][0], g.lim["disco"][1],
           g.lim["gpu_mem"][0], g.lim["gpu_mem"][1], g.n, fila["cpu_max_c"], fila["disco_max_c"], fila["gpu_mem_temp_max_c"]))
    if fila["cpu_max_c"] is None:
        log("no se ve ningún sensor de temperatura de la CPU: se registran memoria, carga y disco, sin corte por CPU", "WARN")
    n = 1
    while not stopping["flag"] and (a.ciclos <= 0 or n < a.ciclos):
        t0 = time.time()
        while not stopping["flag"] and time.time() - t0 < g.interval:
            time.sleep(min(0.2, g.interval))
        if stopping["flag"]:
            break
        try:
            g.cycle()
        except Exception as e:                        # el vigilante no debe morir por una lectura rara
            log("error en el ciclo: %r" % (e,), "ERROR")
        n += 1
    log("terminado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
