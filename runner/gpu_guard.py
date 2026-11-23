#!/usr/bin/env python3
"""Vigilante térmico de las GPU del proyecto paper_new (PLAN.md, Section 9).

Se ejecuta en el SERVIDOR, fuera del contenedor, en su propia ventana de tmux.
Solo biblioteca estándar. Regla: ninguna GPU debe superar 80 °C; con una lectura >= 80 °C
se detienen los trabajos del proyecto y se crea jobs/THERMAL_STOP, que solo retira el usuario.

Uso normal:      python3 runner/gpu_guard.py
Pruebas:         python3 runner/gpu_guard.py --fake-smi temps.txt --no-docker --interval 1 --grace 3
Lectura única:   python3 runner/gpu_guard.py --print-temps
Comprobación:    python3 runner/gpu_guard.py --check-below 65   (código 0 si todas por debajo)
"""
import argparse
import fcntl
import glob
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONTAINER_PREFIX, GRACE_DEFAULT, GRACE_MAX, JOB_LABEL, STOP_C_MAX,  # noqa: E402
                    WARN_C_DEFAULT, WARN_CLEAR_C_DEFAULT, Logger, Paths, SmiError, append_line,
                    atomic_write_json, atomic_write_text, effective_stop_c, now_iso, query_smi,
                    parse_smi_csv, read_json, same_process, signal_group, thermal_event)

CSV_HEADER = "timestamp,epoch,gpu,temp_c,util_pct,mem_used_mib,mem_total_mib,power_w,estado"
DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt_reading(gpus):
    return ", ".join("GPU%d=%.0f °C" % (g["index"], g["temp"]) for g in gpus)


class Guard:
    def __init__(self, paths, interval=5.0, warn_c=WARN_C_DEFAULT, warn_clear_c=WARN_CLEAR_C_DEFAULT,
                 stop_c=None, grace=GRACE_DEFAULT, fake_smi=None, use_docker=True, max_failures=3,
                 csv_max_mb=50.0, log=None):
        self.p = paths
        self.interval = max(0.2, min(float(interval), 10.0))
        self.stop_c, note = effective_stop_c(stop_c)
        self.stop_note = note
        self.warn_c = min(float(warn_c), self.stop_c)
        self.warn_clear_c = min(float(warn_clear_c), self.warn_c)
        self.grace = max(0.5, min(float(grace), GRACE_MAX))
        self.fake_smi = fake_smi
        self.use_docker = use_docker
        self.max_failures = max(1, int(max_failures))
        self.csv_max_bytes = int(csv_max_mb * 1024 * 1024)
        self.log = log or Logger(paths.guard_log, "vigilante")
        self.failures = 0
        self.last_error = None
        self.last_gpus = None
        self.state = "iniciando"
        self.csv_day = None
        # Seguimiento de las paradas en curso
        self.docker_stopping = {}     # id de contenedor -> (instante del docker stop, matado)
        self.pg_signaled = {}         # (id, pgid) -> (instante del SIGTERM, matado)
        self.stop_seen_at = None
        self.last_stop_log = 0.0
        self.last_apps_log = 0.0
        self.apps_logged_once = False

    # ---------------------------------------------------------------- lectura y registro
    def read(self):
        try:
            gpus = parse_smi_csv(query_smi(self.fake_smi))
            if self.failures:
                self.log("lectura recuperada tras %d fallo(s)" % self.failures)
            self.failures = 0
            self.last_error = None
            return gpus
        except SmiError as e:
            self.failures += 1
            self.last_error = str(e)
            self.log("lectura fallida %d/%d: %s" % (self.failures, self.max_failures, e), "ERROR")
            return None

    def _rotate_if_needed(self, now):
        path = self.p.gpu_csv
        today = time.strftime("%Y%m%d", time.localtime(now))
        if not os.path.exists(path):
            self.csv_day = today
            return
        if self.csv_day is None:
            self.csv_day = time.strftime("%Y%m%d", time.localtime(os.path.getmtime(path)))
        size = os.path.getsize(path)
        if self.csv_day != today or size > self.csv_max_bytes:
            dest = os.path.join(self.p.logs, "gpu_monitor_%s.csv" % time.strftime("%Y%m%d_%H%M%S", time.localtime(now)))
            n = 1
            while os.path.exists(dest):
                dest = dest[:-4] + "_%d.csv" % n
                n += 1
            try:
                os.replace(path, dest)
                self.log("gpu_monitor.csv rotado a %s" % os.path.basename(dest))
            except OSError as e:
                self.log("no se pudo rotar gpu_monitor.csv: %s" % e, "ERROR")
            self.csv_day = today

    def write_csv(self, gpus, now, state):
        try:
            self._rotate_if_needed(now)
            new = not os.path.exists(self.p.gpu_csv)
            with open(self.p.gpu_csv, "a", encoding="utf-8") as f:
                if new:
                    f.write(CSV_HEADER + "\n")
                ts = now_iso(now)
                for g in gpus:
                    vals = [ts, "%.3f" % now, str(g["index"]), "%.1f" % g["temp"]]
                    for k in ("util", "mem_used", "mem_total", "power"):
                        vals.append("" if g[k] is None else "%g" % g[k])
                    vals.append(state)
                    f.write(",".join(vals) + "\n")
        except OSError as e:
            self.log("no se pudo escribir gpu_monitor.csv: %s" % e, "ERROR")

    # ---------------------------------------------------------------- parada
    def trigger_stop(self, reason, reading):
        if not os.path.exists(self.p.thermal_stop):
            text = ("THERMAL_STOP creado por gpu_guard.py\n"
                    "fecha: %s\nmotivo: %s\nlectura: %s\numbral de corte: %.1f °C\n"
                    "Para liberar la cola: bash runner/clear_thermal_stop.sh (exige todas las GPU < 65 °C)\n"
                    % (now_iso(), reason, reading, self.stop_c))
            try:
                atomic_write_text(self.p.thermal_stop, text)
            except OSError as e:
                self.log("no se pudo crear THERMAL_STOP: %s" % e, "ERROR")
            thermal_event(self.p, "PARADA", "%s | %s | umbral %.1f °C | se crea jobs/THERMAL_STOP y se detienen los trabajos del proyecto"
                          % (reason, reading, self.stop_c))
            self.log("PARADA: %s | %s" % (reason, reading), "CRIT")
            self.last_stop_log = time.time()
        elif time.time() - self.last_stop_log >= 60:
            thermal_event(self.p, "PARADA", "la condición de parada continúa: %s | %s" % (reason, reading))
            self.last_stop_log = time.time()

    def _docker(self, args, timeout=20):
        return subprocess.run(["docker"] + args, capture_output=True, text=True, timeout=timeout)

    def project_containers(self):
        if not self.use_docker:
            return []
        try:
            out = self._docker(["ps", "-q", "--filter", "label=" + JOB_LABEL])
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log("docker ps falló: %s" % e, "ERROR")
            return None
        if out.returncode != 0:
            self.log("docker ps falló: %s" % out.stderr.strip()[:200], "ERROR")
            return None
        return [x for x in out.stdout.split() if x]

    def project_process_groups(self):
        """Grupos de procesos de trabajos lanzados en modo sin Docker (registrados por el ejecutor)."""
        res = []
        for path in glob.glob(os.path.join(self.p.state_running, "*.json")):
            info = read_json(path)
            if not isinstance(info, dict) or info.get("mode") != "no-docker":
                continue
            pgid, st = info.get("pgid"), info.get("starttime")
            if isinstance(pgid, int) and pgid > 1 and same_process(pgid, st):
                res.append((info.get("id"), pgid))
        return res

    def enforce_stop(self, now):
        """Detiene los trabajos del proyecto. Idempotente: se llama en cada ciclo mientras exista THERMAL_STOP."""
        active = 0
        conts = self.project_containers()
        if conts:
            for cid in conts:
                active += 1
                if cid not in self.docker_stopping:
                    try:
                        subprocess.Popen(["docker", "stop", "-t", str(int(round(self.grace))), cid],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                        self.docker_stopping[cid] = (now, False)
                        thermal_event(self.p, "PARADA", "docker stop -t %d al contenedor %s" % (int(round(self.grace)), cid))
                    except OSError as e:
                        self.log("docker stop falló para %s: %s" % (cid, e), "ERROR")
                else:
                    t0, killed = self.docker_stopping[cid]
                    if not killed and now - t0 > self.grace + 15:
                        try:
                            self._docker(["kill", cid])
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                        self.docker_stopping[cid] = (t0, True)
                        thermal_event(self.p, "PARADA", "el contenedor %s seguía vivo tras docker stop: docker kill" % cid)
        for jid, pgid in self.project_process_groups():
            active += 1
            key = (jid, pgid)
            if key not in self.pg_signaled:
                signal_group(pgid, signal.SIGTERM)
                self.pg_signaled[key] = (now, False)
                thermal_event(self.p, "PARADA", "SIGTERM al grupo de procesos %d (trabajo %s)" % (pgid, jid))
            else:
                t0, killed = self.pg_signaled[key]
                if not killed and now - t0 >= self.grace:
                    signal_group(pgid, signal.SIGKILL)
                    self.pg_signaled[key] = (t0, True)
                    thermal_event(self.p, "PARADA", "SIGKILL al grupo de procesos %d (trabajo %s) tras %.0f s de gracia"
                                  % (pgid, jid, self.grace))
        return active

    def log_foreign_apps(self, gpus, now):
        """Si tras detener lo nuestro la temperatura sigue alta, registra quién usa la GPU. No mata nada."""
        if now - self.last_apps_log < 60:
            return
        self.last_apps_log = now
        if self.fake_smi:
            apps = "(no disponible en modo de prueba --fake-smi)"
        else:
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                      "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
                apps = out.stdout.strip() or "(ningún proceso de cómputo listado)"
                idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=15)
                apps += " | mapa índice-uuid: " + idx.stdout.strip().replace("\n", "; ")
            except (OSError, subprocess.TimeoutExpired) as e:
                apps = "(no se pudo consultar: %s)" % e
        thermal_event(self.p, "AVISO", "los trabajos del proyecto están detenidos pero la temperatura sigue alta (%s). "
                      "Procesos en GPU (NO se matan, revisar a mano): %s" % (fmt_reading(gpus), apps.replace("\n", "; ")))
        self.log("temperatura alta sin trabajos del proyecto; procesos ajenos registrados en thermal_events.log", "WARN")

    # ---------------------------------------------------------------- aviso
    def handle_warn(self, gpus):
        hot = [g for g in gpus if g["temp"] >= self.warn_c]
        exists = os.path.exists(self.p.warn)
        if hot and not exists:
            atomic_write_text(self.p.warn, "aviso térmico %s: %s (umbral de aviso %.1f °C)\n"
                              % (now_iso(), fmt_reading(gpus), self.warn_c))
            thermal_event(self.p, "AVISO", "%s >= %.1f °C: se crea THERMAL_WARN; no se lanzan trabajos nuevos"
                          % (fmt_reading(hot), self.warn_c))
            self.log("AVISO térmico: %s" % fmt_reading(hot), "WARN")
        elif exists and all(g["temp"] < self.warn_clear_c for g in gpus):
            try:
                os.remove(self.p.warn)
                thermal_event(self.p, "FIN_AVISO", "todas las GPU por debajo de %.1f °C (%s): se retira THERMAL_WARN"
                              % (self.warn_clear_c, fmt_reading(gpus)))
                self.log("aviso retirado: %s" % fmt_reading(gpus))
            except OSError as e:
                self.log("no se pudo retirar THERMAL_WARN: %s" % e, "ERROR")

    # ---------------------------------------------------------------- ciclo
    def cycle(self, now=None):
        now = time.time() if now is None else now
        gpus = self.read()
        if gpus is not None:
            self.last_gpus = gpus
        stop_reason = reading = None
        if gpus is not None:
            hot = [g for g in gpus if g["temp"] >= self.stop_c]
            if hot:
                stop_reason = "temperatura >= %.1f °C en %s" % (self.stop_c, ", ".join("GPU%d" % g["index"] for g in hot))
                reading = fmt_reading(gpus)
        elif self.failures >= self.max_failures:
            stop_reason = "sensor no disponible (%d lecturas fallidas seguidas)" % self.failures
            reading = "último error: %s" % self.last_error
        if stop_reason:
            self.trigger_stop(stop_reason, reading)
        if gpus is not None:
            try:
                self.handle_warn(gpus)
            except OSError as e:
                self.log("error al gestionar THERMAL_WARN: %s" % e, "ERROR")

        stop_active = os.path.exists(self.p.thermal_stop)
        active = 0
        if stop_active:
            if self.stop_seen_at is None:
                self.stop_seen_at = now
            active = self.enforce_stop(now)
            if (gpus is not None and active == 0 and now - self.stop_seen_at > self.grace + 10
                    and any(g["temp"] >= self.warn_c for g in gpus)):
                self.log_foreign_apps(gpus, now)
        else:
            self.stop_seen_at = None
            self.docker_stopping.clear()
            self.pg_signaled.clear()

        if gpus is None:
            self.state = "sensor_fallido" if self.failures >= self.max_failures else "lectura_fallida"
        elif stop_active:
            self.state = "parada"
        elif os.path.exists(self.p.warn):
            self.state = "aviso"
        else:
            self.state = "ok"
        if gpus is not None:
            self.write_csv(gpus, now, self.state)
        self.write_heartbeat(now, active)
        return self.state

    def write_heartbeat(self, now, active=0, state=None):
        gpus = self.last_gpus or []
        hb = {
            "ts": now, "iso": now_iso(now), "pid": os.getpid(), "host": os.uname().nodename,
            "state": state or self.state,
            "temps": {str(g["index"]): g["temp"] for g in gpus} if self.failures == 0 else {},
            "gpus": [g["index"] for g in gpus],
            "stop_c": self.stop_c, "warn_c": self.warn_c, "warn_clear_c": self.warn_clear_c,
            "interval_s": self.interval, "grace_s": self.grace,
            "consecutive_failures": self.failures, "last_error": self.last_error,
            "thermal_stop": os.path.exists(self.p.thermal_stop), "thermal_warn": os.path.exists(self.p.warn),
            "project_jobs_being_stopped": active, "fake_smi": bool(self.fake_smi), "docker": self.use_docker,
        }
        try:
            atomic_write_json(self.p.guard_hb, hb)
        except OSError as e:
            self.log("no se pudo escribir el latido: %s" % e, "ERROR")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vigilante térmico de GPU de paper_new")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="raíz del proyecto (por defecto, la carpeta padre de runner/)")
    ap.add_argument("--interval", type=float, default=5.0, help="segundos entre lecturas (máximo 10)")
    ap.add_argument("--warn", type=float, default=float(os.environ.get("PAPER_NEW_WARN_C", WARN_C_DEFAULT)))
    ap.add_argument("--warn-clear", type=float, default=WARN_CLEAR_C_DEFAULT)
    ap.add_argument("--stop", type=float, default=None,
                    help="umbral de corte; solo puede bajar de 80 (también PAPER_NEW_STOP_C)")
    ap.add_argument("--grace", type=float, default=GRACE_DEFAULT, help="segundos entre SIGTERM y SIGKILL (máximo 30)")
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--csv-max-mb", type=float, default=50.0)
    ap.add_argument("--fake-smi", default=None, help="fichero de texto con lecturas falsas (pruebas)")
    ap.add_argument("--no-docker", action="store_true", help="no usar docker (pruebas)")
    ap.add_argument("--print-temps", action="store_true", help="leer una vez y mostrar las temperaturas")
    ap.add_argument("--check-below", type=float, default=None,
                    help="leer una vez; código 0 si todas las GPU están por debajo del valor, 3 si no, 2 si falla")
    a = ap.parse_args(argv)

    if a.print_temps or a.check_below is not None:
        try:
            gpus = parse_smi_csv(query_smi(a.fake_smi))
        except SmiError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            return 2
        for g in gpus:
            print("GPU%d %.1f" % (g["index"], g["temp"]))
        if a.check_below is not None:
            return 0 if all(g["temp"] < a.check_below for g in gpus) else 3
        return 0

    paths = Paths(a.root)
    paths.ensure()
    os.makedirs(paths.logs, exist_ok=True)
    log = Logger(paths.guard_log, "vigilante")
    lockf = open(paths.guard_lock, "a")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("ya hay otro vigilante en marcha (cerrojo %s ocupado); salgo" % paths.guard_lock, "ERROR")
        return 3

    g = Guard(paths, interval=a.interval, warn_c=a.warn, warn_clear_c=a.warn_clear, stop_c=a.stop,
              grace=a.grace, fake_smi=a.fake_smi, use_docker=not a.no_docker,
              max_failures=a.max_failures, csv_max_mb=a.csv_max_mb, log=log)
    stopping = {"flag": False}

    def on_signal(signum, frame):
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    log("en marcha: PID %d, intervalo %.1f s, aviso %.1f °C (se retira < %.1f), corte %.1f °C, gracia %.0f s, docker=%s%s"
        % (os.getpid(), g.interval, g.warn_c, g.warn_clear_c, g.stop_c, g.grace, g.use_docker,
           ", lecturas FALSAS de %s" % a.fake_smi if a.fake_smi else ""))
    if g.stop_note:
        log(g.stop_note, "WARN")
        thermal_event(paths, "CONFIG", "vigilante PID %d: %s" % (os.getpid(), g.stop_note))
    thermal_event(paths, "INICIO", "vigilante PID %d en marcha; aviso %.1f °C, corte %.1f °C"
                  % (os.getpid(), g.warn_c, g.stop_c))
    while not stopping["flag"]:
        t0 = time.time()
        try:
            g.cycle(t0)
        except Exception as e:  # el vigilante no debe caer por un error inesperado
            log("error inesperado en el ciclo: %r" % e, "ERROR")
        while not stopping["flag"] and time.time() - t0 < g.interval:
            time.sleep(min(0.1, g.interval))
    g.write_heartbeat(time.time(), state="detenido")
    thermal_event(paths, "FIN", "vigilante PID %d detenido por señal; sin vigilante el ejecutor no lanza trabajos" % os.getpid())
    log("detenido")
    return 0


if __name__ == "__main__":
    sys.exit(main())
