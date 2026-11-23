#!/usr/bin/env python3
"""Ejecutor de la cola de trabajos de paper_new (PLAN.md, Section 9).

Se ejecuta en el SERVIDOR, fuera del contenedor, en su propia ventana de tmux. Solo biblioteca estándar.
Cola por carpetas: jobs/pending -> jobs/running -> jobs/done | jobs/failed.
Nunca lanza nada sin latido reciente del vigilante, con jobs/THERMAL_STOP presente o con THERMAL_WARN.

Uso normal:  python3 runner/job_runner.py
Pruebas:     python3 runner/job_runner.py --no-docker --tick 0.5 --hb-max-age 5 --grace 3
Estado:      python3 runner/job_runner.py --status
"""
import argparse
import csv
import fcntl
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONTAINER_PREFIX, EXIT_THERMAL, GRACE_DEFAULT, GRACE_MAX, HB_MAX_AGE_MAX,  # noqa: E402
                    JOB_LABEL, Logger, Paths, atomic_write_json, atomic_write_text, heartbeat_status,
                    now_iso, pid_alive, proc_starttime, read_json, same_process, signal_group,
                    thermal_event)

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HOST_PROJECT = "/home/administrador/Proyectos/paper_new"
DEFAULT_DATA_DIR = "/home/administrador/notebooks/covid"
DEFAULT_IMAGE = "paper-new:cu126"
CONTAINER_WORKDIR = "/workspace/paper_new"
CONTAINER_DATA = "/data/covid"
DEFAULT_ALLOW = ("python", "python3", "pytest")
DEFAULT_ALLOW_BASH_DIR = "scripts/"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
REQUIRED = ("id", "comando", "gpu", "prioridad", "tiempo_max_min", "descripcion")
OPTIONAL = ("reencolar_si_termico", "creado", "origen", "interrumpido", "intentos", "historial", "notas")
MAX_MINUTES = 7 * 24 * 60
THERMAL_MARK = "interrumpido por temperatura"


# =============================================================================== validación
def validate_job(obj, stem=None, allow=DEFAULT_ALLOW, allow_bash_dir=DEFAULT_ALLOW_BASH_DIR):
    """Devuelve la lista de errores (vacía si el trabajo es válido)."""
    errs = []
    if not isinstance(obj, dict):
        return ["el JSON no es un objeto"]
    for k in REQUIRED:
        if k not in obj:
            errs.append("falta el campo obligatorio '%s'" % k)
    unknown = [k for k in obj if k not in REQUIRED and k not in OPTIONAL]
    if unknown:
        errs.append("campos desconocidos: %s" % ", ".join(sorted(unknown)))
    jid = obj.get("id")
    if "id" in obj:
        if not isinstance(jid, str) or not ID_RE.match(jid):
            errs.append("id no válido (letras, dígitos, _ . -; máximo 64; empieza por letra o dígito)")
        elif stem is not None and stem != jid:
            errs.append("el nombre del fichero (%s.json) no coincide con el id (%s)" % (stem, jid))
    if "comando" in obj:
        errs += validate_command(obj["comando"], allow, allow_bash_dir)
    if "gpu" in obj:
        g = obj["gpu"]
        if not (g == "any" or (isinstance(g, int) and not isinstance(g, bool) and g in (0, 1))):
            errs.append("gpu debe ser 0, 1 o \"any\"")
    if "prioridad" in obj:
        p = obj["prioridad"]
        if not isinstance(p, int) or isinstance(p, bool) or not (-1000 <= p <= 1000):
            errs.append("prioridad debe ser un entero entre -1000 y 1000 (mayor = antes)")
    if "tiempo_max_min" in obj:
        t = obj["tiempo_max_min"]
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not (0 < t <= MAX_MINUTES):
            errs.append("tiempo_max_min debe ser un número > 0 y <= %d" % MAX_MINUTES)
    if "descripcion" in obj and not isinstance(obj["descripcion"], str):
        errs.append("descripcion debe ser texto")
    if "reencolar_si_termico" in obj and not isinstance(obj["reencolar_si_termico"], bool):
        errs.append("reencolar_si_termico debe ser true o false")
    return errs


def validate_command(cmd, allow=DEFAULT_ALLOW, allow_bash_dir=DEFAULT_ALLOW_BASH_DIR):
    if not isinstance(cmd, list) or not cmd:
        return ["comando debe ser una lista no vacía de argumentos (no una cadena)"]
    if len(cmd) > 256:
        return ["comando con demasiados argumentos"]
    for a in cmd:
        if not isinstance(a, str) or "\x00" in a or len(a) > 8192:
            return ["todos los argumentos del comando deben ser texto sin caracteres nulos"]
    exe = cmd[0]
    if exe in allow:
        return []
    if exe == "bash" and allow_bash_dir:
        if len(cmd) < 2:
            return ["bash necesita un script dentro de %s" % allow_bash_dir]
        script = cmd[1]
        norm = os.path.normpath(script)
        if (script.startswith("/") or not script.startswith(allow_bash_dir) or ".." in script.split("/")
                or not norm.startswith(allow_bash_dir.rstrip("/") + "/") or not script.endswith(".sh")):
            return ["bash solo puede ejecutar un script .sh dentro de %s (recibido %r)" % (allow_bash_dir, script)]
        return []
    return ["ejecutable no permitido: %r (permitidos: %s, bash %s*.sh)" % (exe, ", ".join(allow), allow_bash_dir)]


# =============================================================================== ejecutor
class RunningJob:
    def __init__(self, spec, gpu, proc, log_fh, start, container=None):
        self.spec = spec
        self.id = spec["id"]
        self.gpu = gpu
        self.proc = proc
        self.log_fh = log_fh
        self.start = start
        self.deadline = start + float(spec["tiempo_max_min"]) * 60.0
        self.container = container
        self.stop_kind = None       # "termico" | "timeout" | "parada_ejecutor"
        self.stop_reason = None
        self.stop_t = None
        self.kill_sent = False
        self.client_killed = False
        self.pgid = proc.pid
        self.starttime = proc_starttime(proc.pid)


class Runner:
    def __init__(self, paths, use_docker=True, tick=2.0, hb_max_age=HB_MAX_AGE_MAX, grace=GRACE_DEFAULT,
                 image=DEFAULT_IMAGE, host_project=None, data_dir=DEFAULT_DATA_DIR, allow=DEFAULT_ALLOW,
                 allow_bash_dir=DEFAULT_ALLOW_BASH_DIR, only_job=None, exit_after_job=False,
                 shm_size="16g", log=None):
        self.p = paths
        self.use_docker = use_docker
        self.tick = max(0.2, min(float(tick), 10.0))
        self.hb_max_age = max(1.0, min(float(hb_max_age), HB_MAX_AGE_MAX))
        self.grace = max(0.5, min(float(grace), GRACE_MAX))
        self.image = image
        self.host_project = host_project or paths.root
        self.data_dir = data_dir
        self.allow = tuple(allow)
        self.allow_bash_dir = allow_bash_dir
        self.only_job = only_job
        self.exit_after_job = exit_after_job
        self.shm_size = shm_size
        self.log = log or Logger(paths.runner_log, "ejecutor")
        self.running = {}            # id -> RunningJob
        self.shutting_down = False
        self.finished_only_job = False
        self.last_block = None
        self.ignored_logged = set()
        self.hb = None
        self.hb_ok = False
        self.hb_msg = ""
        self.counts = {}

    # ------------------------------------------------------------------ utilidades
    def _unique(self, folder, name):
        dest = os.path.join(folder, name)
        if not os.path.exists(dest):
            return dest
        base, ext = os.path.splitext(name)
        return os.path.join(folder, "%s__%s%s" % (base, time.strftime("%Y%m%d_%H%M%S"), ext))

    def reject(self, path, reasons):
        name = os.path.basename(path)
        dest = self._unique(self.p.failed, name)
        os.replace(path, dest)
        atomic_write_text(os.path.splitext(dest)[0] + ".motivo.txt",
                          "rechazado %s\nfichero: %s\nmotivos:\n- %s\n" % (now_iso(), name, "\n- ".join(reasons)))
        self.log("trabajo rechazado %s -> failed/: %s" % (name, "; ".join(reasons)), "WARN")

    def ids_in(self, folder):
        return {os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(".json") and not f.startswith(".")}

    # ------------------------------------------------------------------ cola
    def scan_pending(self):
        """Valida la carpeta pending y devuelve los trabajos válidos ordenados por prioridad."""
        valid = []
        done_ids = self.ids_in(self.p.done)
        for name in sorted(os.listdir(self.p.pending)):
            path = os.path.join(self.p.pending, name)
            if name.startswith(".") or not os.path.isfile(path):
                continue
            if not name.endswith(".json"):
                if name not in self.ignored_logged:
                    self.log("se ignora %s en pending/ (solo se leen ficheros .json)" % name, "WARN")
                    self.ignored_logged.add(name)
                continue
            stem = name[:-5]
            if self.only_job and stem != self.only_job:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    obj = json.load(f)
            except ValueError as e:
                self.reject(path, ["JSON mal formado: %s" % e])
                continue
            except OSError:
                continue
            errs = validate_job(obj, stem, self.allow, self.allow_bash_dir)
            if not errs and (stem in self.running or stem in done_ids):
                errs = ["id duplicado: ya existe un trabajo %s en running/ o done/" % stem]
            if errs:
                self.reject(path, errs)
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            valid.append((-int(obj["prioridad"]), mtime, stem, obj, path))
        valid.sort(key=lambda x: (x[0], x[1], x[2]))
        return [(v[3], v[4]) for v in valid]

    def gpus_available(self):
        gpus = self.hb.get("gpus") if isinstance(self.hb, dict) else None
        if not isinstance(gpus, list):
            return []
        busy = {j.gpu for j in self.running.values()}
        return [g for g in gpus if isinstance(g, int) and g not in busy]

    def launch_ready(self):
        if self.shutting_down:
            return "parada del ejecutor en curso"
        if self.finished_only_job:
            return "trabajo único ya terminado"
        if not self.hb_ok:
            return "sin vigilante: " + self.hb_msg
        if os.path.exists(self.p.thermal_stop):
            return "jobs/THERMAL_STOP presente (parada térmica; solo la retira el usuario)"
        if os.path.exists(self.p.warn):
            return "THERMAL_WARN presente (aviso térmico)"
        if self.hb.get("state") != "ok":
            return "estado del vigilante: %s" % self.hb.get("state")
        return None

    def try_launch(self, pending):
        block = self.launch_ready()
        if block:
            if pending and block != self.last_block:
                self.log("no se lanzan trabajos: %s" % block)
            self.last_block = block
            return
        self.last_block = None
        temps = self.hb.get("temps") or {}
        for spec, path in pending:
            free = self.gpus_available()
            if not free:
                return
            want = spec["gpu"]
            if want == "any":
                gpu = sorted(free, key=lambda g: (temps.get(str(g), 999), g))[0]
            elif want in free:
                gpu = want
            else:
                continue
            # Comprobación inmediatamente antes de lanzar
            if self.launch_ready():
                return
            self.launch(spec, path, gpu)

    def build_command(self, spec, gpu):
        jid = spec["id"]
        cmd = list(spec["comando"])
        env = dict(os.environ)
        env.update({"PAPER_NEW_JOB_ID": jid, "PAPER_NEW_GPU_INDEX": str(gpu), "PYTHONUNBUFFERED": "1"})
        if not self.use_docker:
            if cmd[0] in ("python", "python3"):
                cmd[0] = sys.executable
            elif cmd[0] == "pytest":
                cmd = [sys.executable, "-m", "pytest"] + cmd[1:]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            return cmd, env, self.p.root, None
        name = CONTAINER_PREFIX + jid
        docker = ["docker", "run", "--rm",
                  "--label", JOB_LABEL, "--label", "paper_new.job_id=" + jid,
                  "--name", name,
                  "--gpus", "device=%d" % gpu,
                  "--shm-size", self.shm_size,
                  "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
                  "--user", "%d:%d" % (os.getuid(), os.getgid()),
                  "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                  "-e", "PAPER_NEW_JOB_ID=" + jid, "-e", "PAPER_NEW_GPU_INDEX=%d" % gpu,
                  "-e", "PYTHONUNBUFFERED=1", "-e", "HOME=/tmp",
                  "-e", "TORCH_HOME=/cache/torch", "-e", "HF_HOME=/cache/hf",
                  "-e", "MPLCONFIGDIR=/tmp/matplotlib",
                  "-v", "%s:%s:rw" % (self.host_project, CONTAINER_WORKDIR),
                  "-v", "%s:%s:ro" % (self.data_dir, CONTAINER_DATA),
                  "-v", "paper_new_torch_cache:/cache/torch",
                  "-v", "paper_new_hf_cache:/cache/hf",
                  "-w", CONTAINER_WORKDIR]
        stop_env = os.environ.get("PAPER_NEW_STOP_C", "").strip()
        if stop_env:
            docker += ["-e", "PAPER_NEW_STOP_C=" + stop_env]
        docker += [self.image] + cmd
        return docker, env, self.p.root, name

    def launch(self, spec, path, gpu):
        jid = spec["id"]
        run_path = os.path.join(self.p.running, jid + ".json")
        os.replace(path, run_path)
        cmd, env, cwd, cname = self.build_command(spec, gpu)
        log_path = os.path.join(self.p.jobs_logs, jid + ".log")
        fh = open(log_path, "a", encoding="utf-8")
        attempt = int(spec.get("intentos", 0)) + 1
        fh.write("\n===== %s inicio del trabajo %s (intento %d) en GPU %d, modo %s =====\n"
                 % (now_iso(), jid, attempt, gpu, "docker" if self.use_docker else "no-docker"))
        fh.write("comando: %s\n" % json.dumps(cmd, ensure_ascii=False))
        fh.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            fh.write("no se pudo lanzar: %s\n" % e)
            fh.close()
            spec["intentos"] = attempt
            atomic_write_json(run_path, spec)
            dest = self._unique(self.p.failed, jid + ".json")
            os.replace(run_path, dest)
            self.log("no se pudo lanzar %s: %s" % (jid, e), "ERROR")
            return
        start = time.time()
        job = RunningJob(spec, gpu, proc, fh, start, cname)
        job.attempt = attempt
        self.running[jid] = job
        atomic_write_json(os.path.join(self.p.state_running, jid + ".json"), {
            "id": jid, "pid": proc.pid, "pgid": proc.pid, "starttime": job.starttime, "gpu": gpu,
            "mode": "docker" if self.use_docker else "no-docker", "container": cname,
            "start": start, "start_iso": now_iso(start)})
        self.log("lanzado %s en GPU %d (PID %d%s, prioridad %s, límite %s min)"
                 % (jid, gpu, proc.pid, ", contenedor " + cname if cname else "", spec["prioridad"], spec["tiempo_max_min"]))

    # ------------------------------------------------------------------ parada
    def request_stop(self, job, kind, reason):
        now = time.time()
        if job.stop_kind is None:
            job.stop_kind, job.stop_reason, job.stop_t = kind, reason, now
            self.log("deteniendo %s: %s" % (job.id, reason), "WARN")
            try:
                job.log_fh.write("%s [ejecutor] parada solicitada: %s\n" % (now_iso(now), reason))
                job.log_fh.flush()
            except (OSError, ValueError):
                pass
            if self.use_docker:
                try:
                    subprocess.Popen(["docker", "stop", "-t", str(int(round(self.grace))), job.container],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                except OSError as e:
                    self.log("docker stop falló: %s" % e, "ERROR")
            else:
                if same_process(job.pgid, job.starttime):
                    signal_group(job.pgid, signal.SIGTERM)
        elif kind == "termico" and job.stop_kind != "termico":
            # Una causa térmica prevalece sobre otra (afecta a dónde vuelve el trabajo)
            job.stop_kind, job.stop_reason = kind, reason

    def escalate(self):
        now = time.time()
        for job in self.running.values():
            if job.stop_t is None:
                continue
            if not job.kill_sent and now - job.stop_t >= self.grace:
                job.kill_sent = True
                if self.use_docker:
                    subprocess.Popen(["docker", "kill", job.container], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
                else:
                    signal_group(job.pgid, signal.SIGKILL)
                self.log("%s sigue vivo tras %.0f s de gracia: SIGKILL" % (job.id, self.grace), "WARN")
            if self.use_docker and job.kill_sent and not job.client_killed and now - job.stop_t >= self.grace + 15:
                job.client_killed = True
                signal_group(job.pgid, signal.SIGKILL)

    def create_stop(self, reason):
        if not os.path.exists(self.p.thermal_stop):
            atomic_write_text(self.p.thermal_stop, "THERMAL_STOP creado por job_runner.py\nfecha: %s\nmotivo: %s\n"
                              "Para liberar la cola: bash runner/clear_thermal_stop.sh\n" % (now_iso(), reason))
            thermal_event(self.p, "PARADA", "ejecutor: %s; se crea jobs/THERMAL_STOP" % reason)
            self.log("THERMAL_STOP creado: %s" % reason, "CRIT")

    # ------------------------------------------------------------------ fin de trabajo
    def temps_during(self, gpu, start, end):
        vals = []
        files = glob.glob(os.path.join(self.p.logs, "gpu_monitor*.csv"))
        for path in files:
            try:
                if os.path.getmtime(path) < start - 5:
                    continue
                # errors="replace" y el filtro de bytes nulos: un apagado brusco puede dejar nulos al final del
                # registro, y csv.DictReader falla con ellos (18-11-2026: el ejecutor quedó bloqueado por eso)
                with open(path, newline="", encoding="utf-8", errors="replace") as f:
                    for row in csv.DictReader(line.replace("\0", "") for line in f):
                        try:
                            t = float(row["epoch"])
                            if start <= t <= end + 1 and int(row["gpu"]) == gpu:
                                vals.append(float(row["temp_c"]))
                        except (KeyError, ValueError, TypeError):
                            continue
            except (OSError, csv.Error):
                continue
        if not vals:
            return None, None, 0
        return max(vals), round(sum(vals) / len(vals), 2), len(vals)

    def finalize(self, job, rc):
        end = time.time()
        jid = job.id
        kind, reason = job.stop_kind, job.stop_reason
        stop_exists = os.path.exists(self.p.thermal_stop)
        if kind is None:
            if stop_exists:
                kind, reason = "termico", "detenido por el vigilante (jobs/THERMAL_STOP)"
            elif rc == EXIT_THERMAL:
                kind, reason = "termico", "el entrenamiento se detuvo por su doble seguro térmico (código 75)"
                self.create_stop("el trabajo %s salió con código 75 (límite térmico alcanzado dentro del entrenamiento)" % jid)
            elif rc == 0:
                kind, reason = "ok", "terminado"
            else:
                kind, reason = "error", "código de salida %d" % rc
        requeue = kind in ("termico", "parada_ejecutor") and job.spec.get("reencolar_si_termico", True)
        if kind == "ok":
            dest_dir = self.p.done
        elif requeue:
            dest_dir = self.p.pending
        else:
            dest_dir = self.p.failed
        tmax, tmean, n = self.temps_during(job.gpu, job.start, end)
        summary = {
            "id": jid, "descripcion": job.spec.get("descripcion"), "comando": job.spec.get("comando"),
            "gpu": job.gpu, "modo": "docker" if self.use_docker else "no-docker", "contenedor": job.container,
            "intento": job.attempt, "inicio": now_iso(job.start), "fin": now_iso(end),
            "duracion_s": round(end - job.start, 1), "codigo_salida": rc,
            "senal": (-rc if rc < 0 else (rc - 128 if rc > 128 else None)),
            "motivo_parada": reason, "tipo_parada": kind,
            "destino": os.path.basename(dest_dir),
            "temp_max_c": tmax, "temp_media_c": tmean, "n_lecturas_temp": n,
            "fuente_temperaturas": "logs/gpu_monitor*.csv",
        }
        sum_path = os.path.join(self.p.jobs_logs, jid + ".json")
        prev = read_json(sum_path)
        history = []
        if isinstance(prev, dict):
            history = prev.get("intentos_previos", [])
            prev.pop("intentos_previos", None)
            history.append(prev)
        summary["intentos_previos"] = history
        atomic_write_json(sum_path, summary)
        try:
            job.log_fh.write("===== %s fin del trabajo %s: código %s, %s -> %s =====\n"
                             % (now_iso(end), jid, rc, reason, os.path.basename(dest_dir)))
            job.log_fh.close()
        except (OSError, ValueError):
            pass
        spec = dict(job.spec)
        spec["intentos"] = job.attempt
        hist = list(spec.get("historial", []))
        hist.append({"intento": job.attempt, "inicio": now_iso(job.start), "fin": now_iso(end),
                     "codigo_salida": rc, "motivo": reason, "gpu": job.gpu})
        spec["historial"] = hist
        if requeue:
            spec["interrumpido"] = THERMAL_MARK if kind == "termico" else "interrumpido por parada del ejecutor"
        else:
            spec.pop("interrumpido", None)
        run_path = os.path.join(self.p.running, jid + ".json")
        dest = os.path.join(dest_dir, jid + ".json") if dest_dir == self.p.pending else self._unique(dest_dir, jid + ".json")
        atomic_write_json(run_path, spec)
        os.replace(run_path, dest)
        try:
            os.remove(os.path.join(self.p.state_running, jid + ".json"))
        except OSError:
            pass
        del self.running[jid]
        self.log("fin de %s: código %s, %s -> %s (T máx %s °C, media %s °C)"
                 % (jid, rc, reason, os.path.basename(dest_dir), tmax, tmean), "INFO" if kind == "ok" else "WARN")
        if self.only_job and jid == self.only_job:
            self.finished_only_job = True

    # ------------------------------------------------------------------ recuperación al arrancar
    def recover(self):
        for name in sorted(os.listdir(self.p.running)):
            if not name.endswith(".json") or name.startswith("."):
                continue
            jid = name[:-5]
            info = read_json(os.path.join(self.p.state_running, name)) or {}
            self.log("al arrancar se encontró %s en running/: se detiene si sigue vivo y vuelve a pending" % jid, "WARN")
            if self.use_docker:
                try:
                    subprocess.run(["docker", "stop", "-t", str(int(round(self.grace))), CONTAINER_PREFIX + jid],
                                   capture_output=True, timeout=self.grace + 30)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            pgid, st = info.get("pgid"), info.get("starttime")
            if isinstance(pgid, int) and same_process(pgid, st):
                signal_group(pgid, signal.SIGTERM)
                t0 = time.time()
                while time.time() - t0 < self.grace and same_process(pgid, st):
                    time.sleep(0.2)
                if same_process(pgid, st):
                    signal_group(pgid, signal.SIGKILL)
            spec = read_json(os.path.join(self.p.running, name))
            if isinstance(spec, dict):
                spec["interrumpido"] = "interrumpido por reinicio del ejecutor"
                atomic_write_json(os.path.join(self.p.running, name), spec)
            os.replace(os.path.join(self.p.running, name), os.path.join(self.p.pending, name))
            try:
                os.remove(os.path.join(self.p.state_running, name))
            except OSError:
                pass
        if self.use_docker:
            try:
                out = subprocess.run(["docker", "ps", "-q", "--filter", "label=" + JOB_LABEL],
                                     capture_output=True, text=True, timeout=20)
                for cid in out.stdout.split():
                    self.log("contenedor del proyecto huérfano %s: docker stop" % cid, "WARN")
                    subprocess.run(["docker", "stop", "-t", str(int(round(self.grace))), cid],
                                   capture_output=True, timeout=self.grace + 30)
            except (OSError, subprocess.TimeoutExpired) as e:
                self.log("no se pudo revisar contenedores huérfanos: %s" % e, "ERROR")

    # ------------------------------------------------------------------ bucle
    def step(self):
        now = time.time()
        self.hb = read_json(self.p.guard_hb)
        self.hb_ok, self.hb_msg = heartbeat_status(self.hb, self.hb_max_age, now)
        stop_exists = os.path.exists(self.p.thermal_stop)

        for job in list(self.running.values()):
            rc = job.proc.poll()
            if rc is not None:
                self.finalize(job, rc)

        if self.running:
            if stop_exists:
                for job in self.running.values():
                    self.request_stop(job, "termico", "parada térmica (jobs/THERMAL_STOP)")
            elif not self.hb_ok:
                self.create_stop("%s con trabajos en curso" % self.hb_msg)
                for job in self.running.values():
                    self.request_stop(job, "termico", self.hb_msg)
            for job in self.running.values():
                if job.stop_kind is None and now > job.deadline:
                    self.request_stop(job, "timeout", "tiempo máximo de %s min superado" % job.spec["tiempo_max_min"])
            if self.shutting_down:
                for job in self.running.values():
                    self.request_stop(job, "parada_ejecutor", "parada ordenada del ejecutor")
            self.escalate()

        pending = self.scan_pending()
        self.try_launch(pending)
        self.write_state(pending)

    def write_state(self, pending):
        now = time.time()
        hb = {"ts": now, "iso": now_iso(now), "pid": os.getpid(), "host": os.uname().nodename,
              "state": "deteniendo" if self.shutting_down else "ok",
              "docker": self.use_docker, "guard_ok": self.hb_ok, "guard_msg": self.hb_msg,
              "blocked": self.launch_ready(),
              "running": {j.id: {"gpu": j.gpu, "pid": j.proc.pid, "start": now_iso(j.start),
                                 "stopping": j.stop_reason} for j in self.running.values()},
              "pending": [s["id"] for s, _ in pending]}
        try:
            atomic_write_json(self.p.runner_hb, hb)
            atomic_write_text(self.p.runner_status, status_text(self.p))
        except OSError as e:
            self.log("no se pudo escribir el estado: %s" % e, "ERROR")

    def run(self):
        stop = {"n": 0}

        def on_signal(signum, frame):
            stop["n"] += 1
            self.shutting_down = True

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGHUP, on_signal)
        self.recover()
        while True:
            t0 = time.time()
            try:
                self.step()
            except Exception as e:
                self.log("error inesperado en el ciclo: %r" % e, "ERROR")
            if self.shutting_down and not self.running:
                break
            if self.exit_after_job and self.finished_only_job and not self.running:
                break
            while time.time() - t0 < self.tick:
                time.sleep(0.05)
        now = time.time()
        atomic_write_json(self.p.runner_hb, {"ts": now, "iso": now_iso(now), "pid": os.getpid(),
                                             "host": os.uname().nodename, "state": "detenido"})
        try:
            atomic_write_text(self.p.runner_status, status_text(self.p))
        except OSError:
            pass
        self.log("detenido")
        return 0


# =============================================================================== estado legible
def _age(hb):
    try:
        return "%.1f s" % (time.time() - float(hb.get("ts", 0)))
    except (TypeError, ValueError, AttributeError):
        return "?"


def status_text(paths):
    L = []
    L.append("Estado de paper_new (%s)" % now_iso())
    g = read_json(paths.guard_hb)
    ok, msg = heartbeat_status(g, HB_MAX_AGE_MAX)
    if isinstance(g, dict):
        temps = ", ".join("GPU%s %.0f °C" % (k, v) for k, v in sorted((g.get("temps") or {}).items()))
        L.append("Vigilante: %s (%s), estado %s, latido hace %s, PID %s, aviso %s °C, corte %s °C%s"
                 % ("VÁLIDO" if ok else "NO VÁLIDO", msg, g.get("state"), _age(g), g.get("pid"),
                    g.get("warn_c"), g.get("stop_c"), ", LECTURAS FALSAS" if g.get("fake_smi") else ""))
        L.append("Temperaturas: %s" % (temps or "sin lectura"))
    else:
        L.append("Vigilante: sin latido (no está en marcha)")
    r = read_json(paths.runner_hb)
    if isinstance(r, dict):
        L.append("Ejecutor: estado %s, latido hace %s, PID %s, docker=%s" % (r.get("state"), _age(r), r.get("pid"), r.get("docker")))
        if r.get("blocked"):
            L.append("Lanzamientos bloqueados: %s" % r.get("blocked"))
    else:
        L.append("Ejecutor: sin latido")
    L.append("THERMAL_STOP: %s" % ("PRESENTE" if os.path.exists(paths.thermal_stop) else "no"))
    if os.path.exists(paths.thermal_stop):
        try:
            with open(paths.thermal_stop, encoding="utf-8") as f:
                L += ["    " + x for x in f.read().strip().splitlines()]
        except OSError:
            pass
    L.append("THERMAL_WARN: %s" % ("PRESENTE" if os.path.exists(paths.warn) else "no"))
    for name, d in (("pending", paths.pending), ("running", paths.running), ("done", paths.done), ("failed", paths.failed)):
        try:
            items = sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and not f.startswith("."))
        except OSError:
            items = []
        shown = ", ".join(items[-8:])
        L.append("Cola %-8s %3d  %s" % (name + ":", len(items), shown + (" ..." if len(items) > 8 else "")))
    try:
        with open(paths.thermal_log, encoding="utf-8") as f:
            ev = f.read().strip().splitlines()[-8:]
        L.append("Últimos eventos térmicos:")
        L += ["    " + x for x in ev]
    except OSError:
        L.append("Últimos eventos térmicos: ninguno")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ejecutor de la cola de trabajos de paper_new")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--no-docker", action="store_true", help="ejecuta los comandos como subprocesos (pruebas)")
    ap.add_argument("--tick", type=float, default=2.0, help="segundos entre comprobaciones (máximo 10)")
    ap.add_argument("--hb-max-age", type=float, default=HB_MAX_AGE_MAX, help="antigüedad máxima del latido (máximo 30)")
    ap.add_argument("--grace", type=float, default=GRACE_DEFAULT)
    ap.add_argument("--image", default=os.environ.get("PAPER_NEW_IMAGE", DEFAULT_IMAGE))
    ap.add_argument("--host-project", default=None,
                    help="ruta del proyecto en el anfitrión (por defecto la raíz real; en el servidor %s)" % DEFAULT_HOST_PROJECT)
    ap.add_argument("--data-dir", default=os.environ.get("PAPER_NEW_DATA_DIR", DEFAULT_DATA_DIR))
    ap.add_argument("--allow", default=",".join(DEFAULT_ALLOW), help="ejecutables permitidos, separados por comas")
    ap.add_argument("--allow-bash-dir", default=DEFAULT_ALLOW_BASH_DIR)
    ap.add_argument("--only-job", default=None, help="solo procesa este id (prueba del servidor)")
    ap.add_argument("--exit-after-job", action="store_true")
    ap.add_argument("--status", action="store_true", help="muestra el estado y sale")
    ap.add_argument("--check-guard", action="store_true", help="código 0 si el latido del vigilante es válido")
    a = ap.parse_args(argv)
    paths = Paths(a.root)
    if a.status:
        sys.stdout.write(status_text(paths))
        return 0
    if a.check_guard:
        ok, msg = heartbeat_status(read_json(paths.guard_hb), HB_MAX_AGE_MAX)
        print(msg)
        return 0 if ok else 1
    paths.ensure()
    log = Logger(paths.runner_log, "ejecutor")
    lockf = open(paths.runner_lock, "a")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("ya hay otro ejecutor en marcha (cerrojo ocupado); salgo", "ERROR")
        return 3
    r = Runner(paths, use_docker=not a.no_docker, tick=a.tick, hb_max_age=a.hb_max_age, grace=a.grace,
               image=a.image, host_project=a.host_project, data_dir=a.data_dir,
               allow=[x.strip() for x in a.allow.split(",") if x.strip()], allow_bash_dir=a.allow_bash_dir,
               only_job=a.only_job, exit_after_job=a.exit_after_job, log=log)
    log("en marcha: PID %d, modo %s, tick %.1f s, latido máximo %.0f s, gracia %.0f s, imagen %s, proyecto %s, datos %s (solo lectura)"
        % (os.getpid(), "docker" if r.use_docker else "no-docker", r.tick, r.hb_max_age, r.grace, r.image,
           r.host_project, r.data_dir))
    return r.run()


if __name__ == "__main__":
    sys.exit(main())
