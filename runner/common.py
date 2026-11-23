"""Utilidades compartidas por gpu_guard.py, job_runner.py y submit_job.py.

Solo biblioteca estándar de Python 3. Se ejecuta en el servidor, fuera del contenedor.
"""
import json
import os
import signal
import tempfile
import time

# Límite duro de corte. Ninguna configuración puede subirlo: solo se puede bajar.
# Para cambiarlo hay que editar esta línea (decisión del autor, PLAN.md, Section 9).
STOP_C_MAX = 80.0
WARN_C_DEFAULT = 75.0
WARN_CLEAR_C_DEFAULT = 72.0
CLEAR_STOP_MAX_C = 65.0          # clear_thermal_stop.sh exige todas las GPU por debajo de este valor
HB_MAX_AGE_MAX = 30.0            # latido del vigilante: nunca se acepta uno más viejo que esto
GRACE_DEFAULT = 30.0             # periodo de gracia entre SIGTERM y SIGKILL
GRACE_MAX = 30.0
EXIT_THERMAL = 75                # código de salida de un entrenamiento detenido por temperatura
JOB_LABEL = "paper_new.job=1"
CONTAINER_PREFIX = "paper_new_job_"

SMI_FIELDS = ["index", "temperature.gpu", "utilization.gpu", "memory.used",
              "memory.total", "power.draw"]


class Paths:
    def __init__(self, root):
        self.root = os.path.realpath(root)
        r = self.root
        self.jobs = os.path.join(r, "jobs")
        self.pending = os.path.join(self.jobs, "pending")
        self.running = os.path.join(self.jobs, "running")
        self.done = os.path.join(self.jobs, "done")
        self.failed = os.path.join(self.jobs, "failed")
        self.thermal_stop = os.path.join(self.jobs, "THERMAL_STOP")
        self.state = os.path.join(r, "runner", "state")
        self.state_running = os.path.join(self.state, "running")
        self.warn = os.path.join(self.state, "THERMAL_WARN")
        self.guard_hb = os.path.join(self.state, "guard_heartbeat.json")
        self.runner_hb = os.path.join(self.state, "runner_heartbeat.json")
        self.guard_lock = os.path.join(self.state, "guard.lock")
        self.runner_lock = os.path.join(self.state, "runner.lock")
        self.logs = os.path.join(r, "logs")
        self.jobs_logs = os.path.join(self.logs, "jobs")
        self.gpu_csv = os.path.join(self.logs, "gpu_monitor.csv")
        self.thermal_log = os.path.join(self.logs, "thermal_events.log")
        self.guard_log = os.path.join(self.logs, "guard.log")
        self.runner_log = os.path.join(self.logs, "runner.log")
        self.runner_status = os.path.join(self.logs, "runner_status.txt")
        self.stop_archive = os.path.join(self.logs, "thermal_stop_archive")

    def ensure(self):
        for d in (self.pending, self.running, self.done, self.failed, self.state,
                  self.state_running, self.jobs_logs):
            os.makedirs(d, exist_ok=True)


def now_iso(t=None):
    t = time.time() if t is None else t
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


def atomic_write_text(path, text):
    """Escribe en un temporal de la misma carpeta y lo renombra (os.replace es atómico)."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".part", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n")


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def append_line(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()


class Logger:
    """Escribe en la salida estándar (ventana de tmux) y en un fichero de registro."""

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def __call__(self, msg, level="INFO"):
        line = "%s [%s] %s: %s" % (now_iso(), level, self.name, msg)
        print(line, flush=True)
        try:
            append_line(self.path, line)
        except OSError:
            pass


def thermal_event(paths, level, msg):
    append_line(paths.thermal_log, "%s [%s] %s" % (now_iso(), level, msg))


def effective_stop_c(requested=None):
    """Umbral de corte efectivo: el menor entre lo pedido, la variable PAPER_NEW_STOP_C y 80 °C.

    Devuelve (valor, nota). Un valor pedido por encima de 80 se ignora y se deja constancia.
    """
    notes = []
    value = STOP_C_MAX
    cands = []
    if requested is not None:
        cands.append(("argumento", requested))
    env = os.environ.get("PAPER_NEW_STOP_C", "").strip()
    if env:
        cands.append(("PAPER_NEW_STOP_C", env))
    for src, raw in cands:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            notes.append("%s=%r no es un número; se ignora" % (src, raw))
            continue
        if v != v:  # NaN
            notes.append("%s es NaN; se ignora" % src)
            continue
        if v > STOP_C_MAX:
            notes.append("%s=%.1f supera el máximo de %.0f °C; se usa %.0f" % (src, v, STOP_C_MAX, STOP_C_MAX))
            continue
        if v < value:
            value = v
            if v < STOP_C_MAX:
                notes.append("umbral de corte bajado a %.1f °C por %s" % (v, src))
    return value, "; ".join(notes)


class SmiError(Exception):
    pass


def _num(s):
    s = s.strip()
    if s in ("", "[N/A]", "N/A", "[Not Supported]", "[Unknown Error]"):
        return None
    return float(s)


def parse_smi_csv(text):
    """Interpreta la salida de nvidia-smi --query-gpu=... --format=csv,noheader,nounits.

    Cada línea: índice, temperatura[, uso, memoria usada, memoria total, potencia].
    Lanza SmiError si no hay GPU o si alguna temperatura no es interpretable.
    """
    gpus = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise SmiError("línea no interpretable: %r" % line)
        try:
            idx = int(parts[0])
        except ValueError:
            raise SmiError("índice no interpretable: %r" % line)
        try:
            temp = _num(parts[1])
        except ValueError:
            raise SmiError("temperatura no interpretable: %r" % line)
        if temp is None or not (-30.0 <= temp <= 130.0):
            raise SmiError("temperatura no válida: %r" % line)
        extra = []
        for p in parts[2:6]:
            try:
                extra.append(_num(p))
            except ValueError:
                extra.append(None)
        extra += [None] * (4 - len(extra))
        if idx in seen:
            raise SmiError("índice repetido: %d" % idx)
        seen.add(idx)
        gpus.append({"index": idx, "temp": temp, "util": extra[0], "mem_used": extra[1],
                     "mem_total": extra[2], "power": extra[3]})
    if not gpus:
        raise SmiError("nvidia-smi no devolvió ninguna GPU")
    gpus.sort(key=lambda g: g["index"])
    return gpus


def query_smi(fake_path=None, timeout=15):
    """Devuelve el texto CSV de nvidia-smi (o del fichero falso de pruebas)."""
    if fake_path:
        try:
            with open(fake_path, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            raise SmiError("no se pudo leer el fichero falso: %s" % e)
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(SMI_FIELDS), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise SmiError("nvidia-smi no encontrado")
    except subprocess.TimeoutExpired:
        raise SmiError("nvidia-smi no respondió en %d s" % timeout)
    if out.returncode != 0:
        raise SmiError("nvidia-smi terminó con código %d: %s" % (out.returncode, (out.stderr or out.stdout).strip()[:300]))
    return out.stdout


def read_gpus(fake_path=None):
    return parse_smi_csv(query_smi(fake_path))


def pid_alive(pid):
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Un zombi cuenta como muerto
    try:
        with open("/proc/%d/stat" % pid) as f:
            st = f.read()
        state = st[st.rindex(")") + 2:].split()[0]
        if state == "Z":
            return False
    except (OSError, ValueError, IndexError):
        pass
    return True


def proc_starttime(pid):
    """Campo 22 de /proc/<pid>/stat: identifica el proceso aunque el PID se reutilice."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            st = f.read()
        return int(st[st.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def proc_cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def same_process(pid, starttime):
    """Verdadero si el PID sigue vivo y es el mismo proceso que se registró."""
    if not pid_alive(pid):
        return False
    if starttime is None:
        return False
    return proc_starttime(pid) == starttime


def signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def heartbeat_status(hb, max_age, now=None):
    """Devuelve (válido, motivo) para un latido del vigilante."""
    now = time.time() if now is None else now
    max_age = min(float(max_age), HB_MAX_AGE_MAX)
    if not isinstance(hb, dict):
        return False, "sin latido del vigilante"
    if hb.get("state") == "detenido":
        return False, "el vigilante se detuvo"
    try:
        age = now - float(hb.get("ts", 0))
    except (TypeError, ValueError):
        return False, "latido ilegible"
    if age > max_age or age < -5:
        return False, "latido del vigilante caducado (%.1f s)" % age
    pid = hb.get("pid")
    if isinstance(pid, int) and hb.get("host") == os.uname().nodename and not pid_alive(pid):
        return False, "el proceso del vigilante (PID %d) ya no existe" % pid
    return True, "latido de hace %.1f s" % age


SIG_NAMES = {getattr(signal, n): n for n in ("SIGTERM", "SIGKILL", "SIGINT", "SIGHUP")}
