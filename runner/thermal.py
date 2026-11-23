"""Doble seguro térmico para el bucle de entrenamiento (PLAN.md, Section 9).

Se importa desde el código de entrenamiento, dentro del contenedor:

    import sys; sys.path.insert(0, "runner")      # con /workspace/paper_new como directorio de trabajo
    from thermal import ThermalSelfCheck, ThermalLimitReached, GracefulStop, EXIT_THERMAL

    stop = GracefulStop().install()                # SIGTERM del vigilante o de docker stop
    guard = ThermalSelfCheck(every=50)             # consulta la temperatura cada 50 iteraciones
    try:
        for it, batch in enumerate(loader):
            ...paso de entrenamiento...
            guard.step()                           # levanta ThermalLimitReached con >= 80 °C
            if stop.requested:
                save_checkpoint(); sys.exit(EXIT_THERMAL)
    except ThermalLimitReached as e:
        print(e); save_checkpoint(); sys.exit(EXIT_THERMAL)

Lectura: pynvml (paquete nvidia-ml-py) y, si falla, nvidia-smi. Dentro de un contenedor lanzado con
--gpus device=N solo se ve esa GPU; fuera del contenedor se toma la máxima de las GPU visibles, que es
lo más conservador. El límite nunca puede pasar de 80 °C; PAPER_NEW_STOP_C solo puede bajarlo.
Si no hay ninguna GPU ni nvidia-smi (por ejemplo en la VM de trabajo), no hay nada que vigilar y
check_gpu_temperature devuelve una lectura vacía, salvo que require_sensor sea verdadero (por defecto lo
es cuando el trabajo lo lanzó el ejecutor, es decir, cuando existe PAPER_NEW_JOB_ID).
"""
import os
import signal
import subprocess
import threading

STOP_C_MAX = 80.0
EXIT_THERMAL = 75

__all__ = ["STOP_C_MAX", "EXIT_THERMAL", "ThermalLimitReached", "ThermalReading", "check_gpu_temperature",
           "ThermalSelfCheck", "GracefulStop", "effective_limit"]


class ThermalLimitReached(RuntimeError):
    """Temperatura >= límite (o sensor perdido cuando es obligatorio). Guardar checkpoint y salir con 75."""

    def __init__(self, temp, limit, reason="", temps=None):
        self.temp, self.limit, self.reason, self.temps = temp, limit, reason, temps or {}
        msg = reason or ("GPU a %.1f °C >= límite %.1f °C" % (temp, limit))
        super().__init__(msg + "; guardar checkpoint y salir con código %d" % EXIT_THERMAL)


class ThermalReading:
    def __init__(self, temps, source, limit):
        self.temps = temps                      # {índice visible: °C}
        self.source = source                    # "pynvml", "nvidia-smi", "fichero falso" o "no disponible"
        self.limit = limit
        self.max_temp = max(temps.values()) if temps else None
        self.over_limit = self.max_temp is not None and self.max_temp >= limit
        self.available = bool(temps)

    def __repr__(self):
        return "ThermalReading(max=%s, temps=%s, fuente=%s, límite=%.1f)" % (self.max_temp, self.temps, self.source, self.limit)


def effective_limit(limit=STOP_C_MAX):
    try:
        v = float(limit)
    except (TypeError, ValueError):
        v = STOP_C_MAX
    if v != v or v > STOP_C_MAX:
        v = STOP_C_MAX
    env = os.environ.get("PAPER_NEW_STOP_C", "").strip()
    if env:
        try:
            e = float(env)
            if e == e and e < v:
                v = e
        except ValueError:
            pass
    return v


_nvml = {"mod": None, "ok": None}


def _read_pynvml(index=None):
    if _nvml["ok"] is False:
        return None
    try:
        if _nvml["mod"] is None:
            import pynvml  # noqa: F401  (paquete nvidia-ml-py)
            pynvml.nvmlInit()
            _nvml["mod"] = pynvml
        nv = _nvml["mod"]
        n = nv.nvmlDeviceGetCount()
        idxs = [index] if index is not None else list(range(n))
        temps = {}
        for i in idxs:
            h = nv.nvmlDeviceGetHandleByIndex(i)
            temps[i] = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        _nvml["ok"] = True
        return temps or None
    except Exception:
        if _nvml["ok"] is None:
            _nvml["ok"] = False      # no hay NVML en este entorno: no se reintenta
        return None


def _parse(text, index=None):
    temps = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].strip():
            continue
        try:
            i, t = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if index is None or i == index:
            temps[i] = t
    return temps or None


def _read_smi(index=None):
    fake = os.environ.get("PAPER_NEW_FAKE_SMI")
    if fake:
        try:
            with open(fake, encoding="utf-8") as f:
                return _parse(f.read(), index), "fichero falso"
        except OSError:
            return None, "fichero falso"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None, "nvidia-smi"
    if out.returncode != 0:
        return None, "nvidia-smi"
    return _parse(out.stdout, index), "nvidia-smi"


def _sensor_exists():
    if os.environ.get("PAPER_NEW_FAKE_SMI"):
        return True
    if _nvml["ok"]:
        return True
    from shutil import which
    return which("nvidia-smi") is not None or os.path.exists("/dev/nvidiactl")


def check_gpu_temperature(limit=STOP_C_MAX, index=None):
    """Lee la temperatura de la GPU visible (o de todas las visibles) y la compara con el límite.

    Devuelve un ThermalReading con .max_temp, .temps, .over_limit (max_temp >= límite) y .available.
    """
    lim = effective_limit(limit)
    if not os.environ.get("PAPER_NEW_FAKE_SMI"):
        temps = _read_pynvml(index)
        if temps:
            return ThermalReading(temps, "pynvml", lim)
    temps, src = _read_smi(index)
    return ThermalReading(temps or {}, src if temps else "no disponible", lim)


class ThermalSelfCheck:
    """Llamar a step() en cada iteración; consulta la temperatura cada `every` llamadas."""

    def __init__(self, every=50, limit=STOP_C_MAX, index=None, require_sensor=None, max_failures=3, log=print):
        self.every = max(1, int(every))
        self.limit = effective_limit(limit)
        self.index = index
        self.require_sensor = (bool(os.environ.get("PAPER_NEW_JOB_ID")) if require_sensor is None
                               else bool(require_sensor))
        self.max_failures = max(1, int(max_failures))
        self.log = log
        self.n = 0
        self.failures = 0
        self.last = None
        self.max_seen = None
        self._warned = False

    def step(self):
        self.n += 1
        if self.n % self.every:
            return None
        return self.check_now()

    def check_now(self):
        r = check_gpu_temperature(self.limit, self.index)
        self.last = r
        if not r.available:
            if not self.require_sensor and not _sensor_exists():
                if not self._warned and self.log:
                    self.log("[thermal] sin GPU ni nvidia-smi en este entorno: el doble seguro no tiene nada que vigilar")
                    self._warned = True
                return r
            self.failures += 1
            if self.log:
                self.log("[thermal] lectura de temperatura fallida %d/%d" % (self.failures, self.max_failures))
            if self.failures >= self.max_failures:
                raise ThermalLimitReached(None, self.limit, "sensor de temperatura no disponible (%d lecturas fallidas)"
                                          % self.failures)
            return r
        self.failures = 0
        self.max_seen = r.max_temp if self.max_seen is None else max(self.max_seen, r.max_temp)
        if r.over_limit:
            raise ThermalLimitReached(r.max_temp, self.limit, temps=r.temps)
        return r


class GracefulStop:
    """Manejador reutilizable de SIGTERM (y SIGINT): marca una bandera para guardar checkpoint y salir limpio.

    stop = GracefulStop().install()
    if stop.requested: guardar checkpoint; sys.exit(EXIT_THERMAL)
    """

    def __init__(self, signals=(signal.SIGTERM, signal.SIGINT), callback=None):
        self.signals = tuple(signals)
        self.callback = callback
        self.requested = False
        self.signum = None
        self._event = threading.Event()
        self._old = {}

    def _handler(self, signum, frame):
        self.requested = True
        self.signum = signum
        self._event.set()
        if self.callback:
            self.callback(signum)

    def install(self):
        for s in self.signals:
            self._old[s] = signal.signal(s, self._handler)
        return self

    def restore(self):
        for s, h in self._old.items():
            signal.signal(s, h)
        self._old.clear()

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    def __bool__(self):
        return self.requested

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.restore()
        return False
