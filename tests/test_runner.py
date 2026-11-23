"""Pruebas del ejecutor y del vigilante térmico (PLAN.md, Section 9).

Se ejecutan en una VM, sin GPU ni Docker: lecturas falsas (--fake-smi) y modo --no-docker.
Cada prueba crea su propio proyecto de juguete en un directorio temporal FUERA de las carpetas
montadas (PAPER_NEW_TEST_TMP o $HOME/tmp_runner_test), copia allí runner/ y tests/dummy_job.py,
y arranca procesos reales de gpu_guard.py y job_runner.py.

    python3 -m pytest -v tests/test_runner.py
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(HERE)
TMP_BASE = os.environ.get("PAPER_NEW_TEST_TMP", os.path.join(os.path.expanduser("~"), "tmp_runner_test"))
PY = sys.executable

GUARD_INTERVAL = 1.0
GRACE = 3.0
HB_MAX_AGE = 5.0
TICK = 0.5


def wait_until(cond, timeout, step=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return cond()


class Sandbox:
    def __init__(self, name):
        os.makedirs(TMP_BASE, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix=name + "_", dir=TMP_BASE)
        shutil.copytree(os.path.join(SRC_ROOT, "runner"), os.path.join(self.root, "runner"),
                        ignore=shutil.ignore_patterns("state", "__pycache__"))
        os.makedirs(os.path.join(self.root, "tests"))
        shutil.copy(os.path.join(SRC_ROOT, "tests", "dummy_job.py"), os.path.join(self.root, "tests"))
        for d in ("jobs/pending", "jobs/running", "jobs/done", "jobs/failed", "logs/jobs", "runner/state"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.smi = os.path.join(self.root, "fake_smi.txt")
        self.set_temps(60, 60)
        self.guard = None
        self.runner = None
        self.env = dict(os.environ)
        self.env.pop("PAPER_NEW_STOP_C", None)
        self.env.pop("PAPER_NEW_WARN_C", None)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    # -------------------------------------------------------------- utilidades
    def path(self, *p):
        return os.path.join(self.root, *p)

    def set_temps(self, t0, t1=60):
        self.write_smi("0, %s, 50, 1000, 81920, 200\n1, %s, 10, 500, 81920, 60\n" % (t0, t1))

    def write_smi(self, text):
        tmp = self.smi + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, self.smi)

    def start_guard(self, extra=()):
        self.guard_out = open(self.path("logs", "guard_stdout.txt"), "a")
        self.guard = subprocess.Popen([PY, self.path("runner", "gpu_guard.py"), "--root", self.root,
                                       "--fake-smi", self.smi, "--no-docker", "--interval", str(GUARD_INTERVAL),
                                       "--grace", str(GRACE)] + list(extra),
                                      stdout=self.guard_out, stderr=subprocess.STDOUT, env=self.env)
        assert wait_until(lambda: self.hb("guard_heartbeat.json"), 10), "el vigilante no escribió latido"

    def start_runner(self):
        self.runner_out = open(self.path("logs", "runner_stdout.txt"), "a")
        self.runner = subprocess.Popen([PY, self.path("runner", "job_runner.py"), "--root", self.root, "--no-docker",
                                        "--tick", str(TICK), "--hb-max-age", str(HB_MAX_AGE), "--grace", str(GRACE)],
                                       stdout=self.runner_out, stderr=subprocess.STDOUT, env=self.env)
        assert wait_until(lambda: self.hb("runner_heartbeat.json"), 10), "el ejecutor no escribió latido"

    def hb(self, name):
        try:
            with open(self.path("runner", "state", name)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def submit(self, jid, cmd, gpu=0, prio=0, tmax=10):
        out = subprocess.run([PY, self.path("runner", "submit_job.py"), "--root", self.root, "--id", jid,
                              "--gpu", str(gpu), "--prioridad", str(prio), "--tiempo-max-min", str(tmax),
                              "--descripcion", "prueba " + jid, "--"] + cmd,
                             capture_output=True, text=True, env=self.env)
        assert out.returncode == 0, out.stderr
        return out

    def dummy(self, jid, seconds=60, extra=()):
        return ["python", "tests/dummy_job.py", "--seconds", str(seconds), "--pid-file", "logs/%s.pid" % jid] + list(extra)

    def where(self, jid):
        for d in ("pending", "running", "done", "failed"):
            if os.path.exists(self.path("jobs", d, jid + ".json")):
                return d
        return None

    def job_log(self, jid):
        try:
            with open(self.path("logs", "jobs", jid + ".log")) as f:
                return f.read()
        except OSError:
            return ""

    def summary(self, jid):
        try:
            with open(self.path("logs", "jobs", jid + ".json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def spec(self, jid, folder):
        with open(self.path("jobs", folder, jid + ".json")) as f:
            return json.load(f)

    def events(self):
        try:
            with open(self.path("logs", "thermal_events.log")) as f:
                return f.read()
        except OSError:
            return ""

    def pid_of(self, jid):
        try:
            with open(self.path("logs", jid + ".pid")) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def close(self, keep_logs_as=None):
        for p in (self.runner, self.guard):
            if p and p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:
                    p.wait(15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        # Ningún proceso de prueba debe quedar vivo
        for f in os.listdir(self.path("logs")):
            if f.endswith(".pid"):
                try:
                    pid = int(open(self.path("logs", f)).read().strip())
                    os.killpg(pid, signal.SIGKILL)
                except (OSError, ValueError):
                    pass
        shutil.rmtree(self.root, ignore_errors=True)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(self._testMethodName)

    def tearDown(self):
        sb = self.sb
        print("\n--- registro térmico de %s ---\n%s" % (self._testMethodName, sb.events().strip()))
        try:
            print("--- runner.log ---\n" + open(sb.path("logs", "runner.log")).read().strip())
        except OSError:
            pass
        sb.close()

    def start(self):
        self.sb.start_guard()
        self.sb.start_runner()

    # ------------------------------------------------------------------ 1
    def test_1_normal_job_reaches_done(self):
        sb = self.sb
        self.start()
        sb.submit("ok1", sb.dummy("ok1", seconds=3))
        self.assertTrue(wait_until(lambda: sb.where("ok1") == "done", 25), "no llegó a done: %s" % sb.where("ok1"))
        s = sb.summary("ok1")
        self.assertEqual(s["codigo_salida"], 0)
        self.assertEqual(s["gpu"], 0)
        self.assertEqual(s["temp_max_c"], 60.0)
        self.assertEqual(s["temp_media_c"], 60.0)
        self.assertGreater(s["n_lecturas_temp"], 0)
        self.assertIn("progreso", sb.job_log("ok1"))
        self.assertIn("terminado sin interrupción", sb.job_log("ok1"))
        self.assertFalse(os.path.exists(sb.path("jobs", "THERMAL_STOP")))
        print("PRUEBA 1 SUPERADA: trabajo a 60 °C terminó en done; resumen %s" % json.dumps(
            {k: s[k] for k in ("inicio", "fin", "codigo_salida", "gpu", "temp_max_c", "temp_media_c")}, ensure_ascii=False))

    # ------------------------------------------------------------------ 2
    def test_2_warning_blocks_new_jobs_only(self):
        sb = self.sb
        self.start()
        sb.submit("largo", sb.dummy("largo", seconds=40), gpu=0)
        self.assertTrue(wait_until(lambda: sb.where("largo") == "running", 15))
        sb.set_temps(76, 60)
        warn = sb.path("runner", "state", "THERMAL_WARN")
        self.assertTrue(wait_until(lambda: os.path.exists(warn), 6), "no apareció THERMAL_WARN")
        sb.submit("nuevo", sb.dummy("nuevo", seconds=3), gpu=1)
        time.sleep(4)
        self.assertEqual(sb.where("nuevo"), "pending", "se lanzó un trabajo nuevo con aviso térmico")
        self.assertEqual(sb.where("largo"), "running", "el trabajo en curso no debía detenerse con el aviso")
        self.assertNotIn("SIGTERM", sb.job_log("largo"))
        sb.set_temps(73, 60)   # entre 72 y 75: el aviso se mantiene (histéresis)
        time.sleep(3)
        self.assertTrue(os.path.exists(warn), "el aviso se retiró por encima de 72 °C")
        self.assertEqual(sb.where("nuevo"), "pending")
        sb.set_temps(70, 60)
        self.assertTrue(wait_until(lambda: not os.path.exists(warn), 6), "no se retiró THERMAL_WARN a 70 °C")
        self.assertTrue(wait_until(lambda: sb.where("nuevo") in ("running", "done"), 8), "no se lanzó tras retirar el aviso")
        self.assertTrue(wait_until(lambda: sb.where("nuevo") == "done", 15))
        self.assertEqual(sb.where("largo"), "running")
        self.assertIn("[AVISO]", sb.events())
        self.assertIn("[FIN_AVISO]", sb.events())
        self.assertFalse(os.path.exists(sb.path("jobs", "THERMAL_STOP")))
        print("PRUEBA 2 SUPERADA: a 76 °C aparece THERMAL_WARN, el trabajo en curso sigue, el nuevo espera; "
              "a 73 °C se mantiene; a 70 °C se retira y se lanza el nuevo")

    # ------------------------------------------------------------------ 3
    def test_3_stop_at_80(self):
        sb = self.sb
        self.start()
        sb.submit("entreno", sb.dummy("entreno", seconds=120, extra=["--checkpoint-file", "logs/ckpt_entreno.txt"]), gpu=0)
        self.assertTrue(wait_until(lambda: sb.where("entreno") == "running" and "progreso" in sb.job_log("entreno"), 15))
        sb.set_temps(79, 60)             # 79 °C: aviso, sin corte
        time.sleep(3)
        self.assertFalse(os.path.exists(sb.path("jobs", "THERMAL_STOP")), "cortó a 79 °C")
        self.assertEqual(sb.where("entreno"), "running")
        t_hot = time.time()
        sb.set_temps(80, 60)
        ok = wait_until(lambda: "checkpoint guardado" in sb.job_log("entreno"), 15)
        dt = time.time() - t_hot
        self.assertTrue(ok, "el trabajo no escribió 'checkpoint guardado' en 15 s")
        self.assertLess(dt, 15)
        self.assertIn("SIGTERM recibido", sb.job_log("entreno"))
        self.assertTrue(os.path.exists(sb.path("jobs", "THERMAL_STOP")))
        stop_txt = open(sb.path("jobs", "THERMAL_STOP")).read()
        self.assertIn("GPU0=80", stop_txt)
        self.assertIn("[PARADA]", sb.events())
        self.assertIn("80", sb.events())
        self.assertTrue(wait_until(lambda: sb.where("entreno") == "pending", 10), "no volvió a pending")
        spec = sb.spec("entreno", "pending")
        self.assertEqual(spec["interrumpido"], "interrumpido por temperatura")
        self.assertTrue(os.path.exists(sb.path("logs", "ckpt_entreno.txt")))
        self.assertEqual(sb.summary("entreno")["tipo_parada"], "termico")
        self.assertIn("THERMAL_STOP", sb.summary("entreno")["motivo_parada"])
        # La temperatura baja: no hay reanudación automática
        sb.set_temps(50, 50)
        sb.submit("otro", sb.dummy("otro", seconds=2), gpu=1)
        time.sleep(6)
        self.assertEqual(sb.where("entreno"), "pending")
        self.assertEqual(sb.where("otro"), "pending")
        self.assertTrue(os.path.exists(sb.path("jobs", "THERMAL_STOP")))
        self.assertEqual(sb.job_log("entreno").count("inicio del trabajo"), 1, "se relanzó solo")
        print("PRUEBA 3 SUPERADA: 79 °C no corta; con 80 °C el trabajo recibió SIGTERM y guardó checkpoint en %.1f s; "
              "THERMAL_STOP creado, evento registrado, trabajo devuelto a pending con la marca '%s'; a 50 °C no se lanzó nada"
              % (dt, spec["interrumpido"]))

    # ------------------------------------------------------------------ 4
    def test_4_sigkill_after_grace(self):
        sb = self.sb
        self.start()
        sb.submit("terco", sb.dummy("terco", seconds=120, extra=["--ignore-sigterm"]), gpu=1)
        self.assertTrue(wait_until(lambda: sb.where("terco") == "running" and sb.pid_of("terco"), 15))
        pid = sb.pid_of("terco")
        t_hot = time.time()
        sb.set_temps(60, 85)
        self.assertTrue(wait_until(lambda: "IGNORADO" in sb.job_log("terco"), 8), "no recibió SIGTERM")
        dead = wait_until(lambda: not os.path.exists("/proc/%d" % pid) or
                          open("/proc/%d/stat" % pid).read().split(") ")[1][0] == "Z", GRACE + 8)
        dt = time.time() - t_hot
        self.assertTrue(dead, "el proceso que ignora SIGTERM sigue vivo")
        self.assertGreaterEqual(dt, GRACE - 0.5, "murió antes del periodo de gracia")
        self.assertTrue(wait_until(lambda: sb.summary("terco") is not None, 8))
        s = sb.summary("terco")
        self.assertEqual(s["codigo_salida"], -9)
        self.assertEqual(s["senal"], 9)
        self.assertNotIn("checkpoint guardado", sb.job_log("terco"))
        self.assertIn("SIGKILL", sb.events() + open(sb.path("logs", "runner.log")).read())
        print("PRUEBA 4 SUPERADA: el trabajo que ignora SIGTERM murió por SIGKILL (código %d) %.1f s después del corte "
              "(gracia de prueba %.0f s)" % (s["codigo_salida"], dt, GRACE))

    # ------------------------------------------------------------------ 5
    def test_5_guard_killed(self):
        sb = self.sb
        self.start()
        sb.submit("huerfano", sb.dummy("huerfano", seconds=120), gpu=0)
        self.assertTrue(wait_until(lambda: sb.where("huerfano") == "running" and "progreso" in sb.job_log("huerfano"), 15))
        t_kill = time.time()
        sb.guard.send_signal(signal.SIGKILL)   # muerte súbita: no puede escribir nada al salir
        sb.guard.wait()
        self.assertTrue(wait_until(lambda: "checkpoint guardado" in sb.job_log("huerfano"), HB_MAX_AGE + 10),
                        "el ejecutor no detuvo el trabajo al caducar el latido")
        dt = time.time() - t_kill
        self.assertTrue(os.path.exists(sb.path("jobs", "THERMAL_STOP")))
        self.assertIn("vigilante", open(sb.path("jobs", "THERMAL_STOP")).read())
        self.assertTrue(wait_until(lambda: sb.where("huerfano") == "pending", 8))
        sb.submit("sin_vigilante", sb.dummy("sin_vigilante", seconds=2), gpu=1)
        time.sleep(4)
        self.assertEqual(sb.where("sin_vigilante"), "pending", "se lanzó un trabajo sin vigilante")
        print("PRUEBA 5 SUPERADA: vigilante matado con SIGKILL; el ejecutor detuvo el trabajo en %.1f s "
              "(checkpoint guardado), creó THERMAL_STOP y no lanzó trabajos nuevos" % dt)

    def test_5c_guard_frozen_heartbeat_expires(self):
        """Vigilante congelado con SIGSTOP: el proceso existe pero el latido envejece; corta al superar la antigüedad."""
        sb = self.sb
        self.start()
        sb.submit("congelado", sb.dummy("congelado", seconds=120), gpu=0)
        self.assertTrue(wait_until(lambda: sb.where("congelado") == "running" and "progreso" in sb.job_log("congelado"), 15))
        t_stop = time.time()
        sb.guard.send_signal(signal.SIGSTOP)
        try:
            time.sleep(HB_MAX_AGE - 2)
            self.assertNotIn("checkpoint guardado", sb.job_log("congelado"), "cortó antes de caducar el latido")
            self.assertTrue(wait_until(lambda: "checkpoint guardado" in sb.job_log("congelado"), 10))
            dt = time.time() - t_stop
            self.assertGreaterEqual(dt, HB_MAX_AGE - 1.5)
            self.assertIn("caducado", open(sb.path("jobs", "THERMAL_STOP")).read())
            self.assertTrue(wait_until(lambda: sb.where("congelado") == "pending", 8))
        finally:
            sb.guard.send_signal(signal.SIGCONT)
        print("PRUEBA 5c SUPERADA: con el vigilante congelado el latido caducó y el ejecutor detuvo el trabajo a los %.1f s "
              "(antigüedad máxima de prueba %.0f s); THERMAL_STOP con motivo de latido caducado" % (dt, HB_MAX_AGE))

    def test_5b_no_guard_no_launch(self):
        """Sin vigilante desde el principio no se lanza nada (y no hay THERMAL_STOP porque no había trabajos)."""
        sb = self.sb
        sb.start_runner()
        sb.submit("espera", sb.dummy("espera", seconds=2), gpu=0)
        time.sleep(4)
        self.assertEqual(sb.where("espera"), "pending")
        sb.start_guard()
        self.assertTrue(wait_until(lambda: sb.where("espera") == "done", 15))
        print("PRUEBA 5b SUPERADA: sin latido no se lanza; con vigilante en marcha se lanza")

    # ------------------------------------------------------------------ 6
    def test_6_sensor_failure(self):
        sb = self.sb
        self.start()
        sb.submit("ciego", sb.dummy("ciego", seconds=120), gpu=0)
        self.assertTrue(wait_until(lambda: sb.where("ciego") == "running" and "progreso" in sb.job_log("ciego"), 15))
        sb.write_smi("No devices were found\n")
        self.assertTrue(wait_until(lambda: os.path.exists(sb.path("jobs", "THERMAL_STOP")), 8))
        txt = open(sb.path("jobs", "THERMAL_STOP")).read()
        self.assertIn("sensor no disponible", txt)
        self.assertTrue(wait_until(lambda: "checkpoint guardado" in sb.job_log("ciego"), 8))
        self.assertTrue(wait_until(lambda: sb.where("ciego") == "pending", 8))
        self.assertIn("sensor no disponible", sb.events())
        print("PRUEBA 6 SUPERADA: 3 lecturas no interpretables seguidas -> THERMAL_STOP con motivo 'sensor no disponible' "
              "y trabajo detenido con checkpoint")

    def test_6b_two_failures_do_not_stop(self):
        """Prueba determinista del contador: 2 fallos no paran; 1 lectura buena reinicia; 3 seguidos paran."""
        sys.path.insert(0, self.sb.path("runner"))
        try:
            import importlib
            import common
            import gpu_guard
            importlib.reload(common)
            importlib.reload(gpu_guard)
            p = common.Paths(self.sb.root)
            p.ensure()
            g = gpu_guard.Guard(p, fake_smi=self.sb.smi, use_docker=False, log=lambda *a, **k: None)
            for content, expect_stop in (("basura", False), ("0, [N/A]\n", False), ("0, 60\n1, 61\n", False),
                                         ("x", False), ("0, abc\n", False), ("", True)):
                self.sb.write_smi(content)
                g.cycle()
                self.assertEqual(os.path.exists(p.thermal_stop), expect_stop, "contenido %r" % content)
            self.assertIn("sensor no disponible", open(p.thermal_stop).read())
        finally:
            sys.path.remove(self.sb.path("runner"))
        print("PRUEBA 6b SUPERADA: 2 fallos no paran, una lectura válida reinicia el contador, 3 seguidos paran")

    # ------------------------------------------------------------------ 7
    def test_7_invalid_jobs_rejected(self):
        sb = self.sb
        self.start()
        pend = sb.path("jobs", "pending")
        with open(os.path.join(pend, "roto.json"), "w") as f:
            f.write('{"id": "roto", "comando": ["python", ')
        bad = {
            "prohibido": {"id": "prohibido", "comando": ["rm", "-rf", "/"], "gpu": 0, "prioridad": 0,
                          "tiempo_max_min": 5, "descripcion": "x"},
            "cadena": {"id": "cadena", "comando": "python x.py", "gpu": 0, "prioridad": 0,
                       "tiempo_max_min": 5, "descripcion": "x"},
            "bashfuera": {"id": "bashfuera", "comando": ["bash", "../evil.sh"], "gpu": 0, "prioridad": 0,
                          "tiempo_max_min": 5, "descripcion": "x"},
            "sinvalor": {"id": "sinvalor", "comando": ["python", "a.py"], "gpu": 3, "prioridad": 0, "descripcion": "x"},
            "nombre_distinto": {"id": "otro_id", "comando": ["python", "a.py"], "gpu": 0, "prioridad": 0,
                                "tiempo_max_min": 5, "descripcion": "x"},
        }
        for name, obj in bad.items():
            with open(os.path.join(pend, name + ".json"), "w") as f:
                json.dump(obj, f)
        names = ["roto"] + list(bad)
        self.assertTrue(wait_until(lambda: all(os.path.exists(sb.path("jobs", "failed", n + ".json")) for n in names), 8))
        motivos = {n: open(sb.path("jobs", "failed", n + ".motivo.txt")).read() for n in names}
        self.assertIn("JSON mal formado", motivos["roto"])
        self.assertIn("ejecutable no permitido", motivos["prohibido"])
        self.assertIn("lista", motivos["cadena"])
        self.assertIn("scripts/", motivos["bashfuera"])
        self.assertIn("tiempo_max_min", motivos["sinvalor"])
        self.assertIn("gpu", motivos["sinvalor"])
        self.assertIn("no coincide", motivos["nombre_distinto"])
        self.assertEqual(os.listdir(pend), [])
        # submit_job.py también se niega
        out = subprocess.run([PY, sb.path("runner", "submit_job.py"), "--root", sb.root, "--id", "x1", "--",
                              "curl", "http://example.com"], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("ejecutable no permitido", out.stderr)
        self.assertFalse(os.path.exists(sb.path("jobs", "pending", "x1.json")))
        # Un trabajo válido con bash scripts/... sí se acepta en la validación
        sys.path.insert(0, sb.path("runner"))
        try:
            import job_runner
            self.assertEqual(job_runner.validate_command(["bash", "scripts/train/run.sh"]), [])
            self.assertNotEqual(job_runner.validate_command(["bash", "scripts/../../x.sh"]), [])
        finally:
            sys.path.remove(sb.path("runner"))
        print("PRUEBA 7 SUPERADA: JSON roto, ejecutable prohibido, comando como cadena, bash fuera de scripts/, "
              "campos inválidos e id distinto van a failed con motivo; submit_job.py rechaza comandos no permitidos")
        for n in names:
            print("    %s: %s" % (n, motivos[n].splitlines()[-1]))

    # ------------------------------------------------------------------ 8
    def test_8_clear_thermal_stop(self):
        sb = self.sb
        stop = sb.path("jobs", "THERMAL_STOP")
        with open(stop, "w") as f:
            f.write("motivo: prueba\n")
        env = dict(sb.env, PAPER_NEW_FAKE_SMI=sb.smi)
        script = sb.path("runner", "clear_thermal_stop.sh")
        for t0, t1 in ((65, 40), (40, 70), (80, 80)):
            sb.set_temps(t0, t1)
            out = subprocess.run(["bash", script], input="SI\n", capture_output=True, text=True, env=env)
            self.assertNotEqual(out.returncode, 0, "se retiró con GPU a %s/%s °C" % (t0, t1))
            self.assertTrue(os.path.exists(stop))
            self.assertIn("NO SE RETIRA", out.stderr)
        sb.write_smi("basura\n")
        out = subprocess.run(["bash", script], input="SI\n", capture_output=True, text=True, env=env)
        self.assertNotEqual(out.returncode, 0)
        self.assertTrue(os.path.exists(stop))
        sb.set_temps(64, 50)
        out = subprocess.run(["bash", script], input="no\n", capture_output=True, text=True, env=env)
        self.assertNotEqual(out.returncode, 0)
        self.assertTrue(os.path.exists(stop), "se retiró sin confirmación")
        out = subprocess.run(["bash", script], input="SI\n", capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertFalse(os.path.exists(stop))
        self.assertEqual(len(os.listdir(sb.path("logs", "thermal_stop_archive"))), 1)
        self.assertIn("[LIBERADO]", sb.events())
        print("PRUEBA 8 SUPERADA: clear_thermal_stop se niega con 65, 70 y 80 °C, con lectura ilegible y sin "
              "confirmación; con 64/50 °C y SI lo retira y lo archiva")

    # ------------------------------------------------------------------ extra: límites y doble seguro
    def test_9_thresholds_and_thermal_module(self):
        sb = self.sb
        sys.path.insert(0, sb.path("runner"))
        try:
            import importlib
            import common
            import thermal
            importlib.reload(common)
            importlib.reload(thermal)
            os.environ.pop("PAPER_NEW_STOP_C", None)
            self.assertEqual(common.effective_stop_c(95)[0], 80.0)
            self.assertEqual(common.effective_stop_c(None)[0], 80.0)
            self.assertEqual(common.effective_stop_c(70)[0], 70.0)
            os.environ["PAPER_NEW_STOP_C"] = "90"
            self.assertEqual(common.effective_stop_c(None)[0], 80.0)
            self.assertEqual(thermal.effective_limit(80), 80.0)
            os.environ["PAPER_NEW_STOP_C"] = "41"
            self.assertEqual(common.effective_stop_c(None)[0], 41.0)
            self.assertEqual(thermal.effective_limit(80), 41.0)
            os.environ.pop("PAPER_NEW_STOP_C")
            self.assertEqual(thermal.effective_limit(120), 80.0)
            os.environ["PAPER_NEW_FAKE_SMI"] = sb.smi
            try:
                sb.write_smi("0, 79\n")
                chk = thermal.ThermalSelfCheck(every=5, limit=90, log=None)
                self.assertEqual(chk.limit, 80.0)
                for _ in range(10):
                    chk.step()
                self.assertEqual(chk.last.max_temp, 79.0)
                sb.write_smi("0, 80\n")
                for _ in range(4):
                    self.assertIsNone(chk.step())
                with self.assertRaises(thermal.ThermalLimitReached):
                    chk.step()
                sb.write_smi("ilegible\n")
                chk2 = thermal.ThermalSelfCheck(every=1, require_sensor=True, log=None)
                chk2.step()
                chk2.step()
                with self.assertRaises(thermal.ThermalLimitReached):
                    chk2.step()
            finally:
                os.environ.pop("PAPER_NEW_FAKE_SMI", None)
            # GracefulStop
            gs = thermal.GracefulStop(signals=(signal.SIGUSR1,)).install()
            os.kill(os.getpid(), signal.SIGUSR1)
            self.assertTrue(gs.wait(2))
            self.assertTrue(gs.requested)
            gs.restore()
            self.assertEqual(thermal.EXIT_THERMAL, 75)
        finally:
            sys.path.remove(sb.path("runner"))
        print("PRUEBA 9 (extra) SUPERADA: el corte nunca sube de 80 °C (95 y PAPER_NEW_STOP_C=90 -> 80), sí baja; "
              "thermal.ThermalSelfCheck no corta a 79 °C, corta a 80 °C cada N iteraciones, corta tras 3 lecturas "
              "fallidas si el sensor es obligatorio; GracefulStop marca la bandera")

    def test_10_exit75_creates_stop(self):
        """Un trabajo que sale con 75 (doble seguro dentro del entrenamiento) bloquea la cola y vuelve a pending."""
        sb = self.sb
        self.start()
        code = "import sys; print('ThermalLimitReached simulado'); sys.exit(75)"
        sb.submit("autocorte", ["python", "-c", code], gpu=0)
        self.assertTrue(wait_until(lambda: sb.where("autocorte") == "pending" and sb.summary("autocorte"), 15))
        self.assertTrue(os.path.exists(sb.path("jobs", "THERMAL_STOP")))
        self.assertEqual(sb.spec("autocorte", "pending")["interrumpido"], "interrumpido por temperatura")
        self.assertIn("doble seguro", sb.summary("autocorte")["motivo_parada"])
        print("PRUEBA 10 (extra) SUPERADA: salida con código 75 crea THERMAL_STOP y devuelve el trabajo a pending")

    def test_11_one_job_per_gpu_and_timeout(self):
        sb = self.sb
        self.start()
        for i in range(3):
            sb.submit("par%d" % i, sb.dummy("par%d" % i, seconds=6), gpu="any", prio=i)
        self.assertTrue(wait_until(lambda: sum(sb.where("par%d" % i) == "running" for i in range(3)) == 2, 10))
        time.sleep(1)
        running = [i for i in range(3) if sb.where("par%d" % i) == "running"]
        self.assertEqual(sorted(running), [1, 2], "la prioridad más alta debía ir primero")
        gpus = {sb.hb("runner_heartbeat.json")["running"]["par%d" % i]["gpu"] for i in running}
        self.assertEqual(gpus, {0, 1})
        self.assertTrue(wait_until(lambda: all(sb.where("par%d" % i) == "done" for i in range(3)), 30))
        sb.submit("lento", sb.dummy("lento", seconds=120), gpu=0, tmax=0.05)
        self.assertTrue(wait_until(lambda: sb.where("lento") == "failed", 15))
        self.assertIn("tiempo máximo", sb.summary("lento")["motivo_parada"])
        print("PRUEBA 11 (extra) SUPERADA: dos trabajos en paralelo, uno por GPU, por orden de prioridad; "
              "el tiempo máximo detiene y manda a failed")

    def test_12_docker_command_line(self):
        """Comprobación estática de la orden docker run (Docker no existe en la VM: no se ejecuta)."""
        sb = self.sb
        sys.path.insert(0, sb.path("runner"))
        try:
            import importlib
            import common
            import job_runner
            importlib.reload(common)
            importlib.reload(job_runner)
            r = job_runner.Runner(common.Paths(sb.root), use_docker=True,
                                  host_project="/home/administrador/Proyectos/paper_new",
                                  data_dir="/home/administrador/notebooks/covid", log=lambda *a, **k: None)
            spec = {"id": "j1", "comando": ["python", "src/train/train.py", "--x", "1"], "gpu": 1, "prioridad": 0,
                    "tiempo_max_min": 5, "descripcion": ""}
            cmd, env, cwd, name = r.build_command(spec, 1)
        finally:
            sys.path.remove(sb.path("runner"))
        line = " ".join(cmd)
        for frag in ("docker run --rm", "--label paper_new.job=1", "--name paper_new_job_j1", "--gpus device=1",
                     "--shm-size 16g", "-e NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                     "--user %d:%d" % (os.getuid(), os.getgid()),
                     "-v /home/administrador/Proyectos/paper_new:/workspace/paper_new:rw",
                     "-v /home/administrador/notebooks/covid:/data/covid:ro",
                     "-v paper_new_torch_cache:/cache/torch", "-v paper_new_hf_cache:/cache/hf",
                     "-w /workspace/paper_new", "paper-new:cu126 python src/train/train.py --x 1"):
            self.assertIn(frag, line)
        self.assertNotIn("-p ", line)
        self.assertNotIn("--privileged", line)
        self.assertEqual(name, "paper_new_job_j1")
        print("PRUEBA 12 (extra, estática) SUPERADA: orden docker generada: " + line)

    def test_13_csv_rotation_and_heartbeat(self):
        sb = self.sb
        sys.path.insert(0, sb.path("runner"))
        try:
            import importlib
            import common
            import gpu_guard
            importlib.reload(common)
            importlib.reload(gpu_guard)
            p = common.Paths(sb.root)
            p.ensure()
            g = gpu_guard.Guard(p, fake_smi=sb.smi, use_docker=False, csv_max_mb=0.0003, log=lambda *a, **k: None)
            for _ in range(6):
                g.cycle()
            rotated = [f for f in os.listdir(p.logs) if f.startswith("gpu_monitor_") and f.endswith(".csv")]
            self.assertTrue(rotated, "no hubo rotación por tamaño")
            with open(p.gpu_csv) as f:
                head = f.readline().strip()
            self.assertEqual(head, gpu_guard.CSV_HEADER)
            hb = json.load(open(p.guard_hb))
            for k in ("ts", "temps", "state", "stop_c", "warn_c", "pid"):
                self.assertIn(k, hb)
            self.assertEqual(hb["temps"], {"0": 60.0, "1": 60.0})
            self.assertEqual(hb["state"], "ok")
            # Rotación diaria: se simula un fichero del día anterior
            g2 = gpu_guard.Guard(p, fake_smi=sb.smi, use_docker=False, log=lambda *a, **k: None)
            g2.csv_day = "19990101"
            n0 = len(rotated)
            g2.cycle()
            rotated2 = [f for f in os.listdir(p.logs) if f.startswith("gpu_monitor_") and f.endswith(".csv")]
            self.assertEqual(len(rotated2), n0 + 1, "no hubo rotación diaria")
        finally:
            sys.path.remove(sb.path("runner"))
        print("PRUEBA 13 (extra) SUPERADA: gpu_monitor.csv rota por tamaño y por cambio de día; latido con marca de tiempo, "
              "temperaturas y estado")


if __name__ == "__main__":
    unittest.main(verbosity=2)
