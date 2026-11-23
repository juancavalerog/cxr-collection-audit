#!/usr/bin/env python3
"""Trabajo ficticio para probar el ejecutor y el vigilante.

Bucle que escribe progreso; con SIGTERM escribe "checkpoint guardado" y sale con código 75.
--ignore-sigterm ignora la señal (para probar el SIGKILL tras el periodo de gracia).
--gpu-load genera carga real en la GPU con torch (solo en el servidor, dentro del contenedor).
Sin --gpu-load solo usa la biblioteca estándar.
"""
import argparse
import os
import signal
import sys
import time

EXIT_THERMAL = 75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--progress-file", default=None)
    ap.add_argument("--checkpoint-file", default=None)
    ap.add_argument("--pid-file", default=None)
    ap.add_argument("--ignore-sigterm", action="store_true")
    ap.add_argument("--gpu-load", action="store_true")
    ap.add_argument("--matrix", type=int, default=8192)
    a = ap.parse_args()

    flag = {"term": False}

    def on_term(signum, frame):
        if a.ignore_sigterm:
            print("SIGTERM recibido e IGNORADO (prueba de SIGKILL)", flush=True)
            return
        print("SIGTERM recibido", flush=True)
        flag["term"] = True

    signal.signal(signal.SIGTERM, on_term)
    if a.pid_file:
        with open(a.pid_file, "w") as f:
            f.write(str(os.getpid()))
    print("dummy_job PID %d inicio, %.0f s, job=%s, gpu=%s" % (os.getpid(), a.seconds, os.environ.get("PAPER_NEW_JOB_ID"),
                                                           os.environ.get("PAPER_NEW_GPU_INDEX")), flush=True)
    work = None
    if a.gpu_load:
        import torch
        assert torch.cuda.is_available(), "no hay GPU visible en el contenedor"
        x = torch.randn(a.matrix, a.matrix, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(a.matrix, a.matrix, device="cuda", dtype=torch.bfloat16)

        def work():
            for _ in range(20):
                torch.matmul(x, y)
            torch.cuda.synchronize()
        print("carga en %s" % torch.cuda.get_device_name(0), flush=True)
    t0 = time.time()
    i = 0
    last_print = 0.0
    while time.time() - t0 < a.seconds:
        if flag["term"]:
            if a.checkpoint_file:
                with open(a.checkpoint_file, "w") as f:
                    f.write("paso %d\n" % i)
            print("checkpoint guardado (paso %d)" % i, flush=True)
            sys.exit(EXIT_THERMAL)
        if work:
            work()
        else:
            time.sleep(a.step)
        i += 1
        if time.time() - last_print >= 1.0:
            last_print = time.time()
            msg = "progreso paso %d t=%.1f s" % (i, time.time() - t0)
            print(msg, flush=True)
            if a.progress_file:
                with open(a.progress_file, "a") as f:
                    f.write(msg + "\n")
    print("dummy_job terminado sin interrupción", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
