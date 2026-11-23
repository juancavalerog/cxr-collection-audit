"""Pruebas del vigilante del anfitrión (runner/host_guard.py) con árboles /sys y /proc falsos. Nada toca el proyecto real."""
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "runner"))
import host_guard as H  # noqa: E402
from common import Paths  # noqa: E402


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(text) + "\n")


def arbol(tmp, cpu=(55000, 60000), nvme=40000, crit_cpu=100000, mem_disp=50000000, throttle=0):
    s, p = str(tmp / "sys"), str(tmp / "proc")
    h0 = os.path.join(s, "class", "hwmon", "hwmon0")
    _w(os.path.join(h0, "name"), "coretemp")
    for i, v in enumerate(cpu, 1):
        _w(os.path.join(h0, "temp%d_input" % i), v)
        _w(os.path.join(h0, "temp%d_label" % i), "Package id %d" % (i - 1))
        if crit_cpu:
            _w(os.path.join(h0, "temp%d_crit" % i), crit_cpu)
    h1 = os.path.join(s, "class", "hwmon", "hwmon1")
    _w(os.path.join(h1, "name"), "nvme")
    _w(os.path.join(h1, "temp1_input"), nvme)
    _w(os.path.join(h1, "fan1_input"), 1200)
    h2 = os.path.join(s, "class", "hwmon", "hwmon2")
    _w(os.path.join(h2, "name"), "acpitz")
    _w(os.path.join(h2, "temp1_input"), 30000)
    c0 = os.path.join(s, "devices", "system", "cpu", "cpu0")
    _w(os.path.join(c0, "topology", "physical_package_id"), 0)
    _w(os.path.join(c0, "thermal_throttle", "package_throttle_count"), throttle)
    _w(os.path.join(p, "meminfo"), "MemTotal: 100000000 kB\nMemAvailable: %d kB\nSwapTotal: 1000 kB\nSwapFree: 1000 kB" % mem_disp)
    _w(os.path.join(p, "loadavg"), "3.50 2.00 1.00 1/100 1")
    root = str(tmp / "proyecto")
    paths = Paths(root)
    paths.ensure()
    os.makedirs(paths.logs, exist_ok=True)
    return s, p, paths


def guard(s, p, paths, **kw):
    kw.setdefault("disk_free_warn_gb", 0.0)          # el disco de la máquina de pruebas no debe generar avisos
    return H.HostGuard(paths, s, p, use_smi=False, **kw)


def test_lectura_normal_no_detiene(tmp_path):
    s, p, paths = arbol(tmp_path)
    g = guard(s, p, paths)
    estado, fila, r = g.cycle()
    assert estado == "ok" and fila["cpu_max_c"] == 60.0 and fila["cpu_n_sensores"] == 2
    assert fila["disco_max_c"] == 40.0 and fila["otros_max_c"] == 30.0 and fila["vent_min_rpm"] == 1200.0
    assert abs(fila["mem_disp_pct"] - 50.0) < 1e-6 and fila["carga_1m"] == 3.5 and fila["throttle_total"] == 0
    assert not os.path.exists(paths.thermal_stop)
    assert os.path.exists(os.path.join(paths.logs, "host_monitor.csv")) and os.path.exists(g.hb)
    lineas = open(os.path.join(paths.logs, "host_monitor.csv")).read().splitlines()
    assert lineas[0] == H.CSV_HEADER and len(lineas[1].split(",")) == len(H.CSV_HEADER.split(","))


def test_corte_de_cpu_exige_lecturas_seguidas(tmp_path):
    s, p, paths = arbol(tmp_path, cpu=(86000, 60000))
    g = guard(s, p, paths, consecutivas=3)
    g.cycle(); g.cycle()
    assert not os.path.exists(paths.thermal_stop)
    estado, _, _ = g.cycle()
    assert os.path.exists(paths.thermal_stop) and estado == "parada"
    assert "host_guard" in open(paths.thermal_stop).read()
    assert "PARADA" in open(paths.thermal_log).read()


def test_un_pico_aislado_no_detiene(tmp_path):
    s, p, paths = arbol(tmp_path, cpu=(95000,))
    g = guard(s, p, paths, consecutivas=3)
    g.cycle(); g.cycle()
    _w(os.path.join(s, "class", "hwmon", "hwmon0", "temp1_input"), 70000)
    g.cycle()
    _w(os.path.join(s, "class", "hwmon", "hwmon0", "temp1_input"), 95000)
    g.cycle(); g.cycle()
    assert not os.path.exists(paths.thermal_stop)


def test_aviso_sin_parada_y_fin_de_aviso(tmp_path):
    s, p, paths = arbol(tmp_path, cpu=(77000,))
    g = guard(s, p, paths)
    estado, _, _ = g.cycle()
    assert estado == "aviso" and not os.path.exists(paths.thermal_stop)
    assert open(paths.thermal_log).read().count("AVISO") == 1
    g.cycle()
    assert open(paths.thermal_log).read().count("[AVISO]") == 1          # no se repite mientras dura
    _w(os.path.join(s, "class", "hwmon", "hwmon0", "temp1_input"), 60000)
    g.cycle()
    assert "FIN_AVISO" in open(paths.thermal_log).read()


def test_el_critico_del_sensor_baja_el_corte(tmp_path):
    s, p, paths = arbol(tmp_path, cpu=(82000,), crit_cpu=86000)          # corte efectivo 81 °C
    g = guard(s, p, paths, consecutivas=1)
    g.cycle()
    assert os.path.exists(paths.thermal_stop)


def test_corte_de_disco(tmp_path):
    s, p, paths = arbol(tmp_path, nvme=76000)
    g = guard(s, p, paths, consecutivas=2)
    g.cycle()
    assert not os.path.exists(paths.thermal_stop)
    g.cycle()
    assert os.path.exists(paths.thermal_stop)


def test_sin_sensores_no_falla_ni_detiene(tmp_path):
    paths = Paths(str(tmp_path / "proyecto")); paths.ensure(); os.makedirs(paths.logs, exist_ok=True)
    g = guard(str(tmp_path / "nada"), str(tmp_path / "nada"), paths)
    estado, fila, _ = g.cycle()
    assert estado == "ok" and fila["cpu_max_c"] is None and not os.path.exists(paths.thermal_stop)


def test_memoria_y_estrangulamiento_solo_avisan(tmp_path):
    s, p, paths = arbol(tmp_path, mem_disp=1000000)
    g = guard(s, p, paths)
    g.cycle()
    _w(os.path.join(s, "devices", "system", "cpu", "cpu0", "thermal_throttle", "package_throttle_count"), 7)
    g.cycle()
    txt = open(paths.thermal_log).read()
    assert "memoria disponible" in txt and "estrangulamiento" in txt and not os.path.exists(paths.thermal_stop)


def test_zona_termica_como_respaldo_y_check_below(tmp_path, capsys):
    s = str(tmp_path / "sys")
    _w(os.path.join(s, "class", "thermal", "thermal_zone0", "type"), "x86_pkg_temp")
    _w(os.path.join(s, "class", "thermal", "thermal_zone0", "temp"), 66000)
    assert H.main(["--root", str(tmp_path / "proyecto"), "--sysfs-root", s, "--check-below", "70"]) == 0
    assert H.main(["--root", str(tmp_path / "proyecto"), "--sysfs-root", s, "--check-below", "65"]) == 3
    assert H.main(["--root", str(tmp_path / "proyecto"), "--sysfs-root", str(tmp_path / "nada"), "--check-below", "65"]) == 0


def test_no_pisa_una_parada_existente(tmp_path):
    s, p, paths = arbol(tmp_path, cpu=(99000,))
    with open(paths.thermal_stop, "w") as f:
        f.write("creado por gpu_guard.py\n")
    g = guard(s, p, paths, consecutivas=1)
    estado, _, _ = g.cycle()
    assert estado == "parada" and open(paths.thermal_stop).read() == "creado por gpu_guard.py\n"


def _smi(mem, red="0x0000000000000004"):
    txt, grave = H.motivos_reloj(red)
    return lambda: [{"index": "0", "mem_temp": mem, "vent_pct": None, "reduccion": txt, "reduccion_grave": grave}]


def test_limite_de_potencia_no_avisa_y_reduccion_termica_si(tmp_path):
    assert H.motivos_reloj("0x0000000000000004") == ("límite de potencia por software", False)
    assert H.motivos_reloj("0x0000000000000060")[1] is True and H.motivos_reloj("[N/A]") == ("", False)
    s, p, paths = arbol(tmp_path)
    g = H.HostGuard(paths, s, p, disk_free_warn_gb=0.0, smi_reader=_smi(77.0))
    estado, fila, _ = g.cycle()
    assert estado == "ok" and fila["gpu_mem_temp_max_c"] == 77.0 and "potencia" in fila["gpu_reduccion_reloj"]
    assert not os.path.exists(paths.thermal_log)
    g.smi_reader = _smi(77.0, "0x0000000000000040")
    g.cycle()
    assert "por temperatura o por hardware" in open(paths.thermal_log).read() and not os.path.exists(paths.thermal_stop)


def test_corte_por_memoria_de_gpu(tmp_path):
    s, p, paths = arbol(tmp_path)
    g = H.HostGuard(paths, s, p, disk_free_warn_gb=0.0, consecutivas=2, smi_reader=_smi(86.0))
    estado, _, _ = g.cycle()
    assert estado == "aviso" and not os.path.exists(paths.thermal_stop)
    g.smi_reader = _smi(91.0)
    g.cycle()
    assert not os.path.exists(paths.thermal_stop)
    g.cycle()
    assert os.path.exists(paths.thermal_stop) and "memoria" in open(paths.thermal_stop).read()


def test_config_baja_el_corte_pero_no_lo_sube(tmp_path):
    import json
    s, p, paths = arbol(tmp_path, cpu=(72000,))
    g = H.HostGuard(paths, s, p, disk_free_warn_gb=0.0, consecutivas=1, use_smi=False)
    g.cycle()
    assert not os.path.exists(paths.thermal_stop) and g.lim["cpu"] == (75.0, 85.0)
    os.makedirs(os.path.dirname(g.config), exist_ok=True)
    with open(g.config, "w") as f:
        json.dump({"cpu_stop": 95, "gpu_mem_stop": 99, "disk_stop": 500}, f)      # 500 es inválido y se ignora
    g.cycle()
    assert g.lim["cpu"][1] == 90.0 and g.lim["gpu_mem"][1] == 90.0 and not os.path.exists(paths.thermal_stop)
    with open(g.config, "w") as f:
        json.dump({"cpu_warn": 60, "cpu_stop": 70}, f)
    g.cycle()
    assert g.lim["cpu"] == (60.0, 70.0) and os.path.exists(paths.thermal_stop)
