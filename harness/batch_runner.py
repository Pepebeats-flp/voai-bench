"""Corredor por tandas dentro de los límites de the API provider.

Ejecuta las fases de results/batch_plan.json en orden. Cada fase llama a run.py (reanudable);
si la corrida se detiene por cuota agotada, lee "resets in Xd Yh Zm" del último error,
espera hasta el reinicio (+ margen) y continúa. El plan se relee en cada vuelta: se puede
editar (modelos, repeticiones) antes de que una fase empiece o entre tandas.

Estado en results/batch_status.json (lo muestra el dashboard). Log en results/batch.log.
Uso: .venv/bin/python -I -u harness/batch_runner.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLAN = RES / "batch_plan.json"
STATUS = RES / "batch_status.json"
MARGIN_S = 6 * 60
RESET_RE = re.compile(r"resets in\s*(?:(\d+)\s*d)?\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?", re.I)


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    with (RES / "batch.log").open("a") as fh:
        fh.write(line + "\n")


def status(**kw):
    STATUS.write_text(json.dumps({"updated": time.time(), **kw}, default=str))


def reset_seconds(jsonl: Path, since: float):
    """Segundos hasta el reinicio de cuota según el último error quota_exhausted escrito tras `since`."""
    if not jsonl.exists():
        return None
    best = None
    sys.path.insert(0, str(ROOT / "harness"))
    from jsonl_io import tail_records
    for r in tail_records(jsonl, 400):
        err = str(r.get("error") or "")
        if (r.get("ts") or 0) < since:
            continue
        if "quota_exhausted" in err or "usage limit" in err.lower():
            m = RESET_RE.search(err)
            if m and any(m.groups()):
                d, h, mi = (int(x) if x else 0 for x in m.groups())
                best = d * 86400 + h * 3600 + mi * 60 - (time.time() - r["ts"])  # descontar lo ya transcurrido
    if best is not None:
        best = max(0, best)
    return best


def run_phase(ph):
    args = [sys.executable, "-I", "-u", str(ROOT / "harness/run.py"), "--name", ph.get("run", ph["name"]), "--bench", ph["bench"],
            "--models", ",".join(ph["models"]), "--archs", ph["archs"], "--langs", ph.get("langs", "en,es"),
            "--repeats", str(ph.get("repeats", 1)), "--workers", str(ph.get("workers", 24))]
    if ph.get("max_cost") is not None:
        args += ["--max-cost", str(ph["max_cost"])]
    if ph.get("max_allowance") is not None:
        args += ["--max-allowance", str(ph["max_allowance"])]
    t0 = time.time()
    with (RES / f"{ph.get('run', ph['name'])}.log").open("a") as fh:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        pending = None
        budget = False
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if "BUDGET_REACHED" in line:
                budget = True
            m = re.match(r"(\d+) episodios pendientes", line)
            if m and pending is None:
                pending = int(m.group(1))
                status(phase=ph["name"], state="running", pending_at_start=pending, started=t0)
        proc.wait()
    return pending, t0, budget


def main():
    log("corredor por tandas iniciado")
    idle_retries = 0
    while True:
        plan = json.loads(PLAN.read_text())
        pending_phases = [ph for ph in plan["phases"] if not ph.get("done")]
        if not pending_phases:
            log("todas las fases terminadas; generando experimental_log y figuras")
            for run in sorted({p.get("run", p["name"]) for p in plan["phases"] if p.get("bench") == "v1"}):
                for script, extra in (("make_experimental_log.py", ["--label", run]), ("make_figures.py", [])):
                    rc = subprocess.run([sys.executable, "-I", str(ROOT / "harness" / script), str(RES / f"{run}.jsonl"), *extra],
                                        cwd=ROOT, capture_output=True, text=True)
                    log(f"{script}: {'ok' if rc.returncode == 0 else 'error ' + rc.stderr[-300:]}")
            status(state="finished")
            return
        todo = [ph for ph in pending_phases if not ph.get("paused")]
        if not todo:  # fases en pausa (p. ej. cuota mensual): esperar a que se quite "paused" del plan
            status(state="paused", phase=pending_phases[0]["name"])
            time.sleep(300)
            continue
        ph = todo[0]
        log(f"fase {ph['name']}: lanzando run.py")
        pending, t0, budget = run_phase(ph)
        if budget:  # tope de presupuesto alcanzado: pausar la fase (se reanuda quitando "paused" o subiendo el tope)
            log(f"fase {ph['name']}: tope de presupuesto alcanzado; fase en pausa")
            plan = json.loads(PLAN.read_text())
            for p in plan["phases"]:
                if p["name"] == ph["name"]:
                    p["paused"] = True
            PLAN.write_text(json.dumps(plan, indent=1))
            continue
        if pending == 0:
            log(f"fase {ph['name']}: completa")
            plan = json.loads(PLAN.read_text())
            gate_ok = True
            if ph.get("gate"):  # control de calidad automático antes de gastar créditos
                g = ph["gate"]
                rc = subprocess.run([sys.executable, "-I", str(ROOT / "harness/smoke_gate.py"),
                                     str(RES / f"{ph.get('run', ph['name'])}.jsonl"), g["model"], g["lang"]],
                                    cwd=ROOT, capture_output=True, text=True)
                gate_ok = rc.returncode == 0
                log(f"control de calidad {'APROBADO' if gate_ok else 'RECHAZADO'}: {rc.stdout.strip()[:400]}")
            for p in plan["phases"]:
                if p["name"] == ph["name"]:
                    p["done"] = True
                if gate_ok and p["name"] in ph.get("unlock", []):
                    p.pop("paused", None)
            PLAN.write_text(json.dumps(plan, indent=1))
            if not gate_ok:
                status(state="gate_failed", phase=ph["name"])
                log("fases siguientes quedan en pausa: revisar results/smoke_gate.json")
            idle_retries = 0
            continue
        wait = reset_seconds(RES / f"{ph.get('run', ph['name'])}.jsonl", t0)
        if wait is not None:
            resume = time.time() + wait + MARGIN_S
            log(f"fase {ph['name']}: cuota agotada; reanudación a las {dt.datetime.fromtimestamp(resume):%H:%M} "
                f"({(wait + MARGIN_S) / 60:.0f} min)")
            status(phase=ph["name"], state="waiting_quota", resume_at=resume)
            time.sleep(wait + MARGIN_S)
            idle_retries = 0
        else:
            idle_retries += 1
            delay = min(1800, 120 * idle_retries)
            log(f"fase {ph['name']}: terminó con {pending} pendientes sin aviso de cuota; reintento en {delay // 60} min")
            status(phase=ph["name"], state="retry_wait", resume_at=time.time() + delay)
            time.sleep(delay)


if __name__ == "__main__":
    main()
