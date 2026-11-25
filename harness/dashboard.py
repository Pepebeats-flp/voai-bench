"""Dashboard local en vivo para VOAI-Bench.

Uso: .venv/bin/python -I harness/dashboard.py [--port 8765]   ->  http://localhost:8765
Lee results/*.jsonl, results/*.plan.json y bench/gold_v0.json en cada petición (sin estado).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def rate(lst):
    return {"n": len(lst), "ok": sum(lst), "rate": (sum(lst) / len(lst)) if lst else None}


def summary(run: str | None):
    items = yaml.safe_load((ROOT / "bench/queries_v0.yaml").read_text())["items"]
    gold_path = ROOT / "bench/gold_v0.json"
    gold = json.loads(gold_path.read_text()) if gold_path.exists() else {}
    runs = sorted([p.stem for p in RES.glob("*.jsonl")], key=lambda s: -(RES / f"{s}.jsonl").stat().st_mtime)
    out = {"now": time.time(), "gold": {"done": len(gold), "total": len(items),
                                        "missing": [i["id"] for i in items if i["id"] not in gold]},
           "runs": runs}
    bs = RES / "batch_status.json"
    if bs.exists():
        try:
            out["batch"] = json.loads(bs.read_text())
            pl = RES / "batch_plan.json"
            if pl.exists():
                out["batch"]["phases"] = [{"name": p["name"], "done": bool(p.get("done")), "bench": p["bench"],
                                           "repeats": p.get("repeats"), "n_models": len(p["models"])}
                                          for p in json.loads(pl.read_text())["phases"]]
        except Exception:
            pass
    sg = RES / "smoke_gate.json"
    if sg.exists():
        try:
            out.setdefault("batch", {})["gate"] = json.loads(sg.read_text())
        except Exception:
            pass
    run = run or (runs[0] if runs else None)
    if not run:
        return out
    out["run"] = run
    plan_p = RES / f"{run}.plan.json"
    plan = json.loads(plan_p.read_text()) if plan_p.exists() else {}
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from jsonl_io import latest_records
    allrecs = list(latest_records(RES / f"{run}.jsonl").values())
    recs = [r for r in allrecs if not str(r.get("error") or "").startswith(("infra_error", "quota_exhausted", "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailableError"))]
    out["infra_errors"] = len(allrecs) - len(recs)
    jsonl_mtime = (RES / f"{run}.jsonl").stat().st_mtime
    by = defaultdict(list)
    for r in recs:
        s = bool((r.get("score") or {}).get("success"))
        m = r["model"].split("/", 1)[-1]
        by[("ma", m, r["arch"])].append(s)
        by[("a", r["arch"])].append(s)
        by[("pa", r["protocol"], r["arch"])].append(s)
        by[("la", r["lang"], r["arch"])].append(s)
        by[("da", r["difficulty"], r["arch"])].append(s)
        by[("i", r["item"], r["arch"])].append(s)
        by[("m", m)].append(s)
    models = sorted({k[1] for k in by if k[0] == "m"})
    archs = sorted({k[1] for k in by if k[0] == "a"})
    cost = defaultdict(float)
    wall = defaultdict(list)
    for r in recs:
        m = r["model"].split("/", 1)[-1]
        cost[(m, r["arch"])] += r.get("cost_usd") or 0
        wall[(m, r["arch"])].append(r.get("wall_s") or 0)
    errs = Counter()
    for r in recs:
        sc = r.get("score") or {}
        if r.get("error"):
            errs["error: " + str(r["error"])[:70]] += 1
        elif sc.get("hallucinated"):
            errs["URL alucinada"] += 1
        elif sc.get("false_abstain"):
            errs["abstención indebida"] += 1
        elif sc.get("exec_error"):
            errs["ADQL/ejecución falla"] += 1
        elif sc.get("no_answer"):
            errs["sin respuesta"] += 1
        elif not sc.get("success") and not sc.get("protocol_ok"):
            errs["protocolo incorrecto"] += 1
        elif not sc.get("success"):
            errs["resultado incorrecto"] += 1
    started = plan.get("started")
    done = len(recs)
    total = plan.get("total") or done
    elapsed = (time.time() - started) if started else None
    eta = (elapsed / done * (total - done)) if (elapsed and done) else None
    out.update({
        "plan": plan, "done": done, "total": total, "elapsed_s": elapsed, "eta_s": eta,
        "active": (time.time() - jsonl_mtime) < 120 and done < total,
        "overall": rate([bool((r.get("score") or {}).get("success")) for r in recs]),
        "cost_total": sum(r.get("cost_usd") or 0 for r in recs),
        "tokens_total": sum((r.get("tok_in") or 0) + (r.get("tok_out") or 0) for r in recs),
        "models": models, "archs": archs,
        "matrix": {f"{m}|{a}": {**rate(by[("ma", m, a)]), "cost": cost[(m, a)],
                                "wall_med": sorted(wall[(m, a)])[len(wall[(m, a)]) // 2] if wall[(m, a)] else None}
                   for m in models for a in archs},
        "by_arch": {a: rate(by[("a", a)]) for a in archs},
        "by_protocol": {f"{p}|{a}": rate(by[("pa", p, a)]) for p in ["sia", "scs", "tap", "ssa", "none"] for a in archs},
        "by_lang": {f"{l}|{a}": rate(by[("la", l, a)]) for l in ["en", "es"] for a in archs},
        "by_item": {f"{i['id']}|{a}": rate(by[("i", i["id"], a)]) for i in items for a in archs},
        "items": [{"id": i["id"], "protocol": i["protocol"], "en": i["en"]} for i in items],
        "errors": errs.most_common(12),
        "recent": [{"t": r.get("wall_s"), "model": r["model"].split("/", 1)[-1], "arch": r["arch"], "item": r["item"],
                    "lang": r["lang"], "ok": bool((r.get("score") or {}).get("success")),
                    "target": ((r.get("answer") or {}).get("target_name")),
                    "err": r.get("error"), "calls": r.get("n_tool_calls"), "cost": r.get("cost_usd")}
                   for r in recs[-25:]][::-1],
    })
    return out





_ACACHE = {}
_ALOCK = __import__("threading").Lock()
_ALAST = {}  # último resultado por corrida (se sirve mientras se recalcula)


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except Exception:
        pass
    return str(o)


def analysis_cached(run):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib
    import analysis
    runs = sorted([p.stem for p in RES.glob("*.jsonl")], key=lambda s: -(RES / f"{s}.jsonl").stat().st_mtime)
    run = run or (runs[0] if runs else None)
    if not run:
        return {"empty": True}
    path = RES / f"{run}.jsonl"
    key = (run, path.stat().st_mtime, Path(analysis.__file__).stat().st_mtime)
    if key in _ACACHE:
        return _ACACHE[key]
    # Un solo cálculo a la vez (cada uno usa ~250 MB): si ya hay uno en curso, servir el último resultado.
    if not _ALOCK.acquire(blocking=run not in _ALAST):
        return {**_ALAST[run], "stale": True}
    try:
        if key not in _ACACHE:
            importlib.reload(analysis)
            res = analysis.analyze(path)
            res["run"] = run
            _ACACHE.clear()
            _ACACHE[key] = json.loads(json.dumps(res, default=_jsonable).replace("NaN", "null"))
            _ALAST[run] = _ACACHE[key]
        return _ACACHE[key]
    finally:
        _ALOCK.release()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/summary":
            run = parse_qs(u.query).get("run", [None])[0]
            try:
                body, ct, code = json.dumps(summary(run), default=str).encode(), "application/json", 200
            except Exception as e:
                body, ct, code = json.dumps({"error": str(e)}).encode(), "application/json", 500
        elif u.path == "/api/analysis":
            run = parse_qs(u.query).get("run", [None])[0]
            try:
                body, ct, code = json.dumps(analysis_cached(run), default=_jsonable).encode(), "application/json", 200
            except Exception as e:
                import traceback
                body, ct, code = json.dumps({"error": f"{type(e).__name__}: {e}",
                                             "tb": traceback.format_exc()[-1500:]}).encode(), "application/json", 500
        elif u.path == "/":
            page = Path(__file__).with_name("dashboard_page.html")
            body = page.read_text().encode()
            ct, code = "text/html; charset=utf-8", 200
        else:
            body, ct, code = b"not found", "text/plain", 404
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"VOAI-Bench monitor en http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
