"""Ejecuta experimentos VOAI-Bench de forma automatizada y reanudable.

Ejemplo:
  python -I harness/run.py --name pilot --models qwen3.8-flash,glm-5.3 \
      --archs A0,A1,A2 --langs en,es --repeats 1 --workers 6
Cada episodio queda en results/<name>.jsonl (una línea por episodio). Re-ejecutar con el mismo
--name salta los episodios ya hechos.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import agents  # noqa: E402
from llm import LLM  # noqa: E402
from score import score  # noqa: E402

LOCK = threading.Lock()
STOP = threading.Event()  # se activa cuando el proveedor informa cuota agotada
INFRA_PREFIXES = ("infra_error", "quota_exhausted", "RateLimitError", "APIConnectionError", "APITimeoutError",
                  "InternalServerError", "ServiceUnavailableError")


def is_infra(err) -> bool:
    return bool(err) and str(err).startswith(INFRA_PREFIXES)


def slim_trace(trace):
    out = []
    for t in trace:
        t = dict(t)
        o = t.get("out")
        if isinstance(o, dict):
            o = dict(o)
            if isinstance(o.get("rows"), list) and len(o["rows"]) > 50:
                o["rows"], o["rows_truncated"] = o["rows"][:50], True
            for k in ("images", "spectra"):  # el modelo solo ve las primeras 25 (truncate_for_llm)
                if isinstance(o.get(k), list) and len(o[k]) > 25:
                    o[k], o[k + "_truncated"] = o[k][:25], True
            t["out"] = o
        out.append(t)
    return out


def episode(model, arch, item, lang, rep, gold, temperature):
    key = f"{model}|{arch}|{item['id']}|{lang}|{rep}"
    if STOP.is_set():
        return None
    llm = None if arch == "R0" else LLM(model, temperature=temperature)
    t0 = time.time()
    try:
        ep = agents.run(arch, llm, item[lang], session=f"voai-{abs(hash(key)) % 10**10}")
        err = ep.get("error") or ep.get("parse_error")
    except Exception as e:
        ep, err = {"answer": None, "trace": [], "stats": None}, f"{type(e).__name__}: {str(e)[:300]}"
        if "usage limit" in str(e).lower() or "quota" in str(e).lower():
            STOP.set()
            err = "quota_exhausted: " + str(e)[:200]
        else:
            traceback.print_exc()
    wall = time.time() - t0
    st = ep.get("stats")
    trace = ep.get("trace") or []
    infra = [t for t in trace if "transient" in str(t.get("error") or (t.get("out") or {}).get("error") or "")]
    if infra and err is None:  # caída de un servicio durante el episodio: no es fallo del modelo, se repite
        err = "infra_error: " + str(infra[0].get("error") or infra[0]["out"].get("error"))[:200]
    tool_s = sum(t.get("s") or 0 for t in trace)
    rec = {
        "key": key, "ts": time.time(), "model": model, "arch": arch, "item": item["id"], "protocol": item["protocol"],
        "category": item["category"], "difficulty": item["difficulty"], "lang": lang, "rep": rep,
        "temperature": temperature, "answer": ep.get("answer"), "error": err,
        "wall_s": wall, "llm_s": st.llm_s if st else None, "tool_s": tool_s,
        "llm_calls": st.calls if st else 0, "tok_in": st.tok_in if st else 0, "tok_out": st.tok_out if st else 0,
        "tok_reason": st.tok_reason if st else 0, "cost_usd": st.cost_usd if st else 0.0, "allowance": st.allowance if st else 0.0,
        "n_tool_calls": len(trace), "n_tool_errors": sum(1 for t in trace if not t.get("ok", True)),
        "steps": ep.get("steps"), "raw": ep.get("raw"), "trace": slim_trace(trace),
    }
    try:
        rec["score"] = score(gold, {"answer": ep.get("answer"), "trace": trace}, item["id"])
    except Exception as e:
        rec["score"] = {"scoring_error": f"{type(e).__name__}: {e}"}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bench", default="v0", help="v0 (desarrollo) | v1 (prueba)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--archs", default="A0,A1,A2")
    ap.add_argument("--langs", default="en,es")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--items", default=None, help="IDs separados por coma o prefijos (S,T,...)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-cost", type=float, default=None,
                    help="tope de costo de lista (USD) acumulado en esta corrida (cubre créditos de pago por uso)")
    ap.add_argument("--max-allowance", type=float, default=None,
                    help="tope de cuota the API provider (USD de allowance) acumulada en esta corrida; al superarlo se detiene")
    a = ap.parse_args()

    items = yaml.safe_load((ROOT / f"bench/queries_{a.bench}.yaml").read_text())["items"]
    gold = json.loads((ROOT / f"bench/gold_{a.bench}.json").read_text())
    gold = {k: v for k, v in gold.items() if not v.get("invalid")}
    if a.items:
        sel = a.items.split(",")
        items = [i for i in items if any(i["id"] == s or (len(s) == 1 and i["id"].startswith(s)) for s in sel)]
    items = [i for i in items if i["id"] in gold]
    out = ROOT / "results" / f"{a.name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        from jsonl_io import iter_records
        for r in iter_records(out):
            if not is_infra(r.get("error")):  # errores del modelo cuentan como hechos; los de infraestructura se repiten
                done.add(r["key"])
            else:
                done.discard(r["key"])
    jobs = [(m, ar, it, lg, rp) for rp in range(a.repeats) for m in a.models.split(",") for ar in a.archs.split(",")
            for it in items for lg in a.langs.split(",")
            if f"{m}|{ar}|{it['id']}|{lg}|{rp}" not in done]
    import random
    random.Random(7).shuffle(jobs)  # intercalar modelos/arquitecturas: reparte carga entre proveedores
    jobs.sort(key=lambda j: j[4])  # orden por repetición: completar rep 0 de todo antes de rep 1 (sort estable)
    print(f"{len(jobs)} episodios pendientes ({len(done)} ya hechos) -> {out}")
    (ROOT / "results" / f"{a.name}.plan.json").write_text(json.dumps({
        "name": a.name, "total": len(jobs) + len(done), "models": a.models.split(","), "archs": a.archs.split(","),
        "langs": a.langs.split(","), "repeats": a.repeats, "n_items": len(items), "started": time.time()}))
    n_ok = 0
    spent, cost = 0.0, 0.0
    if out.exists():
        from jsonl_io import iter_records
        for r in iter_records(out):
            spent += r.get("allowance") or 0.0
            cost += r.get("cost_usd") or 0.0
    print(f"gastado en esta corrida: costo lista US$ {cost:.2f} (tope {a.max_cost}) · cuota {spent:.2f} (tope {a.max_allowance})", flush=True)
    if (a.max_allowance is not None and spent >= a.max_allowance) or (a.max_cost is not None and cost >= a.max_cost):
        print("BUDGET_REACHED", flush=True)
        return
    with cf.ThreadPoolExecutor(a.workers) as ex, out.open("a") as fh:
        futs = [ex.submit(episode, m, ar, it, lg, rp, gold[it["id"]], a.temperature) for m, ar, it, lg, rp in jobs]
        for k, f in enumerate(cf.as_completed(futs), 1):
            rec = f.result()
            if rec is None:  # saltado por cuota agotada: queda pendiente para la próxima corrida
                continue
            with LOCK:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
            n_ok += bool(rec["score"].get("success"))
            spent += rec.get("allowance") or 0.0
            cost += rec.get("cost_usd") or 0.0
            over = (a.max_allowance is not None and spent >= a.max_allowance) or (a.max_cost is not None and cost >= a.max_cost)
            if over and not STOP.is_set():
                STOP.set()
                print(f"BUDGET_REACHED: costo US$ {cost:.2f}, cuota {spent:.2f}", flush=True)
            if k % 10 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}  success acumulado={n_ok / k:.2f}", flush=True)


if __name__ == "__main__":
    main()
