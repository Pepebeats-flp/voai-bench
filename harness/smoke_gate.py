"""Control de calidad automático tras la prueba de humo en v1 (decide si es seguro gastar créditos).

Distingue errores del HARNESS (bloquean) de errores del MODELO (normales):
  - episodios completados ≥ 90% de lo esperado
  - 0 errores del evaluador (scoring_error)
  - errores internos de herramientas (excepciones Python del harness) ≤ 2% de las llamadas
  - errores no-infra del episodio (p. ej. salida sin JSON) ≤ 15%
  - cada protocolo tiene éxito > 0 en al menos una arquitectura (si un protocolo da 0 en todo, sospechar del gold)
  - ítems con 0 éxitos en las 3 arquitecturas ≤ 40%
Escribe results/smoke_gate.json y sale con código 0 si pasa, 1 si no.
Uso: python -I harness/smoke_gate.py results/test.jsonl gpt-6-luna en
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INFRA = ("infra_error", "quota_exhausted", "RateLimitError", "APIConnectionError", "APITimeoutError",
         "InternalServerError", "ServiceUnavailableError")
PYEXC = re.compile(r"\b(TypeError|KeyError|AttributeError|NameError|IndexError|UnboundLocalError|ZeroDivisionError)\b")


def main():
    path, model, lang = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    gold = {k: v for k, v in json.loads((ROOT / "bench/gold_v1.json").read_text()).items() if not v.get("invalid")}
    sys.path.insert(0, str(ROOT / "harness"))
    from jsonl_io import iter_records
    latest = {}
    for r in iter_records(path, slim=False):  # streaming; se conserva solo un resumen de la traza
        if r["model"] == model and r["lang"] == lang and r["rep"] == 0:
            r["trace"] = [{"tool": t.get("tool"), "error": t.get("error") or (t.get("out") or {}).get("error")}
                          for t in (r.get("trace") or [])]
            r.pop("raw", None)
            latest[r["key"]] = r
    recs = [r for r in latest.values() if not str(r.get("error") or "").startswith(INFRA)]
    expected = len(gold) * 3
    calls = sum(r.get("n_tool_calls") or 0 for r in recs)
    internal = []
    for r in recs:
        for t in r.get("trace") or []:
            e = str(t.get("error") or (t.get("out") or {}).get("error") or "")
            if PYEXC.search(e):
                internal.append(f"{r['item']} {r['arch']} {t.get('tool')}: {e[:160]}")
    scoring = [f"{r['item']} {r['arch']}: {r['score'].get('scoring_error')}" for r in recs if (r.get("score") or {}).get("scoring_error")]
    ep_err = [r for r in recs if r.get("error")]
    by_prot = defaultdict(lambda: defaultdict(list))
    by_item = defaultdict(list)
    for r in recs:
        s = bool((r.get("score") or {}).get("success"))
        by_prot[r["protocol"]][r["arch"]].append(s)
        by_item[r["item"]].append(s)
    prot_rates = {p: {a: (sum(v) / len(v) if v else None) for a, v in d.items()} for p, d in by_prot.items()}
    dead_prot = [p for p, d in prot_rates.items() if all((x or 0) == 0 for x in d.values())]
    zero_items = sorted(i for i, v in by_item.items() if len(v) >= 3 and not any(v))
    checks = {
        "completados": (len(recs) >= 0.9 * expected, f"{len(recs)}/{expected}"),
        "errores_evaluador": (len(scoring) == 0, len(scoring)),
        "errores_internos_herramientas": (len(internal) <= max(2, 0.02 * calls), f"{len(internal)}/{calls} llamadas"),
        "errores_de_episodio": (len(ep_err) <= 0.15 * max(1, len(recs)), f"{len(ep_err)}/{len(recs)}"),
        "protocolos_sin_exito": (not dead_prot, dead_prot),
        "items_sin_exito": (len(zero_items) <= 0.4 * max(1, len(by_item)), f"{len(zero_items)}/{len(by_item)}"),
    }
    ok = all(c[0] for c in checks.values())
    report = {"passed": ok, "checks": {k: {"ok": v[0], "value": v[1]} for k, v in checks.items()},
              "success_by_protocol": prot_rates, "zero_items": zero_items,
              "internal_errors_sample": internal[:15], "scoring_errors_sample": scoring[:10],
              "episode_errors_sample": [f"{r['item']} {r['arch']}: {str(r['error'])[:120]}" for r in ep_err[:10]]}
    (ROOT / "results/smoke_gate.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"passed": ok, **{k: v for k, v in report["checks"].items()}}, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
