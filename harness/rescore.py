"""Re-puntúa un JSONL con la versión actual de score.py (sin volver a llamar a los LLMs).
Uso: python -I harness/rescore.py results/test.jsonl --bench v1   (escribe in-place con respaldo .bak)
No ejecutar mientras el corredor escribe en ese mismo archivo."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
from score import score  # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("jsonl")
_ap.add_argument("--bench", default=None, help="v0 | v1 (por defecto: v1 si el archivo se llama test*)")
_a = _ap.parse_args()
path = Path(_a.jsonl)
bench = _a.bench or ("v1" if path.stem.startswith("test") else "v0")
gold = json.loads((ROOT / f"bench/gold_{bench}.json").read_text())
shutil.copy(path, path.with_suffix(".jsonl.bak"))
# Dos pasadas en streaming (el archivo puede pesar cientos de MB): 1) última línea de cada episodio;
# 2) re-puntuar y escribir solo esas líneas.
last = {}
with open(path) as fh:
    for i, line in enumerate(fh):
        try:
            last[json.loads(line)["key"]] = i
        except (json.JSONDecodeError, KeyError):
            pass
keep = set(last.values())
tmp = path.with_suffix(".jsonl.tmp")
n_out, changed, skipped = 0, 0, 0
with open(path) as fh, open(tmp, "w") as out:
    for i, line in enumerate(fh):
        if i not in keep:
            continue
        r = json.loads(line)
        if r["item"] not in gold:
            skipped += 1
            continue
        new = score(gold[r["item"]], {"answer": r.get("answer"), "trace": r.get("trace") or []}, r["item"])
        changed += new.get("success") != (r.get("score") or {}).get("success")
        r["score"] = new
        out.write(json.dumps(r, default=str) + "\n")
        n_out += 1
        if n_out % 500 == 0:
            print(f"  {n_out}/{len(keep)}", flush=True)
tmp.replace(path)
print(f"re-puntuados {n_out} | cambios de éxito {changed} | descartados sin gold {skipped}")
