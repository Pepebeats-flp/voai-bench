"""Lectura de resultados JSONL en streaming (el archivo puede pesar cientos de MB por las trazas).

Nunca cargar el archivo completo en memoria: se lee línea por línea y, por defecto, se descartan
los campos pesados (trace, raw). Así el dashboard, el runner y el análisis caben en 1.2 GB.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HEAVY = ("trace", "raw")


def iter_records(path, slim=True):
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if slim:
                for k in HEAVY:
                    r.pop(k, None)
            yield r


def latest_records(path, slim=True) -> dict:
    """Último registro por episodio (key)."""
    out = {}
    if not Path(path).exists():
        return out
    for r in iter_records(path, slim):
        out[r["key"]] = r
    return out


def tail_records(path, n=400):
    """Últimos n registros leyendo solo el final del archivo."""
    p = Path(path)
    if not p.exists():
        return []
    size = p.stat().st_size
    chunk = min(size, 8 * 1024 * 1024)
    with open(p, "rb") as fh:
        fh.seek(size - chunk)
        data = fh.read().decode("utf-8", errors="ignore").splitlines()
    recs = []
    for line in data[-n:]:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for k in HEAVY:
            r.pop(k, None)
        recs.append(r)
    return recs
