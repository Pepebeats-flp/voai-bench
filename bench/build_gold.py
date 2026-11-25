"""Construye el gold de VOAI-Bench: resuelve objetos, ejecuta ADQL de referencia y
verifica factibilidad (que exista al menos una respuesta correcta en el catálogo de servicios).

Uso: python -I bench/build_gold.py [--only S01,T02]
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import vo_tools as vt  # noqa: E402


def overlaps(a_lo, a_hi, band):
    if band is None:
        return True
    if a_lo is None or a_hi is None:
        return False
    return a_lo <= band[1] and a_hi >= band[0]


def feasible_images(ra, dec, band):
    """Imágenes candidatas que contienen la posición y cumplen la banda."""
    found = []
    try:
        kw = {"band_um_min": band[0], "band_um_max": band[1]} if band else {}
        r = vt.sia_search(ra, dec, 0.001, **kw)
        for im in r["images"]:
            if overlaps(im["em_min_um"], im["em_max_um"], band):
                found.append(im["collection"])
    except vt.ToolError as e:
        print("   sia2 error:", e)
    for s, (lo, hi) in vt.SKYVIEW_SURVEYS.items():
        if band and not overlaps(lo, hi, band):
            continue
        if not band and s not in ("DSS2 Red", "2MASS-K"):
            continue
        try:
            if vt.skyview_image(s, ra, dec, 0.1)["n"]:
                found.append(f"SkyView:{s}")
        except vt.ToolError as e:
            print("   skyview error:", s, e)
    return sorted(set(found))


def feasible_spectra(ra, dec, band, diam):
    found = []
    for svc in vt.SSA_SERVICES:
        try:
            r = vt.ssa_search(svc, ra, dec, diam)
            ok = [s for s in r["spectra"] if overlaps(s["em_min_um"], s["em_max_um"], band) or s["em_min_um"] is None]
            if ok:
                found.append(f"{svc}:{len(ok)}")
        except vt.ToolError as e:
            print("   ssa error:", svc, e)
    return found


def result_signature(res, compare):
    rows = res["rows"]
    if compare == "scalar":
        return [list(r.values()) for r in rows]
    col = res["columns"][0]
    vals = [r[col] for r in rows]
    return vals if compare == "ordered" else sorted(set(map(str, vals)))


def process(it):
    """Construye el gold de un ítem. Devuelve (id, entrada | None, [problemas], log)."""
    g, log, probs = dict(it["gold"]), [f"[{it['id']}] {it['protocol']} {it['gold'].get('target')}"], []
    if g.get("abstain"):
        return it["id"], {"protocol": "none", **g}, probs, log
    try:
        if g.get("target"):
            r = vt.resolve_name(g["target"])
            if not r["found"]:
                return it["id"], None, ["sesame no resuelve"], log
            g.update(ra=r["ra"], dec=r["dec"], otype=r["otype"], main_id=r["main_id"])
        elif "l" in g:
            g["ra"], g["dec"] = vt.galactic_to_icrs(g["l"], g["b"])
        if g.get("accept_targets"):
            g["accept_resolved"] = {}
            for t in g["accept_targets"]:
                r = vt.resolve_name(t)
                if r["found"]:
                    g["accept_resolved"][t] = r
    except vt.ToolError as e:
        return it["id"], None, [f"resolver: {e}"], log
    p = it["protocol"]
    if p in ("tap", "scs"):
        adql = g["adql"].format(ra=f"{g['ra']:.6f}", dec=f"{g['dec']:.6f}") if "ra" in g else g["adql"]
        g["adql_resolved"] = adql
        try:
            res = vt.tap_query(g["service"], adql, maxrec=5000)
            g["result"] = result_signature(res, g["compare"])
            g["n_rows"] = res["n"]
            log.append(f"   gold rows={res['n']} sample={str(g['result'])[:100]}")
            if res["n"] == 0 or (g["compare"] == "scalar" and g["result"] in ([[0]], [[None]])):
                probs.append("resultado gold vacío/cero")
            if res["n"] >= 5000:
                probs.append("resultado gold truncado (>=5000 filas)")
        except vt.ToolError as e:
            probs.append(f"ADQL falla: {e}")
    elif p == "sia" and "ra" in g:
        g["feasible"] = feasible_images(g["ra"], g["dec"], g.get("band_um"))
        log.append(f"   feasible: {g['feasible'][:8]}")
        if not g["feasible"]:
            probs.append("sin imagen factible")
    elif p == "ssa" and "ra" in g:
        g["feasible"] = feasible_spectra(g["ra"], g["dec"], g.get("band_um"), g.get("radius_deg", 0.05))
        log.append(f"   feasible: {g['feasible']}")
        if not g["feasible"]:
            probs.append("sin espectro factible")
    if probs:
        g["invalid"] = "; ".join(probs)  # el runner excluye ítems inválidos
    return it["id"], {"protocol": p, **g}, probs, log


def main():
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--queries", default="bench/queries_v0.yaml")
    ap.add_argument("--out", default="bench/gold_v0.json")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    items = yaml.safe_load((ROOT / args.queries).read_text())["items"]
    out_path = ROOT / args.out
    gold = json.loads(out_path.read_text()) if out_path.exists() else {}
    only = set(args.only.split(",")) if args.only else None
    todo = [it for it in items if not only or it["id"] in only]
    problems = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, (iid, entry, probs, log) in enumerate(ex.map(process, todo), 1):
            print("\n".join(log) + f"   ({k}/{len(todo)})", flush=True)
            if entry is not None:
                gold[iid] = entry
            else:
                gold.pop(iid, None)
            problems += [(iid, p) for p in probs]
            out_path.write_text(json.dumps(gold, indent=1, default=str))
    print("\nPROBLEMAS:")
    for p in problems:
        print(" ", p)
    print(f"gold válidos: {sum(1 for v in gold.values() if not v.get('invalid'))}/{len(items)}")


if __name__ == "__main__":
    main()
