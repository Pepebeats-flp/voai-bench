"""Puntuación de episodios VOAI-Bench (basada en ejecución).

Métricas por episodio:
  protocol_ok     protocolo predicho == gold
  coord_err_as    error angular (arcsec) entre coordenadas enviadas y gold (si aplica)
  success         éxito de punta a punta:
                    sia/ssa: el recurso elegido existe en la traza (no alucinado), contiene la
                             posición objetivo y su rango espectral solapa la banda pedida
                    scs:     conjunto de IDs devuelto == gold (exacto); además jaccard
                    tap:     resultado == gold (scalar con rtol / ordered / set)  [Execution Accuracy]
                    none:    el sistema se abstiene
  false_abstain   se abstuvo en un ítem respondible
  hallucinated    URL enviada que ninguna herramienta devolvió
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

import vo_tools as vt

_TOL = yaml.safe_load((Path(__file__).resolve().parent.parent / "bench/tolerances.yaml").read_text())


def tol_deg(item_id):
    return float((_TOL.get("items") or {}).get(item_id, _TOL.get("default", 0.016667)))


_CIRCLE = re.compile(r"CIRCLE\s*\(\s*'?ICRS'?\s*,\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,", re.I)

POS_TOL_DEG = 0.02  # tolerancia mínima si la imagen no informa fov


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _find_resource(trace, url):
    if not url:
        return None
    for t in trace:
        out = t.get("out") or {}
        for k in ("images", "spectra"):
            for r in out.get(k, []) or []:
                if r.get("url") == url:
                    return r
    return None


def _band_ok(res, band):
    if not band:
        return True
    lo, hi = res.get("em_min_um"), res.get("em_max_um")
    if lo is None or hi is None:
        return False
    return lo <= band[1] and hi >= band[0]


def _contains(res, ra, dec, default_fov=0.1):
    reg = vt.region_contains(res.get("s_region"), ra, dec)
    if reg is not None:
        return reg
    rra, rdec = _num(res.get("ra")), _num(res.get("dec"))
    if rra is None or rdec is None:
        return None
    fov = _num(res.get("fov_deg")) or default_fov
    return vt.ang_sep_deg(rra, rdec, ra, dec) <= max(fov / 2.0, POS_TOL_DEG)


def _name_variants(name):
    """Variantes de un nombre escrito por el modelo: completo, antes del paréntesis, dentro del paréntesis."""
    name = re.sub(r"^\s*the\s+", "", str(name), flags=re.I).strip()
    out = [name]
    m = re.match(r"^(.*?)\s*\((.+?)\)\s*$", name)
    if m:
        out += [m.group(1).strip(), m.group(2).strip()]
    return [v for i, v in enumerate(out) if v and v not in out[:i]]


def _class_target_ok(g, ans):
    """Para consultas por clase: el objeto elegido pertenece a la clase aceptada si CUALQUIERA de sus
    tipos SIMBAD (principal o secundarios, %OTYPELIST) está en el conjunto aceptado.
    Devuelve (ok, ra, dec) del objeto elegido."""
    name = ans.get("target_name")
    if not name:
        return False, None, None
    r = None
    for v in _name_variants(name):
        try:
            rv = vt.resolve_name(v)
        except vt.ToolError:
            continue
        if rv.get("found"):
            r, name = rv, v
            break
    if r is None:
        return False, None, None
    accept = set(g.get("accept_otypes") or [])
    types = {r.get("otype")}
    try:
        types |= set(vt.otype_list(r.get("main_id") or name))
    except vt.ToolError:
        pass
    ok = bool(types & accept)
    for acc in (g.get("accept_resolved") or {}).values():
        if vt.ang_sep_deg(acc["ra"], acc["dec"], r["ra"], r["dec"]) < 5 / 60:
            ok = True
    return ok, r["ra"], r["dec"]


def _first_col(res):
    col = res.get("id_column") or res["columns"][0]
    return [row.get(col) for row in res["rows"]]


def _cmp_scalar(gold_rows, rows, rtol):
    if not rows:
        return False
    pred = list(rows[0].values())
    gold = gold_rows[0] if gold_rows else []
    if len(pred) < len(gold):
        return False
    for gv in gold:  # cada valor gold debe aparecer entre los valores predichos (orden de columnas libre)
        gnum = _num(gv)
        hit = False
        for pv in pred:
            pnum = _num(pv)
            if gnum is not None and pnum is not None:
                if math.isclose(gnum, pnum, rel_tol=rtol or 0.0, abs_tol=1e-9 if not rtol else 0.0):
                    hit = True
                    break
            elif str(gv) == str(pv):
                hit = True
                break
        if not hit:
            return False
    return True


def _compare(g, res, gold):
    """Compara un resultado TAP/SCS con el gold.

    Conjuntos/top-N: igualdad de conjuntos de identificadores de fila, sin orden ni duplicados
    (BIRD EX usa set(); ALeRCE PM_rows = precisión = recall = 1). Las columnas extra se aceptan y la
    columna identificadora se busca en cualquier posición (ALeRCE PM_cols: "all but not necessarily only").
    Escalares (COUNT/AVG): cada valor gold debe aparecer en la primera fila (ALeRCE trata count como id).
    Devuelve (success, info) con precisión/recall/jaccard de la mejor columna.
    """
    if g["compare"] == "scalar":
        return _cmp_scalar(gold, res["rows"], g.get("rtol")), {}
    gs = set(map(str, gold))
    cols = [res.get("id_column")] if res.get("id_column") else []
    cols += [c for c in res["columns"] if c not in cols]
    best = (False, {"precision": 0.0, "recall": 0.0, "jaccard": 0.0})
    for c in cols:
        pred = set(str(r.get(c)) for r in res["rows"])
        inter, union = len(pred & gs), len(pred | gs)
        info = {"precision": inter / len(pred) if pred else 0.0, "recall": inter / len(gs) if gs else 1.0,
                "jaccard": inter / union if union else 1.0, "id_col": c}
        if pred == gs:
            return True, info
        if info["jaccard"] > best[1]["jaccard"]:
            best = (False, info)
    return best


def _gold_at(g, ra, dec):
    """Re-ejecuta la ADQL gold centrada en (ra, dec) y devuelve su firma de resultado."""
    res = vt.tap_query(g["service"], g["adql"].format(ra=f"{ra:.6f}", dec=f"{dec:.6f}"), maxrec=5000)
    if g["compare"] == "scalar":
        return [list(r.values()) for r in res["rows"]]
    col = res["columns"][0]
    vals = [r[col] for r in res["rows"]]
    return sorted(set(map(str, vals)))


def score(item_gold: dict, ep: dict, item_id: str | None = None) -> dict:
    g, ans, trace = item_gold, ep.get("answer"), ep.get("trace") or []
    tol = float(g["tol_deg"]) if g.get("tol_deg") else tol_deg(item_id)
    gp = g["protocol"]
    s = {"protocol_ok": False, "success": False, "success_tolerant": False, "plan_ok": False, "coord_err_as": None, "false_abstain": False,
         "hallucinated": False, "jaccard": None, "exec_error": False}
    if not ans:
        s["no_answer"] = True
        return s
    pp = ans.get("protocol")
    ab = ans.get("abstain")
    if isinstance(ab, str):
        ab = ab.strip().lower() in ("true", "1", "yes")
    # El campo abstain es la señal de abstención; protocol solo se usa si abstain no viene.
    abstain = bool(ab) if ab is not None else pp == "none"
    valid_protocols = {"scs": ("scs", "tap")}.get(gp, (gp,))  # "possible answers" (BFCL)
    s["protocol_ok"] = (pp in valid_protocols) or (gp == "none" and abstain)
    if gp == "none":
        s["success"] = s["success_tolerant"] = s["plan_ok"] = abstain
        return s
    if abstain:
        s["false_abstain"] = True
        return s
    ra, dec = _num(ans.get("ra")), _num(ans.get("dec"))
    if ra is not None and dec is not None and "ra" in g and not g.get("accept_otypes"):
        s["coord_err_as"] = vt.ang_sep_deg(ra, dec, g["ra"], g["dec"]) * 3600
    # Corrección del plan sin ejecutar (≈ BFCL AST): protocolo válido y coordenadas en el conjunto aceptable
    s["plan_ok"] = s["protocol_ok"] and (s["coord_err_as"] is None or s["coord_err_as"] <= tol * 3600)

    if gp in ("sia", "ssa"):
        res = _find_resource(trace, ans.get("url"))
        if ans.get("url") and res is None:
            s["hallucinated"] = True
            return s
        if res is None:
            return s
        if g.get("accept_otypes") or g.get("accept_targets"):
            ok_cls, tra, tdec = _class_target_ok(g, ans)
            pos_ok = ok_cls and tra is not None and (_contains(res, tra, tdec) is not False)
            strict_ok = pos_ok
        else:
            c = _contains(res, g["ra"], g["dec"], default_fov=0.1)
            strict_ok = c if c is not None else (s["coord_err_as"] is not None and s["coord_err_as"] < 60)
            rra, rdec = _num(res.get("ra")), _num(res.get("dec"))
            near = rra is not None and rdec is not None and vt.ang_sep_deg(rra, rdec, g["ra"], g["dec"]) <= tol
            pos_ok = bool(strict_ok) or near
        band_ok = _band_ok(res, g.get("band_um"))
        s["success_tolerant"] = bool(pos_ok and band_ok)
        s["success"] = bool(strict_ok and band_ok)
        return s

    # scs / tap: re-ejecutar (desde caché) lo que el sistema envió
    try:
        if ans.get("adql"):
            res = vt.tap_query(ans.get("service") or "gaia_esa", ans["adql"], maxrec=5000)
        elif ans.get("catalog") and ra is not None:
            res = vt.cone_search(ans["catalog"], ra, dec, _num(ans.get("radius_deg")) or 0.01)
        else:
            return s
    except vt.ToolError:
        s["exec_error"] = True
        return s
    gold = g.get("result")
    if gold is None:
        return s
    ok, info = _compare(g, res, gold)
    s["success_tolerant"] = s["success"] = ok
    s["jaccard"], s["precision"], s["recall"] = info.get("jaccard"), info.get("precision"), info.get("recall")
    if ok or "ra" not in g:
        return s
    # EX tolerante al objetivo: centro usado por el sistema (CIRCLE del ADQL o cono) dentro de tol
    if ans.get("adql"):
        m = _CIRCLE.search(ans["adql"])
        cra, cdec = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    else:
        cra, cdec = ra, dec
    if cra is None or "{ra}" not in g.get("adql", ""):
        return s
    sep = vt.ang_sep_deg(cra, cdec, g["ra"], g["dec"])
    s["center_err_as"] = sep * 3600
    s["plan_ok"] = s["plan_ok"] and sep <= tol
    if sep <= tol:
        try:
            ok2, info2 = _compare(g, res, _gold_at(g, cra, cdec))
            s["success_tolerant"] = ok2
            if info2.get("jaccard") is not None:
                s["jaccard_tolerant"] = info2["jaccard"]
        except vt.ToolError:
            pass
    return s
