"""Línea base sin LLM (R0): reglas genéricas EN/ES + resolución de nombres + el mismo ejecutor que A0.

No usa la tabla de objetos ni las plantillas del generador: extrae el objeto probando fragmentos del texto
con Sesame. Para TAP no intenta generar ADQL (requeriría una regla por tipo de consulta) y se abstiene.
"""
from __future__ import annotations

import re

import vo_tools as vt
from agents import execute_plan

BANDS = [  # (patrón, (lo, hi) en µm) — se evalúan en orden
    (r"h-?alpha|h-?alfa|hα", (0.65, 0.66)),
    (r"x-?ray|rayos x", (0.00052, 0.0124)),
    (r"(\d+(?:[.,]\d+)?)\s*ghz", "ghz"),
    (r"\bradio\b(?!\s+de)(?<!un radio)", (2.0e5, 2.3e5)),  # en español "radio de" = radius
    (r"(\d+(?:[.,]\d+)?)\s*(?:micron|micrones|µm|um)\b", "micron"),
    (r"\bj[- ]band|banda j\b", (1.1, 1.4)),
    (r"\bh[- ]band|banda h\b", (1.5, 1.8)),
    (r"\bk[- ]band|banda k\b|\bks\b", (1.95, 2.36)),
    (r"near[- ]infrared|infrarrojo cercano|2mass", (0.9, 2.5)),
    (r"mid[- ]infrared|infrarrojo medio", (5.0, 30.0)),
    (r"far[- ]infrared|infrarrojo lejano", (30.0, 200.0)),
    (r"ultraviolet|ultravioleta|\buv\b", (0.1, 0.32)),
    (r"optical|visible|óptica|luz visible", (0.4, 0.9)),
    (r"infrared|infrarroj", (1.0, 200.0)),
]
CATS = [(r"gaia", "gaia_dr3"), (r"allwise|\bwise\b", "allwise"), (r"2mass", "2mass_psc"),
        (r"near-?infrared catalog|infrarrojo cercano", "2mass_psc")]
UNITS = {"arcsec": 1 / 3600, "segundos de arco": 1 / 3600, "arcmin": 1 / 60, "minutos de arco": 1 / 60,
         "degree": 1.0, "degrees": 1.0, "deg": 1.0, "grado": 1.0, "grados": 1.0}
STOP = {"the", "a", "an", "of", "la", "el", "los", "las", "de", "del", "una", "un", "me", "please", "por", "favor",
        "image", "picture", "spectrum", "spectra", "imagen", "espectro", "espectros", "photo", "foto"}


def _num(s):
    return float(s.replace(",", "."))


def band_of(t):
    for pat, val in BANDS:
        m = re.search(pat, t)
        if m:
            if val == "ghz":
                lam_um = 299792458 / (_num(m.group(1)) * 1e9) * 1e6
                return (lam_um * 0.8, lam_um * 1.2)
            if val == "micron":
                x = _num(m.group(1))
                return (x * 0.85, x * 1.15)
            return val
    return None


def coords_of(t):
    try:
        return _coords_of(t)
    except ValueError:
        return None


def _coords_of(t):
    m = re.search(r"l\s*=?\s*([\d.]+)\s*(?:deg|°)?[, ]+.*?b\s*=?\s*(-?[\d.]+)", t) or \
        re.search(r"(?:galactic longitude|longitud galáctica)\s*([\d.,]+)\D+?(-?[\d.,]+)", t)
    if m:
        return vt.galactic_to_icrs(_num(m.group(1)), _num(m.group(2)))
    m = re.search(r"(\d{1,2})h(\d{1,2})m([\d.]+)s\s*([+-]?\d{1,2})d(\d{1,2})m([\d.]+)s", t)
    if m:
        h, mi, s, d, dm, ds = m.groups()
        ra = 15 * (int(h) + int(mi) / 60 + float(s) / 3600)
        sign = -1 if d.startswith("-") else 1
        dec = sign * (abs(int(d)) + int(dm) / 60 + float(ds) / 3600)
        return ra, dec
    m = re.search(r"(?:ra|ar)\s*=?\s*(-?\d+(?:[.,]\d+)?)\s*[,;]?\s*dec\s*=?\s*(-?\d+(?:[.,]\d+)?)", t)
    if m:
        return _num(m.group(1)), _num(m.group(2))
    return None


def name_of(text):
    """Prueba fragmentos (de más largo a más corto) tras 'of/de/around/alrededor de' y, si no, todo el texto."""
    t = re.sub(r"[?¿!¡.]", " ", text)
    spans = []
    for kw in [r"\bof\b", r"\bde\b", r"\baround\b", r"alrededor de", r"\bcovering\b", r"cubren"]:
        for m in re.finditer(kw, t, flags=re.I):
            spans.append(t[m.end():])
    spans.append(t)
    for sp in spans:
        words = [w for w in re.split(r"[\s,]+", sp) if w]
        while words and words[0].lower() in STOP:
            words = words[1:]
        for n in range(min(6, len(words)), 0, -1):
            cand = " ".join(words[:n]).strip(" ,")
            if len(cand) < 2 or cand.lower() in STOP or re.fullmatch(r"[\d.,\s]+", cand):
                continue
            try:
                r = vt.resolve_name(cand)
            except vt.ToolError:
                continue
            if r.get("found"):
                return cand, r
    return None, None


def plan_for(text):
    t = text.lower()
    if re.search(r"spectr|espectro", t):
        proto = "ssa"
    elif re.search(r"cone search|búsqueda cónica|\blist\b|\blista\b|which .*(sources|objects|entries)|qué (fuentes|objetos)|"
                   r"give me (all|the) .*(sources|entries)|dame (todas|las) .*(fuentes|entradas)|"
                   r"what .*sources|catalog(ue)? sources|fuentes de catálogo", t) and not re.search(r"how many|cuánt|count|cuenta", t):
        proto = "scs"
    elif re.search(r"how many|cuánt|count|cuenta|mean|media|brightest|más brillantes|largest|mayor|which .*collections|"
                   r"qué colecciones|give me the g magnitude|dame la magnitud|what is the|cuál es", t):
        proto = "tap"
    elif re.search(r"image|picture|photo|view|imagen|foto|vista", t):
        proto = "sia"
    else:
        proto = "none"
    if proto in ("none", "tap"):  # TAP: no se genera ADQL por reglas genéricas
        return {"protocol": "none", "abstain": True, "explanation": f"rules: {proto}"}
    c = coords_of(text.lower())
    name, r = (None, None)
    if c is None:
        name, r = name_of(text)
        if r is None:
            return {"protocol": "none", "abstain": True, "explanation": "rules: no target resolved"}
        c = (r["ra"], r["dec"])
    plan = {"protocol": proto, "target_name": name, "ra": c[0], "dec": c[1], "abstain": False}
    b = band_of(t)
    if b:
        plan["band_um_min"], plan["band_um_max"] = b
    if proto == "scs":
        cat = next((v for p, v in CATS if re.search(p, t)), "gaia_dr3")
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(arcsec|arcmin|degrees?|deg|segundos de arco|minutos de arco|grados?)", t)
        plan["catalog"] = cat
        plan["radius_deg"] = _num(m.group(1)) * UNITS.get(m.group(2), 1 / 60) if m else 1 / 60
    return plan


def run_rules(query: str):
    from llm import CallStats
    trace = []
    plan = plan_for(query)
    ans = execute_plan(plan, trace)
    return {"answer": ans, "plan": plan, "raw": None, "trace": trace, "stats": CallStats()}
