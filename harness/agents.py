"""Arquitecturas evaluadas en VOAI-Bench.

A0  single-shot: el LLM produce un plan JSON (protocolo, objeto, coordenadas, banda, ADQL);
    un ejecutor determinista lo corre una vez. Sin herramientas ni retroalimentación
    (equivalente multi-protocolo del pipeline SPIE).
A1  A0 + resolución determinista de nombres (Sesame sobrescribe las coordenadas del LLM).
A2  agente con herramientas (function calling): resolver, SIA2, SkyView, cone search, TAP,
    SSA, esquema TAP; ve resultados/errores y puede corregir. Termina con submit_answer.
    Ablaciones: A2-noschema (sin describe_tables), A2-noresolve (sin resolve_name).
"""
from __future__ import annotations

import json
import re
import time

import vo_tools as vt
from llm import LLM, CallStats

MAX_STEPS = 10

SYSTEM_COMMON = """You are an assistant that answers astronomers' data requests by querying IVOA Virtual Observatory services.
Available data services (closed catalogue):
- Images: IRSA SIA2 (ObsCore collections such as 2MASS, WISE/AllWISE, Spitzer SEIP, GALEX, SPHEREx, IRAS...) and SkyView cutouts for these surveys: {surveys}.
- Catalogue cone search: gaia_dr3, 2mass_psc, allwise.
- TAP/ADQL services: gaia_esa (tables: gaiadr3.gaia_source, ...), irsa_tap (tables: fp_psc = 2MASS point sources, allwise_p3as_psd = AllWISE, ivoa.obscore, ...).
  ADQL spatial predicate example: 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',<ra_deg>,<dec_deg>,<radius_deg>)). Use TOP n instead of LIMIT.
- Spectra: SSA services 'iso' (ISO spectra) and 'herschel' (Herschel spectra).
Wavelengths are in micrometres. Coordinates are ICRS degrees.
If the request cannot be satisfied with these services (e.g. it is not an astronomical data request, or the data cannot exist), you must abstain.
The user may write in English or Spanish."""

A0_INSTRUCTIONS = """Return ONLY a JSON object (no prose) with this schema:
{
 "protocol": "sia" | "scs" | "tap" | "ssa" | "none",
 "target_name": string or null,      // object name you chose (resolvable by SIMBAD), if any
 "ra": number or null, "dec": number or null,   // ICRS degrees of the target/region
 "band_um_min": number or null, "band_um_max": number or null,  // requested spectral range in micrometres (sia/ssa)
 "radius_deg": number or null,       // search/cone radius (scs)
 "catalog": "gaia_dr3" | "2mass_psc" | "allwise" | null,   // scs only
 "service": "gaia_esa" | "irsa_tap" | null,   // tap only
 "adql": string or null,             // tap only: complete ADQL query with numeric coordinates
 "abstain": boolean, "explanation": string
}
Use "sia" for images, "scs" to list catalogue sources in a cone, "tap" for filtered/aggregated/sorted catalogue queries, "ssa" for spectra, "none" to abstain."""

A2_INSTRUCTIONS = """Work step by step using the tools. Do not guess coordinates of named objects: resolve them.
Inspect tool results; if a call fails or returns nothing useful, fix the parameters and retry.
When done, call submit_answer exactly once:
- images/spectra: give the URL of the selected image/spectrum exactly as returned by a tool;
- catalogue cone listings: give the cone you executed (catalog, ra, dec, radius_deg);
- other catalogue questions: give the final ADQL and service you executed;
- abstain=true if the request cannot be satisfied.
You have at most {max_steps} tool calls."""


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


NUM = {"type": "number"}
STR = {"type": "string"}
TOOLS = {
    "resolve_name": _fn("resolve_name", "Resolve an astronomical object name with Sesame (SIMBAD/NED). Returns ra, dec (deg) and SIMBAD object type.",
                        {"name": STR}, ["name"]),
    "sia_search": _fn("sia_search", "Search IRSA SIA2 images covering a position. Optional spectral range in micrometres and collection.",
                      {"ra": NUM, "dec": NUM, "radius_deg": NUM, "band_um_min": NUM, "band_um_max": NUM, "collection": STR},
                      ["ra", "dec"]),
    "skyview_image": _fn("skyview_image", "Get a SkyView cutout image of a survey centred on a position.",
                         {"survey": {"type": "string", "enum": list(vt.SKYVIEW_SURVEYS)}, "ra": NUM, "dec": NUM, "size_deg": NUM},
                         ["survey", "ra", "dec"]),
    "cone_search": _fn("cone_search", "List catalogue sources within a cone.",
                       {"catalog": {"type": "string", "enum": list(vt.CATALOGS)}, "ra": NUM, "dec": NUM, "radius_deg": NUM},
                       ["catalog", "ra", "dec", "radius_deg"]),
    "tap_query": _fn("tap_query", "Run a synchronous ADQL query on a TAP service.",
                     {"service": {"type": "string", "enum": list(vt.SERVICES)}, "adql": STR}, ["service", "adql"]),
    "describe_tables": _fn("describe_tables", "List main tables of a TAP service, or the columns of one table.",
                           {"service": {"type": "string", "enum": list(vt.SERVICES)}, "table": STR}, ["service"]),
    "ssa_search": _fn("ssa_search", "Search spectra (SSA) around a position.",
                      {"service": {"type": "string", "enum": list(vt.SSA_SERVICES)}, "ra": NUM, "dec": NUM, "diameter_deg": NUM},
                      ["service", "ra", "dec"]),
    "submit_answer": _fn("submit_answer", "Submit the final answer and finish.", {
        "protocol": {"type": "string", "enum": ["sia", "scs", "tap", "ssa", "none"],
                     "description": "Type of answer: 'sia' = image (from sia_search or skyview_image), 'scs' = catalogue cone "
                                    "listing (cone_search), 'tap' = ADQL query result (tap_query), 'ssa' = spectrum "
                                    "(ssa_search), 'none' = only when abstaining."},
        "target_name": STR, "ra": NUM, "dec": NUM, "url": STR,
        "catalog": {"type": "string", "enum": list(vt.CATALOGS)}, "radius_deg": NUM,
        "service": {"type": "string", "enum": list(vt.SERVICES)}, "adql": STR,
        "abstain": {"type": "boolean"}, "explanation": STR}, ["protocol", "abstain"]),
}

ARCHS = {
    "R0": {"kind": "rules"},
    "A0": {"kind": "single", "resolve": False},
    "A1": {"kind": "single", "resolve": True},
    "A2": {"kind": "agent", "tools": [t for t in TOOLS]},
    "A2-noschema": {"kind": "agent", "tools": [t for t in TOOLS if t != "describe_tables"]},
    "A2-noresolve": {"kind": "agent", "tools": [t for t in TOOLS if t != "resolve_name"]},  # prompt sin adaptar (histórico)
    # Ablación limpia: sin resolvedor y con el prompt adaptado (no pide resolver nombres).
    "A2-nores": {"kind": "agent", "tools": [t for t in TOOLS if t != "resolve_name"], "prompt": "noresolve"},
}


def _system():
    return SYSTEM_COMMON.format(surveys=", ".join(vt.SKYVIEW_SURVEYS))


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in output")
    return json.loads(m.group(0))


# ------------------------------------------------------------------ ejecutor determinista (A0/A1)
def execute_plan(plan: dict, trace: list) -> dict:
    """Ejecuta un plan JSON una sola vez y devuelve la respuesta en el formato de submit_answer."""
    p = plan.get("protocol")
    ans = {k: plan.get(k) for k in ("protocol", "target_name", "ra", "dec", "abstain", "explanation",
                                    "catalog", "radius_deg", "service", "adql")}
    if plan.get("abstain") or p == "none":
        ans["abstain"] = True
        return ans
    ra, dec = plan.get("ra"), plan.get("dec")

    def call(name, fn, **kw):
        t0 = time.time()
        try:
            out = fn(**kw)
            trace.append({"tool": name, "args": kw, "ok": True, "n": out.get("n"), "s": time.time() - t0, "out": out})
            return out
        except Exception as e:
            trace.append({"tool": name, "args": kw, "ok": False, "error": str(e)[:300], "s": time.time() - t0})
            return None
    if p == "sia" and ra is not None:
        b0, b1 = plan.get("band_um_min"), plan.get("band_um_max")
        kw = {"band_um_min": b0, "band_um_max": b1} if b0 is not None and b1 is not None else {}
        out = call("sia_search", vt.sia_search, ra=ra, dec=dec, radius_deg=0.001, **kw)
        if out and out["images"]:
            ans["url"] = out["images"][0]["url"]
        else:  # primer survey de SkyView que solape la banda pedida
            for s, (lo, hi) in vt.SKYVIEW_SURVEYS.items():
                if b0 is None or (lo <= b1 and hi >= b0):
                    o2 = call("skyview_image", vt.skyview_image, survey=s, ra=ra, dec=dec, size_deg=0.25)
                    if o2 and o2["images"]:
                        ans["url"] = o2["images"][0]["url"]
                        break
    elif p == "scs" and ra is not None and plan.get("catalog"):
        call("cone_search", vt.cone_search, catalog=plan["catalog"], ra=ra, dec=dec, radius_deg=plan.get("radius_deg") or 0.01)
    elif p == "tap" and plan.get("adql"):
        call("tap_query", vt.tap_query, service=plan.get("service") or "gaia_esa", adql=plan["adql"])
    elif p == "ssa" and ra is not None:
        for svc in vt.SSA_SERVICES:
            out = call("ssa_search", vt.ssa_search, service=svc, ra=ra, dec=dec, diameter_deg=0.05)
            if out and out["spectra"]:
                ans["url"] = out["spectra"][0]["url"]
                break
    return ans


def run_single(llm: LLM, query: str, resolve: bool, session: str):
    st, trace = CallStats(), []
    msgs = [{"role": "system", "content": _system() + "\n\n" + A0_INSTRUCTIONS}, {"role": "user", "content": query}]
    msg = llm.chat(msgs, st, session=session, json_mode=True)
    raw = msg.content
    try:
        plan = _parse_json(raw)
    except Exception as e:
        return {"answer": None, "parse_error": str(e), "raw": raw, "trace": trace, "stats": st}
    if resolve and plan.get("target_name") and not plan.get("abstain"):
        try:
            t0 = time.time()
            r = vt.resolve_name(plan["target_name"])
            trace.append({"tool": "resolve_name", "args": {"name": plan["target_name"]}, "ok": r["found"], "out": r,
                          "s": time.time() - t0})
            if r["found"]:
                plan["ra"], plan["dec"] = r["ra"], r["dec"]
                if plan.get("adql"):  # A1 no reescribe ADQL: el efecto del resolver se mide en sia/scs/ssa
                    pass
        except vt.ToolError as e:
            trace.append({"tool": "resolve_name", "ok": False, "error": str(e)})
    ans = execute_plan(plan, trace)
    return {"answer": ans, "plan": plan, "raw": raw, "trace": trace, "stats": st}


DISPATCH = {
    "resolve_name": lambda a: vt.resolve_name(a["name"]),
    "sia_search": lambda a: vt.sia_search(a["ra"], a["dec"], a.get("radius_deg") or 0.001, a.get("band_um_min"),
                                          a.get("band_um_max"), a.get("collection")),
    "skyview_image": lambda a: vt.skyview_image(a["survey"], a["ra"], a["dec"], a.get("size_deg") or 0.25),
    "cone_search": lambda a: vt.cone_search(a["catalog"], a["ra"], a["dec"], a["radius_deg"]),
    "tap_query": lambda a: vt.tap_query(a["service"], a["adql"]),
    "describe_tables": lambda a: vt.describe_tables(a["service"], a.get("table")),
    "ssa_search": lambda a: vt.ssa_search(a["service"], a["ra"], a["dec"], a.get("diameter_deg") or 0.05),
}


A2_INSTRUCTIONS_NORESOLVE = A2_INSTRUCTIONS.replace(
    "Do not guess coordinates of named objects: resolve them.",
    "No name-resolution tool is available: use your own knowledge of object coordinates when needed.")


def run_agent(llm: LLM, query: str, tool_names: list, session: str, prompt: str | None = None):
    st, trace = CallStats(), []
    tools = [TOOLS[t] for t in tool_names]
    instr = A2_INSTRUCTIONS_NORESOLVE if prompt == "noresolve" else A2_INSTRUCTIONS
    sysmsg = _system() + "\n\n" + instr.format(max_steps=MAX_STEPS)
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": query}]
    for step in range(MAX_STEPS + 2):
        msg = llm.chat(msgs, st, tools=tools, session=session)
        calls = msg.tool_calls or []
        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
        rc = getattr(msg, "reasoning_content", None)
        if rc:  # algunos proveedores exigen devolver el razonamiento en turnos con herramientas
            assistant["reasoning_content"] = rc
        msgs.append(assistant)
        if not calls:
            if step >= MAX_STEPS:
                break
            msgs.append({"role": "user", "content": "Please call submit_answer to finish."})
            continue
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if name == "submit_answer" and args is not None:
                return {"answer": args, "trace": trace, "stats": st, "steps": step + 1}
            t0 = time.time()
            if args is None:
                out, ok = {"error": "arguments are not valid JSON"}, False
            elif name not in DISPATCH or name not in tool_names:
                out, ok = {"error": f"unknown tool {name}"}, False
            else:
                try:
                    out, ok = DISPATCH[name](args), True
                except vt.ToolError as e:
                    out, ok = {"error": str(e)[:500]}, False
                except Exception as e:  # argumentos mal formados, etc.
                    out, ok = {"error": f"{type(e).__name__}: {str(e)[:300]}"}, False
            trace.append({"tool": name, "args": args, "ok": ok, "s": time.time() - t0,
                          "n": out.get("n") if isinstance(out, dict) else None, "out": out})
            content = json.dumps(vt.truncate_for_llm(out) if ok else out, default=str)[:12000]
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": content})
    return {"answer": None, "error": "max_steps_without_submit", "trace": trace, "stats": st, "steps": MAX_STEPS}


def run(arch: str, llm: LLM, query: str, session: str):
    if arch == "R0":  # línea base sin LLM
        from rules_baseline import run_rules
        return run_rules(query)
    cfg = ARCHS[arch]
    if cfg["kind"] == "single":
        return run_single(llm, query, cfg["resolve"], session)
    return run_agent(llm, query, cfg["tools"], session, cfg.get("prompt"))
