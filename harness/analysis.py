"""Análisis estadístico de una corrida VOAI-Bench (devuelve un dict JSON-serializable).

Las métricas y pruebas siguen la literatura (ver bib/metricas_estado_del_arte.md):
- Primaria: éxito estricto basado en ejecución (Spider/BIRD EX; ALeRCE Perfect Match).
- Sensibilidad planificada: éxito tolerante al objetivo ("possible answers", BFCL).
- Unidad estadística independiente = ítem (Miller 2024): las repeticiones/idiomas/modelos se promedian
  por ítem antes de calcular errores; IC95 por bootstrap de clúster; diferencias pareadas por ítem con
  prueba de permutación por cambio de signo (cf. ALeRCE: permutación + Holm); Friedman entre modelos;
  GEE logística con clúster por ítem; pass^k (τ-bench) cuando hay repeticiones; Pareto costo–éxito (Kapoor).
"""
from __future__ import annotations

import itertools
import json
import math
import warnings
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 12345  # cada cálculo usa su propio generador: resultados independientes del orden de ejecución
INFRA = ("infra_error", "quota_exhausted", "RateLimitError", "APIConnectionError", "APITimeoutError",
         "InternalServerError", "ServiceUnavailableError")
ARCH_ORDER = ["R0", "A0", "A1", "A2", "A2-noschema", "A2-nores", "A2-noresolve"]

try:
    import yaml
    _TOL = yaml.safe_load((ROOT / "bench/tolerances.yaml").read_text())
except Exception:
    _TOL = {"default": 0.016667, "items": {}}


_GOLD_TOL = {}
for _g in ("gold_v0.json", "gold_v1.json"):
    try:
        for _k, _v in json.loads((ROOT / "bench" / _g).read_text()).items():
            if _v.get("tol_deg"):
                _GOLD_TOL[_k] = float(_v["tol_deg"])
    except Exception:
        pass


try:
    _PRICES = yaml.safe_load((ROOT / "bench/prices.yaml").read_text())["models"]
except Exception:
    _PRICES = {}


def official_cost(model, tok_in, tok_out, recorded):
    """Costo con precios de lista oficiales (bench/prices.yaml); si el modelo no está, el costo registrado."""
    p = _PRICES.get(model.split("/", 1)[-1])
    if not p:
        return recorded
    return (tok_in or 0) / 1e6 * p["in"] + (tok_out or 0) / 1e6 * p["out"]


def _tol_as(item):
    """Tolerancia del ítem: la del gold (v1) o la de tolerances.yaml (v0); por defecto 1 arcmin."""
    if item in _GOLD_TOL:
        return 3600 * _GOLD_TOL[item]
    return 3600 * float((_TOL.get("items") or {}).get(item, _TOL.get("default", 0.016667)))


# ------------------------------------------------------------------ carga
def load(path) -> tuple[pd.DataFrame, dict]:
    from jsonl_io import latest_records
    latest = latest_records(path)
    rows, n_infra = [], 0
    for r in latest.values():
        if str(r.get("error") or "").startswith(INFRA):
            n_infra += 1
            continue
        s = r.get("score") or {}
        a = r.get("answer") or {}
        rows.append({
            "model": r["model"].split("/", 1)[-1], "arch": r["arch"], "item": r["item"], "protocol": r["protocol"],
            "category": r["category"], "difficulty": int(r["difficulty"]), "lang": r["lang"], "rep": r["rep"],
            "success": bool(s.get("success")), "tolerant": bool(s.get("success_tolerant", s.get("success"))),
            "plan_ok": bool(s.get("plan_ok")), "protocol_ok": bool(s.get("protocol_ok")),
            "coord_err_as": s.get("coord_err_as"), "center_err_as": s.get("center_err_as"),
            "hallucinated": bool(s.get("hallucinated")), "false_abstain": bool(s.get("false_abstain")),
            "exec_error": bool(s.get("exec_error")), "no_answer": bool(s.get("no_answer")) or not r.get("answer"),
            "jaccard": s.get("jaccard"), "has_url": bool(a.get("url")),
            "error": r.get("error"), "wall_s": r["wall_s"], "llm_s": r.get("llm_s") or 0.0, "tool_s": r.get("tool_s") or 0.0,
            "tok_in": r["tok_in"], "tok_out": r["tok_out"], "tok_reason": r["tok_reason"],
            "cost_usd": official_cost(r["model"], r["tok_in"], r["tok_out"], r["cost_usd"]),
            "llm_calls": r["llm_calls"], "n_tool_calls": r["n_tool_calls"], "n_tool_errors": r.get("n_tool_errors", 0),
        })
    return pd.DataFrame(rows), {"n_unique": len(latest), "n_infra_excluded": n_infra}


# ------------------------------------------------------------------ estadística
def cluster_boot(df, col="success", B=2000):
    """Media e IC95% re-muestreando ítems (bootstrap de clúster; Miller 2024)."""
    per = df.groupby("item")[col].mean().values
    if len(per) < 2:
        return (float(df[col].mean()) if len(df) else None), None, None
    rng = np.random.default_rng(SEED)
    b = per[rng.integers(0, len(per), size=(B, len(per)))].mean(axis=1)
    return float(df[col].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(p):
    p = np.asarray(p, float)
    if not len(p):
        return p
    order, m, run = np.argsort(p), len(p), 0.0
    adj = np.empty_like(p)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def signflip_test(d, B=20000):
    """Prueba de permutación por cambio de signo sobre diferencias pareadas por ítem (dos colas)
    + IC95 bootstrap de la diferencia media. Exacta si n ≤ 16."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return None, None, None, 1.0
    obs = abs(d.mean())
    if np.allclose(d, 0):
        p = 1.0
    elif n <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
        p = float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())
    else:
        signs = np.random.default_rng(SEED).choice([-1, 1], size=(B, n))
        p = float(((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum() + 1) / (B + 1))
    boots = d[np.random.default_rng(SEED + 1).integers(0, n, size=(4000, n))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), p


def paired_by_item(df, factor, a, b, col="success"):
    """Promedia por ítem bajo cada nivel y devuelve las diferencias b − a por ítem (solo ítems con ambos)."""
    m = df[df[factor].isin([a, b])].groupby(["item", factor])[col].mean().unstack()
    if a not in m.columns or b not in m.columns:  # p. ej. ablaciones solo en algunos modelos
        return np.array([]), pd.DataFrame(columns=[a, b])
    m = m.dropna()
    return (m[b] - m[a]).values, m


def items_needed(sd, delta, alpha=0.05, power=0.8):
    """Ítems necesarios para detectar Δ con diferencias pareadas por ítem (aprox. normal; Miller 2024, ec. 9)."""
    if sd is None or not sd or delta <= 0:
        return None
    return int(math.ceil(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / delta) ** 2))


# ------------------------------------------------------------------ taxonomía de fallos
def classify(r) -> str:
    """Primera causa de fallo (orden de precedencia fijo; separa ejecución vs. semántica como ALeRCE)."""
    if r["success"]:
        return "éxito"
    if r["protocol"] == "none":
        return "no se abstuvo (fuera de alcance)"
    if r["no_answer"]:
        return "sin respuesta / formato inválido"
    if r["false_abstain"]:
        return "abstención indebida"
    if r["hallucinated"]:
        return "URL alucinada"
    if r["exec_error"]:
        return "ADQL no ejecuta"
    err = r["center_err_as"] if r["center_err_as"] is not None else r["coord_err_as"]
    if err is not None and not (isinstance(err, float) and math.isnan(err)) and err > _tol_as(r["item"]):
        return "objetivo/coordenadas incorrectas"
    if r["tolerant"]:
        return "centro distinto (válido solo con tolerancia)"
    if r["protocol"] in ("sia", "ssa") and not r["has_url"]:
        return "no encontró recurso"
    if r["protocol"] in ("sia", "ssa"):
        return "banda o recurso incorrecto"
    return "resultado de consulta incorrecto"


# ------------------------------------------------------------------ análisis principal
def analyze(path, reps=None, archs_keep=None, models_keep=None, metric="strict", exclude=None) -> dict:
    """reps/archs_keep/models_keep filtran el diseño analizado (p. ej. análisis principal = rep 0, A0–A2).
    metric="tolerant" repite todo el análisis con la métrica tolerante (sensibilidad)."""
    df, meta = load(path)
    if exclude:  # lista de (modelo, idioma) a excluir (p. ej. episodios de la prueba de humo)
        for m_, l_ in exclude:
            df = df[~((df.model == m_.split("/", 1)[-1]) & (df.lang == l_))]
    if metric == "tolerant":
        df["success"], df["tolerant"] = df["tolerant"], df["success"]
    if reps is not None:
        df = df[df.rep.isin(reps)]
    if archs_keep is not None:
        df = df[df.arch.isin(archs_keep)]
    if models_keep is not None:
        df = df[df.model.isin([m.split("/", 1)[-1] for m in models_keep])]
    if df.empty:
        return {"empty": True, **meta}
    archs = [a for a in ARCH_ORDER if a in set(df.arch)] + sorted(set(df.arch) - set(ARCH_ORDER))
    models = sorted(df.model.unique())
    answerable = df[df.protocol != "none"]
    oos = df[df.protocol == "none"]
    out = {"meta": {**meta, "n": len(df), "models": models, "archs": archs, "items": int(df.item.nunique()),
                    "langs": sorted(df.lang.unique()), "reps": int(df.rep.nunique()),
                    "cost_total": float(df.cost_usd.sum())}}

    # 1) tabla principal modelo × arquitectura
    main = []
    for (m, a), g in df.groupby(["model", "arch"]):
        mu, lo, hi = cluster_boot(g)
        ce = g["coord_err_as"].dropna()
        ga, go = g[g.protocol != "none"], g[g.protocol == "none"]
        calls = g.n_tool_calls.sum()
        succ = g[g.success]
        main.append({"model": m, "arch": a, "n": len(g), "success": mu, "ci_lo": lo, "ci_hi": hi,
                     "tolerant": float(g.tolerant.mean()), "plan_ok": float(g.plan_ok.mean()),
                     "protocol_acc": float(g.protocol_ok.mean()),
                     "coord_ok": float((ce <= 60).mean()) if len(ce) else None,
                     "coord_med_as": float(ce.median()) if len(ce) else None,
                     "irrelevance": float(go.success.mean()) if len(go) else None,
                     "false_abstain": float(ga.false_abstain.mean()) if len(ga) else None,
                     "hallucinated": float(g.hallucinated.mean()),
                     "tool_validity": float(1 - g.n_tool_errors.sum() / calls) if calls else None,
                     "cost_per_100": float(100 * g.cost_usd.mean()),
                     "cost_per_success": float(g.cost_usd.sum() / len(succ)) if len(succ) else None,
                     "wall_med": float(g.wall_s.median()), "llm_med": float(g.llm_s.median()),
                     "tool_med": float(g.tool_s.median()),
                     "tok_in_mean": float(g.tok_in.mean()), "tok_out_mean": float(g.tok_out.mean()),
                     "tok_reason_mean": float(g.tok_reason.mean()),
                     "llm_calls": float(g.llm_calls.mean()), "tool_calls": float(g.n_tool_calls.mean()),
                     "steps_success": float(succ.llm_calls.mean()) if len(succ) else None})
    out["main"] = main

    # 2) desgloses
    def grp(keys, data=df):
        res = []
        for k, g in data.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            mu, lo, hi = cluster_boot(g)
            res.append({**dict(zip(keys, [int(x) if isinstance(x, np.integer) else x for x in k])),
                        "n": len(g), "success": mu, "ci_lo": lo, "ci_hi": hi, "tolerant": float(g.tolerant.mean()),
                        "plan_ok": float(g.plan_ok.mean())})
        return res
    out["by_arch"] = grp(["arch"])
    out["by_protocol"] = grp(["protocol", "arch"])
    out["by_difficulty"] = grp(["difficulty", "arch"])
    out["by_lang"] = grp(["lang", "arch"])
    out["by_model"] = grp(["model"])
    sec = []
    for a in archs:
        g = df[df.arch == a]
        ce = g["coord_err_as"].dropna()
        calls = g.n_tool_calls.sum()
        sec.append({"arch": a, "protocol_acc": float(g.protocol_ok.mean()), "plan_ok": float(g.plan_ok.mean()),
                    "hallucinated": float(g.hallucinated.mean()),
                    "coord_med_as": float(ce.median()) if len(ce) else None,
                    "coord_iqr_as": [float(ce.quantile(.25)), float(ce.quantile(.75))] if len(ce) else None,
                    "coord_ok_1arcmin": float((ce <= 60).mean()) if len(ce) else None,
                    "tool_validity": float(1 - g.n_tool_errors.sum() / calls) if calls else None,
                    "steps_success": float(g[g.success].llm_calls.mean()) if g.success.any() else None,
                    "jaccard_scs": float(g[g.protocol == "scs"].jaccard.dropna().mean()) if g[g.protocol == "scs"].jaccard.notna().any() else None})
    out["secondary"] = sec
    out["abstention"] = [{"arch": a, "irrelevance": float(oos[oos.arch == a].success.mean()) if len(oos[oos.arch == a]) else None,
                          "false_abstain": float(answerable[answerable.arch == a].false_abstain.mean())} for a in archs]

    # 3) arquitectura: diferencias pareadas por ítem (permutación + Holm)
    arch_tests = []
    for a, b in itertools.combinations(archs, 2):
        for m in ["(todos)"] + models:
            sub = df if m == "(todos)" else df[df.model == m]
            d, mat = paired_by_item(sub, "arch", a, b)
            if not len(d):
                continue
            mean, lo, hi, p = signflip_test(d)
            arch_tests.append({"model": m, "A": a, "B": b, "items": len(d), "succ_A": float(mat[a].mean()),
                               "succ_B": float(mat[b].mean()), "delta": mean, "lo": lo, "hi": hi,
                               "items_B_better": int((d > 0).sum()), "items_A_better": int((d < 0).sum()), "p": p})
    for t, q in zip(arch_tests, holm([t["p"] for t in arch_tests])):
        t["p_holm"] = float(q)
    out["arch_tests"] = arch_tests

    # 4) modelos dentro de cada arquitectura: Friedman sobre éxito medio por ítem + pares con permutación (Holm)
    model_tests = []
    for a in archs:
        mat = df[df.arch == a].groupby(["item", "model"]).success.mean().unstack().dropna()
        if mat.shape[1] < 3 or len(mat) < 3:
            continue
        try:
            F, pf = friedmanchisquare(*[mat[c].values for c in mat.columns])
            F, pf = float(F), float(pf)
        except ValueError:
            F, pf = None, 1.0
        pairs = []
        for m1, m2 in itertools.combinations(mat.columns, 2):
            mean, lo, hi, p = signflip_test((mat[m2] - mat[m1]).values)
            pairs.append({"m1": m1, "m2": m2, "delta": mean, "lo": lo, "hi": hi, "p": p})
        for t, q in zip(pairs, holm([t["p"] for t in pairs])):
            t["p_holm"] = float(q)
        model_tests.append({"arch": a, "items": len(mat), "chi2": F, "p": pf, "pairs": pairs,
                            "ranking": [{"model": c, "success": float(mat[c].mean())} for c in mat.mean().sort_values(ascending=False).index]})
    out["model_tests"] = model_tests

    # 5) idioma: ES − EN pareado por ítem
    lang_tests = []
    if {"en", "es"} <= set(df.lang):
        for a in ["(todas)"] + archs:
            sub = df if a == "(todas)" else df[df.arch == a]
            d, mat = paired_by_item(sub, "lang", "en", "es")
            mean, lo, hi, p = signflip_test(d)
            lang_tests.append({"arch": a, "items": len(d), "en": float(mat["en"].mean()), "es": float(mat["es"].mean()),
                               "delta": mean, "lo": lo, "hi": hi, "p": p})
        for t, q in zip(lang_tests, holm([t["p"] for t in lang_tests])):
            t["p_holm"] = float(q)
    out["lang_tests"] = lang_tests

    # 6) sensibilidad pre-registrada: estricto vs. tolerante
    out["sensitivity"] = [{"arch": a, "strict": float(df[df.arch == a].success.mean()),
                           "tolerant": float(df[df.arch == a].tolerant.mean()),
                           "flipped": int((df[df.arch == a].tolerant & ~df[df.arch == a].success).sum()),
                           "n": int((df.arch == a).sum())} for a in archs]

    # 7) GEE logística con clúster por ítem
    try:
        import re
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        # Solo tareas respondibles: las fuera de alcance tienen ~100% de éxito en todas las condiciones
        # (separación perfecta); la abstención se reporta aparte (BFCL irrelevance).
        d = df[df.protocol != "none"].copy()
        d["y"] = d.success.astype(int)
        ref_model = max(models, key=lambda m: d[d.model == m].success.mean())
        terms = [f"C(arch, Treatment('{archs[0]}'))", f"C(model, Treatment('{ref_model}'))"]
        if d.lang.nunique() > 1:
            terms.append("C(lang, Treatment('en'))")
        if d.protocol.nunique() > 1:  # referencia: sia
            terms.append("C(protocol, Treatment('sia'))")
        if d.difficulty.nunique() > 1:
            terms.append("C(difficulty, Treatment(1))")
        f = "y ~ " + " + ".join(terms)
        fit = smf.gee(f, groups="item", data=d, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = fit.conf_int()
        coefs = []
        for name in fit.params.index:
            if name == "Intercept":
                continue
            mm = re.match(r"C\((\w+), Treatment\('?([^')]+)'?\)\)\[T\.(.+)\]", name)
            coefs.append({"term": f"{mm.group(1)}: {mm.group(3)} (ref. {mm.group(2)})" if mm else name,
                          "factor": mm.group(1) if mm else name,
                          "OR": float(np.exp(fit.params[name])), "lo": float(np.exp(ci.loc[name, 0])),
                          "hi": float(np.exp(ci.loc[name, 1])), "p": float(fit.pvalues[name])})
        out["gee"] = {"formula": f, "ref_model": ref_model, "coefs": coefs, "n": int(fit.nobs)}
    except Exception as e:
        out["gee"] = {"error": f"{type(e).__name__}: {e}"}

    # 8) Pareto costo–éxito
    pts = sorted(main, key=lambda r: (r["cost_per_100"], -r["success"]))
    best = -1.0
    for r in pts:
        r["pareto"] = r["success"] > best
        best = max(best, r["success"])
    out["pareto"] = [{"model": r["model"], "arch": r["arch"], "cost": r["cost_per_100"], "success": r["success"],
                      "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "wall": r["wall_med"], "pareto": r["pareto"]} for r in pts]

    # 9) fiabilidad pass@k / pass^k (τ-bench; Codex) si hay repeticiones
    passk = []
    if df.rep.nunique() > 1:
        for (m, a), g in df.groupby(["model", "arch"]):
            per = g.groupby(["item", "lang"]).success.agg(["sum", "count"]).values
            n_min = int(per[:, 1].min())
            rec = {"model": m, "arch": a, "n_reps": n_min}
            for k in range(1, n_min + 1):
                rec[f"pass^{k}"] = float(np.mean([comb(int(c), k) / comb(int(n), k) for c, n in per]))
                rec[f"pass@{k}"] = float(np.mean([1 - comb(int(n - c), k) / comb(int(n), k) if n - c >= k else 1.0
                                                  for c, n in per]))
            passk.append(rec)
    out["passk"] = passk
    rep_var = []
    if df.rep.nunique() > 1:
        for (m, a), g in df.groupby(["model", "arch"]):
            per_rep = g.groupby("rep").success.mean()
            per_task = g.groupby(["item", "lang"]).success.agg(lambda v: v.nunique() > 1)
            rep_var.append({"model": m, "arch": a, "reps": int(per_rep.size),
                            "success_by_rep": [float(x) for x in per_rep.values],
                            "range_pp": float(100 * (per_rep.max() - per_rep.min())),
                            "flaky_tasks": float(per_task.mean())})
    out["rep_variation"] = rep_var

    # 10) taxonomía de fallos
    df["cause"] = df.apply(classify, axis=1)
    tax = df[df.cause != "éxito"].groupby(["arch", "cause"]).size().reset_index(name="n")
    tot = df.groupby("arch").size()
    out["taxonomy"] = [{"arch": r.arch, "cause": r.cause, "n": int(r.n), "share": float(r.n / tot[r.arch])}
                       for r in tax.itertuples()]

    # 11) ítems más difíciles
    it = df.groupby(["item", "protocol"]).agg(success=("success", "mean"), tolerant=("tolerant", "mean"),
                                              n=("success", "size")).reset_index()
    out["items"] = it.sort_values("success").to_dict("records")

    # 12) planificación: ítems necesarios (diferencia pareada por ítem) para Δ con 80% de potencia
    plan = []
    for a, b in itertools.combinations(archs[:3], 2):
        d, _ = paired_by_item(df, "arch", a, b)
        if len(d) < 3:
            continue
        sd = float(np.std(d, ddof=1))
        plan.append({"A": a, "B": b, "sd_item_diff": sd, "items_now": len(d),
                     **{f"n_delta_{int(x * 100)}": items_needed(sd, x) for x in (0.03, 0.05, 0.10)}})
    out["power_plan"] = plan
    hw = [(r["ci_hi"] - r["ci_lo"]) / 2 for r in main if r["ci_hi"] is not None]
    out["ci_halfwidth_mean"] = float(np.mean(hw)) if hw else None
    return out


if __name__ == "__main__":
    import sys
    res = analyze(sys.argv[1])
    print(json.dumps({k: res[k] for k in ("meta", "by_arch", "sensitivity", "power_plan")}, indent=1, default=str))
