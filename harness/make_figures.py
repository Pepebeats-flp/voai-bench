"""Figuras del paper (PDF vectorial + PNG 300 dpi) desde una corrida, con estilo de publicación.

Uso: python -I harness/make_figures.py results/test.jsonl   -> paper/figures/*.pdf|png
Todas las cifras salen de harness/analysis.py (las mismas que experimental_log.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import analysis  # noqa: E402

OUT = ROOT / "paper/figures"
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B", "red_strong": "#B64342",
           "red_2": "#E9A6A1", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}
# Semántica: A0 = línea base (rojo), A1 = intermedio (teal), A2 = propuesto (azul); ablaciones violeta/neutral.
ARCH_COLOR = {"A0": PALETTE["red_strong"], "A1": PALETTE["teal"], "A2": PALETTE["blue_main"],
              "A2-noschema": PALETTE["violet"], "A2-nores": PALETTE["neutral"], "A2-noresolve": PALETTE["neutral"]}
ARCH_LABEL = {"A0": "A0 single-shot", "A1": "A1 + resolver", "A2": "A2 tool agent",
              "A2-noschema": "A2 w/o schema", "A2-nores": "A2 w/o resolver", "A2-noresolve": "A2 w/o resolver (unadapted prompt)"}
PROT_LABEL = {"sia": "Images\n(SIA2)", "scs": "Cone search\n(SCS)", "tap": "ADQL\n(TAP)", "ssa": "Spectra\n(SSA)",
              "none": "Out of\nscope"}


MODEL_NAME = {"claude-haiku-5-5": "Claude Haiku 5.5", "deepseek-v4-pro": "DeepSeek V4 Pro",
              "gpt-6-luna": "GPT-6 Luna", "qwen3.8-flash": "Qwen 3.8 Flash", "baseline": "Rule-based"}
CAUSE_EN = {"objetivo/coordenadas incorrectas": "Wrong target / coordinates",
            "centro distinto (válido solo con tolerancia)": "Approximate centre (tolerant only)",
            "banda o recurso incorrecto": "Wrong band or product",
            "ADQL no ejecuta": "ADQL does not execute",
            "resultado de consulta incorrecto": "Executable query, wrong result",
            "no encontró recurso": "No product found",
            "abstención indebida": "False abstention",
            "sin respuesta / formato inválido": "No answer / invalid format",
            "no se abstuvo (fuera de alcance)": "Answered out-of-scope request",
            "URL alucinada": "Hallucinated URL"}


def pretty_term(t):
    for k, v in MODEL_NAME.items():
        t = t.replace(k, v)
    return (t.replace("arch: ", "Architecture ").replace("model: ", "").replace("lang: es", "Spanish")
             .replace("(ref. en)", "(ref. English)").replace("protocol: ", "Protocol ").replace("difficulty: ", "Difficulty ")
             .replace("scs", "SCS").replace("ssa", "SSA").replace("tap", "TAP").replace("sia", "SIA2"))


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"], "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.4,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.frameon": False,
        "legend.fontsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E6E6E6", "grid.linewidth": 0.8, "axes.axisbelow": True})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, pad_inches=0.04)
    plt.close(fig)
    print("  ", OUT / f"{name}.pdf")


def fig_protocol(A):
    archs = A["meta"]["archs"]
    prots = [p for p in ["sia", "scs", "tap", "ssa", "none"] if any(r["protocol"] == p for r in A["by_protocol"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    w = 0.8 / len(archs)
    for i, a in enumerate(archs):
        rows = [next((r for r in A["by_protocol"] if r["protocol"] == p and r["arch"] == a), None) for p in prots]
        x = np.arange(len(prots)) + (i - (len(archs) - 1) / 2) * w
        y = [100 * r["success"] if r else np.nan for r in rows]
        lo = [100 * (r["success"] - r["ci_lo"]) if r and r["ci_lo"] is not None else 0 for r in rows]
        hi = [100 * (r["ci_hi"] - r["success"]) if r and r["ci_hi"] is not None else 0 for r in rows]
        ax.bar(x, y, w * 0.92, color=ARCH_COLOR[a], label=ARCH_LABEL[a], edgecolor="white", linewidth=0.6)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=1, capsize=2.5)
    ax.set_xticks(range(len(prots)), [PROT_LABEL[p] for p in prots])
    ax.set_ylabel("Strict success (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=len(archs), loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig_success_by_protocol")


def fig_pareto(A):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    pts = A["pareto"]
    for p in pts:
        c = max(p["cost"], 1e-4)
        ax.errorbar(c, 100 * p["success"], yerr=[[100 * (p["success"] - (p["ci_lo"] or p["success"]))],
                                                  [100 * ((p["ci_hi"] or p["success"]) - p["success"])]],
                    fmt="o", color=ARCH_COLOR[p["arch"]], ms=7 if p["pareto"] else 5.5, mec="white", mew=1,
                    elinewidth=0.9, ecolor=ARCH_COLOR[p["arch"]] + "4D")
    fr = sorted([p for p in pts if p["pareto"]], key=lambda p: p["cost"])
    ax.plot([max(p["cost"], 1e-4) for p in fr], [100 * p["success"] for p in fr], ls="--", color="#555555", lw=1, zorder=0)
    groups = []  # agrupar puntos de la frontera del mismo modelo con costo casi igual (etiquetas superpuestas)
    for p in fr:
        if groups and groups[-1][0]["model"] == p["model"] and abs(groups[-1][0]["cost"] - p["cost"]) / p["cost"] < 0.1:
            groups[-1].append(p)
        else:
            groups.append([p])
    for k, g in enumerate(groups):
        p = max(g, key=lambda q: q["success"])
        archs_lbl = ", ".join(q["arch"] for q in sorted(g, key=lambda q: q["arch"]))
        ax.annotate(f"{MODEL_NAME.get(p['model'], p['model'])} ({archs_lbl})", (max(p["cost"], 1e-4), 100 * p["success"]),
                    xytext=(-8, 6) if p["success"] > 0.9 else (7, -16 if k % 2 == 0 else 8),
                    textcoords="offset points", fontsize=8, color="#222222", ha="right" if p["success"] > 0.9 else "left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    xs = [max(p["cost"], 1e-4) for p in pts]
    ax.set_xlim(min(xs) / 1.4, max(xs) * 3.2)
    ax.set_xlabel("List-price cost per 100 requests (USD, log scale)")
    ax.set_ylabel("Strict success (%)")
    ax.grid(True, axis="both")
    for a in A["meta"]["archs"]:
        ax.plot([], [], "o", color=ARCH_COLOR[a], label=ARCH_LABEL[a])
    ax.legend(loc="lower right")
    save(fig, "fig_cost_pareto")


def fig_forest(A):
    if not A["gee"].get("coefs"):
        return
    co = A["gee"]["coefs"]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(co) + 0.8))
    y = np.arange(len(co))[::-1]
    for yi, c in zip(y, co):
        sig = c["p"] < 0.05
        col = PALETTE["blue_main"] if c["OR"] >= 1 else PALETTE["red_strong"]
        ax.plot([c["lo"], c["hi"]], [yi, yi], color=col, lw=1.6)
        ax.plot(c["OR"], yi, "o", color=col if sig else "white", mec=col, mew=1.6, ms=6)
    ax.axvline(1, color="#555555", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y, [pretty_term(c["term"]) for c in co], fontsize=8.5)
    ax.set_xlabel("Odds ratio of strict success (95% CI, GEE clustered by task)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    save(fig, "fig_gee_forest")


def fig_taxonomy(A):
    archs = A["meta"]["archs"]
    causes = sorted({t["cause"] for t in A["taxonomy"]},
                    key=lambda c: -sum(t["n"] for t in A["taxonomy"] if t["cause"] == c))
    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(causes) * len(archs) / 2 + 1.2))
    h = 0.8 / len(archs)
    for i, a in enumerate(archs):
        vals = [100 * next((t["share"] for t in A["taxonomy"] if t["arch"] == a and t["cause"] == c), 0) for c in causes]
        ax.barh(np.arange(len(causes)) + (i - (len(archs) - 1) / 2) * h, vals, h * 0.9, color=ARCH_COLOR[a],
                label=ARCH_LABEL[a])
    ax.set_yticks(range(len(causes)), [CAUSE_EN.get(c, c) for c in causes], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Share of episodes (%)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="lower right")
    save(fig, "fig_failure_taxonomy")


def fig_sensitivity(A):
    s = A["sensitivity"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(s))
    ax.bar(x - 0.19, [100 * r["strict"] for r in s], 0.36, color=[ARCH_COLOR[r["arch"]] for r in s], label="Strict")
    ax.bar(x + 0.19, [100 * r["tolerant"] for r in s], 0.36, color="white", edgecolor=[ARCH_COLOR[r["arch"]] for r in s],
           hatch="///", linewidth=1.2, label="Target-tolerant")
    ax.set_xticks(x, [r["arch"] for r in s])
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    save(fig, "fig_strict_vs_tolerant")


def main():
    jsonl = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results/test.jsonl")
    style()
    A = analysis.analyze(jsonl, reps=[0], archs_keep=["A0", "A1", "A2"])  # mismo subconjunto que el análisis principal
    print(f"figuras desde {jsonl} ({A['meta']['n']} episodios):")
    fig_protocol(A)
    fig_pareto(A)
    fig_forest(A)
    fig_taxonomy(A)


if __name__ == "__main__":
    main()
