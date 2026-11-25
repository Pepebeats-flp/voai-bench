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
INK = "#222222"
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B", "red_strong": "#B64342",
           "red_2": "#E9A6A1", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}
# Semántica: A0 = línea base (rojo), A1 = intermedio (teal), A2 = propuesto (azul); ablaciones violeta/neutral.
ARCH_COLOR = {"A0": PALETTE["red_strong"], "A1": PALETTE["teal"], "A2": PALETTE["blue_main"],
              "A2-noschema": PALETTE["violet"], "A2-nores": PALETTE["neutral"], "A2-noresolve": PALETTE["neutral"]}
ARCH_LABEL = {"A0": "A0 VOAI-Plan", "A1": "A1 VOAI-Ground", "A2": "A2 VOAI-Agent",
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


# ------------------------------------------------------------------ figuras adicionales
PROT_COLOR = {"sia": PALETTE["blue_main"], "scs": PALETTE["teal"], "tap": PALETTE["red_strong"],
              "ssa": PALETTE["violet"], "none": PALETTE["neutral"]}
PROT_SHORT = {"sia": "Images (SIA2)", "scs": "Cone search (SCS)", "tap": "ADQL (TAP)", "ssa": "Spectra (SSA)",
              "none": "Out of scope"}


def fig_benchmark():
    """Panorama de VOAI-Bench v1 (157 tareas válidas): posiciones en el cielo y tareas por protocolo y dificultad."""
    import json
    import yaml
    from astropy.coordinates import SkyCoord
    gold = json.loads((ROOT / "bench/gold_v1.json").read_text())
    qs = yaml.safe_load((ROOT / "bench/queries_v1.yaml").read_text())
    qs = qs if isinstance(qs, list) else (qs.get("items") or qs.get("queries"))
    diff = {q["id"]: q["difficulty"] for q in qs}
    valid = {k: v for k, v in gold.items() if not v.get("invalid")}
    fig = plt.figure(figsize=(7.4, 3.0))
    ax = fig.add_axes([0.0, 0.08, 0.6, 0.84], projection="aitoff")
    ax.grid(True, color="#DDDDDD", lw=0.6)
    for p in ["sia", "scs", "tap", "ssa"]:
        pts = [(v["ra"], v["dec"]) for v in valid.values() if v["protocol"] == p and "ra" in v]
        if not pts:
            continue
        c = SkyCoord([a for a, _ in pts], [d for _, d in pts], unit="deg").galactic
        lon = -c.l.wrap_at("180d").radian  # longitud galáctica creciendo hacia la izquierda
        ax.scatter(lon, c.b.radian, s=16, color=PROT_COLOR[p], edgecolor="white", lw=0.4, label=PROT_SHORT[p],
                   zorder=3, alpha=0.9)
    ax.set_xticklabels([f"{x}°" for x in (150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210)], fontsize=6.5,
                       color="#666666")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("(a) Task targets, Galactic coordinates", fontsize=9.5, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=7.2, handletextpad=0.2,
              columnspacing=0.8)
    ax2 = fig.add_axes([0.76, 0.20, 0.23, 0.66])
    prots = ["sia", "tap", "scs", "ssa", "none"]
    shades = {1: "#C9D9EC", 2: "#6E9AC9", 3: "#0F4D92"}
    left = np.zeros(len(prots))
    for dlev in (1, 2, 3):
        vals = np.array([sum(1 for k, v in valid.items() if v["protocol"] == p and diff[k] == dlev) for p in prots])
        ax2.barh(np.arange(len(prots)), vals, left=left, color=shades[dlev], edgecolor="white", lw=0.6,
                 label=f"Difficulty {dlev}")
        for i, (l0, v) in enumerate(zip(left, vals)):
            if v >= 6:
                ax2.text(l0 + v / 2, i, str(v), ha="center", va="center", fontsize=7,
                         color="white" if dlev > 1 else INK)
        left += vals
    for i, t in enumerate(left):
        ax2.text(t + 0.8, i, f"{int(t)}", va="center", fontsize=7.5, fontweight="bold")
    ax2.set_yticks(np.arange(len(prots)), [PROT_SHORT[p] for p in prots], fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Tasks (each in EN and ES)", fontsize=8.5)
    ax2.grid(axis="x", color="#E6E6E6"); ax2.grid(axis="y", visible=False)
    ax2.set_xlim(0, max(left) * 1.15)
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.set_title("(b) Tasks by protocol\nand difficulty", fontsize=9.5)
    ax2.legend(loc="lower right", fontsize=6.8, handlelength=1.0)
    save(fig, "fig_benchmark")


def fig_models_lang(A, R0):
    """(a) éxito por modelo en A0→A1→A2; (b) inglés vs. español por sistema, con la línea base de reglas."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    archs = ["A0", "A1", "A2"]
    mk = {"gpt-6-luna": "o", "claude-haiku-5-5": "s", "deepseek-v4-pro": "D", "qwen3.8-flash": "^"}
    shade = {"gpt-6-luna": "#222222", "claude-haiku-5-5": "#555555", "deepseek-v4-pro": "#888888",
             "qwen3.8-flash": "#AAAAAA"}
    main = {(r["model"], r["arch"]): r for r in A["main"]}
    for m in A["meta"]["models"]:
        ys = [100 * main[(m, a)]["success"] for a in archs]
        ax.plot(range(3), ys, marker=mk.get(m, "o"), color=shade.get(m, "#444444"), lw=1.3, ms=5.5,
                label=MODEL_NAME.get(m, m))
    for i, a in enumerate(archs):
        ax.axvspan(i - 0.18, i + 0.18, color=ARCH_COLOR[a], alpha=0.10, lw=0)
    r0 = 100 * R0["by_arch"][0]["success"]
    ax.axhline(r0, color=PALETTE["neutral"], ls="--", lw=1.2)
    ax.text(-0.3, r0 + 1.5, "rule-based R0 (all languages)", fontsize=7.5, color="#777777", ha="left")
    ax.set_xticks(range(3), [ARCH_LABEL[a].replace(" ", "\n", 1) for a in archs], fontsize=8.5)
    ax.set_xlim(-0.35, 2.35); ax.set_ylim(30, 100)
    ax.set_ylabel("Strict success (%)")
    ax.set_title("(a) Per model", fontsize=10)
    ax.legend(fontsize=7.2, loc="lower right")
    systems = [("R0", R0), ("A0", A), ("A1", A), ("A2", A)]
    x = np.arange(len(systems))
    for j, (lang, hatch) in enumerate((("en", None), ("es", "////"))):
        vals = []
        for s_, src in systems:
            row = next(r for r in src["by_lang"] if r["lang"] == lang and r["arch"] == s_)
            vals.append(100 * row["success"])
        cols = [PALETTE["neutral"] if s_ == "R0" else ARCH_COLOR[s_] for s_, _ in systems]
        ax2.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=cols if j == 0 else "white", edgecolor=cols, hatch=hatch,
                lw=1.0, label="English" if j == 0 else "Spanish")
        for xi, v in zip(x + (j - 0.5) * 0.38, vals):
            ax2.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.8)
    ax2.set_xticks(x, ["R0\nrules", "A0\nPlan", "A1\nGround", "A2\nAgent"], fontsize=8.5)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Strict success (%)")
    ax2.set_title("(b) English vs. Spanish", fontsize=10)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(fc="#777777", ec="#777777", label="English"),
                        Patch(fc="white", ec="#777777", hatch="////", label="Spanish")], fontsize=7.2,
               loc="upper left", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_models_lang")


def fig_ablation_protocol(ABL):
    prots = ["sia", "scs", "tap", "ssa", "none"]
    archs = ["A2", "A2-noschema", "A2-nores"]
    cols = {"A2": PALETTE["blue_main"], "A2-noschema": PALETTE["violet"], "A2-nores": PALETTE["neutral"]}
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    w = 0.26
    for i, a in enumerate(archs):
        rows = {r["protocol"]: r for r in ABL["by_protocol"] if r["arch"] == a}
        y = np.array([100 * rows[p]["success"] for p in prots])
        lo = np.array([100 * rows[p]["ci_lo"] for p in prots]); hi = np.array([100 * rows[p]["ci_hi"] for p in prots])
        xs = np.arange(len(prots)) + (i - 1) * w
        ax.bar(xs, y, w * 0.92, color=cols[a], edgecolor="white", lw=0.6, label=ARCH_LABEL[a])
        ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax.set_xticks(np.arange(len(prots)), [PROT_LABEL[p] for p in prots], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Strict success (%)")
    ax.legend(ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, "fig_ablation_protocol")


def fig_passk(REL):
    archs = ["A0", "A1", "A2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, m in zip(axes, ["gpt-6-luna", "claude-haiku-5-5"]):
        for a in archs:
            r = next(x for x in REL["passk"] if x["model"] == m and x["arch"] == a)
            ks = list(range(1, r["n_reps"] + 1))
            ax.plot(ks, [100 * r[f"pass^{k}"] for k in ks], "-o", color=ARCH_COLOR[a], ms=5, lw=1.6,
                    label=ARCH_LABEL[a])
            ax.plot(ks[-1], 100 * r[f"pass@{ks[-1]}"], marker="*", ms=8, color=ARCH_COLOR[a], ls="none")
        ax.set_xticks(ks, [f"k = {k}" for k in ks])
        ax.set_xlim(0.7, ks[-1] + 0.3)
        ax.set_title(MODEL_NAME[m], fontsize=10)
    axes[0].set_ylabel("pass^k (%)")
    axes[0].set_ylim(60, 100)
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].plot([], [], "*", color="#555555", label="pass@3"); axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_passk")


def main():
    jsonl = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results/test.jsonl")
    style()
    A = analysis.analyze(jsonl, reps=[0], archs_keep=["A0", "A1", "A2"])  # mismo subconjunto que el análisis principal
    print(f"figuras desde {jsonl} ({A['meta']['n']} episodios):")
    fig_protocol(A)
    fig_pareto(A)
    fig_forest(A)
    fig_taxonomy(A)
    fig_benchmark()
    R0 = analysis.analyze(jsonl, reps=[0], archs_keep=["R0"])
    fig_models_lang(A, R0)
    ABL = analysis.analyze(jsonl, reps=[0], archs_keep=["A2", "A2-noschema", "A2-nores"],
                           models_keep=["gpt-6-luna", "qwen3.8-flash"])
    fig_ablation_protocol(ABL)
    REL = analysis.analyze(jsonl, archs_keep=["A0", "A1", "A2"], models_keep=["gpt-6-luna", "claude-haiku-5-5"])
    fig_passk(REL)


if __name__ == "__main__":
    main()
