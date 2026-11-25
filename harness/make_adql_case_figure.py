"""Caso de estudio ADQL (VT15, GPT-6 Luna, inglés, repetición 0): la misma consulta inválida en A0 (sin
retroalimentación) y en A2 (error -> revisión -> éxito). Textos tomados literalmente de results/test.jsonl.
Uso: python -I harness/make_adql_case_figure.py -> paper/figures/fig_adql_case.pdf|png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
from jsonl_io import iter_records  # noqa: E402

RED, BLUE, GREEN, INK, GREY = "#B64342", "#0F4D92", "#1f7a4d", "#222222", "#EDEDED"
ep = {}
for r in iter_records(ROOT / "results/test.jsonl", slim=False):
    if (r["item"], r["lang"], r["rep"]) == ("VT15", "en", 0) and r["model"].endswith("gpt-6-luna") and r["arch"] in ("A0", "A2"):
        ep[r["arch"]] = r
assert set(ep) == {"A0", "A2"}
a0_adql = ep["A0"]["answer"]["adql"]
a2_calls = [t for t in ep["A2"]["trace"]]
assert [t["tool"] for t in a2_calls] == ["resolve_name", "tap_query", "tap_query"]
assert not ep["A0"]["score"]["success"] and ep["A2"]["score"]["success"]


def short(adql):
    """Abrevia la parte común (cláusula WHERE) para que quepa; la cláusula ORDER BY se deja literal."""
    sel, rest = adql.split(" FROM ", 1)
    order = rest.split(" ORDER BY ", 1)[1]
    sel = sel.replace("SELECT TOP 3 source_id, ra, dec, pmra, pmdec, ", "SELECT TOP 3 source_id, …,\n  ")
    return f"{sel}\nFROM gaiadr3.gaia_source\nWHERE CONTAINS(…, CIRCLE(…, 0.5))=1 …\nORDER BY {order}"


plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(16, 100); ax.axis("off")


def box(x, y, w, h, ec, fc="white", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=lw))


def mono(x, y, s, color=INK, size=6.3):
    ax.text(x, y, s, fontsize=size, family="monospace", va="top", color=color, linespacing=1.3)


def arrow(x0, y0, x1, y1, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color))


def badge(x, y, ok, text):
    ax.text(x, y, ("✓ " if ok else "✗ ") + text, fontsize=7, color="white", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=GREEN if ok else RED, ec="none"))


# Solicitud
box(8, 92, 84, 6, "#999999", GREY)
ax.text(50, 95, "“" + "Which 3 Gaia DR3 stars within 30 arcmin of Rigel have the highest total proper motion?" + "”",
        ha="center", va="center", fontsize=7.6, style="italic")
arrow(35, 91.5, 25, 86); arrow(65, 91.5, 75, 86)

# A0 — VOAI-Plan
ax.text(25, 84, "A0 · VOAI-Plan (one call, no feedback)", ha="center", fontsize=8.5, fontweight="bold", color=RED)
box(1, 56, 47, 25, RED)
ax.text(2.2, 79.6, "JSON plan → deterministic executor", fontsize=7, color="#444444")
mono(2.2, 76.6, short(a0_adql))
ax.text(2.2, 60.5, "ORDER BY on an expression: rejected by the\nGaia archive's ADQL parser", fontsize=6.8,
        color=RED, va="center")
arrow(24.5, 55.5, 24.5, 49.5, RED)
box(1, 37, 47, 11.5, RED, "#F7E3E3")
mono(2.2, 47.1, "DALQueryError: Cannot parse query\n'SELECT TOP 3 source_id, ra, dec, …'", RED)
badge(2.2, 39.8, False, "execution error; episode fails")

# A2 — VOAI-Agent
ax.text(75, 84, "A2 · VOAI-Agent (tool loop)", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
r0 = a2_calls[0]["out"]
steps = [
    (78, 6, BLUE, f"[1] resolve_name(\"Rigel\")\n   → {r0['main_id']}  ({r0['ra']:.4f}, {r0['dec']:.4f})", None),
    (69, 15, RED, "[2] tap_query(gaia_esa,\n" + short(a2_calls[1]["args"]["adql"]) + ")", (False, "same parse error returned")),
    (51, 15, GREEN, "[3] tap_query(gaia_esa,\n" + short(a2_calls[2]["args"]["adql"]) + ")", (True, "3 rows")),
    (33, 8.5, BLUE, "[4] submit_answer(protocol=tap, adql=[3])", (True, "same source set as gold")),
]
for y_top, h, ec, txt, b in steps:
    box(52, y_top - h, 47, h, ec, "#EAF4EE" if ec == GREEN else ("#F7E3E3" if ec == RED else "white"))
    mono(53.2, y_top - 0.8, txt)
    if b:
        bx = ax.text(98.2, y_top - 2.0 if h > 10 else y_top - h + 2.3, ("✓ " if b[0] else "✗ ") + b[1], fontsize=7,
                     color="white", va="center", ha="right",
                     bbox=dict(boxstyle="round,pad=0.25", fc=GREEN if b[0] else RED, ec="none"))
for ya, yb in ((71.7, 69.6), (53.7, 51.6), (35.7, 33.6)):
    arrow(75.5, ya, 75.5, yb, BLUE)
ax.text(53, 22.5, "The error message is returned to the model as an observation;\n"
        "it rewrites the ordering through the column alias and resubmits.", fontsize=7, color="#444444", va="top",
        style="italic")
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"paper/figures/fig_adql_case.{ext}", dpi=250, bbox_inches="tight")
print("ok")
