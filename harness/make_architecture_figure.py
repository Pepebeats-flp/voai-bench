"""Diagrama de las tres arquitecturas propuestas (VOAI-Plan, VOAI-Ground, VOAI-Agent), sin cifras.
Uso: python -I harness/make_architecture_figure.py -> paper/figures/fig_architecture.pdf|png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RED, TEAL, BLUE, INK, GREY = "#B64342", "#42949E", "#0F4D92", "#222222", "#EDEDED"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.4, 5.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, ec, fc="white", size=8, bold=False, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25,
            fontweight="bold" if bold else "normal", family="monospace" if mono else "sans-serif")


def arrow(x0, y0, x1, y1, two=False, text=None, dy=1.0, color=INK, style="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if two else "-|>", mutation_scale=9,
                                 lw=1.1, color=color, linestyle=style))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.8, color="#444444")


def lane(y0, color, title, subtitle):
    ax.add_patch(FancyBboxPatch((0.5, y0), 99, 29, boxstyle="round,pad=0.2,rounding_size=1.5", fc=color + "14",
                                ec=color, lw=1.2))
    ax.text(2.0, y0 + 26.2, title, fontsize=9.5, fontweight="bold", color=color, va="center")
    ax.text(2.0, y0 + 22.3, subtitle, fontsize=7.2, color="#444444", va="center")


# A0 — VOAI-Plan
y = 68
lane(y, RED, "A0 · VOAI-Plan", "single-shot structured planning")
box(2, y + 4, 13, 12, "Request\n(EN / ES)", "#999999", GREY)
box(20, y + 4, 17, 12, "LLM\n1 call", RED)
box(42, y + 4, 19, 12, "JSON plan\nprotocol · target ·\ncoords · band · ADQL", RED)
box(66, y + 4, 15, 12, "Deterministic\nexecutor\n(run once)", "#777777")
box(86, y + 4, 12.5, 12, "IVOA\nservices", "#999999", GREY)
arrow(15, y + 10, 20, y + 10); arrow(37, y + 10, 42, y + 10); arrow(61, y + 10, 66, y + 10); arrow(81, y + 10, 86, y + 10)
ax.text(51, y + 18.5, "coordinates and ADQL from the model's own knowledge; no feedback", ha="center", fontsize=6.8,
        color="#444444", style="italic")

# A1 — VOAI-Ground
y = 36
lane(y, TEAL, "A1 · VOAI-Ground", "planning with deterministic grounding")
box(2, y + 4, 13, 12, "Request\n(EN / ES)", "#999999", GREY)
box(20, y + 4, 17, 12, "LLM\n1 call", TEAL)
box(42, y + 3, 19, 10, "JSON plan\n+ object name", TEAL)
box(66, y + 4, 15, 12, "Deterministic\nexecutor\n(run once)", "#777777")
box(86, y + 4, 12.5, 12, "IVOA\nservices", "#999999", GREY)
box(42, y + 18.5, 19, 6, "Sesame resolver", TEAL, fc="#E6F2F3", size=7.5)
arrow(15, y + 10, 20, y + 10); arrow(37, y + 10, 42, y + 10); arrow(61, y + 10, 66, y + 10); arrow(81, y + 10, 86, y + 10)
arrow(51.5, y + 13.3, 51.5, y + 18.2, two=True)
ax.text(70, y + 22, "resolved coordinates override the model's", ha="left", fontsize=6.8, color="#444444", style="italic")

# A2 — VOAI-Agent
y = 2
lane(y, BLUE, "A2 · VOAI-Agent", "tool-calling agent with execution feedback")
box(2, y + 4, 13, 12, "Request\n(EN / ES)", "#999999", GREY)
box(20, y + 3, 17, 14, "LLM\nreason → call\n→ observe → revise\n(≤ 12 turns)", BLUE, size=7.5)
ax.add_patch(FancyBboxPatch((42, y + 2), 28, 17, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec=BLUE, lw=1.1))
for i, row in enumerate(["resolve_name    sia_search", "skyview_image   cone_search", "tap_query       describe_tables",
                         "ssa_search"]):
    ax.text(43.5, y + 13.6 - i * 3.3, row, fontsize=6.6, family="monospace", va="center")
ax.text(56, y + 17.2, "deterministic VO tools", ha="center", fontsize=7, color=BLUE, fontweight="bold")
box(74, y + 4, 11, 12, "IVOA\nservices", "#999999", GREY)
box(88.5, y + 4, 10, 12, "submit_\nanswer", BLUE, size=7.5)
arrow(15, y + 10, 20, y + 10)
arrow(37, y + 10, 42, y + 10, two=True)
arrow(70, y + 10, 74, y + 10, two=True)
ax.add_patch(FancyArrowPatch((32, y + 17), (93.5, y + 16), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>",
                             mutation_scale=9, lw=1.0, color=BLUE, linestyle="--"))
ax.text(83, y + 22.6, "final answer", fontsize=6.8, color=BLUE, ha="center", style="italic")
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"paper/figures/fig_architecture.{ext}", dpi=250, bbox_inches="tight")
print("ok")
