"""Figura cualitativa: misma consulta, producto devuelto por A0 / A1 / A2 (datos FITS reales devueltos).
Uso: python -I harness/make_example_figure.py   -> paper/figures/fig_examples.pdf|png"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "paper/figures/examples_data"
gold = json.loads((ROOT / "bench/gold_v1.json").read_text())
ROWS = [
    ("VS20", "DeepSeek V4 Pro", "“Can you find an H-alpha image of the Carina Nebula?”",
     {"A0": "DSS2 Red (0.59–0.72 µm)", "A1": "DSS2 Red (0.59–0.72 µm)", "A2": "SHASSA Hα (0.656 µm)"}),
    ("VS32", "GPT-6 Luna", "“Retrieve an X-ray image of Cygnus X-1, please.”",
     {"A0": "Spitzer GLIMPSE (4.5 µm)", "A1": "ROSAT RASS (0.1–2.4 keV)", "A2": "ROSAT RASS (0.1–2.4 keV)"}),
]
VERDICT = {("VS20", "A0"): (False, "approximate centre (10′ off), broadband"),
           ("VS20", "A1"): (True, "resolved centre, broadband red"),
           ("VS20", "A2"): (True, "resolved centre, Hα survey"),
           ("VS32", "A0"): (False, "wrong band; no data at target"),
           ("VS32", "A1"): (True, "X-ray"),
           ("VS32", "A2"): (True, "X-ray")}
ARCH = {"A0": "A0 VOAI-Plan", "A1": "A1 VOAI-Ground", "A2": "A2 VOAI-Agent"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4))
for i, (item, model, q, prod) in enumerate(ROWS):
    g = gold[item]
    tgt = SkyCoord(g["ra"], g["dec"], unit="deg")
    for j, a in enumerate(["A0", "A1", "A2"]):
        ax = axes[i, j]
        with fits.open(D / f"{item}_{a}.fits") as h:
            raw = h[0].data.astype(float)
            w = WCS(h[0].header)
        valid = np.isfinite(raw) & (raw != 0)
        if valid.mean() < 0.01:  # el producto no tiene datos en la región del objetivo
            ax.imshow(np.zeros_like(raw), origin="lower", cmap="gray", vmin=-1, vmax=1)
            ax.text(0.5, 0.25, "no data at the target position\n(outside survey coverage)", transform=ax.transAxes,
                    ha="center", color="white", fontsize=7.5)
        else:
            data = np.where(valid, raw, np.nanmedian(raw[valid]))
            norm = ImageNormalize(data, interval=PercentileInterval(99.7), stretch=AsinhStretch(0.08))
            ax.imshow(data, origin="lower", cmap="gray", norm=norm)
        x, y = w.world_to_pixel(tgt)
        ny, nx = raw.shape
        if valid.mean() < 0.01:
            pass
        elif 0 <= x < nx and 0 <= y < ny:
            ax.plot(x, y, marker="+", ms=14, mew=1.6, color="#FFD700")
        else:
            ax.annotate("target outside\nfield of view", (0.5, 0.06), xycoords="axes fraction", ha="center",
                        color="#FFD700", fontsize=7.5)
        ok, why = VERDICT[(item, a)]
        ax.set_title(f"{ARCH[a]}\n{prod[a]}", fontsize=8)
        ax.text(0.03, 0.95, ("✓ " if ok else "✗ ") + why, transform=ax.transAxes, va="top", fontsize=7,
                color="white", bbox=dict(boxstyle="round,pad=0.25", fc="#1f7a4d" if ok else "#B64342", ec="none"))
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.2)
for i, (item, model, q, prod) in enumerate(ROWS):
    bb = axes[i, 0].get_position()
    fig.text(0.5, bb.y1 + 0.075, f"{q}  \u2014  {model}", ha="center", fontsize=9, fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"paper/figures/fig_examples.{ext}", dpi=250, bbox_inches="tight")
print("ok")
