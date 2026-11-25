"""Caso de prueba entre modelos: la misma solicitud (VS20, inglés, repetición 0) respondida por los cuatro
modelos con A0 (VOAI-Plan) y con A2 (VOAI-Agent); se muestran los FITS reales devueltos por cada episodio.
Los FITS están en paper/figures/models_data/ (URLs en urls.json, extraídas de results/test.jsonl).
Uso: python -I harness/make_model_comparison_figure.py -> paper/figures/fig_models_case.pdf|png"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval, LinearStretch, ZScaleInterval  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "paper/figures/models_data"
gold = json.loads((ROOT / "bench/gold_v1.json").read_text())["VS20"]
TGT = SkyCoord(gold["ra"], gold["dec"], unit="deg")
MODELS = [("gpt-6-luna", "GPT-6 Luna"), ("claude-haiku-5-5", "Claude Haiku 5.5"),
          ("deepseek-v4-pro", "DeepSeek V4 Pro"), ("qwen3.8-flash", "Qwen 3.8 Flash")]
RED, BLUE, GREEN, GOLD = "#B64342", "#0F4D92", "#1f7a4d", "#FFD700"


def load(arch, model):
    fn = D / f"VS20_{arch}_{model}.fits"
    if not fn.exists():  # descarga el producto devuelto en el episodio (URL registrada en results/test.jsonl)
        import urllib.request
        url = json.loads((D / "urls.json").read_text())[fn.name]
        req = urllib.request.Request(url, headers={"User-Agent": "voai-bench-figures"})
        fn.write_bytes(urllib.request.urlopen(req, timeout=180).read())
    with fits.open(fn) as h:
        return h[0].data.astype(float), WCS(h[0].header), h[0].header


def show(ax, data, zscale=False):
    valid = np.isfinite(data)
    data = np.where(valid, data, np.nanmedian(data[valid]))
    norm = (ImageNormalize(data, interval=ZScaleInterval(contrast=0.05), stretch=LinearStretch()) if zscale else
            ImageNormalize(data, interval=PercentileInterval(99.9), stretch=AsinhStretch(0.03)))
    ax.imshow(data, origin="lower", cmap="gray", norm=norm)
    ax.set_xticks([]); ax.set_yticks([])


def fov_deg(hdr):
    return abs(hdr["CDELT1"]) * hdr["NAXIS1"] if "CDELT1" in hdr else None


plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.15))
for j, (m, mname) in enumerate(MODELS):
    # Fila superior: A0 (DSS2 Red, recorte de 0.25°, centrado en coordenadas recordadas por el modelo)
    ax = axes[0, j]
    d0, w0, h0 = load("A0", m)
    show(ax, d0, zscale=True)
    c0 = w0.pixel_to_world(d0.shape[1] / 2, d0.shape[0] / 2)
    sep = c0.separation(TGT)
    x, y = w0.world_to_pixel(TGT)
    if 0 <= x < d0.shape[1] and 0 <= y < d0.shape[0]:
        ax.plot(x, y, "+", ms=13, mew=1.6, color=GOLD)
    else:  # flecha hacia el objetivo desde el centro
        pa = c0.position_angle(TGT).deg
        dx, dy = -np.sin(np.radians(pa)), np.cos(np.radians(pa))  # este a la izquierda (RA crece hacia la izquierda)
        cx, cy = d0.shape[1] / 2, d0.shape[0] / 2
        ax.annotate("", xy=(cx + dx * 0.46 * d0.shape[1], cy + dy * 0.46 * d0.shape[0]), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.4))
    off = f"{sep.arcmin:.0f}′ off" if sep.deg < 1 else f"{sep.deg:.0f}° off"
    ax.text(0.03, 0.96, f"✗ centre {off}", transform=ax.transAxes, va="top", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=RED, ec="none"))
    ax.text(0.97, 0.04, f"DSS2 Red · {fov_deg(h0):.2f}°", transform=ax.transAxes, ha="right", fontsize=6.8,
            color="white", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"))
    ax.set_title(mname, fontsize=8.5, fontweight="bold")
    # Fila inferior: A2 (SHASSA Hα, centrado en la posición resuelta por Sesame)
    ax = axes[1, j]
    d2, w2, h2 = load("A2", m)
    show(ax, d2)
    x, y = w2.world_to_pixel(TGT)
    ax.plot(x, y, "+", ms=13, mew=1.6, color=GOLD)
    # huella del recorte A0 del mismo modelo, si cae dentro del campo
    ny, nx = d0.shape
    corners = w0.pixel_to_world([0, nx, nx, 0], [0, 0, ny, ny])
    px, py = w2.world_to_pixel(corners)
    if np.all((px > -nx) & (px < 2 * d2.shape[1]) & (py > -ny) & (py < 2 * d2.shape[0])):
        ax.add_patch(Polygon(np.c_[px, py], closed=True, fill=False, ec=RED, lw=1.1, ls="--"))
    else:
        ax.text(0.5, 0.20, "A0 cutout outside\nthis field", transform=ax.transAxes, ha="center", fontsize=6.8,
                color="white", bbox=dict(boxstyle="round,pad=0.2", fc=RED, alpha=0.8, ec="none"))
    ax.set_xlim(-0.5, d2.shape[1] - 0.5); ax.set_ylim(-0.5, d2.shape[0] - 0.5)
    ax.text(0.03, 0.96, "✓ resolved centre", transform=ax.transAxes, va="top", fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=GREEN, ec="none"))
    ax.text(0.97, 0.04, f"SHASSA Hα · {fov_deg(h2):.0f}°", transform=ax.transAxes, ha="right", fontsize=6.8,
            color="white", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"))
axes[0, 0].set_ylabel("A0 VOAI-Plan", fontsize=9, color=RED, fontweight="bold")
axes[1, 0].set_ylabel("A2 VOAI-Agent", fontsize=9, color=BLUE, fontweight="bold")
fig.suptitle("“Can you find an H-alpha image of the Carina Nebula?”", fontsize=9.5, fontweight="bold", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=0.4)
fig.subplots_adjust(hspace=0.06)
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"paper/figures/fig_models_case.{ext}", dpi=250, bbox_inches="tight")
print("ok")
