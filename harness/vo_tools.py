"""Catálogo cerrado de herramientas VO para VOAI-Bench.

Todas las llamadas a servicios pasan por `cached()`, que guarda la respuesta en disco
(modo record/replay). Así la evaluación de los modelos no depende de la volatilidad de
los servicios: la primera ejecución graba, las siguientes reproducen.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import warnings
from pathlib import Path

import requests

warnings.filterwarnings("ignore")
# PyVO no fija timeout por defecto: un servidor que acepta la conexión y no responde bloquearía el hilo
# indefinidamente. Este límite global convierte esos bloqueos en errores transitorios que se reintentan.
import socket  # noqa: E402
socket.setdefaulttimeout(float(os.environ.get("VOAI_SOCKET_TIMEOUT", "180")))

CACHE_DIR = Path(os.environ.get("VOAI_CACHE", Path(__file__).resolve().parent.parent / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MODE = os.environ.get("VOAI_CACHE_MODE", "record")  # record | replay | off
HTTP_TIMEOUT = 90
MAX_ROWS_TO_LLM = 25  # filas que se muestran al modelo (el resultado completo se guarda)

# ---------------------------------------------------------------- servicios
SERVICES = {
    "gaia_esa": "https://gea.esac.esa.int/tap-server/tap",
    "irsa_tap": "https://irsa.ipac.caltech.edu/TAP",
}
IRSA_SIA2 = "https://irsa.ipac.caltech.edu/SIA"
SKYVIEW_SIA = "https://skyview.gsfc.nasa.gov/current/cgi/vo/sia.pl"
SSA_SERVICES = {
    "iso": "http://nida.esac.esa.int/nida-sl-tap/servlet/ssa",
    "herschel": "http://archives.esac.esa.int/hsa/aio/jsp/ssap.jsp",
}
SESAME_MIRRORS = ["https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV",
                  "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oxp/SNV"]

# Rango espectral (micrómetros) de los surveys de SkyView expuestos al modelo.
SKYVIEW_SURVEYS = {
    "DSS2 Red": (0.59, 0.72), "DSS2 Blue": (0.39, 0.54), "DSS2 IR": (0.70, 0.95),
    "SDSSg": (0.40, 0.55), "SDSSr": (0.55, 0.70), "SDSSi": (0.69, 0.84),
    "2MASS-J": (1.1, 1.4), "2MASS-H": (1.5, 1.8), "2MASS-K": (1.95, 2.36),
    "WISE 3.4": (2.8, 3.8), "WISE 4.6": (4.1, 5.2), "WISE 12": (7.5, 16.5), "WISE 22": (19.5, 27.5),
    "GALEX Near UV": (0.175, 0.28), "GALEX Far UV": (0.135, 0.175),
    "SHASSA H": (0.651, 0.661), "IRIS 100": (83.0, 120.0), "AKARI WIDE-S": (65.0, 105.0),
    "RASS-Cnt Broad": (0.00052, 0.0124), "RASS-Cnt Hard": (0.00052, 0.0062),
    "NVSS": (2.0e5, 2.3e5), "SUMSS 843 MHz": (3.4e5, 3.7e5), "VLA FIRST (1.4 GHz)": (2.0e5, 2.3e5),
    "TGSS ADR1": (1.8e6, 2.0e6),
}
# Columna identificadora y nombres de tabla por catálogo (para cone_search).
CATALOGS = {
    "gaia_dr3": ("gaia_esa", "gaiadr3.gaia_source", "source_id",
                 "source_id, ra, dec, phot_g_mean_mag, parallax, pmra, pmdec"),
    "2mass_psc": ("irsa_tap", "fp_psc", "designation", "designation, ra, dec, j_m, h_m, k_m"),
    "allwise": ("irsa_tap", "allwise_p3as_psd", "designation", "designation, ra, dec, w1mpro, w2mpro, w3mpro, w4mpro"),
}


class ToolError(Exception):
    pass


class TransientError(Exception):
    """Fallo de infraestructura (timeout, 5xx, conexión): no se cachea y se reintenta."""


def _is_transient(e: Exception) -> bool:
    m = f"{type(e).__name__} {e}".lower()
    if "job timeout" in m or "aborted" in m:  # el servicio abortó la consulta por pesada: culpa de la consulta
        return False
    return isinstance(e, TransientError) or any(k in m for k in (
        "timeout", "timed out", "connectionerror", "connection aborted", "max retries", " 502", " 503", " 504",
        "502 server", "503 server", "504 server", "gateway", "temporarily unavailable"))


# ---------------------------------------------------------------- caché
def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def cached(kind: str, params: dict, fn):
    """Ejecuta fn() con caché en disco. Devuelve (resultado, meta)."""
    path = CACHE_DIR / kind / f"{_key(kind, params)}.json"
    if CACHE_MODE != "off" and path.exists():
        rec = json.loads(path.read_text())
        return rec["result"], {"cache": "hit", "service_s": rec["service_s"], "error": rec.get("error")}
    if CACHE_MODE == "replay":
        raise ToolError(f"cache miss in replay mode for {kind} {params}")
    t0 = time.time()
    err = None
    for attempt in range(3):
        try:
            result = fn()
            err = None
            break
        except Exception as e:
            result, err = None, f"{type(e).__name__}: {str(e)[:300]}"
            if not _is_transient(e):
                break  # error semántico (p.ej. ADQL inválido): se graba, es comportamiento observado
            time.sleep(3 * (attempt + 1))
    dt = time.time() - t0
    if err and result is None and attempt == 2 and _is_transient(Exception(err)):
        raise ToolError(f"service unavailable (transient): {err}")
    if CACHE_MODE != "off":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"params": params, "result": result, "service_s": dt, "error": err}, default=str))
    return result, {"cache": "miss", "service_s": dt, "error": err}


# ---------------------------------------------------------------- utilidades astro
def ang_sep_deg(ra1, dec1, ra2, dec2) -> float:
    r = math.radians
    a = math.sin((r(dec2) - r(dec1)) / 2) ** 2 + math.cos(r(dec1)) * math.cos(r(dec2)) * math.sin((r(ra2) - r(ra1)) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def region_contains(s_region, ra, dec):
    """¿El POLYGON/CIRCLE de ObsCore (STC-S) contiene (ra, dec)? Proyección gnomónica local.
    Devuelve None si no se puede interpretar."""
    if not s_region or not isinstance(s_region, str):
        return None
    toks = s_region.replace(",", " ").split()
    up = [t.upper() for t in toks]
    nums = []
    for t in toks:
        try:
            nums.append(float(t))
        except ValueError:
            pass
    if "CIRCLE" in up and len(nums) >= 3:
        return ang_sep_deg(nums[0], nums[1], ra, dec) <= nums[2]
    if "POLYGON" not in up or len(nums) < 6:
        return None
    r = math.radians
    ra0, de0 = r(ra), r(dec)

    def proj(a, d):
        a, d = r(a), r(d)
        c = math.sin(de0) * math.sin(d) + math.cos(de0) * math.cos(d) * math.cos(a - ra0)
        if c <= 0:
            return None
        return (math.cos(d) * math.sin(a - ra0) / c,
                (math.cos(de0) * math.sin(d) - math.sin(de0) * math.cos(d) * math.cos(a - ra0)) / c)
    pts = [proj(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    if any(p is None for p in pts):
        return False
    inside = False  # ray casting desde el origen (= posición objetivo)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if (y1 > 0) != (y2 > 0) and 0 < (x2 - x1) * (0 - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def galactic_to_icrs(l, b):
    from astropy.coordinates import SkyCoord
    c = SkyCoord(l=l, b=b, unit="deg", frame="galactic").icrs
    return float(c.ra.deg), float(c.dec.deg)


# ---------------------------------------------------------------- herramientas
SIMBAD_SCRIPT = ["https://simbad.cfa.harvard.edu/simbad/sim-script", "https://simbad.cds.unistra.fr/simbad/sim-script"]
NED_LOOKUP = "https://ned.ipac.caltech.edu/srs/ObjectLookup"
_SESAME_DOWN_UNTIL = 0.0


def _resolve_fallback(name: str, last_err) -> dict:
    """Respaldo cuando Sesame no responde: SIMBAD sim-script (Harvard/CDS) y luego NED."""
    script = ('output console=off script=off\nformat object "%COO(d;A D)|%OTYPE(3)|%MAIN_ID"\n'
              f"query id {name}")
    for base in SIMBAD_SCRIPT:
        try:
            r = requests.get(base, params={"script": script}, timeout=40)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                parts = line.split("|")
                if len(parts) == 3 and re.match(r"^\s*[\d.]+\s+[-+][\d.]+", parts[0]):
                    ra, dec = parts[0].split()
                    return {"found": True, "ra": float(ra), "dec": float(dec), "otype": parts[1].strip(),
                            "main_id": re.sub(r"\s+", " ", parts[2].replace("~", "")).strip(), "via": "simbad-script"}
            if "error" in r.text.lower() or "not found" in r.text.lower() or "catalogue" in r.text.lower():
                break  # SIMBAD respondió pero no conoce el nombre: probar NED
        except requests.RequestException as e:
            last_err = str(e)
    try:
        r = requests.post(NED_LOOKUP, data={"json": json.dumps({"name": {"v": name}})}, timeout=40)
        d = r.json()
        if d.get("ResultCode") in (2, 3) and d.get("Preferred", {}).get("Position"):
            pos = d["Preferred"]["Position"]
            return {"found": True, "ra": float(pos["RA"]), "dec": float(pos["Dec"]), "otype": None,
                    "main_id": d["Preferred"].get("Name"), "via": "ned"}
        if r.status_code == 200 and "ResultCode" in d:
            return {"found": False, "via": "simbad+ned"}
    except Exception as e:
        last_err = str(e)
    raise TransientError(f"name resolvers unavailable: {last_err}")


def otype_list(name: str) -> list:
    """Todos los tipos de objeto SIMBAD (principal + secundarios) de un objeto (%OTYPELIST)."""
    script = f'output console=off script=off\nformat object "%OTYPELIST(3)"\nquery id {name}'

    def run():
        for base in SIMBAD_SCRIPT:
            try:
                r = requests.get(base, params={"script": script}, timeout=40)
            except requests.RequestException:
                continue
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                if "," in line and " " not in line.strip():
                    return sorted(set(line.strip().split(",")))
            return []
        raise TransientError("SIMBAD script unavailable")
    res, meta = cached("otypes", {"name": name.strip()}, run)
    if res is None:
        raise ToolError(meta["error"])
    return res


def resolve_name(name: str) -> dict:
    """Sesame (SIMBAD/NED/VizieR) -> ra, dec (J2000, deg) y tipo de objeto SIMBAD."""
    def run():
        global _SESAME_DOWN_UNTIL
        txt, last = None, None
        if time.time() < _SESAME_DOWN_UNTIL:  # cortacircuito: Sesame cayó hace poco
            return _resolve_fallback(name, "sesame circuit open")
        for base in SESAME_MIRRORS:
            try:
                r = requests.get(f"{base}?{requests.utils.quote(name)}", timeout=8)
                if r.status_code == 200 and "<Sesame" in r.text:
                    txt = r.text
                    break
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = str(e)
        if txt is None:
            _SESAME_DOWN_UNTIL = time.time() + 300
            return _resolve_fallback(name, last)
        ra = re.search(r"<jradeg>([-\d.eE+]+)</jradeg>", txt)
        de = re.search(r"<jdedeg>([-\d.eE+]+)</jdedeg>", txt)
        if not (ra and de):
            return {"found": False}
        ot = re.search(r"<otype>([^<]+)</otype>", txt)
        on = re.search(r"<oname>([^<]+)</oname>", txt)
        return {"found": True, "ra": float(ra.group(1)), "dec": float(de.group(1)),
                "otype": ot.group(1).strip() if ot else None,
                "main_id": re.sub(r"\s+", " ", on.group(1)).strip() if on else None}
    res, meta = cached("sesame", {"name": name.strip()}, run)
    if res is None:
        raise ToolError(meta["error"])
    return res


def _rows(table, limit=None):
    out = []
    for i, row in enumerate(table):
        if limit and i >= limit:
            break
        rec = {}
        for c in table.colnames:
            v = row[c]
            try:
                v = v.item() if hasattr(v, "item") else v
            except Exception:
                v = str(v)
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, float) and math.isnan(v):
                v = None
            rec[c] = v if isinstance(v, (int, float, str, type(None), bool)) else str(v)
        out.append(rec)
    return out


def tap_query(service: str, adql: str, maxrec: int = 2000) -> dict:
    if service not in SERVICES:
        raise ToolError(f"unknown TAP service '{service}'. Valid: {list(SERVICES)}")

    def run():
        import pyvo
        t = pyvo.dal.TAPService(SERVICES[service]).run_sync(adql, maxrec=maxrec).to_table()
        return {"columns": t.colnames, "rows": _rows(t), "n": len(t)}
    res, meta = cached("tap", {"service": service, "adql": adql.strip(), "maxrec": maxrec}, run)
    if res is None:
        raise ToolError(meta["error"])
    return res


def cone_search(catalog: str, ra: float, dec: float, radius_deg: float) -> dict:
    if catalog not in CATALOGS:
        raise ToolError(f"unknown catalog '{catalog}'. Valid: {list(CATALOGS)}")
    if not (0 < radius_deg <= 0.5):
        raise ToolError("radius_deg must be in (0, 0.5]")
    svc, table, idcol, cols = CATALOGS[catalog]
    contains = "1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{r}))".format(ra=ra, dec=dec, r=radius_deg)
    res = tap_query(svc, f"SELECT {cols} FROM {table} WHERE {contains}", maxrec=5000)
    res["id_column"] = idcol
    return res


def sia_search(ra: float, dec: float, radius_deg: float = 0.01, band_um_min: float | None = None,
               band_um_max: float | None = None, collection: str | None = None) -> dict:
    """IRSA SIA2 (ObsCore). Devuelve imágenes con centro, fov y rango espectral (µm)."""
    def run():
        import pyvo
        kw = {"pos": (ra, dec, radius_deg), "maxrec": 300}
        if band_um_min is not None and band_um_max is not None:
            kw["band"] = (band_um_min * 1e-6, band_um_max * 1e-6)
        if collection:
            kw["collection"] = collection
        t = pyvo.dal.SIA2Service(IRSA_SIA2).search(**kw).to_table()
        rows = []
        for r in _rows(t):
            if r.get("dataproduct_type") not in (None, "image", "cube"):
                continue
            rows.append({"collection": r.get("obs_collection"), "instrument": r.get("instrument_name"),
                         "band": r.get("energy_bandpassname"), "ra": r.get("s_ra"), "dec": r.get("s_dec"),
                         "fov_deg": r.get("s_fov"),
                         "em_min_um": r["em_min"] * 1e6 if r.get("em_min") else None,
                         "em_max_um": r["em_max"] * 1e6 if r.get("em_max") else None,
                         "url": r.get("access_url"), "format": r.get("access_format"),
                         "s_region": r.get("s_region")})
        return {"n": len(rows), "images": rows}
    res, meta = cached("sia2", {"ra": round(ra, 5), "dec": round(dec, 5), "r": radius_deg, "b0": band_um_min,
                                "b1": band_um_max, "c": collection}, run)
    if res is None:
        raise ToolError(meta["error"])
    return res


def skyview_image(survey: str, ra: float, dec: float, size_deg: float = 0.25) -> dict:
    """Recorte de SkyView centrado en (ra, dec) para un survey dado."""
    if survey not in SKYVIEW_SURVEYS:
        raise ToolError(f"unknown survey '{survey}'. Valid: {list(SKYVIEW_SURVEYS)}")

    def run():
        r = requests.get(SKYVIEW_SIA, params={"survey": survey, "POS": f"{ra},{dec}", "SIZE": size_deg,
                                              "FORMAT": "image/fits"}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        urls = re.findall(r"<TD><!\[CDATA\[(https?://[^\]]+)\]\]></TD>|<TD>(https?://[^<]+)</TD>", r.text)
        urls = [a or b for a, b in urls]
        return {"n": len(urls), "url": urls[0] if urls else None}
    res, meta = cached("skyview", {"s": survey, "ra": round(ra, 5), "dec": round(dec, 5), "size": size_deg}, run)
    if res is None:
        raise ToolError(meta["error"])
    lo, hi = SKYVIEW_SURVEYS[survey]
    if res["n"] == 0:
        return {"n": 0, "images": []}
    return {"n": 1, "images": [{"collection": f"SkyView:{survey}", "ra": ra, "dec": dec, "fov_deg": size_deg,
                                "em_min_um": lo, "em_max_um": hi, "url": res["url"]}]}


# Rango nominal (µm) por instrumento cuando el servicio SSA no informa la cobertura espectral.
SSA_INSTRUMENT_RANGES_UM = {"PACS": (51.0, 220.0), "SPIRE": (194.0, 671.0), "HIFI": (157.0, 625.0),
                            "SWS": (2.4, 45.0), "LWS": (43.0, 197.0), "PHT-S": (2.5, 11.6)}


def _num_or_none(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def ssa_search(service: str, ra: float, dec: float, diameter_deg: float = 0.05) -> dict:
    """SSA con mapeo robusto de columnas: los servicios ESA usan nombres propios (ISO: reference,
    center_wave/extent_wave en m; Herschel: DATA_LINK, RA/DEC, WAVE_CENTRAL/WAVE_BANDWIDTH en m o vacíos)."""
    if service not in SSA_SERVICES:
        raise ToolError(f"unknown SSA service '{service}'. Valid: {list(SSA_SERVICES)}")

    def run():
        import pyvo
        t = pyvo.dal.SSAService(SSA_SERVICES[service]).search(pos=(ra, dec), diameter=diameter_deg).to_table()
        rows = []
        for r in _rows(t, limit=300):
            low = {k.lower(): v for k, v in r.items()}

            def g(*names):
                for n in names:
                    v = low.get(n.lower())
                    if v not in (None, "", "--"):
                        return v
                return None
            lo = _num_or_none(g("ssa_specstart", "spectralstart", "spectral_start", "em_min"))
            hi = _num_or_none(g("ssa_specend", "spectralend", "spectral_end", "em_max"))
            if lo is None or hi is None:
                c = _num_or_none(g("ssa_specmid", "center_wave", "wave_central"))
                w = _num_or_none(g("ssa_specext", "extent_wave", "wave_bandwidth"))
                if c is not None and w is not None:
                    lo, hi = c - w / 2, c + w / 2
            lo_um = lo * 1e6 if lo is not None else None
            hi_um = hi * 1e6 if hi is not None else None
            if lo_um is not None and (hi_um is None or hi_um <= 0 or hi_um <= lo_um):
                lo_um = hi_um = None  # cobertura vacía o nula (Herschel informa 0): usar rango del instrumento
            inst = str(g("instrument_name", "instrument", "target_name", "title") or "")
            if lo_um is None:
                for key, (a0, a1) in SSA_INSTRUMENT_RANGES_UM.items():
                    if key.lower() in inst.lower():
                        lo_um, hi_um = a0, a1
                        break
            rows.append({"title": g("ssa_title", "title", "target_name"),
                         "instrument": g("instrument_name", "instrument"),
                         "ra": _num_or_none(g("ssa_location_ra", "ra")), "dec": _num_or_none(g("ssa_location_dec", "dec")),
                         "em_min_um": lo_um, "em_max_um": hi_um,
                         "url": g("ssa_reference", "access_url", "acref", "reference", "data_link")})
        return {"n": len(rows), "spectra": rows}
    res, meta = cached("ssa2", {"svc": service, "ra": round(ra, 5), "dec": round(dec, 5), "d": diameter_deg}, run)
    if res is None:
        raise ToolError(meta["error"])
    return res


def describe_tables(service: str, table: str | None = None) -> dict:
    """Grounding de esquema: tablas o columnas de un servicio TAP (vía TAP_SCHEMA)."""
    if table is None:
        q = ("SELECT table_name, description FROM TAP_SCHEMA.tables WHERE table_name IN "
             "('gaiadr3.gaia_source','gaiadr3.astrophysical_parameters','gaiadr3.nss_two_body_orbit')"
             if service == "gaia_esa" else
             "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE table_name IN "
             "('fp_psc','allwise_p3as_psd','ivoa.obscore','fp_xsc','catwise_2020')")
    else:
        q = f"SELECT column_name, datatype, unit, description FROM TAP_SCHEMA.columns WHERE table_name = '{table}'"
    return tap_query(service, q, maxrec=400)


def truncate_for_llm(res: dict) -> dict:
    """Recorta el resultado que ve el modelo (el completo queda en caché y en el log)."""
    out = dict(res)
    for k in ("rows", "images", "spectra"):
        if k in out and isinstance(out[k], list) and len(out[k]) > MAX_ROWS_TO_LLM:
            out[k] = out[k][:MAX_ROWS_TO_LLM]
            out["truncated"] = True
    return out
