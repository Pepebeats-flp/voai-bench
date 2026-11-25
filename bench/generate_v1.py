"""Genera el conjunto de PRUEBA de VOAI-Bench v1 (≈160 ítems EN/ES) de forma reproducible.

v0 (queries_v0.yaml, 50 ítems) se usó para ajustar prompts en el piloto → conjunto de DESARROLLO.
v1 se genera con plantillas + tabla de objetos reales y semilla fija → conjunto de PRUEBA (Kapoor et al. 2025).
Cada ítem lleva su gold *template*; build_gold.py lo resuelve y descarta los no factibles o con gold vacío.

Uso: python -I bench/generate_v1.py  -> bench/queries_v1.yaml
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RNG = random.Random(2026)

# ------------------------------------------------------------------ objetos
# (sesame, nombre EN, nombre ES, clase, tolerancia_deg, campo_denso)
OBJ = [
    ("M31", "the Andromeda Galaxy", "la galaxia de Andrómeda", "gal", 1.0, False),
    ("M33", "the Triangulum Galaxy", "la galaxia del Triángulo", "gal", 0.3, False),
    ("M81", "Bode's Galaxy", "la galaxia de Bode", "gal", 0.15, False),
    ("M82", "the Cigar Galaxy", "la galaxia del Cigarro", "gal", 0.07, False),
    ("M101", "the Pinwheel Galaxy", "la galaxia del Molinete", "gal", 0.2, False),
    ("M104", "the Sombrero Galaxy", "la galaxia del Sombrero", "gal", 0.05, False),
    ("M87", "M87", "M87", "gal", 0.05, False),
    ("NGC 253", "the Sculptor Galaxy", "la galaxia del Escultor", "gal", 0.2, False),
    ("NGC 4565", "the Needle Galaxy", "la galaxia de la Aguja", "gal", 0.1, False),
    ("M83", "the Southern Pinwheel Galaxy", "la galaxia del Molinete Austral", "gal", 0.1, False),
    ("NGC 1365", "the Great Barred Spiral Galaxy NGC 1365", "la gran galaxia espiral barrada NGC 1365", "gal", 0.08, False),
    ("M64", "the Black Eye Galaxy", "la galaxia del Ojo Negro", "gal", 0.05, False),
    ("NGC 5128", "Centaurus A", "Centaurus A", "gal", 0.15, False),
    ("M8", "the Lagoon Nebula", "la nebulosa de la Laguna", "neb", 0.4, True),
    ("M16", "the Eagle Nebula", "la nebulosa del Águila", "neb", 0.1, True),
    ("M17", "the Omega Nebula", "la nebulosa Omega", "neb", 0.1, True),
    ("M20", "the Trifid Nebula", "la nebulosa Trífida", "neb", 0.1, True),
    ("M27", "the Dumbbell Nebula", "la nebulosa de la Haltera", "pn", 0.05, False),
    ("NGC 7293", "the Helix Nebula", "la nebulosa de la Hélice", "pn", 0.2, False),
    ("NGC 6543", "the Cat's Eye Nebula", "la nebulosa Ojo de Gato", "pn", 0.01, False),
    ("NGC 7027", "the planetary nebula NGC 7027", "la nebulosa planetaria NGC 7027", "pn", 0.01, True),
    ("Cas A", "Cassiopeia A", "Cassiopea A", "snr", 0.05, True),
    ("NGC 3372", "the Carina Nebula", "la nebulosa de Carina", "neb", 1.0, True),
    ("M42", "the Orion Nebula", "la nebulosa de Orión", "neb", 0.25, True),
    ("M1", "the Crab Nebula", "la nebulosa del Cangrejo", "snr", 0.05, False),
    ("M3", "the globular cluster M3", "el cúmulo globular M3", "glc", 0.1, False),
    ("NGC 104", "47 Tucanae", "47 Tucanae", "glc", 0.3, False),
    ("M4", "the globular cluster M4", "el cúmulo globular M4", "glc", 0.2, True),
    ("M13", "the Hercules Globular Cluster", "el cúmulo globular de Hércules", "glc", 0.1, False),
    ("NGC 5139", "Omega Centauri", "Omega Centauri", "glc", 0.3, True),
    ("M45", "the Pleiades", "las Pléyades", "opc", 1.0, False),
    ("M44", "the Beehive Cluster", "el cúmulo del Pesebre", "opc", 0.5, False),
    ("M67", "the open cluster M67", "el cúmulo abierto M67", "opc", 0.25, False),
    ("M11", "the Wild Duck Cluster", "el cúmulo del Pato Salvaje", "opc", 0.1, True),
    ("NGC 869", "h Persei (NGC 869)", "h Persei (NGC 869)", "opc", 0.2, True),
    ("Betelgeuse", "Betelgeuse", "Betelgeuse", "star", 0.01, False),
    ("Rigel", "Rigel", "Rigel", "star", 0.01, False),
    ("Sirius", "Sirius", "Sirio", "star", 0.01, False),
    ("Vega", "Vega", "Vega", "star", 0.01, False),
    ("Polaris", "Polaris", "la estrella Polar", "star", 0.01, False),
    ("Antares", "Antares", "Antares", "star", 0.01, True),
    ("VY CMa", "VY Canis Majoris", "VY Canis Majoris", "star", 0.01, True),
    ("IRC +10216", "the carbon star IRC +10216", "la estrella de carbono IRC +10216", "star", 0.01, False),
    ("KIC 8462852", "Tabby's Star", "la estrella de Tabby", "star", 0.005, True),
    ("T Tau", "T Tauri", "T Tauri", "star", 0.005, False),
    ("Proxima Cen", "Proxima Centauri", "Próxima Centauri", "star", 0.01, True),
    ("Barnard's star", "Barnard's Star", "la estrella de Barnard", "star", 0.01, True),
    ("eta Car", "Eta Carinae", "Eta Carinae", "star", 0.05, True),
    ("Cyg X-1", "Cygnus X-1", "Cygnus X-1", "star", 0.005, True),
    ("3C 273", "the quasar 3C 273", "el cuásar 3C 273", "qso", 0.005, False),
    ("Sgr A*", "Sagittarius A*", "Sagitario A*", "gc", 0.02, True),
    ("Arp 220", "Arp 220", "Arp 220", "gal", 0.02, False),
    ("NGC 7023", "the Iris Nebula", "la nebulosa del Iris", "neb", 0.1, False),
]
BY = {o[0]: o for o in OBJ}

# ------------------------------------------------------------------ bandas (EN, ES, rango µm, dificultad)
BANDS = {
    "optical": ("an optical", "óptica", [0.35, 1.0], 1),
    "visible": ("a visible-light", "en luz visible", [0.35, 1.0], 1),
    "nir": ("a near-infrared", "en el infrarrojo cercano", [0.9, 2.5], 1),
    "J": ("a J-band", "en banda J", [1.1, 1.4], 2),
    "Ks": ("a K-band", "en banda K", [1.9, 2.5], 2),
    "w1": ("a 3.4 micron", "a 3,4 micrones", [2.8, 3.8], 2),
    "w3": ("a 12 micron", "a 12 micrones", [7.5, 16.5], 2),
    "w4": ("a 22 micron", "a 22 micrones", [18.0, 28.0], 2),
    "mir": ("a mid-infrared", "en el infrarrojo medio", [5.0, 30.0], 2),
    "fir": ("a far-infrared", "en el infrarrojo lejano", [30.0, 200.0], 2),
    "uv": ("an ultraviolet", "ultravioleta", [0.1, 0.32], 2),
    "xray": ("an X-ray", "en rayos X", [0.00001, 0.0124], 2),
    "radio": ("a 1.4 GHz radio", "en radio a 1,4 GHz", [150000, 260000], 2),
    "halpha": ("an H-alpha", "en H-alfa", [0.64, 0.67], 2),
}
SIA_EN = ["Show me {band} image of {obj}.", "I'd like {band} image of {obj}.", "Can you find {band} image of {obj}?",
          "Retrieve {band} image of {obj}, please.", "Get me {band} picture of {obj}."]
SIA_ES = ["Muéstrame una imagen {band} de {obj}.", "Quisiera una imagen {band} de {obj}.",
          "¿Puedes encontrar una imagen {band} de {obj}?", "Recupera una imagen {band} de {obj}, por favor.",
          "Consígueme una imagen {band} de {obj}."]
SIA_NOBAND_EN = ["Show me an image of {obj}.", "I want to see a picture of {obj}.", "Find any image of {obj}."]
SIA_NOBAND_ES = ["Muéstrame una imagen de {obj}.", "Quiero ver una imagen de {obj}.", "Encuentra alguna imagen de {obj}."]

# clases para consultas ambiguas (otypes SIMBAD aceptados)
CLASSES = [
    ("a Seyfert galaxy", "una galaxia Seyfert", ["Sy1", "Sy2", "SyG"], None),
    ("an open star cluster", "un cúmulo estelar abierto", ["OpC", "Cl*"], None),
    ("a quasar", "un cuásar", ["QSO", "Bla", "BLL"], None),
    ("an H II region", "una región H II", ["HII"], [5.0, 30.0]),
    ("a Wolf-Rayet star", "una estrella Wolf-Rayet", ["WR*"], [0.9, 2.5]),
    ("a carbon star", "una estrella de carbono", ["C*"], [0.9, 2.5]),
    ("a cluster of galaxies", "un cúmulo de galaxias", ["ClG", "GrG", "CGG"], [0.00001, 0.0124]),
    ("an X-ray binary", "una binaria de rayos X", ["XB", "HXB", "LXB"], [0.00001, 0.0124]),
    ("a white dwarf", "una enana blanca", ["WD*"], None),
    ("a Be star", "una estrella Be", ["Be*"], None),
]


def sexa(ra, dec):
    h = ra / 15
    hh, mm = int(h), int((h - int(h)) * 60)
    ss = ((h - hh) * 60 - mm) * 60
    sgn = "-" if dec < 0 else "+"
    d = abs(dec)
    dd, dm = int(d), int((d - int(d)) * 60)
    ds = ((d - dd) * 60 - dm) * 60
    return f"{hh:02d}h{mm:02d}m{ss:04.1f}s {sgn}{dd:02d}d{dm:02d}m{ds:02.0f}s"


def radius_text(deg, lang):
    if deg < 1 / 60:
        v, u = round(deg * 3600), ("arcsec" if lang == "en" else "segundos de arco")
    elif deg < 1:
        v, u = round(deg * 60, 1), ("arcmin" if lang == "en" else "minutos de arco")
    else:
        v, u = deg, ("degrees" if lang == "en" else "grados")
    v = int(v) if float(v).is_integer() else v
    vs = str(v) if lang == "en" else str(v).replace(".", ",")
    return f"{vs} {u}"


ITEMS = []


def _contract_es(t):
    import re
    t = re.sub(r"\bde el\b", "del", t)
    t = re.sub(r"\ba el\b", "al", t)
    return re.sub(r"\bDe el\b", "Del", t)


def add(protocol, category, difficulty, en, es, gold):
    es = _contract_es(es)
    ITEMS.append({"id": None, "protocol": protocol, "category": category, "difficulty": difficulty,
                  "en": en[0].upper() + en[1:], "es": es[0].upper() + es[1:], "gold": gold, "split": "test"})


def obj(name):
    o = BY[name]
    return o


# ------------------------------------------------------------------ SIA (≈52)
def gen_sia():
    pairs = [("M33", "uv"), ("M81", "optical"), ("M82", "xray"), ("M101", "visible"), ("NGC 253", "w3"),
             ("NGC 4565", "nir"), ("M83", "uv"), ("NGC 1365", "Ks"), ("M64", "optical"), ("NGC 5128", "radio"),
             ("M8", "halpha"), ("M16", "w4"), ("M17", "mir"), ("M20", "optical"), ("M27", "visible"),
             ("NGC 7293", "uv"), ("NGC 6543", "xray"), ("Cas A", "xray"), ("Cas A", "radio"), ("NGC 3372", "halpha"),
             ("M3", "nir"), ("NGC 104", "J"), ("M44", "optical"), ("M11", "Ks"), ("NGC 869", "visible"),
             ("Betelgeuse", "w1"), ("Rigel", "J"), ("Antares", "fir"), ("VY CMa", "w3"), ("IRC +10216", "w4"),
             ("T Tau", "nir"), ("Cyg X-1", "xray"), ("3C 273", "radio"), ("Arp 220", "fir"), ("NGC 7023", "mir"),
             ("M42", "fir"), ("Polaris", "Ks")]
    for name, b in pairs:
        o = obj(name)
        ben, bes, rng, d = BANDS[b]
        add("sia", "named_object_band", d, RNG.choice(SIA_EN).format(band=ben, obj=o[1]),
            RNG.choice(SIA_ES).format(band=bes, obj=o[2]), {"target": name, "band_um": rng, "tol_deg": o[4]})
    for name in ["M31", "M45", "NGC 5139", "Vega"]:
        o = obj(name)
        add("sia", "named_object", 1, RNG.choice(SIA_NOBAND_EN).format(obj=o[1]),
            RNG.choice(SIA_NOBAND_ES).format(obj=o[2]), {"target": name, "band_um": None, "tol_deg": o[4]})
    # coordenadas (decimal / sexagesimal / galácticas) — coordenadas fijas de referencia
    coords = [(83.6331, 22.0145, "dec", "optical"), (266.4168, -29.0078, "sexa", "Ks"),
              (201.3651, -43.0191, "sexa", "radio"), (10.6847, 41.2690, "dec", "uv"),
              (56.75, 24.1167, "dec", "nir"), (284.0, 1.0, "gal_l", "w4")]
    for ra, dec, kind, b in coords:
        ben, bes, rng, d = BANDS[b]
        if kind == "dec":
            cen, ces = f"RA = {ra}, Dec = {dec}", f"AR = {str(ra).replace('.', ',')}, Dec = {str(dec).replace('.', ',')}"
            g = {"target": None, "ra": ra, "dec": dec, "band_um": rng}
        elif kind == "sexa":
            cen = ces = sexa(ra, dec)
            g = {"target": None, "ra": ra, "dec": dec, "band_um": rng}
            d = max(d, 2)
        else:  # galácticas: l=ra, b=dec
            cen, ces = f"Galactic longitude {ra} deg, latitude {dec} deg", f"longitud galáctica {ra}°, latitud {dec}°"
            g = {"target": None, "l": ra, "b": dec, "band_um": rng}
            d = 3
        add("sia", "coordinates_band", d, f"Get {ben} image centred on {cen}.", f"Obtén una imagen {bes} centrada en {ces}.", g)
    for en, es, ot, rng in CLASSES:
        if rng:
            ben = "an X-ray" if rng[1] < 0.02 else "a mid-infrared" if rng[0] >= 5 else "a near-infrared"
            bes = "en rayos X" if rng[1] < 0.02 else "en el infrarrojo medio" if rng[0] >= 5 else "en el infrarrojo cercano"
            add("sia", "object_class_band", 3, f"Show me {ben} image of {en}.", f"Muéstrame una imagen {bes} de {es}.",
                {"target": None, "accept_otypes": ot, "band_um": rng})
        else:
            add("sia", "object_class", 3, f"I want to see an image of {en}.", f"Quiero ver una imagen de {es}.",
                {"target": None, "accept_otypes": ot, "band_um": None})


# ------------------------------------------------------------------ SCS (≈24)
CAT = {"gaia_dr3": ("Gaia DR3", "gaia_esa", "source_id", "gaiadr3.gaia_source", "1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{r}))"),
       "2mass_psc": ("2MASS", "irsa_tap", "designation", "fp_psc", "CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{r}))=1"),
       "allwise": ("AllWISE", "irsa_tap", "designation", "allwise_p3as_psd", "CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{r}))=1")}
SCS_EN = ["List the {cat} sources within {rad} of {obj}.", "Which {cat} objects are within {rad} of {obj}?",
          "Do a cone search in {cat} around {obj} with a radius of {rad}.", "Give me all {cat} entries within {rad} of {obj}."]
SCS_ES = ["Lista las fuentes de {cat} a menos de {rad} de {obj}.", "¿Qué objetos de {cat} hay a menos de {rad} de {obj}?",
          "Haz una búsqueda cónica en {cat} alrededor de {obj} con un radio de {rad}.",
          "Dame todas las entradas de {cat} a menos de {rad} de {obj}."]


def gen_scs():
    spec = [("M81", "allwise", 1 / 60), ("M101", "2mass_psc", 1 / 60), ("M104", "gaia_dr3", 0.5 / 60), ("NGC 4565", "allwise", 1 / 60),
            ("M27", "gaia_dr3", 1 / 60), ("NGC 6543", "2mass_psc", 30 / 3600), ("M3", "gaia_dr3", 0.5 / 60),
            ("M67", "gaia_dr3", 1 / 60), ("Rigel", "2mass_psc", 1 / 60), ("Vega", "gaia_dr3", 2 / 60), ("Polaris", "allwise", 1 / 60),
            ("Sirius", "gaia_dr3", 1 / 60), ("3C 273", "allwise", 30 / 3600), ("3C 273", "gaia_dr3", 20 / 3600),
            ("T Tau", "2mass_psc", 1 / 60), ("KIC 8462852", "gaia_dr3", 30 / 3600), ("Cyg X-1", "allwise", 20 / 3600),
            ("Arp 220", "2mass_psc", 1 / 60), ("M64", "gaia_dr3", 1 / 60), ("NGC 7293", "allwise", 2 / 60),
            ("IRC +10216", "2mass_psc", 1 / 60), ("M44", "2mass_psc", 1 / 60)]
    for name, cat, r in spec:
        o = obj(name)
        cname, svc, idc, table, cond = CAT[cat]
        d = 1 if r >= 1 / 60 and cat != "allwise" else 2
        add("scs", "cone_named", d, RNG.choice(SCS_EN).format(cat=cname, rad=radius_text(r, "en"), obj=o[1]),
            RNG.choice(SCS_ES).format(cat=cname, rad=radius_text(r, "es"), obj=o[2]),
            {"target": name, "catalog": cat, "radius_deg": round(r, 6), "service": svc, "compare": "set", "tol_deg": min(o[4], 0.01),
             "adql": f"SELECT {idc} FROM {table} WHERE " + cond.replace("{r}", f"{r:.6f}")})
    # dos con coordenadas
    for ra, dec, cat, r in [(83.8221, -5.3911, "gaia_dr3", 20 / 3600), (202.4696, 47.1953, "allwise", 1 / 60)]:
        cname, svc, idc, table, cond = CAT[cat]
        add("scs", "cone_coordinates", 1, f"Cone search {cname} at RA {ra}, Dec {dec} with radius {radius_text(r, 'en')}.",
            f"Búsqueda cónica en {cname} en AR {str(ra).replace('.', ',')}, Dec {str(dec).replace('.', ',')} con radio de {radius_text(r, 'es')}.",
            {"target": None, "ra": ra, "dec": dec, "catalog": cat, "radius_deg": round(r, 6), "service": svc, "compare": "set",
             "adql": f"SELECT {idc} FROM {table} WHERE " + cond.replace("{r}", f"{r:.6f}")})


# ------------------------------------------------------------------ TAP (≈56)
G = "gaiadr3.gaia_source"
GC = "1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},%s))"
IC = "CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},%s))=1"


def gen_tap():
    def T(d, cat, en, es, name, adql, compare, service="gaia_esa", rtol=None):
        o = obj(name) if name else None
        g = {"target": name, "service": service, "compare": compare, "adql": adql}
        if o:
            g["tol_deg"] = o[4]
        if rtol:
            g["rtol"] = rtol
        add("tap", cat, d, en.format(obj=o[1] if o else ""), es.format(obj=o[2] if o else ""), g)

    # conteos con corte de magnitud (d2)
    for name, r, mag in [("M3", 0.05, 17), ("NGC 104", 0.1, 15), ("M67", 0.2, 14), ("M11", 0.05, 15), ("NGC 869", 0.1, 12),
                         ("M45", 0.5, 10), ("M4", 0.05, 16)]:
        T(2, "tap_count_mag", f"How many Gaia DR3 sources brighter than G = {mag} lie within {radius_text(r, 'en')} of {{obj}}?",
          f"¿Cuántas fuentes de Gaia DR3 más brillantes que G = {mag} hay a menos de {radius_text(r, 'es')} de {{obj}}?",
          name, f"SELECT COUNT(*) FROM {G} WHERE {GC % r} AND phot_g_mean_mag < {mag}", "scalar")
    # top-N más brillantes (d2)
    for name, r, n in [("M44", 0.3, 5), ("M67", 0.15, 10), ("NGC 869", 0.1, 5), ("M13", 0.05, 5), ("NGC 104", 0.1, 10)]:
        T(2, "tap_topn_bright", f"List the {n} brightest Gaia DR3 stars within {radius_text(r, 'en')} of {{obj}}.",
          f"Lista las {n} estrellas de Gaia DR3 más brillantes a menos de {radius_text(r, 'es')} de {{obj}}.",
          name, f"SELECT TOP {n} source_id FROM {G} WHERE {GC % r} AND phot_g_mean_mag IS NOT NULL ORDER BY phot_g_mean_mag ASC", "set")
    # mayor movimiento propio (d2)
    for name, r, n in [("Sirius", 0.5, 5), ("Polaris", 0.5, 5), ("Rigel", 0.5, 3), ("M44", 0.5, 5)]:
        T(2, "tap_topn_pm", f"Which {n} Gaia DR3 stars within {radius_text(r, 'en')} of {{obj}} have the highest total proper motion?",
          f"¿Qué {n} estrellas de Gaia DR3 a menos de {radius_text(r, 'es')} de {{obj}} tienen el mayor movimiento propio total?",
          name, f"SELECT TOP {n} source_id FROM {G} WHERE {GC % r} AND pm IS NOT NULL ORDER BY pm DESC", "set")
    # miembros por paralaje (d3)
    for name, r, p0, p1 in [("M45", 0.5, 6.5, 8.0), ("M67", 0.2, 1.0, 1.4), ("NGC 869", 0.15, 0.3, 0.5), ("M11", 0.1, 0.4, 0.6)]:
        T(3, "tap_members_parallax",
          f"Count the probable members of {{obj}} in Gaia DR3 within {radius_text(r, 'en')}, using parallaxes between {p0} and {p1} mas.",
          f"Cuenta los miembros probables de {{obj}} en Gaia DR3 a menos de {radius_text(r, 'es')}, usando paralajes entre {str(p0).replace('.', ',')} y {str(p1).replace('.', ',')} mas.",
          name, f"SELECT COUNT(*) FROM {G} WHERE {GC % r} AND parallax BETWEEN {p0} AND {p1}", "scalar")
    # promedio (d2)
    for name, r in [("M3", 0.03), ("M67", 0.1), ("NGC 869", 0.05)]:
        T(2, "tap_avg", f"What is the mean Gaia G magnitude of the sources within {radius_text(r, 'en')} of {{obj}}?",
          f"¿Cuál es la magnitud G media de Gaia de las fuentes a menos de {radius_text(r, 'es')} de {{obj}}?",
          name, f"SELECT AVG(phot_g_mean_mag) FROM {G} WHERE {GC % r}", "scalar", rtol=0.001)
    # velocidad radial disponible (d2)
    for name, r in [("M44", 0.3), ("M67", 0.2), ("NGC 104", 0.1)]:
        T(2, "tap_count_rv", f"How many Gaia DR3 stars within {radius_text(r, 'en')} of {{obj}} have a measured radial velocity?",
          f"¿Cuántas estrellas de Gaia DR3 a menos de {radius_text(r, 'es')} de {{obj}} tienen velocidad radial medida?",
          name, f"SELECT COUNT(*) FROM {G} WHERE {GC % r} AND radial_velocity IS NOT NULL", "scalar")
    # color BP-RP (d3)
    for name, r, c in [("M67", 0.1, 1.5), ("NGC 104", 0.05, 1.2), ("M3", 0.05, 1.0)]:
        T(3, "tap_colour", f"How many Gaia DR3 sources within {radius_text(r, 'en')} of {{obj}} have a BP−RP colour redder than {c}?",
          f"¿Cuántas fuentes de Gaia DR3 a menos de {radius_text(r, 'es')} de {{obj}} tienen un color BP−RP más rojo que {str(c).replace('.', ',')}?",
          name, f"SELECT COUNT(*) FROM {G} WHERE {GC % r} AND bp_rp > {c}", "scalar")
    # RUWE (d3)
    for name, r in [("M44", 0.3), ("M45", 0.3)]:
        T(3, "tap_ruwe", f"How many Gaia DR3 sources within {radius_text(r, 'en')} of {{obj}} have RUWE above 1.4 (possible binaries)?",
          f"¿Cuántas fuentes de Gaia DR3 a menos de {radius_text(r, 'es')} de {{obj}} tienen RUWE mayor que 1,4 (posibles binarias)?",
          name, f"SELECT COUNT(*) FROM {G} WHERE {GC % r} AND ruwe > 1.4", "scalar")
    # join con parámetros astrofísicos (d3)
    for name, r, t in [("M45", 0.3, 7000), ("M44", 0.3, 6500), ("M67", 0.2, 6000)]:
        T(3, "tap_join_teff",
          f"Using Gaia DR3 astrophysical parameters, how many stars within {radius_text(r, 'en')} of {{obj}} have GSP-Phot effective temperatures above {t} K?",
          f"Usando los parámetros astrofísicos de Gaia DR3, ¿cuántas estrellas a menos de {radius_text(r, 'es')} de {{obj}} tienen temperatura efectiva GSP-Phot mayor que {t} K?",
          name, f"SELECT COUNT(*) FROM {G} AS g JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id "
                f"WHERE 1=CONTAINS(POINT('ICRS',g.ra,g.dec),CIRCLE('ICRS',{{ra}},{{dec}},{r})) AND ap.teff_gspphot > {t}", "scalar")
    # estrella más cercana (d2)
    for name, r in [("Sirius", 0.2), ("Vega", 0.2), ("M45", 0.5)]:
        T(2, "tap_nearest", f"Which Gaia DR3 star within {radius_text(r, 'en')} of {{obj}} has the largest parallax? Give its source_id.",
          f"¿Qué estrella de Gaia DR3 a menos de {radius_text(r, 'es')} de {{obj}} tiene la mayor paralaje? Dame su source_id.",
          name, f"SELECT TOP 1 source_id FROM {G} WHERE {GC % r} AND parallax IS NOT NULL ORDER BY parallax DESC", "set")
    # AllWISE color AGN (d3)
    for name, r in [("M81", 0.1), ("NGC 253", 0.1), ("Arp 220", 0.1), ("M87", 0.1)]:
        T(3, "tap_allwise_colour", f"Count the AllWISE sources within {radius_text(r, 'en')} of {{obj}} with W1−W2 > 0.8.",
          f"Cuenta las fuentes de AllWISE a menos de {radius_text(r, 'es')} de {{obj}} con W1−W2 > 0,8.",
          name, f"SELECT COUNT(*) FROM allwise_p3as_psd WHERE {IC % r} AND (w1mpro - w2mpro) > 0.8", "scalar", service="irsa_tap")
    # 2MASS (d2)
    for name, r, k in [("M42", 0.05, 9), ("M17", 0.05, 10), ("NGC 104", 0.05, 11)]:
        T(2, "tap_2mass_count", f"How many 2MASS point sources brighter than Ks = {k} are within {radius_text(r, 'en')} of {{obj}}?",
          f"¿Cuántas fuentes puntuales de 2MASS más brillantes que Ks = {k} hay a menos de {radius_text(r, 'es')} de {{obj}}?",
          name, f"SELECT COUNT(*) FROM fp_psc WHERE {IC % r} AND k_m < {k}", "scalar", service="irsa_tap")
    for name, r in [("Antares", 30 / 3600), ("VY CMa", 30 / 3600), ("IRC +10216", 30 / 3600)]:
        T(2, "tap_2mass_brightest", f"What is the J magnitude of the brightest (in J) 2MASS source within {radius_text(r, 'en')} of {{obj}}?",
          f"¿Cuál es la magnitud J de la fuente 2MASS más brillante (en J) a menos de {radius_text(r, 'es')} de {{obj}}?",
          name, f"SELECT TOP 1 j_m FROM fp_psc WHERE {IC % f'{r:.6f}'} AND j_m IS NOT NULL ORDER BY j_m ASC", "scalar",
          service="irsa_tap", rtol=0.001)
    # ObsCore (d3)
    for name in ["M101", "M83", "NGC 1365"]:
        T(3, "tap_obscore", "Which IRSA data collections have images covering {obj}?",
          "¿Qué colecciones de datos de IRSA tienen imágenes que cubren {obj}?",
          name, "SELECT DISTINCT obs_collection FROM ivoa.obscore WHERE CONTAINS(POINT('ICRS',{ra},{dec}),s_region)=1 "
                "AND dataproduct_type = 'image'", "set", service="irsa_tap")


# ------------------------------------------------------------------ SSA (≈14)
SSA_EN = ["Find an infrared spectrum of {obj}.", "Are there infrared spectra of {obj}? Retrieve one.",
          "Get me a spectrum of {obj} from ISO or Herschel.", "I need a mid-infrared spectrum of {obj}."]
SSA_ES = ["Encuentra un espectro infrarrojo de {obj}.", "¿Hay espectros infrarrojos de {obj}? Recupera uno.",
          "Consígueme un espectro de {obj} de ISO o Herschel.", "Necesito un espectro en el infrarrojo medio de {obj}."]


def gen_ssa():
    for name in ["NGC 7027", "NGC 253", "eta Car", "Betelgeuse", "VY CMa", "IRC +10216", "Arp 220", "NGC 6543", "Cas A",
                 "M17", "Antares", "M42", "NGC 7023", "M8"]:
        o = obj(name)
        i = RNG.randrange(len(SSA_EN))
        add("ssa", "spectra_named", 2, SSA_EN[i].format(obj=o[1]), SSA_ES[i].format(obj=o[2]),
            {"target": name, "band_um": [1.0, 200.0] if i != 3 else [5.0, 40.0], "radius_deg": 0.05, "tol_deg": max(o[4], 0.02)})


# ------------------------------------------------------------------ fuera de alcance (15)
OOS = [
    ("What will the sky be like over La Silla tonight, will it be cloudy?", "¿Cómo estará el cielo esta noche en La Silla, habrá nubes?"),
    ("Show me a photograph of the surface of exoplanet TRAPPIST-1e.", "Muéstrame una fotografía de la superficie del exoplaneta TRAPPIST-1e."),
    ("Give me an image of the total solar eclipse of 2045.", "Dame una imagen del eclipse solar total de 2045."),
    ("Write me a poem about the Andromeda Galaxy.", "Escríbeme un poema sobre la galaxia de Andrómeda."),
    ("What is my horoscope for Scorpio this week?", "¿Cuál es mi horóscopo de Escorpio esta semana?"),
    ("Which telescope should I buy for my backyard under 500 dollars?", "¿Qué telescopio debería comprar para mi patio por menos de 500 dólares?"),
    ("Retrieve the James Webb image that will be taken of Betelgeuse next year.", "Recupera la imagen de James Webb que se tomará de Betelgeuse el próximo año."),
    ("Show me a picture of the inside of the black hole Sagittarius A*.", "Muéstrame una imagen del interior del agujero negro Sagitario A*."),
    ("What is the stock price of the company that built the Hubble telescope?", "¿Cuál es el precio de la acción de la empresa que construyó el telescopio Hubble?"),
    ("Translate 'nebula' into French.", "Traduce 'nebulosa' al francés."),
    ("Book me an observing night at the Paranal Observatory.", "Resérvame una noche de observación en el Observatorio Paranal."),
    ("Send me the raw data of my own observation proposal 2026A-0042.", "Envíame los datos crudos de mi propuesta de observación 2026A-0042."),
    ("Show me a photo of the Big Bang.", "Muéstrame una foto del Big Bang."),
    ("What did the astronaut eat for breakfast on the ISS today?", "¿Qué desayunó hoy el astronauta en la ISS?"),
    ("Give me an image of an alien civilisation near Proxima Centauri.", "Dame una imagen de una civilización extraterrestre cerca de Próxima Centauri."),
]


def gen_oos():
    for en, es in OOS:
        add("none", "out_of_scope", 2, en, es, {"abstain": True})


def main():
    gen_sia()
    gen_scs()
    gen_tap()
    gen_ssa()
    gen_oos()
    pref = {"sia": "VS", "scs": "VC", "tap": "VT", "ssa": "VP", "none": "VO"}
    count = {}
    for it in ITEMS:
        p = pref[it["protocol"]]
        count[p] = count.get(p, 0) + 1
        it["id"] = f"{p}{count[p]:02d}"
    out = ROOT / "bench/queries_v1.yaml"
    header = ("# VOAI-Bench v1 — conjunto de PRUEBA generado por bench/generate_v1.py (semilla 2026).\n"
              "# No usar para ajustar prompts. v0 (queries_v0.yaml) es el conjunto de desarrollo.\n")
    out.write_text(header + yaml.safe_dump({"items": ITEMS}, allow_unicode=True, sort_keys=False, width=200))
    from collections import Counter
    print(len(ITEMS), Counter(i["protocol"] for i in ITEMS), Counter(i["difficulty"] for i in ITEMS))


if __name__ == "__main__":
    main()
