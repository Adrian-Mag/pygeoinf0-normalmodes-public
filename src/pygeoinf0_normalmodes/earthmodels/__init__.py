"""Global mantle Vs and Vp models read from their original files in their native parametrisations: evaluate them, take
depth slices, profiles and great-circle sections, and plot them. From the pack global-models-raw-2026-10-30 (commit ff7cb94),
whose README and CHECKS describe every model and what was verified; only the file locations changed here.

    from pygeoinf0_normalmodes import earthmodels as gm
    gm.list_models()                         # what there is
    gm.info("SAVANI")                        # what one model contains, its sources and files
    m = gm.load("SAVANI")                    # default source; gm.load("SAVANI", source="voxels")
    print(m)                                 # fields as stored, units, references, coverage
    m.evaluate("dvs", lat=-20, lon=20, depth=2800)        # as stored (here percent)
    m.evaluate("dvs", -20, 20, 2800, unit="fraction", about_mean=True)
    lat, lon, v = m.depth_slice("dvs", 2800)
    gm.plot_depth_slice(m, "dvs", 2800)

Needs scipy; matplotlib to plot and netCDF4 for the EMC and grid sources. The originals are read from
$GLOBALMODELS_ORIGINALS; a few slow-to-parse members are cached in $GLOBALMODELS_CACHE. Please cite each model's paper.
"""
from ._model import Model, great_circle, voigt_linear, voigt_vp, voigt_vs
from .catalogue import CATALOGUE, default_source, info, list_models, model_names
from .plotting import plot_compare, plot_cross_section, plot_depth_slice, plot_rms_profiles

__all__ = ["load", "list_models", "info", "model_names", "CATALOGUE", "Model", "great_circle", "voigt_vs",
           "voigt_vp", "voigt_linear", "plot_depth_slice", "plot_cross_section", "plot_compare",
           "plot_rms_profiles"]


def _loader(name, source):
    from .readers import detox, emc, me16, ritsema, savani, semucb, tables
    table = {
        ("DETOX-P2", "grid"): lambda: detox.load("DETOX-P2", "grid"),
        ("DETOX-P2", "mesh"): lambda: detox.load("DETOX-P2", "mesh"),
        ("DETOX-P3", "grid"): lambda: detox.load("DETOX-P3", "grid"),
        ("DETOX-P3", "mesh"): lambda: detox.load("DETOX-P3", "mesh"),
        ("GLAD-M25", "emc"): emc.glad_m25,
        ("GyPSuM", "emc"): emc.gypsum,
        ("HMSL-P06", "author"): lambda: tables.HMSL("HMSL-P06"),
        ("HMSL-P06", "emc"): lambda: emc.hmsl_emc("HMSL-P06"),
        ("HMSL-S06", "author"): lambda: tables.HMSL("HMSL-S06"),
        ("HMSL-S06", "emc"): lambda: emc.hmsl_emc("HMSL-S06"),
        ("ME16", "nc4"): lambda: me16.load("nc4"),
        ("ME16", "epix"): lambda: me16.load("epix"),
        ("MTT2022", "author"): tables.MTT2022,
        ("P12", "sph"): ritsema.load_p12,
        ("S20RTS", "sph"): ritsema.load_s20rts,
        ("S362ANI", "emc"): emc.s362ani,
        ("S40RTS", "sph"): ritsema.load_s40rts,
        ("SAVANI", "sh"): lambda: savani.load("sh"),
        ("SAVANI", "grid"): lambda: savani.load("grid"),
        ("SAVANI", "voxels"): lambda: savani.load("voxels"),
        ("SAW24B16", "emc"): emc.saw24b16,
        ("SEMUCB-WM1", "native"): semucb.load,
        ("SEMUCB-WM1", "emc"): emc.semucb_emc,
        ("SGLOBE-rani", "native"): ritsema.load_sglobe_native,
        ("SGLOBE-rani", "emc"): emc.sglobe_emc,
        ("SP12RTS", "sph"): ritsema.load_sp12rts,
        ("TX2011", "emc"): emc.tx2011,
        ("UU-P07", "author"): tables.UUP07,
    }
    return table[(name, source)]


def load(name, source=None):
    """One model read from one of its sources (default: the first listed by info(name))."""
    key = {n.lower(): n for n in CATALOGUE}.get(str(name).lower())
    if key is None:
        raise KeyError(f"no model {name!r}; models: {', '.join(model_names())}")
    source = source or default_source(key)
    if source not in CATALOGUE[key]["sources"]:
        raise KeyError(f"{key} has sources {list(CATALOGUE[key]['sources'])}, not {source!r}")
    return _loader(key, source)()
