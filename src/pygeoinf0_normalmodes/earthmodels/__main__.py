"""Command line:

    python -m globalmodels list
    python -m globalmodels info MODEL
    python -m globalmodels plot MODEL FIELD DEPTH [--source S] [--out FILE.png] [--about-mean] [--unit U]
                                                  [--step DEG]
    python -m globalmodels eval MODEL FIELD LAT LON DEPTH [--source S] [--about-mean] [--unit U]

FIELD may be a stored or derived field name, or "vs" / "vp" for the model's representative field.
"""
import argparse
import sys

import globalmodels as gm
from globalmodels.plotting import field_for


def _field(m, f):
    if f in m.fields() + m.derived_fields():
        return f
    g = field_for(m, f) if f in ("vs", "vp") else None
    if g is None:
        sys.exit(f"{m.name} ({m.source}) has no field {f!r}; fields: {m.fields()}, derived: {m.derived_fields()}")
    return g


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m globalmodels", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the models")
    pi = sub.add_parser("info", help="describe one model")
    pi.add_argument("model")
    for name in ("plot", "eval"):
        q = sub.add_parser(name)
        q.add_argument("model")
        q.add_argument("field")
        if name == "eval":
            q.add_argument("lat", type=float)
            q.add_argument("lon", type=float)
        q.add_argument("depth", type=float)
        q.add_argument("--source", default=None)
        q.add_argument("--about-mean", action="store_true", help="perturbation about the lateral mean")
        q.add_argument("--unit", default=None, help="fraction or percent (relative fields)")
        if name == "plot":
            q.add_argument("--out", default=None, help="PNG file (default MODEL_FIELD_DEPTHkm.png)")
            q.add_argument("--step", type=float, default=1.0, help="grid step in degrees")
    a = p.parse_args(argv)
    if a.cmd == "list":
        gm.list_models()
    elif a.cmd == "info":
        name = {n.lower(): n for n in gm.model_names()}.get(a.model.lower())
        if name is None:
            sys.exit(f"no model {a.model!r}; models: {', '.join(gm.model_names())}")
        gm.info(name)
        print(f"\nfields of the default source (other sources: gm.load({name!r}, source=...)):")
        print(gm.load(name))
    else:
        m = gm.load(a.model, a.source)
        f = _field(m, a.field)
        if a.cmd == "eval":
            v = m.evaluate(f, a.lat, a.lon, a.depth, unit=a.unit, about_mean=a.about_mean)
            print(f"{m.name} ({m.source}) {f} at lat {a.lat:g}, lon {a.lon:g}, depth {a.depth:g} km: "
                  f"{float(v):.6g} {m.unit_of(f, a.unit, a.about_mean)}")
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ax = gm.plot_depth_slice(m, f, a.depth, step=a.step, unit=a.unit, about_mean=a.about_mean)
            out = a.out or f"{m.name}_{f}_{a.depth:g}km.png"
            ax.figure.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(ax.figure)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
