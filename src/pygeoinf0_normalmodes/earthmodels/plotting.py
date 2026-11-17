"""Plots: depth-slice maps, great-circle sections, radial r.m.s. profiles and a side-by-side comparison of
several models at one depth on a common colour scale. Matplotlib only; coastlines from data/coastlines.npz
(GSHHG crude resolution, Wessel and Smith 1996, LGPL).

Relative perturbations are drawn red (slow) to blue (fast) on a symmetric scale; absolute speeds on a
sequential scale unless about_mean=True turns them into perturbations about the lateral mean.
"""
import os

import numpy as np

from ._model import ABSOLUTE_UNITS

_COAST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coastlines.npz")
WAVE_FIELDS = {"vs": ["dvs", "vs", "vs_voigt", "dvsh", "vsh"], "vp": ["dvp", "vp", "vp_voigt"]}


def field_for(model, wave):
    """The field of a model that best represents isotropic 'vs' or 'vp' (stored first, then derived):
    dvs, vs, vs_voigt, dvsh, vsh (or dvp, vp, vp_voigt). None if the model has neither."""
    have = model.fields() + model.derived_fields()
    return next((f for f in WAVE_FIELDS[wave] if f in have), None)


def _coastlines(ax, lon_range=(-180, 180)):
    if os.path.exists(_COAST):
        z = np.load(_COAST)
        lon, lat = z["lon"], z["lat"]
        if lon_range[0] >= 0:
            lon = np.where(np.isfinite(lon), np.mod(lon, 360.0), np.nan)
            lon = np.where(np.abs(np.diff(lon, prepend=np.nan)) > 180, np.nan, lon)
        ax.plot(lon, lat, color="k", lw=0.4)


def _frame(ax):
    _coastlines(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))


def _label(model, field, unit, about_mean):
    u = model.unit_of(field, unit, about_mean)
    rel = u in ("percent", "fraction")
    name = field + (" (about lateral mean)" if about_mean else "")
    return f"{name} [{u}]", rel


def plot_depth_slice(model, field, depth, step=1.0, unit=None, about_mean=False, vlim=None, ax=None,
                     cmap=None, title=None):
    """A global map of one field at one depth. Returns the axes."""
    import matplotlib.pyplot as plt
    lat, lon, v = model.depth_slice(field, depth, step=step, unit=unit, about_mean=about_mean)
    lab, rel = _label(model, field, unit, about_mean)
    ax = ax or plt.figure(figsize=(9, 4.8)).add_subplot(111)
    if rel:
        vlim = vlim or float(np.nanmax(np.abs(v)))
        im = ax.pcolormesh(lon, lat, v, cmap=cmap or "RdBu", vmin=-vlim, vmax=vlim, shading="auto")
    else:
        lo, hi = np.nanpercentile(v, [1, 99])
        im = ax.pcolormesh(lon, lat, v, cmap=cmap or "viridis", vmin=lo, vmax=hi, shading="auto")
    _frame(ax)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"{model.name} ({model.source})  {field} at {depth:g} km")
    plt.colorbar(im, ax=ax, shrink=0.75, label=lab)
    return ax


def plot_cross_section(model, field, start, end, n=181, depths=None, unit=None, about_mean=False, vlim=None,
                       ax=None, cmap=None):
    """A great-circle section from start=(lat, lon) to end=(lat, lon), distance against depth."""
    import matplotlib.pyplot as plt
    cs = model.cross_section(field, start, end, n=n, depths=depths, unit=unit, about_mean=about_mean)
    v = cs["values"]
    lab, rel = _label(model, field, unit, about_mean)
    ax = ax or plt.figure(figsize=(9, 4.2)).add_subplot(111)
    if rel:
        vlim = vlim or float(np.nanmax(np.abs(v)))
        im = ax.pcolormesh(cs["distance"], cs["depth"], v, cmap=cmap or "RdBu", vmin=-vlim, vmax=vlim,
                           shading="auto")
    else:
        im = ax.pcolormesh(cs["distance"], cs["depth"], v, cmap=cmap or "viridis", shading="auto")
    ax.set_ylim(cs["depth"].max(), cs["depth"].min())
    ax.set_xlabel(f"distance along the great circle (degrees) from ({start[0]:g}, {start[1]:g}) "
                  f"to ({end[0]:g}, {end[1]:g})")
    ax.set_ylabel("depth (km)")
    ax.set_title(f"{model.name} ({model.source})  {field}")
    plt.colorbar(im, ax=ax, shrink=0.8, label=lab)
    return ax


def plot_rms_profiles(items, depths=None, unit="percent", about_mean=True, ax=None, n_lat=90):
    """Lateral r.m.s. against depth for several (model, field) pairs, in percent about the lateral mean.
    Default depths: 80 between 50 km (shallower, some models carry crust, water or air) and the bottom of each
    model's coverage."""
    import matplotlib.pyplot as plt
    ax = ax or plt.figure(figsize=(5, 7)).add_subplot(111)
    for model, field in items:
        top, bot = model.coverage(field)
        d = np.linspace(max(top, 50.0), bot, 80) if depths is None else np.asarray(depths, float)
        d = d[(d >= top) & (d <= bot)]
        d, r = model.rms_profile(field, depths=d, about_mean=about_mean, unit=unit, n_lat=n_lat)
        ax.plot(r, d, label=f"{model.name} {field}")
    ax.set_ylim(2891, 0)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"lateral r.m.s. ({unit}{', mean removed' if about_mean else ''})")
    ax.set_ylabel("depth (km)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return ax


def plot_compare(items, depth, step=2.0, unit="percent", about_mean=True, vlim=None, ncols=3):
    """Maps of several (model, field) pairs at one depth on one colour scale (default: percent about each
    model's lateral mean, so different reference models do not matter). Returns the figure."""
    import matplotlib.pyplot as plt
    data = []
    for model, field in items:
        lat, lon, v = model.depth_slice(field, depth, step=step, unit=unit, about_mean=about_mean)
        data.append((model, field, lat, lon, v))
    vlim = vlim or max(float(np.nanpercentile(np.abs(v), 99.5)) for *_, v in data)
    nrows = -(-len(data) // ncols)
    nc = min(ncols, len(data))
    fig, axes = plt.subplots(nrows, nc, figsize=(3.6 * nc + 1.0, 1.9 * nrows + 0.6), squeeze=False,
                             layout="constrained")
    for ax, (model, field, lat, lon, v) in zip(axes.flat, data):
        im = ax.pcolormesh(lon, lat, v, cmap="RdBu", vmin=-vlim, vmax=vlim, shading="auto")
        _frame(ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"{model.name} {field}", fontsize=9)
    for ax in list(axes.flat)[len(data):]:
        ax.axis("off")
    fig.suptitle(f"{depth:g} km" + (", each model about its lateral mean" if about_mean else ""))
    fig.colorbar(im, ax=axes, shrink=0.6, label=unit)
    return fig
