"""Plots of the catalogue (d.plot.coefficients(), ...) and of the kernels (K.plot, ...). matplotlib is optional: the plot extra.

Each takes ax= and returns its axes; with no ax it makes its own figure. Axes and titles state the units as the README states
them, and what is not settled: the kernels' unit reading is the thesis's, not confirmed by the mode code's author, and the
topography closure is not independently tested.
"""
import textwrap

import numpy as np

COEFFICIENT = "splitting coefficient (as in the files)"
DENSITY = "100 K_l / dr_l, per km, per unit fractional perturbation"
BOUNDARY = "K_d: response to 1 km of height\n(the files' scale)"
UNITS_NOTE = "unit reading as in the thesis, not confirmed by the mode code's author"
TOPOGRAPHY_NOTE = "topography closure not independently tested"


def pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plots need matplotlib: install the plugin's plot extra, or matplotlib itself") from None
    return plt


def _axes(ax):
    return pyplot().subplots(figsize=(8, 5))[1] if ax is None else ax


def _title(ax, main, *notes):
    ax.set_title("\n".join(textwrap.fill(line, 95) for line in (main,) + notes), fontsize=9)


def _rows(data):
    if len(data) == 0:
        raise ValueError("this selection has no rows")
    return data


def _jitter(data):
    """Each degree's rows spread over t, so that they do not sit on one line."""
    width = np.maximum(2 * data.s, 1)
    return data.s + 0.7 * (data.t / width - 0.5)


def _model_note(model):
    return (TOPOGRAPHY_NOTE,) if "_cmb_" in model else ()


class DataPlots:
    """d.plot: the plots of one catalogue, as selected."""

    def __init__(self, data):
        self.data = data

    def coefficients(self, ax=None):
        """The coefficients against degree, with their sigma."""
        d, ax = _rows(self.data), _axes(ax)
        ax.errorbar(_jitter(d), d.value, yerr=d.sigma, fmt=".", ms=3, lw=0.4, alpha=0.6)
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.set(xlabel="degree s (within a degree, t increases)", ylabel=COEFFICIENT)
        _title(ax, f"coefficients: {d.label}")
        return ax

    def snr(self, ax=None):
        """value / sigma against degree."""
        d, ax = _rows(self.data), _axes(ax)
        ax.plot(_jitter(d), d.value / d.sigma, ".", ms=3, alpha=0.6)
        for level, style in ((0.0, "-"), (2.0, ":"), (-2.0, ":")):
            ax.axhline(level, color="0.5", lw=0.7, ls=style)
        ax.set(xlabel="degree s (within a degree, t increases)", ylabel="value / sigma (dimensionless)")
        _title(ax, f"value / sigma: {d.label}")
        return ax

    def sigma(self, ax=None, bins=30):
        """The distribution of sigma, one histogram per degree."""
        d, ax = _rows(self.data), _axes(ax)
        logs = np.log10(d.sigma)
        edges = np.linspace(logs.min(), logs.max() + 1e-9, bins + 1)
        for s in np.unique(d.s).tolist():
            ax.hist(logs[d.s == s], bins=edges, histtype="step", label=f"s = {s}")
        ax.set(xlabel="log10 sigma (as in the files)", ylabel="rows")
        ax.legend(fontsize=8)
        _title(ax, f"sigma: {d.label}")
        return ax

    def mode(self, mode, ax=None, model=None):
        """One mode's coefficients as bars with error bars, in file order; model overlays that model's synthetics."""
        d = self.data.select(modes=[mode], quiet=True)
        if len(d) == 0:
            raise ValueError(f"mode {mode} has no rows in this selection")
        ax = _axes(ax)
        x = np.arange(len(d))
        ax.bar(x, d.value, yerr=d.sigma, color="tab:blue", alpha=0.7, error_kw={"lw": 0.7}, label="data")
        notes = ()
        if model is not None:
            ax.plot(x, d.model_values(model), "x", color="tab:red", ms=4, label=d.model_label(model))
            ax.legend(fontsize=8)
            notes = _model_note(model)
        ticks, names = [], []
        for s in np.unique(d.s).tolist():
            rows = np.flatnonzero(d.s == s)
            ticks.append(rows.mean())
            names.append(f"s = {s}")
            ax.axvline(rows[0] - 0.5, color="0.8", lw=0.7)
        ax.set_xticks(ticks, names)
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.set(xlabel="rows of the file: t = 0..2s within each degree", ylabel=COEFFICIENT)
        _title(ax, f"{mode}: {self.data.label}", *notes)
        return ax

    def across(self, s, t, ax=None, model=None):
        """One coefficient (s, t) across the modes that have it."""
        d = self.data.select(degrees=[s], t=[t], quiet=True)
        if len(d) == 0:
            raise ValueError(f"no rows at (s, t) = ({s}, {t}) in this selection")
        ax = _axes(ax)
        x = np.arange(len(d))
        ax.errorbar(x, d.value, yerr=d.sigma, fmt="o", ms=3, lw=0.7, label="data")
        notes = ()
        if model is not None:
            ax.plot(x, d.model_values(model), "x", color="tab:red", ms=4, label=d.model_label(model))
            ax.legend(fontsize=8)
            notes = _model_note(model)
        step = max(1, len(d) // 40)
        ax.set_xticks(x[::step], d.mode.tolist()[::step], rotation=90, fontsize=7)
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.set(xlabel="mode", ylabel=COEFFICIENT)
        _title(ax, f"(s, t) = ({s}, {t}): {self.data.label}", *notes)
        return ax

    def overview(self, ax=None):
        """A heat map, mode against degree, of the mean |value| / sigma over t; blank where a mode has no row."""
        d, ax = _rows(self.data), _axes(ax)
        degrees = np.unique(d.s)
        grid = np.full((len(d.modes), degrees.size), np.nan)
        for i, mode in enumerate(d.modes):
            for j, s in enumerate(degrees.tolist()):
                rows = (d.mode == mode) & (d.s == s)
                if rows.any():
                    grid[i, j] = np.mean(np.abs(d.value[rows]) / d.sigma[rows])
        image = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        ax.figure.colorbar(image, ax=ax, label="mean |value| / sigma over t (dimensionless)")
        ax.set_xticks(np.arange(degrees.size), degrees.tolist())
        step = max(1, len(d.modes) // 40)
        ax.set_yticks(np.arange(len(d.modes))[::step], list(d.modes)[::step], fontsize=7)
        ax.set(xlabel="degree s", ylabel="mode")
        _title(ax, f"overview: {d.label}")
        return ax

    def versus(self, model, mode=None, ax=None):
        """Data against a model's synthetics, with a residual panel; mode narrows it to one mode, row by row.

        Returns (axes, residual axes). A given ax is divided in two. On a catalogue with Crust5.1 subtracted, Crust5.1 is
        subtracted from a model that holds it, so that both sides are the same quantity."""
        d = self.data if mode is None else self.data.select(modes=[mode], quiet=True)
        _rows(d)
        top = _axes(ax)
        box = top.get_position()
        top.set_position([box.x0, box.y0 + 0.34 * box.height, box.width, 0.66 * box.height])
        bottom = top.figure.add_axes([box.x0, box.y0, box.width, 0.24 * box.height])
        synthetic = d.model_values(model)
        residual = (d.value - synthetic) / d.sigma
        name = d.model_label(model)
        if mode is None:
            top.errorbar(synthetic, d.value, yerr=d.sigma, fmt=".", ms=3, lw=0.3, alpha=0.5)
            span = np.array([min(synthetic.min(), d.value.min()), max(synthetic.max(), d.value.max())])
            top.plot(span, span, color="0.4", lw=0.8, label="data = synthetic")
            top.legend(fontsize=8)
            top.set(xlabel=f"synthetic: {name}", ylabel=f"data: {COEFFICIENT}")
            bottom.plot(_jitter(d), residual, ".", ms=3, alpha=0.5)
            bottom.set_xlabel("degree s")
        else:
            x = np.arange(len(d))
            top.errorbar(x, d.value, yerr=d.sigma, fmt="o", ms=3, lw=0.7, label="data")
            top.plot(x, synthetic, "x", color="tab:red", ms=4, label=name)
            top.legend(fontsize=8)
            top.set(xlabel="rows of the file", ylabel=COEFFICIENT)
            top.xaxis.get_major_locator().set_params(integer=True)
            bottom.plot(x, residual, ".", ms=4)
            bottom.set_xlabel("rows of the file")
            bottom.xaxis.get_major_locator().set_params(integer=True)
        bottom.axhline(0.0, color="0.5", lw=0.7)
        bottom.set_ylabel("(data - syn) / sigma")
        rms = float(np.sqrt(np.mean(residual ** 2)))
        _title(top, f"{mode or 'all rows'} against {name}, rms residual / sigma {rms:.3g}", self.data.label, *_model_note(model))
        return top, bottom


def _radii(ax, kernels, along="y"):
    """The discontinuities of the kernels' table, the named boundaries labelled."""
    from . import BOUNDARIES
    line = ax.axhline if along == "y" else ax.axvline
    for r in kernels.markers:
        if 0.0 < r < kernels.R:
            line(r, color="0.85", lw=0.6, zorder=0)
    transform = ax.get_yaxis_transform() if along == "y" else ax.get_xaxis_transform()
    for name, r in BOUNDARIES.items():
        if r < kernels.R:
            line(r, color="0.65", lw=0.7, zorder=0)
            place = (1.0, r) if along == "y" else (r, 1.0)
            ax.text(*place, name, transform=transform, fontsize=6, color="0.4", va="center", ha="left")


def kernel(kernels, mode, s, fields, ax=None, n=2000):
    ax = _axes(ax)
    r = np.linspace(0.0, kernels.R, n)
    for field in fields:
        ax.plot(kernels.function(mode, field, s)(r), r, label=field)
    _radii(ax, kernels)
    ax.axvline(0.0, color="0.5", lw=0.7)
    ax.set(xlabel=DENSITY, ylabel="radius (km)", ylim=(0.0, kernels.R))
    ax.legend(fontsize=8)
    _title(ax, f"{mode}, s = {s}", UNITS_NOTE)
    return ax


def kernel_modes(kernels, modes, field, s, ax=None, n=2000):
    ax = _axes(ax)
    r = np.linspace(0.0, kernels.R, n)
    for mode in modes:
        ax.plot(kernels.function(mode, field, s)(r), r, label=mode)
    _radii(ax, kernels)
    ax.axvline(0.0, color="0.5", lw=0.7)
    ax.set(xlabel=DENSITY, ylabel="radius (km)", ylim=(0.0, kernels.R))
    ax.legend(fontsize=8, ncol=1 + len(modes) // 12)
    _title(ax, f"{field}, s = {s}", UNITS_NOTE)
    return ax


def kernel_boundaries(kernels, mode, s=None, boundaries=None, ax=None):
    """Bars of the boundary kernels, one group per boundary and one bar per degree. A name or a radius in km."""
    from . import BOUNDARIES
    ax = _axes(ax)
    table = kernels.table(mode, "topo-all")
    degrees = np.unique(table.s).tolist() if s is None else [s]
    wanted = list(BOUNDARIES) if boundaries is None else list(boundaries)
    held = [b for b in wanted if not isinstance(b, str) or BOUNDARIES[b] in table.r_inner
            or np.any(np.abs(table.r_inner - BOUNDARIES[b]) <= 1e-6)]
    if not held:
        raise ValueError(f"{mode}: none of these boundaries is in {table.path}")
    width = 0.8 / len(degrees)
    for j, degree in enumerate(degrees):
        values = []
        for b in held:
            try:
                values.append(kernels.boundary(mode, b, degree))
            except ValueError:
                values.append(np.nan)      # no row at this degree: left blank, not zero
        ax.bar(np.arange(len(held)) + j * width, values, width=width, label=f"s = {degree}")
    names = [(f"{b}\n{BOUNDARIES[b]:.0f} km" if isinstance(b, str) else f"{float(b):.0f} km") for b in held]
    ax.set_xticks(np.arange(len(held)) + 0.4 - width / 2, names, fontsize=7)
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set(xlabel="boundary", ylabel=BOUNDARY)
    ax.legend(fontsize=8)
    _title(ax, f"{mode}: boundary kernels", UNITS_NOTE, TOPOGRAPHY_NOTE)
    return ax


def kernel_map(kernels, field, s, modes=None, ax=None, normalise=True):
    """A mode-by-layer map of the density at degree s; modes without rows at s are left out, and named in the title."""
    ax = _axes(ax)
    modes = kernels.modes(field) if modes is None else list(modes)
    held = [m for m in modes if np.any(kernels.table(m, field).s == s)]
    if not held:
        raise ValueError(f"no mode here has a {field} kernel at s = {s}")
    radii, W = kernels.layer_matrix(field, s, held)
    shell = ~(kernels.prem.zero_thickness | kernels.prem.inverted)
    order = np.argsort(0.5 * (kernels.prem.r_inner[shell] + kernels.prem.r_outer[shell]))
    density = W / (kernels.prem.r_outer[shell] - kernels.prem.r_inner[shell])[order]
    if normalise:
        peak = np.abs(density).max(axis=1, keepdims=True)
        density = np.divide(density, peak, out=np.zeros_like(density), where=peak > 0)
    limit = 1.0 if normalise else float(np.abs(density).max()) or 1.0
    mesh = ax.pcolormesh(np.arange(len(held)), radii, density.T, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit)
    label = f"{field}: density / its largest |value| in the mode (dimensionless)" if normalise else f"{field}: {DENSITY}"
    ax.figure.colorbar(mesh, ax=ax, label=label)
    step = max(1, len(held) // 40)
    ax.set_xticks(np.arange(len(held))[::step], held[::step], rotation=90, fontsize=7)
    _radii(ax, kernels)
    ax.set(xlabel="mode", ylabel="radius (km), PREM layer midpoints", ylim=(0.0, kernels.R))
    left = len(modes) - len(held)
    _title(ax, f"{field}, s = {s}: {len(held)} modes" + (f" ({left} without rows at this degree left out)" if left else ""), UNITS_NOTE)
    return ax
