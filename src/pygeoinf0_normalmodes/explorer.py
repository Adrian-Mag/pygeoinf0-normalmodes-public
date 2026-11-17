"""One window for the data and the kernels: nm.explore(root), or python -m pygeoinf0_normalmodes. Matplotlib widgets only.

The window draws through the same d.select and d.plot / K.plot methods as a script, so what it shows can be repeated by hand.
A plain script needs an interactive backend (macosx or TkAgg, both come with matplotlib and Python here). A notebook needs
%matplotlib tk or %matplotlib osx before the call, which opens a separate window; with the inline backend the controls cannot
respond, and explore says so.
"""
import os
import re
from pathlib import Path

import numpy as np

from .plots import pyplot

DATA_VIEWS = ("coefficients", "value / sigma", "sigma", "one mode", "(s, t) across modes", "overview", "data vs synthetics")
KERNEL_VIEWS = ("volumetric kernels", "modes overlaid", "mode x layer map", "boundary kernels")
VIEWS = DATA_VIEWS + KERNEL_VIEWS
INNER_CORE = {"all modes": None, "without inner-core": False, "inner-core only": True}
PLOT_BOX = [0.25, 0.33, 0.51, 0.53]
_OPEN = []      # a window lives until closed, whether or not the caller keeps it: widgets stop responding when collected


def parse_numbers(text):
    """"2,4" -> values [2, 4]; "0-5", "3-" or "-5" -> an inclusive range; "" -> nothing. Returns (values, range)."""
    text = text.strip()
    if not text:
        return None, None
    found = re.fullmatch(r"(\d*)\s*-\s*(\d*)", text)
    if found and any(found.groups()):
        return None, tuple(int(g) if g else None for g in found.groups())
    if re.fullmatch(r"\d+(\s*,\s*\d+)*", text):
        return [int(x) for x in text.split(",")], None
    raise ValueError(f"not numbers or a range: {text!r}; write 2,4 or 0-5")


class Explorer:
    """The window's state and controls. set(...) is what every control calls, and what a script may call too."""

    def __init__(self, root=None, dataset=None, crust=True, degree_zero=False, view=VIEWS[0], mode=None, s=2, t=0, model=None):
        from . import DEFAULT_DATASET, FIELDS, Kernels, _root
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; known: {', '.join(VIEWS)}")
        self.root = _root(root)
        if not (self.root / "observed").is_dir():
            raise ValueError(f"{self.root} is not the data folder: it has no observed/. Give the folder holding observed/, kernels/ "
                             "and synthetics/, or set PLI_NORMAL_MODE_DATA and give none")
        self._base = {}
        self.state = dict(view=view, dataset=dataset or DEFAULT_DATASET, crust=crust, degree_zero=degree_zero, mode=mode,
                          s=s, t=t, model=model, field="vs", fields=tuple(FIELDS), family="", n="", l="", degrees="",
                          inner_core="all modes", snr_min=0.0, sigma_max=None, overlay="", boundary="", show_model=False)
        self.error, self.selected, self.message = None, None, ""
        try:
            self.kernels, self._kernel_error = Kernels(self.root), None
        except (OSError, ValueError) as e:
            self.kernels, self._kernel_error = None, str(e)
        self._select()
        self.models = self.selected.models() if self.selected is not None else []
        if model is None and self.models:
            self.state["model"] = "S20RTS-Crust5.1" if "S20RTS-Crust5.1" in self.models else self.models[0]
        self.plt = pyplot()
        self.fig = self.plt.figure(figsize=(14.5, 8.8))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("pygeoinf0-normalmodes")
        self._syncing = False
        self._build()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw()

    # selection -------------------------------------------------------------------------------------------------------
    def _catalogue(self):
        from . import read_catalogue
        st = self.state
        key = (st["dataset"], st["crust"], st["degree_zero"])
        if key not in self._base:
            self._base[key] = read_catalogue(self.root, st["dataset"], crust=st["crust"], degree_zero=st["degree_zero"])
        return self._base[key]

    def _select(self):
        """The shared selection: every data and kernel view reads self.selected, and message says what it did."""
        st = self.state
        try:
            base = self._catalogue()
            n, n_range = parse_numbers(st["n"])
            l, l_range = parse_numbers(st["l"])
            degrees, degree_range = parse_numbers(st["degrees"])
            self.selected = base.select(family=st["family"] or None, n=n, n_range=n_range, l=l, l_range=l_range, degrees=degrees,
                                        degree_range=degree_range, inner_core=INNER_CORE[st["inner_core"]],
                                        snr_min=st["snr_min"] or None, sigma_max=st["sigma_max"], quiet=True)
            self.message, self.error = f"{base.size_text} -> {self.selected.size_text}", None
            if st["mode"] not in self.selected.modes:
                st["mode"] = self.selected.modes[0] if self.selected.modes else None
        except (OSError, ValueError) as e:
            self.error = str(e)
            self.message = "selection refused"

    @property
    def modes(self):
        return self.selected.modes if self.selected is not None else ()

    # controls --------------------------------------------------------------------------------------------------------
    def _build(self):
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

        from . import DATASETS, FIELDS, SUPERSEDED_DATASETS
        st, fig = self.state, self.fig
        self.status = fig.text(0.25, 0.99, "", fontsize=9, va="top")
        self.widgets, self.controls = {}, []

        def box(rect, title=None):
            ax = fig.add_axes(rect)
            self.controls.append(ax)
            if title:
                ax.set_title(title, fontsize=8, loc="left", pad=2)
            return ax

        def radio(name, rect, labels, active, title, on):
            w = RadioButtons(box(rect, title), labels, active=labels.index(active) if active in labels else 0)
            for label in w.labels:
                label.set_fontsize(8)
            w.on_clicked(on)
            self.widgets[name] = w

        def text(name, rect, label, initial):
            w = TextBox(box(rect), label, initial=initial, label_pad=0.02)
            w.label.set_fontsize(8)
            w.on_submit(lambda value, key=name: self.set(**{key: value}))
            self.widgets[name] = w

        def slider(name, rect, label, low, high, value, step=None, fmt=None):
            w = Slider(box(rect), label, low, high, valinit=value, valstep=step, valfmt=fmt)
            w.label.set_fontsize(8)
            w.valtext.set_fontsize(8)
            self.widgets[name] = w
            return w

        radio("view", [0.01, 0.53, 0.17, 0.42], list(VIEWS), st["view"], "view (data, then kernels)", lambda v: self.set(view=v))
        fixed = [d for d in DATASETS if d not in SUPERSEDED_DATASETS]
        radio("dataset", [0.01, 0.44, 0.17, 0.07], fixed, st["dataset"], "dataset", lambda v: self.set(dataset=v))
        switches = CheckButtons(box([0.01, 0.35, 0.17, 0.07], "read_catalogue"), ["Crust5.1 subtracted", "keep degree 0"],
                                [st["crust"], st["degree_zero"]])
        switches.on_clicked(self._on_switch)
        self.widgets["switches"] = switches
        radio("inner_core", [0.01, 0.24, 0.17, 0.09], list(INNER_CORE), st["inner_core"], "inner-core modes",
              lambda v: self.set(inner_core=v))
        text("family", [0.05, 0.19, 0.13, 0.03], "family", st["family"])
        text("n", [0.05, 0.15, 0.13, 0.03], "n", st["n"])
        text("l", [0.05, 0.11, 0.13, 0.03], "l", st["l"])
        text("degrees", [0.05, 0.07, 0.13, 0.03], "degrees", st["degrees"])
        fig.text(0.01, 0.045, "family: 00s*   n, l, degrees: 2,4 or 0-5\npress Enter in a box to apply", fontsize=7, va="top")

        high = len(self.modes) - 1
        slider("mode_index", [0.31, 0.235, 0.37, 0.022], "mode", 0, max(high, 1), self._mode_index(), step=1, fmt="%d").on_changed(
            lambda v: self.set(mode_index=int(v)))
        previous, following = Button(box([0.23, 0.195, 0.06, 0.03]), "previous"), Button(box([0.30, 0.195, 0.06, 0.03]), "next")
        previous.on_clicked(lambda _: self.step(-1))
        following.on_clicked(lambda _: self.step(1))
        self.widgets["previous"], self.widgets["next"] = previous, following
        text("mode", [0.44, 0.195, 0.08, 0.03], "mode id", st["mode"] or "")
        slider("s", [0.31, 0.155, 0.37, 0.022], "degree s", 0, 14, st["s"], step=2, fmt="%d").on_changed(lambda v: self.set(s=int(v)))
        slider("t", [0.31, 0.12, 0.37, 0.022], "t (0..2s)", 0, 28, st["t"], step=1, fmt="%d").on_changed(lambda v: self.set(t=int(v)))
        slider("snr_min", [0.31, 0.085, 0.37, 0.022], "min |value|/sigma", 0.0, 10.0, st["snr_min"], step=0.1, fmt="%.1f").on_changed(
            lambda v: self.set(snr_min=float(v)))
        bottom, top = self._sigma_limits()
        slider("sigma_max", [0.31, 0.05, 0.37, 0.022], "log10 max sigma", bottom, top, top, fmt="%.2f").on_changed(
            lambda v: self.set(sigma_max=None if v >= self._sigma_limits()[1] else float(10.0 ** v)))

        if self.models:
            names = [m.replace("S20RTS-Crust5.1_cmb_", "cmb ")[:26] for m in self.models]
            self._model_names = dict(zip(names, self.models))
            active = names[self.models.index(st["model"])] if st["model"] in self.models else names[0]
            radio("model", [0.83, 0.56, 0.16, 0.34], names, active, "synthetics", lambda v: self.set(model=self._model_names[v]))
        overlay = CheckButtons(box([0.83, 0.50, 0.16, 0.04]), ["overlay on data views"], [st["show_model"]])
        overlay.on_clicked(lambda _: self.set(show_model=not self.state["show_model"]))
        self.widgets["show_model"] = overlay
        fields = CheckButtons(box([0.83, 0.37, 0.16, 0.11], "volumetric kernels"), list(FIELDS), [f in st["fields"] for f in FIELDS])
        fields.on_clicked(self._on_field)
        self.widgets["fields"] = fields
        radio("field", [0.83, 0.25, 0.16, 0.10], list(FIELDS), st["field"], "overlay and map field", lambda v: self.set(field=v))
        text("overlay", [0.88, 0.20, 0.11, 0.03], "modes", st["overlay"])
        text("boundary", [0.88, 0.16, 0.11, 0.03], "boundaries", st["boundary"])
        fig.text(0.83, 0.15, "modes: 00s05,01s03 (blank: 5 from the\nmode slider). boundaries: cmb,d670,5000\n(blank: all); a name or a radius in km",
                 fontsize=7, va="top")

    def _sigma_limits(self):
        sigma = self.selected.sigma if self.selected is not None and len(self.selected) else np.array([1e-3, 1.0])
        base = self._base.get((self.state["dataset"], self.state["crust"], self.state["degree_zero"]))
        sigma = base.sigma if base is not None and len(base) else sigma
        low, high = float(np.log10(sigma.min())), float(np.log10(sigma.max()))
        return low, high if high > low else low + 1.0

    def _mode_index(self):
        return self.modes.index(self.state["mode"]) if self.state["mode"] in self.modes else 0

    def _on_switch(self, label):
        if label.startswith("Crust"):
            self.set(crust=not self.state["crust"])
        else:
            self.set(degree_zero=not self.state["degree_zero"])

    def _on_field(self, field):
        current = set(self.state["fields"])
        current.symmetric_difference_update({field})
        order = [label.get_text() for label in self.widgets["fields"].labels]
        self.set(fields=tuple(f for f in order if f in current))

    def _on_key(self, event):
        if event.key in ("left", "right"):
            self.step(-1 if event.key == "left" else 1)

    def step(self, by):
        if self.modes:
            self.set(mode=self.modes[(self._mode_index() + by) % len(self.modes)])

    def set(self, **changes):
        """Change anything in the state (view, mode, s, t, dataset, crust, family, n, l, degrees, snr_min, ...) and redraw."""
        if self._syncing:
            return
        if "mode_index" in changes:
            index = changes.pop("mode_index")
            if self.modes:
                changes["mode"] = self.modes[min(index, len(self.modes) - 1)]
        unknown = set(changes) - set(self.state)
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        self.state.update(changes)
        self.state["t"] = min(self.state["t"], 2 * self.state["s"])
        if set(changes) & {"dataset", "crust", "degree_zero", "family", "n", "l", "degrees", "inner_core", "snr_min", "sigma_max"}:
            self._select()
        self._sync()
        self.draw()

    def _sync(self):
        """The slider and box of the mode, degree and t follow the state, without calling back."""
        self._syncing = True
        try:
            w = self.widgets
            w["mode_index"].valmax = max(len(self.modes) - 1, 1)
            w["mode_index"].ax.set_xlim(0, w["mode_index"].valmax)
            w["mode_index"].set_val(self._mode_index())
            w["mode"].set_val(self.state["mode"] or "")
            w["s"].set_val(self.state["s"])
            w["t"].set_val(self.state["t"])
        finally:
            self._syncing = False

    # drawing ---------------------------------------------------------------------------------------------------------
    def _clear(self):
        for ax in list(self.fig.axes):
            if ax not in self.controls:
                ax.remove()
        return self.fig.add_axes(PLOT_BOX)

    def draw(self):
        """Draw the view through the public plots; what cannot be drawn is said in the plot, it never closes the window."""
        ax = self._clear()
        st = self.state
        try:
            if self.error:
                raise ValueError(self.error)
            self._draw(ax, st["view"])
        except (OSError, ValueError, KeyError) as e:
            for other in list(self.fig.axes):
                if other not in self.controls and other is not ax:
                    other.remove()
            ax.clear()
            ax.set_position(PLOT_BOX)
            ax.text(0.5, 0.5, f"cannot draw this view:\n{e}", ha="center", va="center", wrap=True, fontsize=9, transform=ax.transAxes)
            ax.set_axis_off()
        self.status.set_text(f"{st['view']}   |   {self.message}   |   mode {st['mode']}, s = {st['s']}, t = {st['t']}")
        self.fig.canvas.draw_idle()

    def _kernels(self):
        if self.kernels is None:
            raise ValueError(f"no kernels: {self._kernel_error}")
        return self.kernels

    def _overlay_modes(self):
        chosen = [m.strip() for m in self.state["overlay"].split(",") if m.strip()]
        if chosen:
            return chosen
        i = self._mode_index()
        return list(self.modes[i:i + 5])

    def _boundaries(self):
        chosen = [b.strip() for b in self.state["boundary"].split(",") if b.strip()]
        if not chosen:
            return None
        return [b if not re.fullmatch(r"[\d.]+", b) else float(b) for b in chosen]

    def _draw(self, ax, view):
        st, d = self.state, self.selected
        model = st["model"] if st["show_model"] else None
        if view == "coefficients":
            d.plot.coefficients(ax=ax)
        elif view == "value / sigma":
            d.plot.snr(ax=ax)
        elif view == "sigma":
            d.plot.sigma(ax=ax)
        elif view == "one mode":
            d.plot.mode(st["mode"], ax=ax, model=model)
        elif view == "(s, t) across modes":
            d.plot.across(st["s"], st["t"], ax=ax, model=model)
        elif view == "overview":
            d.plot.overview(ax=ax)
        elif view == "data vs synthetics":
            if st["model"] is None:
                raise ValueError("no synthetics found beside the data")
            d.plot.versus(st["model"], mode=st["mode"], ax=ax)
        elif view == "volumetric kernels":
            self._kernels().plot(st["mode"], st["s"], fields=st["fields"], ax=ax)
        elif view == "modes overlaid":
            self._kernels().plot_modes(self._overlay_modes(), st["field"], st["s"], ax=ax)
        elif view == "mode x layer map":
            self._kernels().plot_map(st["field"], st["s"], modes=self.modes, ax=ax)
        elif view == "boundary kernels":
            self._kernels().plot_boundaries(st["mode"], boundaries=self._boundaries(), ax=ax)

    def save(self, path, dpi=150):
        self.fig.savefig(path, dpi=dpi)
        return Path(path)

    def show(self):
        self.plt.show()


def explore(root=None, dataset=None, crust=True, degree_zero=False, view=VIEWS[0], mode=None, s=2, t=0, model=None, show=True, save=None):
    """Open the window on a data folder (default: PLI_NORMAL_MODE_DATA) and return its Explorer.

    save= writes the first view to an image. A plain script blocks until the window closes; a notebook needs %matplotlib tk or
    %matplotlib osx first."""
    explorer = Explorer(root, dataset, crust, degree_zero, view, mode, s, t, model)
    _OPEN.append(explorer)
    explorer.fig.canvas.mpl_connect("close_event", lambda _: _OPEN.remove(explorer) if explorer in _OPEN else None)
    if save is not None:
        explorer.save(save)
    if show:
        backend = explorer.plt.get_backend().lower()
        if "inline" in backend or backend in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            print(f"matplotlib's backend is {explorer.plt.get_backend()}: this is a picture, and the controls will not respond. In a "
                  "notebook run %matplotlib tk or %matplotlib osx before nm.explore(); in a script, unset MPLBACKEND.")
        explorer.show()
    return explorer


def main(argv=None):
    """python -m pygeoinf0_normalmodes [--root FOLDER] [--view V] [--mode M] [--save IMAGE] [--no-show] [--list]"""
    import argparse

    from . import DATASETS, read_catalogue
    parser = argparse.ArgumentParser(prog="python -m pygeoinf0_normalmodes", description="One window for the data and the kernels.")
    parser.add_argument("--root", default=os.environ.get("PLI_NORMAL_MODE_DATA"), help="the data folder, holding observed/, kernels/ and synthetics/ (default: PLI_NORMAL_MODE_DATA)")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default=None)
    parser.add_argument("--view", choices=VIEWS, default=VIEWS[0])
    parser.add_argument("--mode")
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--no-crust", action="store_true", help="keep the Crust5.1 prediction in the data")
    parser.add_argument("--degree-zero", action="store_true", help="keep degree 0")
    parser.add_argument("--save", help="write the view to this image")
    parser.add_argument("--no-show", action="store_true", help="open no window")
    parser.add_argument("--list", action="store_true", help="print what the folder holds and stop")
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("give --root, or set PLI_NORMAL_MODE_DATA")
    if args.list:
        kwargs = {"dataset": args.dataset} if args.dataset else {}
        d = read_catalogue(args.root, crust=not args.no_crust, degree_zero=args.degree_zero, **kwargs)
        print(f"{d.dataset}: {d.size_text}; degrees {np.unique(d.s).tolist()}")
        print("synthetics:", ", ".join(d.models()))
        return 0
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    try:
        explore(args.root, args.dataset, not args.no_crust, args.degree_zero, args.view, args.mode, args.s, args.t, args.model,
                show=not args.no_show, save=args.save)
    except ValueError as e:
        parser.error(str(e))
    return 0
