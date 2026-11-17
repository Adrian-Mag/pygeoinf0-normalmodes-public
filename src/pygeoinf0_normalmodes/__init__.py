"""pygeoinf0-normalmodes: normal-mode splitting data and PREM-layer sensitivity kernels for pygeoinf0 (a plugin; plan 030).

The clean catalogue, kernels as functions, their plots, and forward maps:

    read_catalogue(root=None, dataset=DEFAULT_DATASET, crust=True, degree_zero=False, exclude=())
                                            the corrected coefficients, Crust5.1 subtracted, degree 0 removed -> SplittingData
    d.select(...), d.plot.overview(), ...   chainable selections; plots of the data, each with ax=
    explore(root)                           one window for the data and the kernels; or python -m pygeoinf0_normalmodes
    Kernels(root=None)                      K.function(mode, field, s) -> pg.Tabulated, the density against dr
                                            K.boundary(mode, boundary, s) -> float; K.plot(mode, s)
    model_space(boundaries, shells=None)    the volumetric fields and six boundaries as a pg.DirectSum (the thesis's form), or
                                            each field a flat sum of shells between named discontinuities
    forward(data, kernels, M=None, summands=None, rule="linear", into=None)
                                            the coefficients as pg.Functionals.separable on M
    problem(root_or_data, ...)              M, G, D from the same rows and kernels; D carries the reported noise
    earthmodels                             19 published Vs and Vp models from their original files: load, evaluate, plot
    to_harmonics(model, field, lmax, radii) a model on a spherical-harmonic basis -> HarmonicModel; in_basis, evaluate, element
    predict(models, data, kernels)          the coefficients of a model; recover_topography from a pair of synthetics

Files become frozen records with read-only NumPy arrays, radii in km, values exactly as written:

    read_prem_layers(path)                  rows "layer r_inner r_outer" -> PremLayers
    read_mode_list(path)                    one mode id per line (00s02) -> tuple of ids
    read_splitting_data(data_dir, dataset=DEFAULT_DATASET, modes=None, superseded=False)
                                            data_dir holds data_list_* and data_<folder>/<mode>.new -> SplittingData
    read_kernel(kernel_dir, mode, param, prem)
                                            kern_<mode>_<param>_PREM-layers-all_PREM-iso.kern rows "s layer value"
                                            -> KernelData, each row with its radius interval from prem
    list_kernel_modes(kernel_dir, param)    the modes with a kernel file for param
    read_synthetics(synthetics_dir, mode, model)
                                            raw_<mode>_<model>.raw rows "l s t value" -> Synthetics
    crust_corrected(data, synthetics_dir, model="Crust5.1"), synthetic_values(data, synthetics_dir, model)

A bad row raises FormatError naming the file and line; a missing file raises FileNotFoundError; an unknown dataset or
param raises ValueError.

Which coefficients are correct. The delivered folders data_arwen-paula and data_SP12RTS hold the same four files
(SUPERSEDED_MODES) about 100-300 times too small; data_arwen-paula-fixed-anomalies replaces those four and is otherwise
byte-identical. The two delivered folders are refused unless superseded=True. "SP12RTS-fixed-anomalies" is the SP12RTS
mode list read from the corrected folder. observed/fixed-data-files is a different delivery, without 22 modes, and is not
read: the corrected folder deliberately keeps the combined data for every other mode.

What the numbers mean, as the submitted thesis uses them (section "What the sensitivity kernels measure"; reproduced
in pygeoinf0's EXP-030), and checked here: with these rules, S20RTS with vp = 0.5 vs and rho = 0.3 vs gives the delivered
S20RTS synthetics to 1e-3 (tests), so no sqrt(4 pi) belongs on the volumetric kernels:
    - a volumetric value is the response to a 1 % perturbation over the layer, so the density against dr of a fractional
      perturbation is 100 K_l / dr_l; no further global factor;
    - a boundary kernel keeps its supplied scale and takes the height in km;
    - coefficients and kernels are in the catalogue's harmonic basis (CATALOGUE: Condon-Shortley phase, and non-zonal
      members 1/sqrt 2 of the orthonormal ones);
    - Crust5.1 (Moho, surface topography, solid-water interface) is subtracted from the data, and those three boundaries
      are then fixed. A problem that frees them must use crust=False.
Not confirmed by the mode code's author; the check above is against the code's own synthetics.

Layer numbers are not identifiers of boundaries. The same boundary is written as different numbers: the older topo
files label the core-mantle boundary by the first shell above it, the topo-all files by a zero-thickness row of the
same radius. Select by radius (KernelData.at_boundary, PremLayers.at_radius), which holds under both.

The tables hold rows that are not shells, kept as given (PremLayers.zero_thickness, PremLayers.inverted): zero-thickness
rows mark discontinuities, and the last row of the full table has r_outer < r_inner (surface radius, then 0). The
no-discontinuity table is not the set of layers the volume kernels use, so reading one against it can refuse a layer.
"""
from __future__ import annotations

import fnmatch
import functools
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pygeoinf0 as pg
from pygeoinf0.core.claims import ASSERTED, Claim
from pygeoinf0.spaces.harmonic import HarmonicConvention

__version__ = "0.9.0"

# name: (mode list, folder under data_dir)
DATASETS = {"arwen-paula-fixed-anomalies": ("data_list_arwen-paula", "data_arwen-paula-fixed-anomalies"),
            "SP12RTS-fixed-anomalies": ("data_list_SP12RTS", "data_arwen-paula-fixed-anomalies"),
            "arwen-paula": ("data_list_arwen-paula", "data_arwen-paula"),
            "SP12RTS": ("data_list_SP12RTS", "data_SP12RTS")}
DEFAULT_DATASET = "arwen-paula-fixed-anomalies"
SUPERSEDED_DATASETS = ("arwen-paula", "SP12RTS")
SUPERSEDED_MODES = ("00s03", "00s04", "00s05", "01s03")
# Inner-core-sensitive modes of this catalogue, after Table 3 of Deuss et al. (2013), as starred in the thesis; and the
# eight with the largest inner-core share of the squared kernels. For exclusion tests: exclude=INNER_CORE_MODES.
INNER_CORE_MODES = ("02s03", "03s01", "03s02", "05s02", "06s03", "08s01", "08s05", "09s02", "09s03", "09s04", "11s04",
                    "11s05", "11s06", "13s01", "13s02", "13s03", "13s06", "14s04", "15s03", "15s04", "16s05", "16s06",
                    "16s07", "17s01", "17s08", "18s03", "18s04", "18s06", "20s01", "20s05", "21s06", "21s07", "22s01",
                    "23s04", "23s05", "25s01", "25s02", "27s02")
INNER_CORE_EIGHT = ("13s01", "17s01", "20s01", "22s01", "23s04", "25s01", "25s02", "27s02")
# Boundary radii (km) of the topo-all kernels, from PREM_all-layers_radius; the first six are free in the thesis's problem.
BOUNDARIES = {"cmb": 3479.95776, "icb": 1221.49072, "dpp": 3630.0, "d670": 5701.0, "d400": 5971.0, "d220": 6151.0,
              "moho": 6346.6289062, "ocean": 6368.0, "surface": 6371.0}
FREE_BOUNDARIES = ("cmb", "icb", "dpp", "d670", "d400", "d220")
# The default of the shell layout and of problem: without dpp, the author's word 2026-11-03 (leaving it out changes the Gram by
# 4e-14). FREE_BOUNDARIES stays the thesis's, so that model_space() with no arguments and what depends on it do not change.
PROBLEM_BOUNDARIES = ("cmb", "icb", "d670", "d400", "d220")
FIELDS = ("vp", "vs", "rho")
R_EARTH, R_CMB, R_ICB = 6371.0, BOUNDARIES["cmb"], BOUNDARIES["icb"]
TAPER_FRACTION = 1e-3     # the centre taper r^2 / (r^2 + eps^2), eps = 1e-3 of the outer radius: it keeps G bounded on L2
CATALOGUE = HarmonicConvention("condon-shortley", scale=lambda l, j: np.where(np.asarray(j) > 0, math.sqrt(2.0), 1.0),
                               name="the normal-mode catalogue's basis")
VOLUME_PARAMS = ("vp", "vs", "rho")
TOPO_PARAMS = ("topo", "topo-all")
PARAMS = VOLUME_PARAMS + TOPO_PARAMS
RADIUS_TOL = 1e-6  # km; the tables repeat a boundary radius to about 1e-8 km

_MODE = re.compile(r"\d\d[st]\d\d")
_INT = re.compile(r"[+-]?\d+")


class FormatError(ValueError):
    """A file does not have the declared format; the message names the file and line."""


def _frozen(values, dtype=None):
    a = np.array(values, dtype=dtype)
    a.setflags(write=False)
    return a


def _rows(path, ncols):
    """Yield (line number, fields) of each non-blank row with exactly ncols fields; a missing file is refused."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != ncols:
                raise FormatError(f"{path}:{number}: expected {ncols} columns, found {len(fields)}")
            yield number, fields


def _int(text, path, number, what):
    if not _INT.fullmatch(text):
        raise FormatError(f"{path}:{number}: {what} is not an integer: {text!r}")
    return int(text)


def _float(text, path, number, what):
    try:
        value = float(text)
    except ValueError:
        raise FormatError(f"{path}:{number}: {what} is not a number: {text!r}") from None
    if not math.isfinite(value):
        raise FormatError(f"{path}:{number}: {what} is not finite: {text!r}")
    return value


def _check_mode(mode, where):
    if not isinstance(mode, str) or not _MODE.fullmatch(mode):
        raise FormatError(f"{where}: not a mode id like 00s02: {mode!r}")


def _check_param(param):
    if param not in PARAMS:
        raise ValueError(f"unknown kernel param {param!r}; known: {', '.join(PARAMS)}")


def _kernel_name(mode, param):
    return f"kern_{mode}_{param}_PREM-layers-all_PREM-iso.kern"


@dataclass(frozen=True, eq=False)
class PremLayers:
    """A table of PREM layers as given: aligned arrays layer, r_inner, r_outer (km)."""
    path: str
    layer: np.ndarray
    r_inner: np.ndarray
    r_outer: np.ndarray

    def __len__(self):
        return len(self.layer)

    @property
    def zero_thickness(self):
        """Mask of rows with r_inner == r_outer: they mark discontinuities."""
        return self.r_inner == self.r_outer

    @property
    def inverted(self):
        """Mask of rows with r_outer < r_inner, which are not shells (see the module docstring)."""
        return self.r_outer < self.r_inner

    def indices(self, layers):
        """Row index of each layer number; a number outside the table is refused."""
        layers = np.asarray(layers)
        order = np.argsort(self.layer)
        found = np.searchsorted(self.layer[order], layers).clip(max=len(self) - 1)
        bad = self.layer[order][found] != layers
        if bad.any():
            raise ValueError(f"layer {int(layers[bad][0])} is outside the table {self.path}")
        return order[found]

    def interval(self, layer):
        """(r_inner, r_outer) of one layer number, as given."""
        i = self.indices([layer])[0]
        return float(self.r_inner[i]), float(self.r_outer[i])

    def at_radius(self, radius, tol=RADIUS_TOL):
        """The layer numbers whose r_inner is this radius; a boundary may have more than one, so the result is an array."""
        return _frozen(self.layer[np.abs(self.r_inner - radius) <= tol])


def read_prem_layers(path):
    """Read rows "layer r_inner r_outer" (km). Layer numbers need not be contiguous, and must be unique."""
    layer, r_inner, r_outer = [], [], []
    for number, (a, b, c) in _rows(path, 3):
        n = _int(a, path, number, "layer")
        if n in layer:
            raise FormatError(f"{path}:{number}: layer {n} is repeated")
        layer.append(n)
        r_inner.append(_float(b, path, number, "r_inner"))
        r_outer.append(_float(c, path, number, "r_outer"))
    if not layer:
        raise FormatError(f"{path}: no layers")
    return PremLayers(str(path), _frozen(layer, int), _frozen(r_inner, float), _frozen(r_outer, float))


def read_mode_list(path):
    """Read one mode id per line, in file order; a malformed or repeated id is refused."""
    modes = []
    for number, (mode,) in _rows(path, 1):
        _check_mode(mode, f"{path}:{number}")
        if mode in modes:
            raise FormatError(f"{path}:{number}: mode {mode} is repeated")
        modes.append(mode)
    return tuple(modes)


def _values(chosen):
    return [chosen] if isinstance(chosen, (str, int, np.integer)) else list(chosen)


def _between(values, bounds):
    """The mask of values within (low, high), inclusive; None leaves an end open."""
    low, high = (None, None) if bounds is None else bounds
    keep = np.ones(np.shape(values), bool)
    if low is not None:
        keep &= values >= low
    if high is not None:
        keep &= values <= high
    return keep


def _mode_passes(mode, family, kind, n, n_range, l, l_range, inner_core):
    order, angular = int(mode[:2]), int(mode[3:])
    return ((family is None or fnmatch.fnmatchcase(mode, family)) and (kind is None or mode[2] == kind)
            and (n is None or order in _values(n)) and (l is None or angular in _values(l))
            and bool(_between(np.array(order), n_range)) and bool(_between(np.array(angular), l_range))
            and (inner_core is None or (mode in INNER_CORE_MODES) == inner_core))


@dataclass(frozen=True, eq=False)
class SplittingData:
    """Observed splitting coefficients: aligned arrays mode, s, t, value, sigma; smax is aligned with modes.

    t is one index 0..2s over the 2s+1 real harmonics, kept as given. corrections names what was subtracted from value.
    source holds read_catalogue's arguments and steps the selections since, so that d.select(crust=False) can read it again.
    """
    dataset: str
    modes: tuple
    smax: np.ndarray
    mode: np.ndarray
    s: np.ndarray
    t: np.ndarray
    value: np.ndarray
    sigma: np.ndarray
    corrections: tuple = ()
    source: dict | None = None
    steps: tuple = ()

    def __len__(self):
        return len(self.mode)

    @property
    def size_text(self):
        return f"{len(self.modes)} modes, {len(self)} rows"

    @property
    def label(self):
        """What this catalogue is, for a title."""
        crust = ", ".join(f"{c} subtracted" for c in self.corrections) or "no crustal correction"
        steps = [", ".join(f"{k}={v}" for k, v in step.items()) for step in self.steps]
        return "; ".join([f"{self.dataset}, {crust}", *steps, self.size_text])

    @property
    def plot(self):
        """The plots of this catalogue: d.plot.coefficients(), snr, sigma, mode, across, overview, versus."""
        from .plots import DataPlots
        return DataPlots(self)

    def _rows(self, keep):
        left = [i for i, m in enumerate(self.modes) if np.any(self.mode[keep] == m)]
        columns = {k: _frozen(getattr(self, k)[keep]) for k in ("mode", "s", "t", "value", "sigma")}
        return replace(self, modes=tuple(self.modes[i] for i in left), smax=_frozen(self.smax[left]), **columns)

    def select(self, modes=None, exclude=(), degrees=None, *, family=None, kind=None, n=None, n_range=None, l=None, l_range=None,
               degree_range=None, t=None, sigma_min=None, sigma_max=None, snr_min=None, snr_max=None, inner_core=None,
               dataset=None, crust=None, degree_zero=None, quiet=False):
        """A new catalogue of the rows chosen, each argument narrowing it; prints what it did unless quiet. A mode with no row
        left is dropped.

        modes, exclude     mode ids; degrees, t, n, l: one value or several (n, l from the id nnTll; t is 0..2s as given)
        family             a pattern on the id, e.g. "00s*"; kind: "s" or "t"
        n_range, l_range, degree_range
                           (low, high), inclusive; None leaves an end open
        sigma_min, sigma_max, snr_min, snr_max
                           bounds on sigma and on |value| / sigma, row by row
        inner_core         False leaves out INNER_CORE_MODES, True keeps only them
        dataset, crust, degree_zero
                           read_catalogue's switches: read the files again and repeat the selections so far
        """
        filters = dict(modes=modes, exclude=exclude or None, degrees=degrees, family=family, kind=kind, n=n, n_range=n_range,
                       l=l, l_range=l_range, degree_range=degree_range, t=t, sigma_min=sigma_min, sigma_max=sigma_max,
                       snr_min=snr_min, snr_max=snr_max, inner_core=inner_core)
        filters = {k: v for k, v in filters.items() if v is not None}
        switches = {k: v for k, v in (("dataset", dataset), ("crust", crust), ("degree_zero", degree_zero)) if v is not None}
        if kind not in (None, "s", "t"):
            raise ValueError(f"kind is 's' or 't', not {kind!r}")
        base = self
        if switches:
            if self.source is None:
                raise ValueError("dataset, crust and degree_zero read the files again: this catalogue did not come from "
                                 "read_catalogue, so read one with them")
            base = read_catalogue(**{**self.source, **switches})
            for step in self.steps:
                base = base.select(quiet=True, **step)
        keep = ~np.isin(base.mode, list(exclude))
        if modes is not None:
            keep &= np.isin(base.mode, _values(modes))
        if degrees is not None:
            keep &= np.isin(base.s, _values(degrees))
        if t is not None:
            keep &= np.isin(base.t, _values(t))
        keep &= _between(base.s, degree_range)
        keep &= _between(base.sigma, (sigma_min, sigma_max))
        keep &= _between(np.abs(base.value) / base.sigma, (snr_min, snr_max))
        allowed = [m for m in base.modes if _mode_passes(m, family, kind, n, n_range, l, l_range, inner_core)]
        keep &= np.isin(base.mode, allowed)
        out = replace(base._rows(keep), steps=base.steps + ((filters,) if filters else ()))
        if not quiet:
            print(f"{self.size_text} -> {out.size_text}")
        return out

    def without_degree_zero(self):
        """Degree 0 corrects the spherical reference, not its lateral variations."""
        return self._rows(self.s != 0)

    def _synthetics_dir(self, synthetics_dir=None):
        if synthetics_dir is not None:
            return synthetics_dir
        if self.source is None:
            raise ValueError("give synthetics_dir: this catalogue did not come from read_catalogue")
        return _root(self.source["root"]) / "synthetics"

    def model_values(self, model, synthetics_dir=None):
        """A model's synthetics at these rows, as the same quantity as value: a correction already subtracted from the data
        is subtracted from a model that holds it (S20RTS-Crust5.1 is S20RTS plus Crust5.1, tested on the real files)."""
        folder = self._synthetics_dir(synthetics_dir)
        out = synthetic_values(self, folder, model)
        for correction in self.corrections:
            if correction in model:
                out = out - synthetic_values(self, folder, correction)
        return _frozen(out)

    def model_label(self, model):
        """The model's name, saying what model_values took away from it."""
        taken = [c for c in self.corrections if c in model]
        return model + "".join(f" minus {c}" for c in taken)

    def models(self, synthetics_dir=None):
        """The names of the synthetic models beside this catalogue."""
        return list_synthetic_models(self._synthetics_dir(synthetics_dir))

    def smax_of(self, mode):
        """The smax in the file of this mode."""
        if mode not in self.modes:
            raise ValueError(f"mode {mode!r} is not in this selection of {self.dataset}")
        return int(self.smax[self.modes.index(mode)])


def _read_mode_file(path):
    """(smax, rows) of one .new file: first line smax, then rows "s t value sigma"."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    with path.open(encoding="utf-8") as f:
        lines = [(n, line.split()) for n, line in enumerate(f, 1) if line.strip()]
    if not lines or len(lines[0][1]) != 1:
        raise FormatError(f"{path}:1: the first line must hold only smax")
    smax = _int(lines[0][1][0], path, lines[0][0], "smax")
    rows, seen = [], set()
    for number, fields in lines[1:]:
        if len(fields) != 4:
            raise FormatError(f"{path}:{number}: expected 4 columns, found {len(fields)}")
        s = _int(fields[0], path, number, "s")
        t = _int(fields[1], path, number, "t")
        value = _float(fields[2], path, number, "value")
        sigma = _float(fields[3], path, number, "sigma")
        if not 0 <= s <= smax:
            raise FormatError(f"{path}:{number}: s = {s} is outside 0..smax = {smax}")
        if not 0 <= t <= 2 * s:
            raise FormatError(f"{path}:{number}: t = {t} is outside 0..2s for s = {s}")
        if sigma <= 0:
            raise FormatError(f"{path}:{number}: sigma = {sigma} is not positive")
        if (s, t) in seen:
            raise FormatError(f"{path}:{number}: (s, t) = ({s}, {t}) is repeated")
        seen.add((s, t))
        rows.append((s, t, value, sigma))
    return smax, rows


def read_splitting_data(data_dir, dataset=DEFAULT_DATASET, modes=None, superseded=False):
    """Read a dataset's modes, in the order of its mode list; modes selects a subset, in the given order.

    Only the listed modes are opened, so other files beside them are ignored. A delivered dataset with the uncorrected
    SUPERSEDED_MODES is refused unless superseded=True.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {', '.join(DATASETS)}")
    if dataset in SUPERSEDED_DATASETS and not superseded:
        raise ValueError(f"dataset {dataset!r} holds uncorrected values for {', '.join(SUPERSEDED_MODES)}: use "
                         f"{dataset}-fixed-anomalies, or superseded=True to read the delivery as it is")
    data_dir = Path(data_dir)
    list_name, folder = DATASETS[dataset]
    listed = read_mode_list(data_dir / list_name)
    if modes is None:
        modes = listed
    else:
        modes = (modes,) if isinstance(modes, str) else tuple(modes)
        for mode in modes:
            if mode not in listed:
                raise ValueError(f"mode {mode!r} is not listed for dataset {dataset}")
    smax, mode_column, rows = [], [], []
    for mode in modes:
        top, these = _read_mode_file(data_dir / folder / f"{mode}.new")
        smax.append(top)
        mode_column += [mode] * len(these)
        rows += these
    s, t, value, sigma = (list(c) for c in zip(*rows)) if rows else ([], [], [], [])
    return SplittingData(dataset, tuple(modes), _frozen(smax, int), _frozen(mode_column, "U5"), _frozen(s, int),
                         _frozen(t, int), _frozen(value, float), _frozen(sigma, float))


@dataclass(frozen=True, eq=False)
class KernelData:
    """One kernel file: aligned arrays s, layer, value, with each row's r_inner, r_outer (km) from the PREM table.

    Use the radii, not the layer numbers. A volumetric value is the response to 1 % over the layer (module docstring).
    """
    mode: str
    param: str
    path: str
    s: np.ndarray
    layer: np.ndarray
    value: np.ndarray
    r_inner: np.ndarray
    r_outer: np.ndarray

    def __len__(self):
        return len(self.s)

    @property
    def thickness(self):
        """r_outer - r_inner, with 0 on a row that is not a shell."""
        return _frozen(np.maximum(self.r_outer - self.r_inner, 0.0))

    @property
    def boundary_radii(self):
        """The distinct boundary radii of a topography kernel, ascending: r_inner under either numbering."""
        self._require_topography()
        return _frozen(np.unique(self.r_inner))

    def at_boundary(self, radius, tol=RADIUS_TOL):
        """The rows of a topography kernel at this boundary radius; a radius it does not hold is refused."""
        self._require_topography()
        keep = np.abs(self.r_inner - radius) <= tol
        if not keep.any():
            raise ValueError(f"no boundary at radius {radius} km in {self.path}")
        return replace(self, **{k: _frozen(getattr(self, k)[keep]) for k in ("s", "layer", "value", "r_inner", "r_outer")})

    def _require_topography(self):
        if self.param not in TOPO_PARAMS:
            raise ValueError(f"param {self.param!r} is a volume kernel: its rows are layers, not boundaries")


def read_kernel(kernel_dir, mode, param, prem):
    """Read one kernel file against a PremLayers table; a layer number outside the table is refused."""
    _check_param(param)
    _check_mode(mode, "read_kernel")
    if not isinstance(prem, PremLayers):
        raise TypeError("prem must be the PremLayers of read_prem_layers")
    path = Path(kernel_dir) / _kernel_name(mode, param)
    s, layer, value, seen = [], [], [], set()
    for number, (a, b, c) in _rows(path, 3):
        degree = _int(a, path, number, "s")
        n = _int(b, path, number, "layer")
        if degree < 0 or degree % 2:
            raise FormatError(f"{path}:{number}: s = {degree} is not even and non-negative")
        if (degree, n) in seen:
            raise FormatError(f"{path}:{number}: (s, layer) = ({degree}, {n}) is repeated")
        seen.add((degree, n))
        s.append(degree)
        layer.append(n)
        value.append(_float(c, path, number, "value"))
    if not s:
        raise FormatError(f"{path}: no rows")
    try:
        rows = prem.indices(layer)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    return KernelData(mode, param, str(path), _frozen(s, int), _frozen(layer, int), _frozen(value, float),
                      _frozen(prem.r_inner[rows]), _frozen(prem.r_outer[rows]))


def list_kernel_modes(kernel_dir, param):
    """The modes with a kernel file for param in this folder, sorted."""
    _check_param(param)
    kernel_dir = Path(kernel_dir)
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"no such folder: {kernel_dir}")
    suffix = _kernel_name("", param)[len("kern_"):]
    return tuple(sorted(p.name[len("kern_"):-len(suffix)] for p in kernel_dir.glob(f"kern_*{suffix}")))


@dataclass(frozen=True, eq=False)
class Synthetics:
    """Synthetic splitting coefficients of one mode and model: aligned arrays l, s, t, value, as given."""
    mode: str
    model: str
    path: str
    l: np.ndarray
    s: np.ndarray
    t: np.ndarray
    value: np.ndarray

    def __len__(self):
        return len(self.s)


def read_synthetics(synthetics_dir, mode, model):
    """Read raw_<mode>_<model>.raw, rows "l s t value"."""
    _check_mode(mode, "read_synthetics")
    path = Path(synthetics_dir) / f"raw_{mode}_{model}.raw"
    l, s, t, value = [], [], [], []
    for number, (a, b, c, d) in _rows(path, 4):
        l.append(_int(a, path, number, "l"))
        s.append(_int(b, path, number, "s"))
        t.append(_int(c, path, number, "t"))
        value.append(_float(d, path, number, "value"))
    if not s:
        raise FormatError(f"{path}: no rows")
    return Synthetics(mode, model, str(path), _frozen(l, int), _frozen(s, int), _frozen(t, int), _frozen(value, float))


def list_synthetic_models(synthetics_dir):
    """The model names of the raw_<mode>_<model>.raw files in this folder, sorted."""
    synthetics_dir = Path(synthetics_dir)
    if not synthetics_dir.is_dir():
        raise FileNotFoundError(f"no such folder: {synthetics_dir}")
    found = (re.fullmatch(r"raw_\d\d[st]\d\d_(.+)\.raw", p.name) for p in synthetics_dir.iterdir())
    return sorted({m.group(1) for m in found if m})


@functools.lru_cache(maxsize=8192)
def _synthetic_table(folder, mode, model, stamp):
    syn = read_synthetics(folder, mode, model)
    return syn.path, {(s, t): v for s, t, v in zip(syn.s.tolist(), syn.t.tolist(), syn.value.tolist())}


def synthetic_values(data, synthetics_dir, model):
    """The synthetics of model at the rows of data, aligned with them; a row the synthetics do not hold is refused."""
    out = np.zeros(len(data))
    for mode in data.modes:
        file = Path(synthetics_dir) / f"raw_{mode}_{model}.raw"
        path, table = _synthetic_table(str(synthetics_dir), mode, model, file.stat().st_mtime_ns if file.exists() else None)
        for i in np.flatnonzero(data.mode == mode):
            key = (int(data.s[i]), int(data.t[i]))
            if key not in table:
                raise ValueError(f"{path}: no row (s, t) = {key}")
            out[i] = table[key]
    return _frozen(out)


def crust_corrected(data, synthetics_dir, model="Crust5.1"):
    """data minus the crustal model's prediction (sigma unchanged: the correction's own uncertainty is not represented)."""
    if model in data.corrections:
        raise ValueError(f"{model} has already been subtracted")
    return replace(data, value=_frozen(data.value - synthetic_values(data, synthetics_dir, model)),
                   corrections=data.corrections + (model,))


def _root(root):
    root = root if root is not None else os.environ.get("PLI_NORMAL_MODE_DATA")
    if root is None:
        raise ValueError("give the data folder, or set PLI_NORMAL_MODE_DATA")
    return Path(root)


def read_catalogue(root=None, dataset=DEFAULT_DATASET, crust=True, degree_zero=False, exclude=(), superseded=False):
    """The coefficients to use: corrected, with Crust5.1 subtracted and degree 0 removed unless asked otherwise.

    root is the folder with observed/, kernels/ and synthetics/ (default: PLI_NORMAL_MODE_DATA). The files are not changed.
    """
    root = _root(root)
    data = read_splitting_data(root / "observed", dataset, superseded=superseded)
    if not degree_zero:
        data = data.without_degree_zero()
    if exclude:
        data = data._rows(~np.isin(data.mode, list(exclude)))
    source = dict(root=root, dataset=dataset, crust=crust, degree_zero=degree_zero, exclude=tuple(exclude), superseded=superseded)
    return replace(crust_corrected(data, root / "synthetics") if crust else data, source=source)


class Kernels:
    """The kernels of one data folder as functions (cached); the files' readers underneath.

    K.function(mode, field, s)     a pg.Tabulated on [0, R]: the density against dr of a fractional perturbation
    K.boundary(mode, boundary, s)  the response to 1 km of height on a named boundary (BOUNDARIES), or a radius in km
    K.plot(mode, s)                the three volumetric kernels against radius; plot_modes, plot_boundaries, plot_map
    """

    def __init__(self, root=None, volume="kernels/kernels-all_PREM-layers_Adrian",
                 topography="kernels/kernels-topo-all_PREM-layers_Adrian", table="PREM_all-layers_radius"):
        root = _root(root)
        self.volume, self.topography = root / volume, root / topography
        self.prem = read_prem_layers(self.volume / table)
        self._tables, self._functions, self._matrices, self._vectors = {}, {}, {}, {}
        rows = ~self.prem.inverted
        self.R = float(self.prem.r_outer[rows].max())
        self.markers = tuple(np.unique(self.prem.r_inner[self.prem.zero_thickness]).tolist())

    def modes(self, param="vs"):
        """The modes with a kernel of this param (topography for topo-all, the volume folder otherwise)."""
        return list_kernel_modes(self.topography if param == "topo-all" else self.volume, param)

    def table(self, mode, param):
        """The KernelData of one file."""
        key = (mode, param)
        if key not in self._tables:
            folder = self.topography if param == "topo-all" else self.volume
            self._tables[key] = read_kernel(folder, mode, param, self.prem)
        return self._tables[key]

    def function(self, mode, field, s, rule="linear"):
        """The kernel as a density against dr. rule "linear": between the layer midpoints and the surface, split at the
        discontinuities, flat to their edges (the thesis's computation). rule "constant": constant on each PREM layer."""
        if field not in FIELDS:
            raise ValueError(f"unknown field {field!r}; known: {', '.join(FIELDS)}")
        key = (mode, field, s, rule)
        if key not in self._functions:
            self._functions[key] = self._build(mode, field, s, rule)
        return self._functions[key]

    def _build(self, mode, field, s, rule):
        k = self.table(mode, field)
        row = k.s == s
        if not row.any():
            raise ValueError(f"{k.path}: no rows at s = {s}")
        layer, value, lo, dr = k.layer[row], k.value[row], k.r_inner[row], k.thickness[row]
        if np.any((dr == 0) & (value != 0)):
            raise ValueError(f"{k.path}: a nonzero value at s = {s} on a row that is not a shell")
        density = np.divide(100.0 * value, dr, out=np.zeros_like(value), where=dr > 0)
        name = f"K_{field}[{mode}, s={s}]"
        if rule == "linear":
            point = lo + 0.5 * dr                       # the surface row sits at its r_inner, with dr = 0
            order = np.argsort(point, kind="stable")
            inner = [r for r in self.markers if 0.0 < r < self.R]
            return pg.Tabulated((0.0, self.R), point[order], density[order], rule="linear", segments=inner,
                                extrapolation="flat", name=name)
        if rule == "constant":
            shell = dr > 0
            order = np.argsort(lo[shell])
            edges = np.r_[lo[shell][order], (lo[shell] + dr[shell]).max()]
            return pg.Tabulated((0.0, self.R), edges, density[shell][order], rule="constant", name=name)
        raise ValueError("rule is 'linear' or 'constant'")

    def boundary(self, mode, boundary, s):
        """The boundary kernel at degree s; boundary is a name in BOUNDARIES or a radius in km, never a layer number."""
        if isinstance(boundary, str) and boundary not in BOUNDARIES:
            raise ValueError(f"unknown boundary {boundary!r}; known: {', '.join(BOUNDARIES)}")
        radius = BOUNDARIES[boundary] if isinstance(boundary, str) else float(boundary)
        at = self.table(mode, "topo-all").at_boundary(radius)
        row = at.s == s
        if not row.any():
            raise ValueError(f"{at.path}: no row at s = {s}, radius {radius} km")
        return float(at.value[row][0])

    def layer_matrix(self, field, s, modes):
        """(midpoints, W): W[i, l] = 100 K_l of modes[i] at degree s, on the PREM shells, zero where a file lists no row.
        Kept, so repeating a forward costs a matrix product. A mode without rows at s is refused."""
        key = (field, s, tuple(modes))
        if key not in self._matrices:
            shell = np.flatnonzero(~(self.prem.zero_thickness | self.prem.inverted))
            order = shell[np.argsort(0.5 * (self.prem.r_inner[shell] + self.prem.r_outer[shell]))]
            position = np.full(self.prem.layer.max() + 1, -1)
            position[self.prem.layer[order]] = np.arange(order.size)
            W = np.zeros((len(modes), order.size))
            for i, mode in enumerate(modes):
                table = self.table(mode, field)
                row = table.s == s
                keep = row & (table.thickness > 0)
                if not row.any():
                    raise ValueError(f"{table.path}: no rows at s = {s}")
                W[i, position[table.layer[keep]]] = 100.0 * table.value[keep]
            W.setflags(write=False)
            self._matrices[key] = (0.5 * (self.prem.r_inner[order] + self.prem.r_outer[order]), W)
        return self._matrices[key]

    def boundary_vector(self, boundary, s, modes):
        """K_d at degree s for each of modes (kept); a mode without a row at s is refused."""
        key = (boundary, s, tuple(modes))
        if key not in self._vectors:
            self._vectors[key] = np.array([self.boundary(m, boundary, s) for m in modes])
        return self._vectors[key]

    def midpoints(self, within=None):
        """The midpoints of the PREM shells (km), optionally within (a, b): the radii at which predict reads a model."""
        shell = ~(self.prem.zero_thickness | self.prem.inverted)
        mid = np.sort(0.5 * (self.prem.r_inner[shell] + self.prem.r_outer[shell]))
        return mid if within is None else mid[(mid >= within[0]) & (mid <= within[1])]

    def plot(self, mode, s, fields=FIELDS, ax=None, n=2000):
        """The volumetric kernels against radius, the discontinuities drawn. Returns the axes; with no ax, a new figure."""
        from .plots import kernel
        return kernel(self, mode, s, fields, ax=ax, n=n)

    def plot_modes(self, modes, field, s, ax=None, n=2000):
        """One field at degree s for several modes, overlaid."""
        from .plots import kernel_modes
        return kernel_modes(self, modes, field, s, ax=ax, n=n)

    def plot_boundaries(self, mode, s=None, boundaries=None, ax=None):
        """The boundary kernels as bars, one per degree; boundaries are names in BOUNDARIES or radii in km."""
        from .plots import kernel_boundaries
        return kernel_boundaries(self, mode, s=s, boundaries=boundaries, ax=ax)

    def plot_map(self, field, s, modes=None, ax=None, normalise=True):
        """A mode-by-layer map of the density at degree s; each mode scaled to its largest value unless normalise=False."""
        from .plots import kernel_map
        return kernel_map(self, field, s, modes=modes, ax=ax, normalise=normalise)


def data_space(data, noise=False):
    """R^n with the rows' labels. noise=True gives it the covariance diag(sigma^2): this assumes independent errors, as
    reported, with no correlation between coefficients, and no error for the crustal correction."""
    labels = {"mode": data.mode.tolist(), "s": data.s.tolist(), "t": data.t.tolist()}
    return pg.Euclidean(len(data), covariance=data.sigma ** 2 if noise else None, labels=labels)


def _shell_name(field, inner, outer):
    return f"{field}_{inner or 'centre'}_{outer or 'surface'}"


def _shells(cuts, R):
    """(inner name, outer name, a, b) from the centre to R, cut at these names of BOUNDARIES."""
    unknown = [c for c in cuts if c not in BOUNDARIES]
    if unknown:
        raise ValueError(f"unknown cuts {unknown}; known: {', '.join(BOUNDARIES)}")
    if len(set(cuts)) != len(cuts):
        raise ValueError(f"a cut is repeated: {list(cuts)}")
    inside = sorted((c for c in cuts if BOUNDARIES[c] < R), key=BOUNDARIES.get)
    names = [None] + inside + [None]
    radii = [0.0] + [BOUNDARIES[c] for c in inside] + [R]
    return [(names[i], names[i + 1], radii[i], radii[i + 1]) for i in range(len(radii) - 1)]


class ModeSpace(pg.PiecewiseSpace):
    """The volumetric fields on the shells (pg.PiecewiseSpace) and the boundary heights after them, in one flat direct sum. The heights
    are summands by name; M.fields, M.split, M.glue and M.functional concern the volumetric fields only."""

    def __init__(self, partition, fields, heights, height_weights, where=None, weights=None):
        super().__init__(partition, fields, where=where, weights=weights)
        pg.DirectSum.__init__(self, {**dict(zip(self.names, self.summands)), **heights},
                              weights={**dict(zip(self.names, self.weights)), **height_weights})
        self.heights = tuple(heights)

    def model(self, models, name="m"):
        """The element of M for {field or boundary: HarmonicModel}, as nm.predict takes them; what is absent is zero. A field is
        built on the whole ball and restricted to its pieces, so a piece needs no nodes of its own; a height is the model's
        coefficients at the boundary's radius."""
        from pygeoinf0.realisations.series.spherical import SphericalHarmonicSeries as SH
        unknown = set(models) - set(self.fields) - set(self.heights)
        if unknown:
            raise ValueError(f"unknown field or boundary {sorted(unknown)}: fields {list(self.fields)}, heights {list(self.heights)}")
        parts = {}
        for key, hm in models.items():
            if key in self.fields:
                u = hm.element(self.fields[key])
                parts.update({self.key(key, n): self.restriction(key, n)(u) for n in self.pieces_of(key)})
            else:
                q = hm.in_basis("package")
                C = np.array([q(l, t, np.array([BOUNDARIES[key]]))[0] for l in range(q.lmax + 1) for t in range(2 * l + 1)])
                parts[key] = pg.Function(self.summand(key), pl=SH(q.lmax, C[:, None], np.zeros(1)), name=f"{hm.name}_{key}")
        return self(parts, name=name)

    def __repr__(self):
        return f"ModeSpace({', '.join(self.fields)} on {', '.join(self.partition.names)}; heights {', '.join(self.heights)})"


def model_space(boundaries=None, R=R_EARTH, shells=None):
    """The volumetric fields and boundary heights as the thesis poses them: vp and rho on the ball, vs on the inner core
    and the mantle. This is one choice of model space; forward takes any direct sum with a summands map.

    shells=("icb", "cmb", "d670", "d400", "d220") makes each field a flat direct sum of shells between those names of
    BOUNDARIES, e.g. vp_centre_icb, vp_icb_cmb, ..., vp_d220_surface; vs is always cut at the ICB and the CMB and has no
    outer-core summand. G does not change; a prior can then decide interface by interface whether the shells are correlated.
    boundaries defaults to FREE_BOUNDARIES, and with shells to PROBLEM_BOUNDARIES, without dpp.

    With shells cutting at both the ICB and the CMB the result is a ModeSpace, a pg.PiecewiseSpace with the same keys and the heights
    after it: a target across shells is M.functional("vp", bump.measure()), and M.split / M.glue move a whole-ball model in and
    out. Without both cuts vs is cut where vp is not, which is no single partition, and the result is a plain pg.DirectSum.

    The pairing is the volume average, 1/V with the Earth's V on every volume summand whatever its shell, so the shells of
    one field join isometrically. On a height it is (1 / (4 pi r_b^2)) int_{S^2} h k dOmega, an integral over the unit sphere:
    1/r_b^2 times the area average, not the area average itself.
    """
    if boundaries is None:
        boundaries = FREE_BOUNDARIES if shells is None else PROBLEM_BOUNDARIES
    V = 4.0 * math.pi * R ** 3 / 3.0
    heights = {name: pg.L2(pg.Sphere(1.0)) for name in boundaries}
    height_weights = {name: 1.0 / (4.0 * math.pi * BOUNDARIES[name] ** 2) for name in boundaries}
    if shells is not None and {"icb", "cmb"} <= set(shells):
        rows = _shells(shells, R)
        names = [f"{inner or 'centre'}_{outer or 'surface'}" for inner, outer, _, _ in rows]
        P = pg.LayeredBall([b for _, _, _, b in rows], names=names).partition
        X = pg.L2(P.domain, realisation="radial_harmonics")
        return ModeSpace(P, {f: X for f in FIELDS}, heights, height_weights, where={"vs": [n for n in names if n != "icb_cmb"]},
                         weights={f: 1.0 / V for f in FIELDS})
    if shells is None:
        domains = {"vp": pg.Ball(R), "vs_ic": pg.Ball(R_ICB), "vs_m": pg.Shell(R_CMB, R), "rho": pg.Ball(R)}
    else:
        domains = {}
        for field in FIELDS:
            cuts = tuple(shells) + tuple(c for c in ("icb", "cmb") if field == "vs" and c not in shells)
            for inner, outer, a, b in _shells(cuts, R):
                if field == "vs" and inner == "icb" and outer == "cmb":
                    continue
                domains[_shell_name(field, inner, outer)] = pg.Ball(b) if a == 0.0 else pg.Shell(a, b)
    spaces = {k: pg.L2(d, realisation="radial_harmonics") for k, d in domains.items()}
    weights = {k: 1.0 / V for k in domains}
    return pg.DirectSum({**spaces, **heights}, weights={**weights, **height_weights})


def _taper(outer):
    eps2 = (TAPER_FRACTION * outer) ** 2

    def tau(r):
        r = np.asarray(r, float)
        return r * r / (r * r + eps2)
    tau.__name__ = f"tau(eps={TAPER_FRACTION * outer:g} km)"
    return tau


def forward(data, kernels, M=None, summands=None, rule="linear", name="G", into=None):
    """The rows of data as functionals on M: c = sum_f int K_f m_f,st dr + sum_d K_d h_d,st, in the catalogue's basis.

    summands maps each summand of M to a field or boundary name; a summand left out does not enter. Default: M =
    model_space(), and a summand named vs_ic, vp_cmb_d670 and so on is read as its field. A mode without a kernel at (s) is
    refused, not taken as zero. into is the data space: by default data_space(data), without the noise; see problem.
    """
    M = model_space() if M is None else M
    summands = {k: (k.split("_")[0] if k.split("_")[0] in FIELDS else k) for k in M.names} if summands is None else summands
    radial, against, taper = {}, {}, {}
    rows = list(zip(data.mode.tolist(), data.s.tolist()))
    for key, what in summands.items():
        if what in FIELDS:
            radial[key] = [kernels.function(m, what, s, rule) for m, s in rows]
            backend = M.summand(key).backend
            against[key] = pg.LebesgueMeasure(pg.Interval(backend.a, backend.b))
            # the field's outer radius, not the summand's: splitting a space into shells must not change G
            taper[key] = _taper(R_ICB if what == "vs" and backend.b <= R_ICB + 1e-6 else kernels.R)
        else:
            radial[key] = [kernels.boundary(m, what, s) for m, s in rows]
    return pg.Functionals.separable(M, radial, np.c_[data.s, data.t], against=against, taper=taper,
                                    into=data_space(data) if into is None else into, name=name, convention=CATALOGUE)


@dataclass(frozen=True, eq=False)
class ModeProblem:
    """M, G, D = nm.problem(...), all from the same rows and kernels; p.data, p.kernels, p.d (the data as an element of D) too.
    p.noise is the noise as a measure on D, p.plain() the same problem on the unweighted D, and p.decompose(C) the data-space
    decomposition for a prior covariance C."""
    M: pg.DirectSum
    G: object
    D: pg.Euclidean
    data: SplittingData
    kernels: Kernels
    rule: str = "linear"

    def __iter__(self):
        return iter((self.M, self.G, self.D))

    @property
    def noise(self):
        """The reported noise on D: D's standard Gaussian when D carries it, else N(0, diag(sigma^2)) on the unweighted D."""
        return self.D.standard_gaussian() if not self.D.is_standard else pg.GaussianMeasure(
            self.D, pg.Diagonal(np.asarray(self.data.sigma, float) ** 2, self.D), name="N(0, sigma^2)")

    def plain(self):
        """The same M and rows with the unweighted D and the noise as a separate measure (p.noise)."""
        if self.D.is_standard:
            return self
        D = data_space(self.data)
        return replace(self, G=forward(self.data, self.kernels, self.M, rule=self.rule, into=D), D=D)

    def decompose(self, middle=None):
        """pg.decompose on the plain problem, the route for a prior covariance by degree (module README): the structured Gram
        needs a standard codomain. Its data are p.plain().d."""
        p = self.plain()
        return pg.decompose(p.G, p.noise, middle=middle)

    @property
    def d(self):
        return self.D(self.data.value)


def problem(source=None, boundaries=None, shells=None, noise=True, kernels=None, rule="linear", dataset=DEFAULT_DATASET, crust=True,
            degree_zero=False, exclude=()):
    """The model space, the forward map and the data space together: M, G, D = nm.problem(root, shells=(...)).

    source is the data folder (default: PLI_NORMAL_MODE_DATA) or a SplittingData, which keeps its own dataset and crust setting.
    noise=True makes G's codomain the labelled R^n with covariance diag(sigma^2), so pg.decompose(G, D.standard_gaussian()) uses the
    reported errors: this assumes they are independent, as reported, and gives the crustal correction no error. It changes the
    adjoint: G G* is then Gram W, with W = diag(1 / sigma^2). noise=False is today's unweighted D.

    With a prior covariance by degree (Covariance.glued, Covariance.angular), pg.decompose(G, D.standard_gaussian(), middle=C) works on the
    default D with pygeoinf0 plan 045; on an older pygeoinf0, p.decompose(C).

    boundaries defaults to PROBLEM_BOUNDARIES, with moho, ocean and surface added on data that keep the Crust5.1 prediction. A
    subtracted crust and a free crustal boundary are refused: they would count the crust twice.
    """
    if isinstance(source, SplittingData):
        data = source
    else:
        data = read_catalogue(source, dataset, crust=crust, degree_zero=degree_zero, exclude=exclude)
    subtracted = "Crust5.1" in data.corrections
    crustal = ("moho", "ocean", "surface")
    if boundaries is None:
        boundaries = PROBLEM_BOUNDARIES + (() if subtracted else crustal)
    elif subtracted and set(boundaries) & set(crustal):
        raise ValueError(f"Crust5.1 is subtracted from these data, which fixes {', '.join(crustal)}: use crust=False to free them")
    if kernels is None:
        if data.source is None:
            raise ValueError("give kernels: this catalogue did not come from read_catalogue")
        kernels = Kernels(data.source["root"])
    M = model_space(boundaries, shells=shells)
    D = data_space(data, noise=noise)
    return ModeProblem(M, forward(data, kernels, M, rule=rule, into=D), D, data, kernels, rule)


from . import earthmodels  # noqa: E402  (they need nothing above; here so the plugin's objects can name them)
from .explorer import explore  # noqa: E402
from .harmonic_models import BASES, HarmonicModel, predict, recover_topography, to_harmonics  # noqa: E402

PLUGIN = pg.port.plugins.Plugin(
    name="normalmodes", version=__version__, port=">=1,<2",
    objects={"Kernels": Kernels, "forward": forward, "model_space": model_space, "read_catalogue": read_catalogue,
             "to_harmonics": to_harmonics, "predict": predict, "problem": problem},
    claims=(Claim("first-order self-coupled splitting of an isotropic reference", ASSERTED,
                  "the kernels are the mode code's; anisotropy and coupling are not in the relation", key="relation"),
            Claim("the kernels' unit input", ASSERTED, "read as 1 % per layer and 1 km of height, as in the thesis; closes on "
                  "the S20RTS synthetics to 1e-3, not confirmed by the mode code's author", key="units"),),
    licence="BSD-3-Clause (coastlines.npz: LGPL-3.0-or-later, THIRD_PARTY.md)", source="github.com/Adrian-Mag/pygeoinf0-normalmodes-public")
pg.port.plugins.register(PLUGIN)

__all__ = ["BASES", "BOUNDARIES", "CATALOGUE", "HarmonicModel", "earthmodels", "predict", "recover_topography", "to_harmonics",
            "DATASETS", "DEFAULT_DATASET", "FIELDS", "FREE_BOUNDARIES", "FormatError",
           "INNER_CORE_EIGHT", "INNER_CORE_MODES", "KernelData", "Kernels", "PARAMS", "PLUGIN", "PROBLEM_BOUNDARIES", "PremLayers",
           "SUPERSEDED_DATASETS", "SUPERSEDED_MODES", "SplittingData", "Synthetics", "crust_corrected", "data_space",
           "ModeProblem", "ModeSpace", "explore", "forward", "list_kernel_modes", "problem", "list_synthetic_models", "model_space", "read_catalogue", "read_kernel", "read_mode_list",
           "read_prem_layers", "read_splitting_data", "read_synthetics", "synthetic_values"]
