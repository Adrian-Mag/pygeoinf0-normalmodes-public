"""Earth models on a spherical-harmonic basis, moved between bases and back to the sphere (plan 030).

    hm = to_harmonics("S20RTS", "dvs", lmax=20, radii=K.midpoints())    -> HarmonicModel: coefficients per radius
    hm.in_basis("catalogue"), hm.truncated(14), hm.scaled(0.5), hm.without_degree_zero()
    hm(s, t, radii), hm.evaluate(lat, lon, radius), hm.element(space), hm.function(), hm.as_model()
    predict({"vs": hm, "vp": hm.scaled(0.5), "rho": hm.scaled(0.3)}, data, K)    -> the splitting coefficients
    recover_topography(K, synthetics_dir, "S20RTS-Crust5.1_cmb_tanaka-2010", "S20RTS-Crust5.1")    -> a boundary's HarmonicModel

Bases of the real harmonics, by what they do to a coefficient of the orthonormal basis (t = 0 zonal, t = 2m - 1 cos m phi,
t = 2m sin m phi, Condon-Shortley phase):

    "orthonormal"   1              the pack's sh_coefficients; int Y^2 = 1
    "catalogue"     sqrt 2, t > 0  the normal-mode data and kernels; also the Ritsema files' raw a, b
    "package"       (-1)^m         pygeoinf0's realisations (no Condon-Shortley phase)
    "4pi"           1 / sqrt(4 pi) int Y^2 = 4 pi

A coefficient is the same field in every basis: only hm.basis says how to read the numbers.

What a coefficient is. Projection is by Gauss-Legendre quadrature of the native field, at each radius, with undefined points
counted as zero. Point sampling converges slowly on piecewise-constant models (the pack's CHECKS: 2-3e-2 at 0.5 degrees for
cell models; raise n_lat). Between radii, coefficients are linear; outside them, zero.
"""
from dataclasses import dataclass, replace

import numpy as np

from .earthmodels import _sh
from .earthmodels import load as _load_model
from .earthmodels._model import ABSOLUTE_UNITS, R_EARTH_KM, Model

BASES = ("orthonormal", "catalogue", "package", "4pi")


def _indices(lmax):
    s = np.concatenate([np.full(2 * l + 1, l) for l in range(lmax + 1)])
    t = np.concatenate([np.arange(2 * l + 1) for l in range(lmax + 1)])
    return s, t


def _factors(lmax, basis):
    """c_basis = c_orthonormal * factor."""
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}; known: {', '.join(BASES)}")
    _, t = _indices(lmax)
    if basis == "catalogue":
        return np.where(t > 0, np.sqrt(2.0), 1.0)
    if basis == "package":
        return np.where(((t + 1) // 2) % 2 == 1, -1.0, 1.0)
    return np.full(t.shape, 1.0 / np.sqrt(4.0 * np.pi) if basis == "4pi" else 1.0)


def _harmonics(lmax, lat, lon):
    """The orthonormal harmonics at points: (n, (lmax + 1)^2), in the order s^2 + t."""
    lat, lon = np.ravel(lat).astype(float), np.ravel(lon).astype(float)
    theta, phi = np.radians(90.0 - lat), np.radians(lon)
    cost, sint = np.cos(theta), np.sin(theta)
    out = np.zeros((lat.size, (lmax + 1) ** 2))
    for m in range(lmax + 1):
        X = _sh.legendre_m(lmax, m, cost, sint)
        for l in range(m, lmax + 1):
            if m == 0:
                out[:, l * l] = X[l]
            else:
                out[:, l * l + 2 * m - 1] = np.sqrt(2.0) * X[l - m] * np.cos(m * phi)
                out[:, l * l + 2 * m] = np.sqrt(2.0) * X[l - m] * np.sin(m * phi)
    return out


@dataclass(frozen=True, eq=False)
class HarmonicModel:
    """A field as coefficients: coefficients[i, s^2 + t] at radii[i] (ascending, km), in one basis."""
    name: str
    field: str
    unit: str
    lmax: int
    radii: np.ndarray
    coefficients: np.ndarray
    basis: str = "orthonormal"

    def __post_init__(self):
        r, c = np.array(self.radii, float), np.array(self.coefficients, float)
        if c.shape != (r.size, (self.lmax + 1) ** 2):
            raise ValueError(f"coefficients have shape {c.shape}, expected {(r.size, (self.lmax + 1) ** 2)}")
        if r.size == 0 or np.any(np.diff(r) <= 0):
            raise ValueError("radii must be non-empty and increasing")
        _factors(self.lmax, self.basis)
        r.setflags(write=False)
        c.setflags(write=False)
        object.__setattr__(self, "radii", r)
        object.__setattr__(self, "coefficients", c)

    def __repr__(self):
        return (f"HarmonicModel({self.name} {self.field}, {self.unit}, lmax {self.lmax}, {self.radii.size} radii "
                f"{self.radii[0]:g}-{self.radii[-1]:g} km, basis {self.basis})")

    def in_basis(self, basis):
        """The same field in another basis."""
        factor = _factors(self.lmax, basis) / _factors(self.lmax, self.basis)
        return replace(self, coefficients=self.coefficients * factor, basis=basis)

    def truncated(self, lmax):
        """Degrees up to lmax."""
        if not 0 <= lmax <= self.lmax:
            raise ValueError(f"lmax must be in 0..{self.lmax}")
        return replace(self, lmax=lmax, coefficients=self.coefficients[:, :(lmax + 1) ** 2])

    def scaled(self, factor, field=None):
        """factor times this field, e.g. a scaled density from a shear model; a choice, recorded only in field."""
        return replace(self, coefficients=factor * self.coefficients, field=field or f"{factor:g} x {self.field}")

    def without_degree_zero(self):
        """The lateral variations: degree 0 set to zero."""
        c = np.array(self.coefficients)
        c[:, 0] = 0.0
        return replace(self, coefficients=c)

    def __call__(self, s, t, radii=None):
        """The radial profile of one coefficient, in this basis; zero outside the radii and above lmax."""
        if not 0 <= t <= 2 * s:
            raise ValueError(f"t = {t} is outside 0..2s for s = {s}")
        radii = self.radii if radii is None else np.asarray(radii, float)
        if s > self.lmax:
            return np.zeros(np.shape(radii))
        if self.radii.size == 1:
            return np.where(np.abs(radii - self.radii[0]) < 1e-6, self.coefficients[0, s * s + t], 0.0)
        return np.interp(radii, self.radii, self.coefficients[:, s * s + t], left=0.0, right=0.0)

    def evaluate(self, lat, lon, radius):
        """The field at points (degrees, degrees, km; arrays broadcast); zero outside the radii."""
        lat, lon, radius = np.broadcast_arrays(*(np.asarray(x, float) for x in (lat, lon, radius)))
        c = self.in_basis("orthonormal").coefficients
        Y = _harmonics(self.lmax, lat, lon)
        out = np.zeros(Y.shape[0])
        flat = radius.ravel()
        if self.radii.size == 1:
            on = np.abs(flat - self.radii[0]) < 1e-6
            out[on] = Y[on] @ c[0]
            return out.reshape(lat.shape)
        inside = (flat >= self.radii[0]) & (flat <= self.radii[-1])
        hi = np.clip(np.searchsorted(self.radii, flat[inside]), 1, self.radii.size - 1)
        w = (flat[inside] - self.radii[hi - 1]) / (self.radii[hi] - self.radii[hi - 1])
        out[inside] = np.sum(Y[inside] * ((1 - w)[:, None] * c[hi - 1] + w[:, None] * c[hi]), axis=1)
        return out.reshape(lat.shape)

    def element(self, space, name=None):
        """This field in a radial-harmonic pygeoinf0 space, directly from the coefficients: linear between the radii, zero
        outside them, no fitting. Radii outside the space are dropped. Give radii spanning the model's coverage: layer
        midpoints alone stop half a layer short at each end."""
        from pygeoinf0.realisations.series.radial_direct import harmonic_element  # plan 031: the rest works on an older pygeoinf0

        a, b = space.backend.a, space.backend.b
        keep =(self.radii >= a) & (self.radii <= b)
        return harmonic_element(space, self.radii[keep], self.in_basis("package").coefficients[keep],
                                name=name or f"{self.name}_{self.field}")

    def function(self):
        """The field as a function of Cartesian coordinates in km, for what else needs one; for a space, use element.
        Give radii spanning the model's coverage: it is zero outside them, so layer midpoints alone stop half a layer short."""
        def f(x, y, z):
            x, y, z = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, z)))
            r = np.sqrt(x * x + y * y + z * z)
            lat = np.degrees(np.arcsin(np.clip(z / np.where(r > 0, r, 1.0), -1.0, 1.0)))
            return self.evaluate(lat, np.degrees(np.arctan2(y, x)), r)
        f.__name__ = f"{self.name}_{self.field}"
        return f

    def as_model(self):
        """This band-limited field with the pack's Model interface, so its slices, sections and plots apply."""
        return _BandLimited(self)


class _BandLimited(Model):
    def __init__(self, hm):
        super().__init__()
        self.hm, self.name, self.source = hm, hm.name, f"harmonics to degree {hm.lmax}"
        self._fields[hm.field] = dict(unit=hm.unit, quantity=hm.field, reference="as projected", description="band-limited",
                                      coverage=(R_EARTH_KM - hm.radii[-1], R_EARTH_KM - hm.radii[0]),
                                      native_depths=R_EARTH_KM - hm.radii[::-1])

    def _eval(self, field, lat, lon, depth):
        return self.hm.evaluate(lat, lon, R_EARTH_KM - depth)


def to_harmonics(model, field, lmax=20, radii=None, unit="fraction", about_mean=False, basis="orthonormal", n_lat=None,
                 source=None):
    """Move a model's field onto harmonics. model: a name in earthmodels.list_models() or a Model. radii (km) default to
    the model's native radial nodes, or 100 uniform radii over its coverage; a radius outside the coverage is refused.

    unit: "fraction" or "percent" for relative perturbations, None as stored. about_mean=True removes the lateral mean at
    each radius; an absolute speed then becomes v / mean - 1, a fraction, and it needs this, since it has no other reference.
    """
    m = _load_model(model, source) if isinstance(model, str) else model
    stored = m.field_info(field)["unit"]
    absolute = stored in ABSOLUTE_UNITS
    if absolute and not about_mean and unit is not None:
        raise ValueError(f"{m.name} {field} is an absolute speed in {stored}: use about_mean=True, or unit=None")
    top, bottom = m.coverage(field)
    if radii is None:
        native = m.native_depths(field)
        depths = np.linspace(top, bottom, 100) if native is None or native.size < 2 else native
        radii = R_EARTH_KM - np.asarray(depths, float)[::-1]
    radii = np.sort(np.atleast_1d(np.asarray(radii, float)))
    if np.any(R_EARTH_KM - radii < top - 1e-9) or np.any(R_EARTH_KM - radii > bottom + 1e-9):
        raise ValueError(f"{m.name} {field} covers radii {R_EARTH_KM - bottom:g}-{R_EARTH_KM - top:g} km")
    rows = []
    for r in radii:
        c = m.sh_coefficients(field, R_EARTH_KM - r, lmax=lmax, n_lat=n_lat, unit=None if absolute else unit)
        if about_mean:
            if absolute:
                c = c / (c[0] / np.sqrt(4.0 * np.pi))     # c_00 = mean sqrt(4 pi)
            c[0] = 0.0
        rows.append(c)
    final = stored if unit is None and not (absolute and about_mean) else ("fraction" if absolute else unit)
    return HarmonicModel(m.name, field, final, lmax, radii, np.array(rows)).in_basis(basis)


def predict(models, data, kernels):
    """Splitting coefficients of a model at the rows of data, in the catalogue's basis. models maps "vs", "vp", "rho" to a
    fractional HarmonicModel and a boundary name to a HarmonicModel of its height in km; what is absent contributes nothing.

    Volumetric: sum_l 100 K_l a_st(r_l), the kernels' own layers at their midpoints. With S20RTS, vp = 0.5 vs and rho = 0.3 vs
    this reproduces the synthetics of the mode code (see the tests). Heights: K_d h_st, h at the boundary's radius. By the
    degree blocks, one matrix per degree, kept on kernels: repeat it freely, e.g. over an ensemble.
    """
    from . import BOUNDARIES, FIELDS
    catalogue = {key: hm.in_basis("catalogue") for key, hm in models.items()}
    for key in catalogue:
        if key not in FIELDS and key not in BOUNDARIES:
            raise ValueError(f"unknown field or boundary {key!r}")
    out = np.zeros(len(data))
    for s in np.unique(data.s).tolist():
        rows = np.flatnonzero(data.s == s)
        modes, ms = np.unique(data.mode[rows], return_inverse=True)
        modes, ts = tuple(modes.tolist()), data.t[rows]
        for key, hm in catalogue.items():
            if key in FIELDS:
                radii, W = kernels.layer_matrix(key, s, modes)           # (modes, layers): 100 K_l
                a = np.stack([hm(s, t, radii) for t in range(2 * s + 1)], axis=1)
                out[rows] += (W @ a)[ms, ts]
            else:
                w = kernels.boundary_vector(key, s, modes)
                h = np.array([hm(s, t, np.array([BOUNDARIES[key]]))[0] for t in range(2 * s + 1)])
                out[rows] += w[ms] * h[ts]
    out.setflags(write=False)
    return out


def recover_topography(kernels, synthetics_dir, model, base, boundary="cmb", modes=None, lmax=14):
    """A boundary's height coefficients (km, catalogue's basis) from two synthetics that differ only by it: the difference is
    K_d(s) h_st across modes, so h_st is the least-squares ratio. The published models' coefficients as the mode code used them."""
    from . import BOUNDARIES, read_synthetics
    modes = kernels.modes("topo-all") if modes is None else modes
    top, bottom = np.zeros((lmax + 1) ** 2), np.zeros((lmax + 1) ** 2)
    for mode in modes:
        a = read_synthetics(synthetics_dir, mode, model)
        b = read_synthetics(synthetics_dir, mode, base)
        if not (np.array_equal(a.s, b.s) and np.array_equal(a.t, b.t)):
            raise ValueError(f"{model} and {base} of {mode} have different rows")
        keep = a.s <= lmax
        at = kernels.table(mode, "topo-all").at_boundary(BOUNDARIES[boundary])
        K = dict(zip(at.s.tolist(), at.value.tolist()))
        k = np.array([K.get(s, 0.0) for s in a.s[keep].tolist()])
        idx = a.s[keep] ** 2 + a.t[keep]
        np.add.at(top, idx, k * (a.value[keep] - b.value[keep]))
        np.add.at(bottom, idx, k * k)
    h = np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)
    return HarmonicModel(model, boundary, "km", lmax, [BOUNDARIES[boundary]], h[None, :], basis="catalogue")
