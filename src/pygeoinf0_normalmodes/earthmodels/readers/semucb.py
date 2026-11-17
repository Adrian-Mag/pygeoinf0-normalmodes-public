"""SEMUCB-WM1 (French and Romanowicz 2014) in its native A3d form, read from the authors' archive
SEMUCB-WM1-<commit>.zip (UCB_a3d_dist.SEMUCB-WM1.r20151019/data/), a Python port of the authors' evaluation
code (src/ucb_A3d.c and src/main.c of the same archive; not compiled).

Model-2.6S6X4-ZeroMean.A3d holds two parameters, S (10242 level-6 spherical-spline knots, grid.S) and X (642
knots, grid.X), each on 20 radial cubic B-splines with knots from 3480 to 6341 km radius (fspl in
ucb_A3d.c). Value at (r, lat, lon) = sum_j v_j(r) sum_k c_jk h_k, with h_k the spherical spline
0.75 d^3 - 1.5 d^2 + 1 (d < 1), 0.25 (2 - d)^3 (1 <= d < 2), 0 beyond, d = angular distance to knot k divided
by the knot spacing of its level.
  S = relative perturbation of the Voigt-average vs = sqrt((2 vsv^2 + vsh^2)/3) about the reference model
      data/model.ref (= Model-2.6S6X4-ZeroMean_1D), a fraction (main.c prints 100 S as "dlnVs (%)").
  X = relative perturbation of xi = vsh^2 / vsv^2 about the same reference.
The README stresses that model.ref is a parameterisation convenience, not a 1-D Earth model. Inside the
homogenised crust (r >= r_moho(lat, lon), moho.dat, 30-60 km deep) the tool reports nan; so does this reader.
"""
import functools

import numpy as np

from .. import _paths
from .._model import Model, R_EARTH_KM

ZIP = ("SEMUCB-WM1", "raw", "SEMUCB-WM1-870eb96ad2ae05a095deae33321f12cf5a63d884.zip")
DATA = "bsl-group-SEMUCB-WM1-Model-870eb96/UCB_a3d_dist.SEMUCB-WM1.r20151019/data/"
# Knot spacing (radians) and cos(2 spacing) of spherical-spline levels 1..7 (ucb_A3d.c).
ADEL = (1.106538745764405, 0.5532693728822025, 0.27576202181510406, 0.13788101090755203,
        0.06894050545377602, 0.034557519189487726, 0.017278759594743863)
COS2ADEL = (-0.5990235985155857, 0.4477590878387697, 0.8517269341430476, 0.9622179935292854,
            0.9905094632383088, 0.9976125063612252, 0.9994029483549729)


def fspl(o, knot, xi):
    """Radial cubic B-spline number o at radius xi (km), knots `knot` (literal port of ucb_A3d.c fspl)."""
    hh = np.diff(knot)
    nx = knot.size - 1
    if (xi - 1.5) > knot[nx] or (xi + 1.5) < knot[0] or o > nx:
        return 0.0

    def poly(a, b, c, d, x0):
        u = xi - x0
        return a * u ** 3 + b * u ** 2 + c * u + d

    if o == 0:
        den = 3 * hh[o] ** 2 + 3 * hh[o] * hh[o + 1] + hh[o + 1] ** 2
        if knot[o] <= xi <= knot[o + 1]:
            return poly(4 / (hh[o] * (hh[o] + hh[o + 1]) * den), 0.0, -12 / den,
                        4 * (2 * hh[o] + hh[o + 1]) / den, knot[o])
        if knot[o + 1] < xi <= knot[o + 2]:
            s = hh[o] + hh[o + 1]
            return poly(-4 / (hh[o + 1] * s * den), 12 / (s * den), -12 * hh[o + 1] / (s * den),
                        4 * hh[o + 1] ** 2 / (s * den), knot[o + 1])
        return 0.0
    if o == 1:
        h0, h1, h2 = hh[o - 1], hh[o], hh[o + 1]
        den = 3 * h0 * h0 + 4 * h0 * h1 + h1 * h1 + 2 * h0 * h2 + h1 * h2
        dd = (h0 + h1 + h2) * den
        if knot[o - 1] <= xi <= knot[o]:
            return poly(-4 * (3 * h0 + 2 * h1 + h2) / (h0 * (h0 + h1) * dd), 0.0, 12 / den, 0.0, knot[o - 1])
        if knot[o] <= xi <= knot[o + 1]:
            a = 4 * (2 * h0 * h0 + 6 * h0 * h1 + 3 * h1 * h1 + 3 * h0 * h2 + 3 * h1 * h2 + h2 * h2) \
                / (h1 * (h0 + h1) * (h1 + h2) * dd)
            b = -12 * (3 * h0 + 2 * h1 + h2) / ((h0 + h1) * dd)
            c = 12 * (-2 * h0 * h0 + h1 * h1 + h1 * h2) / ((h0 + h1) * dd)
            d = 4 * h0 * (4 * h0 * h1 + 3 * h1 * h1 + 2 * h0 * h2 + 3 * h1 * h2) / ((h0 + h1) * dd)
            return poly(a, b, c, d, knot[o])
        if knot[o + 1] <= xi <= knot[o + 2]:
            dd2 = dd * (h1 + h2)
            e = 2 * h0 + h1
            return poly(-4 * e / (h2 * dd2), 12 * e / dd2, -12 * e * h2 / dd2, 4 * e * h2 * h2 / dd2, knot[o + 1])
        return 0.0
    if o == nx - 1:
        g0, g1, g2 = hh[o - 2], hh[o - 1], hh[o]
        den = g0 * g1 + g1 * g1 + 2 * g0 * g2 + 4 * g1 * g2 + 3 * g2 * g2
        dd = (g0 + g1 + g2) * den
        if knot[o - 2] <= xi <= knot[o - 1]:
            return poly(4 * (g1 + 2 * g2) / (g0 * (g0 + g1) * dd), 0.0, 0.0, 0.0, knot[o - 2])
        if knot[o - 1] <= xi <= knot[o]:
            a = -4 * (g0 * g0 + 3 * g0 * g1 + 3 * g1 * g1 + 3 * g0 * g2 + 6 * g1 * g2 + 2 * g2 * g2) \
                / (g1 * (g0 + g1) * (g1 + g2) * dd)
            b = 12 * (g1 + 2 * g2) / ((g0 + g1) * dd)
            c = 12 * g0 * (g1 + 2 * g2) / ((g0 + g1) * dd)
            d = 4 * g0 * g0 * (g1 + 2 * g2) / ((g0 + g1) * dd)
            return poly(a, b, c, d, knot[o - 1])
        if knot[o] <= xi <= knot[o + 1]:
            dd2 = dd * (g1 + g2)
            e = g0 + 2 * g1 + 3 * g2
            a = 4 * e / (g2 * dd2)
            b = -12 * e / dd2
            c = 12 * (-g0 * g1 - g1 * g1 + 2 * g2 * g2) / dd2
            d = 4 * g2 * (3 * g0 * g1 + 3 * g1 * g1 + 2 * g0 * g2 + 4 * g1 * g2) / dd2
            return poly(a, b, c, d, knot[o])
        return 0.0
    if o == nx:
        g0, g1 = hh[o - 2], hh[o - 1]
        den = (g0 + g1) * (g0 * g0 + 3 * g0 * g1 + 3 * g1 * g1)
        if knot[o - 2] <= xi <= knot[o - 1]:
            return poly(4 / (g0 * den), 0.0, 0.0, 0.0, knot[o - 2])
        if knot[o - 1] <= xi <= knot[o]:
            return poly(-4 / (g1 * den), 12 / den, 12 * g0 / den, 4 * g0 * g0 / den, knot[o - 1])
        return 0.0
    g0, g1, g2, g3 = hh[o - 2], hh[o - 1], hh[o], hh[o + 1]
    den1 = g0 + g1 + g2 + g3
    if knot[o - 2] <= xi <= knot[o - 1]:
        return poly(4 / (g0 * (g0 + g1) * (g0 + g1 + g2) * den1), 0.0, 0.0, 0.0, knot[o - 2])
    if knot[o - 1] <= xi <= knot[o]:
        den = den1 * (g0 + g1) * (g0 + g1 + g2)
        a = -4 * (g0 * g0 + 3 * g0 * g1 + 3 * g1 * g1 + 2 * g0 * g2 + 4 * g1 * g2 + g2 * g2
                  + g0 * g3 + 2 * g1 * g3 + g2 * g3) / (g1 * (g1 + g2) * (g1 + g2 + g3) * den)
        return poly(a, 12 / den, 12 * g0 / den, 4 * g0 * g0 / den, knot[o - 1])
    if knot[o] <= xi <= knot[o + 1]:
        den = den1 * (g1 + g2) * (g0 + g1 + g2) * (g1 + g2 + g3)
        a = 4 * (g0 * g1 + g1 * g1 + 2 * g0 * g2 + 4 * g1 * g2 + 3 * g2 * g2 + g0 * g3
                 + 2 * g1 * g3 + 3 * g2 * g3 + g3 * g3) / (g2 * (g2 + g3) * den)
        b = -12 * (g0 + 2 * g1 + 2 * g2 + g3) / den
        c = 12 * (-g0 * g1 - g1 * g1 + g2 * g2 + g2 * g3) / den
        d = 4 * (2 * g0 * g1 * g2 + 2 * g1 * g1 * g2 + g0 * g2 * g2 + 2 * g1 * g2 * g2
                 + g0 * g1 * g3 + g1 * g1 * g3 + g0 * g2 * g3 + 2 * g1 * g2 * g3) / den
        return poly(a, b, c, d, knot[o])
    if knot[o + 1] <= xi <= knot[o + 2]:
        den = den1 * (g2 + g3) * (g1 + g2 + g3)
        return poly(-4 / (g3 * den), 12 / den, -12 * g3 / den, 4 * g3 * g3 / den, knot[o + 1])
    return 0.0


def _unit_vectors(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


class SemucbNative(Model):
    name, source = "SEMUCB-WM1", "native"

    def __init__(self):
        super().__init__()
        z = _paths.original(*ZIP)
        tok = _paths.read_text(z, DATA + "Model-2.6S6X4-ZeroMean.A3d").split()
        n = int(tok[0])
        specs = [(int(tok[1 + 3 * p]), int(tok[2 + 3 * p]), tok[3 + 3 * p]) for p in range(n)]
        pos = 1 + 3 * n
        if int(tok[pos]) != 0:
            raise ValueError("A3d radial discontinuities are not supported (nor by ucb_A3d.c)")
        pos += 1
        nb = specs[0][1]
        self.knots = np.array(tok[pos:pos + nb], float)
        pos += nb
        self.coef = {}
        for nsspl, _, desc in specs:
            self.coef[desc] = np.array(tok[pos:pos + nb * nsspl], float).reshape(nb, nsspl)
            pos += nb * nsspl
        if pos != len(tok):
            raise ValueError(f"A3d file: {len(tok) - pos} trailing tokens")
        self.grids = {}
        for desc in self.coef:
            g = _paths.numbers(_paths.read_text(z, DATA + f"grid.{desc}").split("\n", 1)[1]).reshape(-1, 3)
            if g.shape[0] != self.coef[desc].shape[1]:
                raise ValueError(f"grid.{desc} does not match the A3d header")
            self.grids[desc] = (g[:, 1], g[:, 0], g[:, 2].astype(int))      # lat, lon, level
        mt = _paths.read_text(z, DATA + "moho.dat").split()
        self.moho = np.array(mt[2:], float).reshape(int(mt[0]), int(mt[1]))
        ref = _paths.numbers("\n".join(_paths.read_text(z, DATA + "Model-2.6S6X4-ZeroMean_1D")
                                       .splitlines()[3:])).reshape(-1, 9)
        self.ref_r_km = ref[:, 0] / 1000.0
        self.ref_vs_voigt = np.sqrt((2 * ref[:, 3] ** 2 + ref[:, 7] ** 2) / 3.0) / 1000.0
        cov = (R_EARTH_KM - self.knots[-1], R_EARTH_KM - self.knots[0])
        ref_txt = "model.ref = Model-2.6S6X4-ZeroMean_1D (the authors' parameterisation reference, not PREM)"
        nd = np.sort(R_EARTH_KM - self.knots)
        self._fields["dvs"] = dict(unit="fraction", quantity="d ln vs of the Voigt average sqrt((2vsv^2+vsh^2)/3)"
                                   " (A3d parameter S)", reference=ref_txt, coverage=cov, native_depths=nd,
                                   description="A3d S: 20 radial B-splines x 10242 level-6 spherical splines; "
                                               "nan inside the homogenised crust (moho.dat)")
        self._fields["dxi"] = dict(unit="fraction", quantity="d ln xi, xi = vsh^2/vsv^2 (A3d parameter X)",
                                   reference=ref_txt, coverage=cov, native_depths=nd,
                                   description="A3d X: 20 radial B-splines x 642 spherical splines; nan in the crust")
        self._derived["vs_abs"] = dict(inputs=["dvs"], func=None, unit="km/s",
                                       quantity="absolute Voigt vs = (1 + dvs) x Voigt vs of model.ref",
                                       reference=ref_txt, description="computed on request")

    # radial and lateral bases
    def radial_basis(self, depth):
        r = R_EARTH_KM - np.atleast_1d(np.asarray(depth, float))
        return np.array([[fspl(j, self.knots, x) for x in r] for j in range(self.knots.size)])   # (20, n)

    def ref_vs(self, depth):
        """Voigt vs (km/s) of model.ref, linear between its nodes (the first bracketing pair wins)."""
        r = R_EARTH_KM - np.atleast_1d(np.asarray(depth, float))
        rr = self.ref_r_km
        out = np.empty(r.size)
        for i, x in enumerate(r):
            k = int(np.nonzero((rr[:-1] <= x) & (rr[1:] >= x))[0][0])
            beta = (x - rr[k]) / (rr[k + 1] - rr[k])
            out[i] = (1 - beta) * self.ref_vs_voigt[k] + beta * self.ref_vs_voigt[k + 1]
        return out

    def moho_depth(self, lat, lon):
        """Moho depth (km) of the homogenised crust, bilinear on moho.dat (ucb_A3d.c evaluate_moho_depth)."""
        lat = np.asarray(lat, float)
        plon = np.mod(np.asarray(lon, float), 360.0)
        nlat, nlon = self.moho.shape
        dlat, dlon = 180.0 / (nlat - 1), 360.0 / (nlon - 1)
        clat = 90.0 - lat
        ix = np.clip((plon / dlon).astype(int), 0, nlon - 2)
        iy = np.clip(((180.0 - clat) / dlat).astype(int), 0, nlat - 2)
        ay = ((180.0 - clat) - iy * dlat) / dlat
        ax = (plon - ix * dlon) / dlon
        m = self.moho
        return ((1 - ay) * (1 - ax) * m[iy, ix] + ay * (1 - ax) * m[iy + 1, ix]
                + (1 - ay) * ax * m[iy, ix + 1] + ay * ax * m[iy + 1, ix + 1])

    def lateral_matrix(self, desc, lat, lon):
        """Sparse (n_points, n_knots) matrix of spherical-spline values h_k at the points."""
        from scipy.sparse import csr_matrix
        from scipy.spatial import cKDTree
        klat, klon, klev = self.grids[desc]
        pts = _unit_vectors(np.ravel(lat), np.ravel(lon))
        kn = _unit_vectors(klat, klon)
        tree = cKDTree(pts)
        rows, cols, vals = [], [], []
        for lev in np.unique(klev):
            adel, cos2 = ADEL[lev - 1], COS2ADEL[lev - 1]
            idx = np.nonzero(klev == lev)[0]
            hits = tree.query_ball_point(kn[idx], r=2 * np.sin(adel) * 1.0001)
            for k, h in zip(idx, hits):
                if not h:
                    continue
                h = np.asarray(h, dtype=np.int64)
                arg = np.clip(pts[h] @ kn[k], -1.0, 1.0)
                ok = arg > cos2
                h, arg = h[ok], arg[ok]
                d = np.arccos(arg) / adel
                rows.append(h)
                cols.append(np.full(h.size, k))
                vals.append(np.where(d < 1.0, 0.75 * d ** 3 - 1.5 * d ** 2 + 1.0, 0.25 * (2.0 - d) ** 3))
        if not rows:
            return csr_matrix((pts.shape[0], kn.shape[0]))
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(pts.shape[0], kn.shape[0]))

    @functools.lru_cache(maxsize=4)
    def _lateral_fields(self, desc, lat_key, lon_key):
        lat, lon = np.frombuffer(lat_key), np.frombuffer(lon_key)
        H = self.lateral_matrix(desc, lat, lon)
        return np.asarray(H @ self.coef[desc].T)            # (n_points, 20)

    def _param(self, field):
        return {"dvs": "S", "dxi": "X"}[field]

    def _eval(self, field, lat, lon, depth):
        if field == "vs_abs":
            return (1.0 + self._eval("dvs", lat, lon, depth)) * self.ref_vs(depth)
        pts = np.stack([lat, lon], axis=1)
        upts, pinv = np.unique(pts, axis=0, return_inverse=True)
        pinv = pinv.ravel()
        udep, dinv = np.unique(depth, return_inverse=True)
        G = self._lateral_fields(self._param(field), np.ascontiguousarray(upts[:, 0]).tobytes(),
                                 np.ascontiguousarray(upts[:, 1]).tobytes())
        V = self.radial_basis(udep)                          # (20, n_udep)
        out = np.einsum("ij,ji->i", G[pinv], V[:, dinv])
        top, bot = self.coverage(field)
        out[(depth < top - 1e-9) | (depth > bot + 1e-9)] = np.nan
        out[depth <= self.moho_depth(lat, lon)] = np.nan
        return out

    def _grid(self, field, depth, lat, lon):
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        return self._eval(field, LA.ravel(), LO.ravel(), np.full(LA.size, float(depth))).reshape(LA.shape)

    def _raw(self, field, lat, lon, depth):
        return self._eval(field, lat, lon, depth)

    def _raw_grid(self, field, depth, lat, lon):
        return self._grid(field, depth, lat, lon)


def load(source="native"):
    return SemucbNative()
