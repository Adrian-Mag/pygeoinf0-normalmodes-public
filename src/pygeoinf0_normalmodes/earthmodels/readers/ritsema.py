"""Ritsema-format models: real spherical harmonics x 21 radial cubic splines (S20RTS, S40RTS, SP12RTS, P12) and
SGLOBE-rani's SPECFEM files, which use the same layout.

.sph file (the authors' plotting package, lib/libS20/rsphhead.f, getfdp.f, splh.f, rspln.f; src/raw2xyz_jr.f):
header "lmax lmask nspl smask"; smask "000" + 21 ones, i.e. 24 splines of which the first three (crust) are
absent. Then 21 blocks of coefficients, top spline first; each block degree by degree: a_l0, a_l1, b_l1, ...,
a_ll, b_ll. SGLOBE-rani's .dat files have no header and degree 35, in the same order (SPECFEM
model_sglobe.f90).

Value: f(r, theta, phi) = sum_k S_k(x) sum_l [a_kl0 X_l0 + sum_m X_lm (a_klm cos m phi + b_klm sin m phi)],
X_lm fully normalised with the Condon-Shortley phase (SPECFEM lgndr), phi east longitude.
x = -1 + 2 (r - r_cmb) / (r_moho - r_cmb), r_cmb = 3480 km, r_moho = 6346.619 km (getfdp.f; SGLOBE: 6346.6 km,
model_sglobe.f90). S_k is the interpolating cubic spline through the 21 knots SPKNT that is 1 at knot 20 - k and
0 at the others, with end derivatives fixed by the parabola through the three end knots (rspln.f, "three-point
clamped"). Values are relative perturbations d ln v as fractions (the plotting code multiplies by 100).
"""
import numpy as np
from scipy.interpolate import CubicSpline

from .. import _paths, _sh
from .._model import Model, R_EARTH_KM, voigt_linear

SPKNT = np.array([-1.00000, -0.78631, -0.59207, -0.41550, -0.25499, -0.10909, 0.02353, 0.14409, 0.25367,
                  0.35329, 0.44384, 0.52615, 0.60097, 0.66899, 0.73081, 0.78701, 0.83810, 0.88454, 0.92675,
                  0.96512, 1.00000])
R_CMB_KM = 3480.0
R_MOHO_RITSEMA_KM = 6346.619
R_MOHO_SGLOBE_KM = 6346.6

_SPLINES = []


def _cardinal_splines():
    if not _SPLINES:
        eye = np.eye(SPKNT.size)
        for j in range(SPKNT.size):
            y = eye[j]
            d0 = np.polyval(np.polyder(np.polyfit(SPKNT[:3], y[:3], 2)), SPKNT[0])
            dn = np.polyval(np.polyder(np.polyfit(SPKNT[-3:], y[-3:], 2)), SPKNT[-1])
            _SPLINES.append(CubicSpline(SPKNT, y, bc_type=((1, d0), (1, dn))))
    return _SPLINES


def radial_basis(depth_km, r_moho_km):
    """(n, 21): value of file block k's spline at each depth (block 0 = top spline); NaN outside
    [r_cmb, r_moho]."""
    r = R_EARTH_KM - np.atleast_1d(np.asarray(depth_km, float))
    x = -1.0 + 2.0 * (r - R_CMB_KM) / (r_moho_km - R_CMB_KM)
    out = np.full((r.size, 21), np.nan)
    inside = (x >= -1.0 - 1e-12) & (x <= 1.0 + 1e-12)
    xs = np.clip(x[inside], -1.0, 1.0)
    spl = _cardinal_splines()
    for k in range(21):
        out[inside, k] = spl[20 - k](xs)
    return out


def read_sph(text):
    """(lmax, raw (21, (lmax+1)^2)) from the text of a Ritsema .sph file."""
    head, body = text.split("\n", 1)
    h = head.split()
    lmax_rd, lmask, nspl, smask = int(h[0]), h[1], int(h[2]), h[3]
    lmax = max(i for i, ch in enumerate(lmask[:lmax_rd + 1]) if ch == "1")
    if lmask[:lmax + 1] != "1" * (lmax + 1):
        raise ValueError(f"degree mask {lmask} has holes; not supported")
    if nspl != 24 or smask[:nspl] != "000" + "1" * 21:
        raise ValueError(f"spline mask {nspl} {smask} is not the 21-spline mantle layout")
    v = _paths.numbers(body)
    n = (lmax + 1) ** 2
    if v.size != 21 * n:
        raise ValueError(f"{v.size} values, expected 21 x {n}")
    return lmax, v.reshape(21, n)


def read_headerless(text, lmax):
    v = _paths.numbers(text)
    n = (lmax + 1) ** 2
    if v.size != 21 * n:
        raise ValueError(f"{v.size} values, expected 21 x {n} for degree {lmax}")
    return v.reshape(21, n)


class RitsemaModel(Model):
    """Fields given as raw coefficient arrays (21, (lmax+1)^2) on the 21 Ritsema splines."""

    def __init__(self, name, source, fields, r_moho_km):
        """fields: {field: dict(raw=..., lmax=..., unit, quantity, reference, description)}."""
        super().__init__()
        self.name, self.source, self.r_moho_km = name, source, r_moho_km
        self._raw_coeffs = {}
        knot_depths = R_EARTH_KM - (R_CMB_KM + (SPKNT + 1.0) / 2.0 * (r_moho_km - R_CMB_KM))
        top = R_EARTH_KM - r_moho_km
        for f, d in fields.items():
            self._raw_coeffs[f] = (d.pop("lmax"), d.pop("raw"))
            d["coverage"] = (top, R_EARTH_KM - R_CMB_KM)
            d["native_depths"] = np.sort(knot_depths)
            self._fields[f] = d

    def lmax(self, field):
        return self._raw_coeffs[field][0]

    def coefficients(self, field, depth):
        """(A, B) arrays of the expansion at one depth (package convention, _sh module)."""
        lmax, raw = self._raw_coeffs[field]
        b = radial_basis(depth, self.r_moho_km)[0]
        if not np.isfinite(b).all():
            return None
        return _sh.raw_to_ab(b @ raw, lmax)

    def _eval(self, field, lat, lon, depth):
        lmax, raw = self._raw_coeffs[field]
        out = np.full(lat.shape, np.nan)
        udep, dinv = np.unique(depth, return_inverse=True)
        pts = np.stack([lat, lon], axis=1)
        upts, pinv = np.unique(pts, axis=0, return_inverse=True)
        pinv = pinv.ravel()
        if 21 * upts.shape[0] < lat.size:
            # factorised: the 21 lateral fields at the distinct points, times the radial basis
            G = np.stack([_sh.synth_points(*_sh.raw_to_ab(raw[k], lmax), upts[:, 0], upts[:, 1])
                          for k in range(21)])                     # (21, n_upts)
            Bm = radial_basis(udep, self.r_moho_km)                # (n_udep, 21)
            out = np.einsum("ik,ki->i", Bm[dinv], G[:, pinv])
            return out
        for i, d in enumerate(udep):
            sel = dinv == i
            ab = self.coefficients(field, d)
            if ab is not None:
                out[sel] = _sh.synth_points(*ab, lat[sel], lon[sel])
        return out

    def _grid(self, field, depth, lat, lon):
        ab = self.coefficients(field, depth)
        if ab is None:
            return np.full((len(lat), len(lon)), np.nan)
        return _sh.synth_grid(*ab, lat, lon)

    def _mean(self, field, depth):
        ab = self.coefficients(field, depth)
        return np.nan if ab is None else float(ab[0][0, 0] / np.sqrt(4.0 * np.pi))


# ---------------------------------------------------------------------- the models

_REF_PREM_TI = "transversely isotropic PREM (as perturbed by SPECFEM3D_GLOBE's model_s20rts.f90)"


def _sph_field(text, unit, quantity, reference, description):
    lmax, raw = read_sph(text)
    return dict(lmax=lmax, raw=raw, unit=unit, quantity=quantity, reference=reference, description=description)


def load_s20rts(source="sph"):
    text = _paths.read_text(_paths.original("S20RTS", "S20RTS.sph"))
    return RitsemaModel("S20RTS", "sph", {"dvs": _sph_field(
        text, "fraction", "d ln vs (isotropic shear speed), degree 20", _REF_PREM_TI,
        "S20RTS.sph: degree-20 SH x 21 radial splines")}, R_MOHO_RITSEMA_KM)


def load_s40rts(source="sph"):
    text = _paths.read_text(_paths.original("S40RTS", "raw", "S40RTS.sph"))
    return RitsemaModel("S40RTS", "sph", {"dvs": _sph_field(
        text, "fraction", "d ln vs (isotropic shear speed), degree 40",
        "PREM (Ritsema et al. 2011: isotropic shear velocity with respect to PREM)",
        "S40RTS.sph: degree-40 SH x 21 radial splines")}, R_MOHO_RITSEMA_KM)


def load_sp12rts(source="sph"):
    tar = _paths.original("SP12RTS", "raw", "SP12RTS_plotting_GMT5.tar.gz")
    ref = "anisotropic PREM (Koelemeijer et al. 2016: isotropic velocity perturbations w.r.t. anisotropic PREM)"
    fields = {}
    for f, member, q in (("dvs", "SP12RTS..ES.sph", "d ln vs (isotropic shear speed), degree 12"),
                         ("dvp", "SP12RTS..EP.sph", "d ln vp (isotropic P speed), degree 12")):
        fields[f] = _sph_field(_paths.read_text(tar, "SP12RTS_plotting_GMT5/" + member), "fraction", q, ref,
                               f"SP12RTS_plotting_GMT5.tar.gz member SP12RTS_plotting_GMT5/{member}")
    return RitsemaModel("SP12RTS", "sph", fields, R_MOHO_RITSEMA_KM)


def load_p12(source="sph"):
    text = _paths.read_text(_paths.original("P12", "P12.sph"))
    return RitsemaModel("P12", "sph", {"dvp": _sph_field(
        text, "fraction", "d ln vp (isotropic P speed), degree 12",
        "PREM (as used by SPECFEM3D_GLOBE, which adds it to PREM); not stated in the file",
        "P12.sph: degree-12 SH x 21 radial splines")}, R_MOHO_RITSEMA_KM)


def load_sglobe_native():
    base = ("SGLOBE-rani", "raw", "specfem", "DATA", "sglobe")
    ref = "transversely isotropic PREM (readme: 'its reference model is PREM')"
    fields = {}
    for f, fn, q in (("dvs", "dvs_iso.dat", "d ln vs, isotropic (Voigt) version supplied by the authors"),
                     ("dvsv", "dvsv.dat", "d ln vsv"), ("dvsh", "dvsh.dat", "d ln vsh")):
        raw = read_headerless(_paths.read_text(_paths.original(*base, fn)), 35)
        fields[f] = dict(lmax=35, raw=raw, unit="fraction", quantity=q + ", degree 35", reference=ref,
                         description=f"SPECFEM DATA/sglobe/{fn}: degree-35 SH x 21 radial splines")
    m = RitsemaModel("SGLOBE-rani", "native", fields, R_MOHO_SGLOBE_KM)
    m._derived["dvs_linear"] = dict(
        inputs=["dvsv", "dvsh"], func=voigt_linear, unit="fraction",
        quantity="(2 d ln vsv + d ln vsh) / 3, first-order Voigt average of the two polarisations",
        reference=ref, description="computed on request from dvsv and dvsh")
    return m
