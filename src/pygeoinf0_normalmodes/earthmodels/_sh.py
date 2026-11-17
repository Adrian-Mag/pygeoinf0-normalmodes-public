"""Spherical-harmonic routines: synthesis of real expansions at points and on grids, projection onto
orthonormal real harmonics, and Gauss-Legendre quadrature grids.

Convention used inside the package. X_lm(theta) is the fully normalised associated Legendre function with the
Condon-Shortley phase, X_lm = (-1)^m sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) P_l^m(cos theta), so that
X_lm(theta) exp(i m phi) is orthonormal on the unit sphere and sum_m |X_lm|^2 e^{...} adds to (2l+1)/(4 pi).
An expansion is stored as two arrays A[l, m], B[l, m] (0 <= m <= l <= lmax) with

    f(theta, phi) = sum_l sum_m X_lm(theta) (A[l, m] cos m phi + B[l, m] sin m phi).

This is the form of Ritsema's .sph files and SPECFEM's lgndr (A = a, B = b directly). Real orthonormal
harmonics Y_st (int Y_st Y_s't' dOmega = delta; t = 0 zonal, t = 2m-1 the cos(m phi) part, t = 2m the
sin(m phi) part) are Y_s0 = X_s0, Y_s,2m-1 = sqrt(2) X_sm cos m phi, Y_s,2m = sqrt(2) X_sm sin m phi.
"""
import numpy as np


def legendre_m(lmax, m, cost, sint):
    """X_lm for l = m..lmax at points with cos(theta) = cost, sin(theta) = sint: array (lmax - m + 1, n)."""
    cost = np.asarray(cost, float)
    sint = np.asarray(sint, float)
    out = np.empty((lmax - m + 1,) + cost.shape)
    pmm = np.full(cost.shape, 1.0 / np.sqrt(4.0 * np.pi))
    for k in range(1, m + 1):
        pmm = -np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * sint * pmm
    out[0] = pmm
    if lmax == m:
        return out
    out[1] = np.sqrt(2.0 * m + 3.0) * cost * pmm
    for l in range(m + 2, lmax + 1):
        a = np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
        b = np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
        out[l - m] = a * (cost * out[l - m - 1] - b * out[l - m - 2])
    return out


def raw_to_ab(raw, lmax):
    """Ritsema/SPECFEM coefficient vector (per degree: a_l0, a_l1, b_l1, ..., a_ll, b_ll) to A, B arrays."""
    raw = np.asarray(raw, float)
    A = np.zeros((lmax + 1, lmax + 1))
    B = np.zeros((lmax + 1, lmax + 1))
    pos = 0
    for l in range(lmax + 1):
        blk = raw[pos:pos + 2 * l + 1]
        pos += 2 * l + 1
        A[l, 0] = blk[0]
        A[l, 1:l + 1] = blk[1::2]
        B[l, 1:l + 1] = blk[2::2]
    if pos != raw.size:
        raise ValueError(f"coefficient vector of length {raw.size} does not match lmax {lmax}")
    return A, B


def synth_points(A, B, lat, lon):
    """sum_lm X_lm (A cos m phi + B sin m phi) at points (lat, lon in degrees, same shape)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    theta = np.radians(90.0 - lat)
    phi = np.radians(lon)
    cost, sint = np.cos(theta), np.sin(theta)
    lmax = A.shape[0] - 1
    out = np.zeros(lat.shape)
    for m in range(lmax + 1):
        a, b = A[m:, m], B[m:, m]
        if not (a.any() or b.any()):
            continue
        X = legendre_m(lmax, m, cost, sint)
        ga = np.tensordot(a, X, axes=(0, 0))
        if m == 0:
            out += ga
        else:
            gb = np.tensordot(b, X, axes=(0, 0))
            out += ga * np.cos(m * phi) + gb * np.sin(m * phi)
    return out


def synth_grid(A, B, lat, lon):
    """The expansion on the grid lat (n_lat,) x lon (n_lon,), degrees; returns (n_lat, n_lon)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    theta = np.radians(90.0 - lat)
    phi = np.radians(lon)
    cost, sint = np.cos(theta), np.sin(theta)
    lmax = A.shape[0] - 1
    out = np.zeros((lat.size, lon.size))
    for m in range(lmax + 1):
        a, b = A[m:, m], B[m:, m]
        if not (a.any() or b.any()):
            continue
        X = legendre_m(lmax, m, cost, sint)
        out += np.outer(a @ X, np.cos(m * phi))
        if m:
            out += np.outer(b @ X, np.sin(m * phi))
    return out


def gauss_grid(n_lat, n_lon=None):
    """Gauss-Legendre grid: lat (n_lat,) deg, lon (n_lon,) deg, area weights w (n_lat, n_lon) summing to 4 pi.
    Exact for band-limited fields of degree < n_lat (and < n_lon / 2 in longitude)."""
    n_lon = 2 * n_lat if n_lon is None else n_lon
    x, wl = np.polynomial.legendre.leggauss(n_lat)
    lat = np.degrees(np.arcsin(x))
    lon = np.arange(n_lon) * 360.0 / n_lon
    w = np.outer(wl, np.full(n_lon, 2.0 * np.pi / n_lon))
    return lat, lon, w


def real_sh_indices(lmax):
    """(s, t) pairs of the real orthonormal basis, s = 0..lmax, t = 0..2s."""
    return [(s, t) for s in range(lmax + 1) for t in range(2 * s + 1)]


def project(values, lat, lon, weights, lmax):
    """Coefficients c_st of the real orthonormal harmonics (real_sh_indices order) of a field on a quadrature
    grid (values (n_lat, n_lon), weights summing to 4 pi): c_st = sum w f Y_st."""
    theta = np.radians(90.0 - np.asarray(lat, float))
    phi = np.radians(np.asarray(lon, float))
    cost, sint = np.cos(theta), np.sin(theta)
    wf = np.asarray(weights) * np.asarray(values)
    out = {}
    for m in range(lmax + 1):
        X = legendre_m(lmax, m, cost, sint)               # (lmax - m + 1, n_lat)
        if m == 0:
            col = wf.sum(axis=1)
            for l in range(lmax + 1):
                out[(l, 0)] = float(X[l] @ col)
        else:
            cm = wf @ np.cos(m * phi)
            sm = wf @ np.sin(m * phi)
            for l in range(m, lmax + 1):
                out[(l, 2 * m - 1)] = float(np.sqrt(2.0) * X[l - m] @ cm)
                out[(l, 2 * m)] = float(np.sqrt(2.0) * X[l - m] @ sm)
    return np.array([out[st] for st in real_sh_indices(lmax)])


def real_to_ab(coeffs, lmax):
    """Real orthonormal coefficients (real_sh_indices order) to the package's A, B arrays."""
    A = np.zeros((lmax + 1, lmax + 1))
    B = np.zeros((lmax + 1, lmax + 1))
    for c, (s, t) in zip(coeffs, real_sh_indices(lmax)):
        if t == 0:
            A[s, 0] = c
        elif t % 2 == 1:
            A[s, (t + 1) // 2] = np.sqrt(2.0) * c
        else:
            B[s, t // 2] = np.sqrt(2.0) * c
    return A, B
