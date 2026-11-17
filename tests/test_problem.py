"""The shell layout of the model space and nm.problem, on tiny invented files; the same equalities on the real rows only behind
PLI_NORMAL_MODE_DATA. No real data is committed."""
import math
import os

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0_normalmodes as nm
from pygeoinf0.realisations.series.radial_direct import harmonic_element

CUTS = ("icb", "cmb", "d670", "d400", "d220")
R = nm.R_EARTH
MODES = ("00s02", "01s03")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def root(tmp_path):
    """Every number invented; degrees 0, 2 and 4, a kernel on each shell and at each named boundary."""
    base = tmp_path / "data"
    write(base / "observed" / "data_list_arwen-paula", "".join(m + "\n" for m in MODES))
    vol = base / "kernels/kernels-all_PREM-layers_Adrian"
    # a shell, then a zero-thickness row at each named radius, and the last row with r_outer < r_inner, as in the real table
    radii = sorted(r for r in nm.BOUNDARIES.values() if r < R) + [R]
    table, shells, markers, inner = [], [], [], 0.0
    for r in radii:
        table.append(f"{len(table) + 1} {inner} {r}\n")
        shells.append(len(table))
        if r < R:
            table.append(f"{len(table) + 1} {r} {r}\n")
        else:
            table.append(f"{len(table) + 1} {R} 0.0\n")
        markers.append(len(table))
        inner = r
    write(vol / "PREM_all-layers_radius", "".join(table))
    for j, mode in enumerate(MODES):
        rows = [(s, t) for s in (0, 2, 4) for t in range(2 * s + 1)]
        write(base / "observed" / "data_arwen-paula-fixed-anomalies" / f"{mode}.new",
              "4\n" + "".join(f"{s} {t} {1.0 + s + 0.1 * t + j} {0.25 + 0.05 * s}\n" for s, t in rows))
        write(base / "synthetics" / f"raw_{mode}_Crust5.1.raw", "".join(f"1 {s} {t} 0.125\n" for s, t in rows))
        for scale, field in enumerate(nm.FIELDS, 1):
            write(vol / f"kern_{mode}_{field}_PREM-layers-all_PREM-iso.kern",
                  "".join(f"{s} {layer} {scale * (s + 1 + j) * (1.0 + 0.1 * layer)}\n" for s in (0, 2, 4) for layer in shells))
        write(base / "kernels/kernels-topo-all_PREM-layers_Adrian" / f"kern_{mode}_topo-all_PREM-layers-all_PREM-iso.kern",
              "".join(f"{s} {n} {(s + 1 + j) * 0.5 + n * 0.01}\n" for s in (0, 2, 4) for n in markers))
    return base


def dense(A):
    return A.matrix() if hasattr(A, "matrix") else np.asarray(A)


# the layout -------------------------------------------------------------------------------------------------------------
def test_the_default_is_unchanged():
    assert nm.model_space().names == ("vp", "vs_ic", "vs_m", "rho") + nm.FREE_BOUNDARIES


def test_a_flat_sum_with_no_vs_in_the_outer_core_and_the_weights_of_today():
    M = nm.model_space(shells=CUTS)
    assert len(M.names) == 17 + 5 and "dpp" not in M.names and M.names[:6] == ("vp_centre_icb", "vp_icb_cmb", "vp_cmb_d670", "vp_d670_d400", "vp_d400_d220", "vp_d220_surface")
    assert "vs_icb_cmb" not in M.names and "vs_centre_icb" in M.names and "rho_icb_cmb" in M.names
    assert all(not isinstance(M.summand(k), pg.DirectSum) for k in M.names)
    V = 4 * math.pi * R ** 3 / 3
    for k, w in zip(M.names, M.weights):
        assert w == pytest.approx(1 / (4 * math.pi * nm.BOUNDARIES[k] ** 2) if k in nm.BOUNDARIES else 1 / V)


def test_a_cut_is_easy_to_drop_and_vs_stays_out_of_the_outer_core():
    M = nm.model_space(shells=("icb", "cmb", "d670", "d400"))
    assert "vp_d400_surface" in M.names and "vp_d220_surface" not in M.names
    only = nm.model_space(boundaries=(), shells=("d670",))
    assert only.names == ("vp_centre_d670", "vp_d670_surface", "vs_centre_icb", "vs_cmb_d670", "vs_d670_surface", "rho_centre_d670", "rho_d670_surface")


def test_crustal_names_are_cuts_and_surface_adds_no_shell():
    assert "vp_moho_ocean" in nm.model_space(boundaries=(), shells=("moho", "ocean")).names
    assert nm.model_space(boundaries=(), shells=("cmb", "surface")).names == nm.model_space(boundaries=(), shells=("cmb",)).names


@pytest.mark.parametrize("cuts, message", [(("mantle",), "unknown cuts"), (("cmb", "cmb"), "repeated")])
def test_bad_cuts_are_refused(cuts, message):
    with pytest.raises(ValueError, match=message):
        nm.model_space(shells=cuts)


def test_the_pairing_of_a_height_is_the_weight_on_the_unit_sphere_integral():
    """1 / (4 pi r^2) int_{S^2}: the norm^2 of the kernel Y_00 is 4 pi r^2, where an area average would give 1."""
    r = nm.BOUNDARIES["cmb"]
    M = pg.DirectSum({"cmb": pg.L2(pg.Sphere(1.0))}, weights={"cmb": nm.model_space(boundaries=("cmb",)).weights[-1]})
    G = pg.Functionals.separable(M, {"cmb": [1.0]}, [(0, 0)])
    assert dense(G.gram())[0, 0] == pytest.approx(4 * math.pi * r ** 2)


# G does not change ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cuts", [CUTS, ("icb", "cmb", "d670", "d400"), ("cmb",)])
def test_the_gram_is_the_same_on_every_layout(root, cuts):
    d, K = nm.read_catalogue(root), nm.Kernels(root)
    g0 = dense(nm.forward(d, K, nm.model_space()).gram())
    g1 = dense(nm.forward(d, K, nm.model_space(nm.FREE_BOUNDARIES, shells=cuts)).gram())
    assert np.linalg.norm(g1 - g0) <= 1e-13 * np.linalg.norm(g0)


def _element(S, c, lmax, value, jumps):
    """value(r) c on one summand, in the catalogue's basis, linear between many radii, with a true step at jumps."""
    a, b = S.backend.a, S.backend.b
    knots = [a] + [x for x in jumps if a < x < b] + [b]
    radii, values = [], []
    for lo, hi in zip(knots[:-1], knots[1:]):
        g = np.linspace(lo, hi, 400)
        g[0] = lo + (1e-6 if lo != a else 0.0)
        g[-1] = hi - (1e-6 if hi != b else 0.0)
        radii.append(g)
        values.append(value(np.linspace(lo + 1e-9, hi - 1e-9, 400)))
    radii, values = np.concatenate(radii), np.concatenate(values)
    hm = nm.HarmonicModel("m", "x", "fraction", lmax, radii, np.outer(values, c), basis="catalogue")
    return harmonic_element(S, radii, hm.in_basis("package").coefficients, name="m")


def _forward_of_a_jumpy_model(d, K, cuts, lmax=4):
    c = np.random.default_rng(1).standard_normal((lmax + 1) ** 2)
    jumps = (4500.0, nm.BOUNDARIES["d670"])                      # inside a shell, and at a cut
    value = lambda r: (1 + r / R) * (1 + 0.7 * (r > jumps[0])) * (1 + 0.3 * (r > jumps[1]))
    M = nm.model_space(boundaries=(), shells=cuts)
    return nm.forward(d, K, M)(M({k: _element(M.summand(k), c, lmax, value, jumps) for k in M.names})).values


def test_forward_of_a_model_that_jumps_is_the_same_on_every_layout_and_right(root):
    d, K = nm.read_catalogue(root), nm.Kernels(root)
    y0, y1 = _forward_of_a_jumpy_model(d, K, None), _forward_of_a_jumpy_model(d, K, CUTS)
    assert np.linalg.norm(y1 - y0) <= 1e-9 * np.linalg.norm(y0)
    # against an independent quadrature, one zonal row, vp and rho and the two pieces of vs, without the small centre taper
    row = np.flatnonzero((d.s == 2) & (d.t == 0))[0]
    c = np.random.default_rng(1).standard_normal(25)
    jumps = np.sort(np.r_[[r for r in nm.BOUNDARIES.values() if r < R], 4500.0])
    breaks = np.r_[1.0, jumps, R]
    total = 0.0
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        r, w = np.polynomial.legendre.leggauss(64)
        r, w = 0.5 * (hi - lo) * r + 0.5 * (hi + lo), 0.5 * (hi - lo) * w
        a = (1 + r / R) * (1 + 0.7 * (r > 4500.0)) * (1 + 0.3 * (r > nm.BOUNDARIES["d670"])) * c[4]
        for field in nm.FIELDS:
            if field == "vs" and nm.R_ICB <= lo and hi <= nm.R_CMB:
                continue
            total += np.sum(w * K.function(str(d.mode[row]), field, 2)(r) * a)
    assert y1[row] == pytest.approx(total, rel=1e-3)


# a prior can then choose -------------------------------------------------------------------------------------------------
def _covariances(degrees):
    def bessel(domain, bc):
        return pg.Laplacian(domain, bc=bc).bessel_potential(1 / 500.0, 2.0, degrees=degrees, modes=16)
    ball = {"boundary": "dirichlet"}
    return bessel(pg.Ball(R), ball), bessel(pg.Ball(nm.R_ICB), ball), bessel(pg.Shell(nm.R_CMB, R), {"inner": "dirichlet", "outer": "dirichlet"}), \
        pg.Covariance.angular(pg.L2(pg.Sphere(1.0)), {s: 0.5 for s in degrees})


def _glued_grams(d, K, degrees):
    ball, inner, mantle, angular = _covariances(degrees)
    M0, M1 = nm.model_space(), nm.model_space(nm.FREE_BOUNDARIES, shells=CUTS)
    pieces = lambda field: tuple(k for k in M1.names if k.startswith(field + "_"))
    heights = {b: angular for b in nm.FREE_BOUNDARIES}
    C0 = pg.Covariance.glued(M0, {"vp": ball, "rho": ball, "vs_ic": inner, "vs_m": mantle, **heights})
    C1 = pg.Covariance.glued(M1, {pieces("vp"): ball, pieces("rho"): ball, "vs_centre_icb": inner,
                                  tuple(k for k in pieces("vs") if k != "vs_centre_icb"): mantle, **heights})
    return dense(nm.forward(d, K, M0).structured_gram(C0)), dense(nm.forward(d, K, M1).structured_gram(C1))


def test_glued_pieces_give_the_undivided_structured_gram(root):
    d = nm.read_catalogue(root)
    g0, g1 = _glued_grams(d, nm.Kernels(root), [2, 4])
    assert np.linalg.norm(g1 - g0) <= 1e-10 * np.linalg.norm(g0) and np.linalg.norm(g0) > 0    # 2e-11 here, the quadrature on pieces


# nm.problem --------------------------------------------------------------------------------------------------------------
def test_problem_unpacks_to_m_g_d_with_the_noise_on_d(root):
    p = nm.problem(root, shells=CUTS)
    M, G, D = p
    data = p.data
    assert G.domain == M and G.codomain == D and D.dim == len(data)
    assert np.allclose(D.weights, 1 / data.sigma ** 2) and D.select(mode="00s02").size == np.sum(data.mode == "00s02")
    assert np.allclose(p.d.values, data.value) and p.d.space == D


def test_the_noise_changes_the_adjoint_and_not_the_gram(root):
    M, G, D = nm.problem(root)
    _, G0, D0 = nm.problem(root, noise=False)
    assert D0.is_standard and D0.select(mode="00s02").tolist() == D.select(mode="00s02").tolist()
    g = dense(G.gram())
    assert np.allclose(g, dense(G0.gram()))
    y = np.arange(1.0, D.dim + 1.0)
    assert np.allclose((G @ G.adjoint)(D(y)).values, g @ (y * D.weights))


def test_decompose_uses_the_noise_on_d(root):
    p = nm.problem(root)
    M, G, D = p
    dec = pg.decompose(G, D.standard_gaussian())
    assert dec.n == D.dim


def test_forward_keeps_its_unweighted_d(root):
    d = nm.read_catalogue(root)
    assert nm.forward(d, nm.Kernels(root)).codomain.is_standard
    assert nm.data_space(d).is_standard and not nm.data_space(d, noise=True).is_standard


def test_problem_from_a_selection_uses_its_rows(root):
    d = nm.read_catalogue(root).select(modes=["00s02"], degrees=[2], quiet=True)
    M, G, D = nm.problem(d)
    assert D.dim == 5 and M.names == nm.model_space(nm.PROBLEM_BOUNDARIES).names


def test_the_crust_and_its_boundaries_go_together(root):
    assert not set(nm.problem(root).M.names) & {"moho", "ocean", "surface"}
    assert {"moho", "ocean", "surface"} <= set(nm.problem(root, crust=False).M.names)
    with pytest.raises(ValueError, match="fixes"):
        nm.problem(root, boundaries=("cmb", "moho"))
    assert "moho" in nm.problem(root, crust=False, boundaries=("cmb", "moho")).M.names
    assert "dpp" not in nm.problem(root, boundaries=("cmb", "icb")).M.names


def test_a_catalogue_without_a_folder_needs_kernels(root):
    d = nm.read_splitting_data(root / "observed")
    with pytest.raises(ValueError, match="give kernels"):
        nm.problem(d)
    assert nm.problem(d, kernels=nm.Kernels(root), noise=False).D.dim == len(d)


# the real rows, only behind the variable ----------------------------------------------------------------------------------
DATA = os.environ.get("PLI_NORMAL_MODE_DATA")
real = pytest.mark.skipif(not DATA, reason="PLI_NORMAL_MODE_DATA is not set")


@real
def test_real_the_layout_changes_nothing_on_all_rows():
    d, K = nm.read_catalogue(DATA), nm.Kernels(DATA)
    assert len(d) == 8002
    g0 = dense(nm.forward(d, K, nm.model_space()).gram())
    g1 = dense(nm.forward(d, K, nm.model_space(nm.FREE_BOUNDARIES, shells=CUTS)).gram())
    assert np.linalg.norm(g1 - g0) <= 1e-13 * np.linalg.norm(g0)
    y0, y1 = _forward_of_a_jumpy_model(d, K, None, 14), _forward_of_a_jumpy_model(d, K, CUTS, 14)
    assert np.linalg.norm(y1 - y0) <= 1e-9 * np.linalg.norm(y0)
    b0, b1 = _glued_grams(d, K, [2, 4, 6, 8, 10, 12, 14])
    assert np.linalg.norm(b1 - b0) <= 1e-11 * np.linalg.norm(b0)


def test_shells_give_on_pieces_with_the_keys_and_weights_of_a_plain_sum():
    M = nm.model_space(shells=CUTS)
    assert isinstance(M, pg.PiecewiseSpace) and isinstance(M, nm.ModeSpace)
    assert M.key("vp", "cmb_d670") == "vp_cmb_d670" and "vs_icb_cmb" not in M.names
    assert M.pieces_of("vs") == [n for n in M.partition.names if n != "icb_cmb"]
    assert M.names[-5:] == nm.PROBLEM_BOUNDARIES and M.heights == nm.PROBLEM_BOUNDARIES
    for k in ("vp_centre_icb", "vp_d220_surface"):
        b = M.summand(k).backend
        assert (b.a, b.b) in {(0.0, nm.BOUNDARIES["icb"]), (nm.BOUNDARIES["d220"], R)}
    assert M.weights[M.index("cmb")] == pytest.approx(1 / (4 * math.pi * nm.BOUNDARIES["cmb"] ** 2))
    assert not isinstance(nm.model_space(shells=("d670",)), pg.PiecewiseSpace)


def test_a_target_across_shells_on_a_constant_field_is_its_mass():
    """A unit-mass bump straddling the CMB against vp = c, with no hand cutting; zero on the other fields."""
    M = nm.model_space(shells=CUTS)
    centre = (nm.BOUNDARIES["cmb"] + 100.0, 0.0, 0.0)
    import pygeoinf0.functions as F
    target = F.on(M.partition.domain).smooth_indicator(centre=centre, radius=600.0).normalised()
    l = M.functional("vp", target.measure())
    x = M.split({"vp": lambda *a: 0.01 + 0.0 * a[0]})
    assert l(x) == pytest.approx(0.01, rel=1e-6)
    assert M.functional("vp", target.measure())(M.split({"rho": lambda *a: 1.0 + 0.0 * a[0]})) == 0.0
    assert len(M.parts("vp", target.measure())) == 2


def test_problem_on_on_pieces_keeps_g_and_gives_the_noise_and_the_plain_problem(root):
    p = nm.problem(root, shells=CUTS, boundaries=nm.PROBLEM_BOUNDARIES)
    assert isinstance(p.M, nm.ModeSpace) and not p.D.is_standard
    q = p.plain()
    assert q.D.is_standard and q.M is p.M and q.plain() is q and q.rule == p.rule
    assert np.asarray(q.noise.covariance.matrix()).diagonal() == pytest.approx(p.data.sigma ** 2)
    assert q.d.values == pytest.approx(p.d.values)


def test_a_prior_by_degree_works_on_the_weighted_d_and_through_p_decompose(root):
    p = nm.problem(root, shells=CUTS, boundaries=nm.PROBLEM_BOUNDARIES)
    ball, inner, mantle, angular = _covariances([2, 4])
    pieces = lambda field: tuple(k for k in p.M.names if k.startswith(field + "_"))
    C = pg.Covariance.glued(p.M, {pieces("vp"): ball, pieces("rho"): ball, "vs_centre_icb": inner,
                                  tuple(k for k in pieces("vs") if k != "vs_centre_icb"): mantle,
                                  **{b: angular for b in nm.PROBLEM_BOUNDARIES}})
    q = p.plain()
    assert q.G.structured_gram(C) is not None and p.G.structured_gram(C) is not None
    dec = p.decompose(C)
    assert np.sort(pg.decompose(p.G, p.D.standard_gaussian(), middle=C).lam) == pytest.approx(np.sort(dec.lam), rel=1e-9, abs=1e-9 * np.abs(dec.lam).max())
    K = dense(q.G.structured_gram(C))
    W = np.diag(1.0 / p.data.sigma) @ K @ np.diag(1.0 / p.data.sigma)
    assert np.sort(dec.lam) == pytest.approx(np.sort(np.linalg.eigvalsh(W)), rel=1e-9, abs=1e-9 * np.linalg.norm(W))


def test_a_model_of_harmonic_models_is_the_element_on_the_pieces_and_the_heights(root):
    d, K = nm.read_catalogue(root), nm.Kernels(root)
    M = nm.model_space(shells=CUTS)
    G = nm.forward(d, K, M)
    lmax = 4
    c = np.random.default_rng(5).standard_normal((lmax + 1) ** 2)
    radii = np.array([0.0, nm.R_CMB - 1.0, nm.R_CMB + 1.0, R])                # no node in several shells
    a = np.outer([1.0, 0.5, 2.0, 1.0], c)
    hm = nm.HarmonicModel("m", "dvs", "", lmax, radii, a, "package")
    h = nm.HarmonicModel("t", "h", "km", lmax, np.array([nm.BOUNDARIES["cmb"]]), c[None], "package")

    def on_piece(key):
        S = M.summand(key)
        nodes = np.r_[S.backend.a, radii[(radii > S.backend.a) & (radii < S.backend.b)], S.backend.b]
        return harmonic_element(S, nodes, np.stack([np.interp(nodes, radii, a[:, k]) for k in range(a.shape[1])], axis=1))

    pieces = M({M.key("vs", n): on_piece(M.key("vs", n)) for n in M.pieces_of("vs")})
    x = M.model({"vs": hm, "cmb": h})
    assert G(M.model({"vs": hm})).values == pytest.approx(G(pieces).values, rel=1e-10, abs=1e-12)
    only = M.model({"cmb": h})
    assert G(only).values == pytest.approx(nm.predict({"cmb": h}, d, K), rel=1e-10, abs=1e-12)
    assert np.linalg.norm(G(x).values) > 0
    with pytest.raises(ValueError, match="unknown field"):
        M.model({"mantle": hm})
