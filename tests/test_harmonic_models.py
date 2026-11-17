"""Models on harmonic bases: invented fields with known coefficients, and optional checks on the real models and data."""
import os
from pathlib import Path

import numpy as np
import pytest

import pygeoinf0 as pg
import pygeoinf0_normalmodes as nm
from pygeoinf0_normalmodes import earthmodels as gm


class Field(gm.Model):
    """f = a(r) Y_21 + b Y_00, with Y_21 = sqrt(2) X_21 cos phi, Condon-Shortley phase (negative at mid northern latitudes)."""
    name, source = "invented", "formula"

    def __init__(self, unit="percent", mean=0.0):
        super().__init__()
        self.mean = mean
        self._fields["dvs"] = dict(unit=unit, quantity="d ln vs", reference="none", coverage=(100.0, 2000.0),
                                   native_depths=np.array([100.0, 1000.0, 2000.0]), description="")

    def _eval(self, field, lat, lon, depth):
        la, lo = np.radians(lat), np.radians(lon)
        y21 = -np.sqrt(15.0 / (4.0 * np.pi)) * np.sin(la) * np.cos(la) * np.cos(lo)
        return (6371.0 - depth) / 1000.0 * y21 + self.mean


def test_projection_finds_the_coefficient_and_the_phase():
    hm = nm.to_harmonics(Field(), "dvs", lmax=4, unit="percent")
    assert hm.radii.tolist() == [4371.0, 5371.0, 6271.0]
    assert hm(2, 1) == pytest.approx(hm.radii / 1000.0)
    other = np.delete(hm.coefficients, 4 + 1, axis=1)
    assert np.abs(other).max() < 1e-12


def test_units_and_the_mean():
    frac = nm.to_harmonics(Field(), "dvs", lmax=2)
    assert frac.unit == "fraction" and frac(2, 1)[0] == pytest.approx(0.04371)
    assert nm.to_harmonics(Field(mean=3.0), "dvs", lmax=2, about_mean=True).coefficients[:, 0] == pytest.approx(0.0)
    absolute = Field(unit="km/s", mean=10.0)
    with pytest.raises(ValueError, match="absolute speed"):
        nm.to_harmonics(absolute, "dvs", lmax=2)
    a = nm.to_harmonics(absolute, "dvs", lmax=2, about_mean=True)
    assert a.unit == "fraction" and a(2, 1)[-1] == pytest.approx(6.271 / 10.0) and a.coefficients[0, 0] == 0.0
    assert nm.to_harmonics(absolute, "dvs", lmax=2, unit=None).unit == "km/s"


def test_a_radius_outside_the_coverage_is_refused():
    with pytest.raises(ValueError, match="covers radii"):
        nm.to_harmonics(Field(), "dvs", radii=[100.0])


@pytest.mark.parametrize("basis, factor", [("orthonormal", 1.0), ("catalogue", np.sqrt(2.0)), ("package", -1.0),
                                           ("4pi", 1.0 / np.sqrt(4.0 * np.pi))])
def test_bases_change_the_numbers_not_the_field(basis, factor):
    hm = nm.to_harmonics(Field(), "dvs", lmax=4, unit="percent")
    other = hm.in_basis(basis)
    assert other(2, 1) == pytest.approx(factor * hm(2, 1))
    assert other.in_basis("orthonormal").coefficients == pytest.approx(hm.coefficients)
    assert other.evaluate(30.0, 40.0, 5000.0) == pytest.approx(hm.evaluate(30.0, 40.0, 5000.0))


def test_the_package_basis_is_pygeoinf0s():
    """A coefficient in "package" is a pygeoinf0 inner product with its own harmonic: the phase is right."""
    S = pg.L2(pg.Sphere(1.0))
    hm = nm.to_harmonics(Field(), "dvs", lmax=2, unit="percent", radii=[5371.0]).in_basis("package")
    h = S(lambda x, y, z: Field()._eval(None, np.degrees(np.arcsin(z)), np.degrees(np.arctan2(y, x)), 1000.0))
    G = pg.Functionals.separable(S, [1.0], [(2, 1)])
    assert G(h).values[0] == pytest.approx(hm(2, 1)[0], rel=1e-6)


def test_unknown_basis():
    with pytest.raises(ValueError, match="unknown basis"):
        nm.to_harmonics(Field(), "dvs", lmax=2).in_basis("schmidt")


def test_synthesis_returns_the_field_and_interpolates_in_radius():
    field = Field()
    hm = nm.to_harmonics(field, "dvs", lmax=4, unit="percent")
    lat, lon = np.array([10.0, -50.0, 80.0]), np.array([0.0, 120.0, -30.0])
    assert hm.evaluate(lat, lon, 5000.0) == pytest.approx(field._eval("dvs", lat, lon, np.full(3, 1371.0)))
    assert hm.evaluate(10.0, 0.0, 7000.0) == 0.0 and hm.evaluate(10.0, 0.0, 100.0) == 0.0
    assert hm(2, 1, np.array([100.0, 5000.0])) == pytest.approx([0.0, 5.0])
    assert hm(30, 0, np.array([5000.0])) == 0.0


def test_truncated_scaled_and_degree_zero():
    hm = nm.to_harmonics(Field(mean=2.0), "dvs", lmax=4, unit="percent")
    assert hm.truncated(1).coefficients.shape == (3, 4) and hm.truncated(2)(2, 1)[0] == pytest.approx(hm(2, 1)[0])
    assert hm.scaled(0.5)(2, 1)[0] == pytest.approx(0.5 * hm(2, 1)[0]) and hm.scaled(0.5).field == "0.5 x dvs"
    assert hm.coefficients[0, 0] != 0.0 and hm.without_degree_zero().coefficients[0, 0] == 0.0
    with pytest.raises(ValueError, match="lmax must be"):
        hm.truncated(9)
    with pytest.raises(ValueError):
        hm.coefficients[0, 0] = 1.0


def test_record_refuses_a_bad_shape():
    with pytest.raises(ValueError, match="expected"):
        nm.HarmonicModel("m", "f", "km", 2, [1.0], np.zeros((1, 3)))
    with pytest.raises(ValueError, match="increasing"):
        nm.HarmonicModel("m", "f", "km", 0, [2.0, 1.0], np.zeros((2, 1)))


def test_as_model_gives_the_slices_and_function_the_cartesian_field():
    hm = nm.to_harmonics(Field(), "dvs", lmax=4, unit="percent")
    m = hm.as_model()
    lat, lon, v = m.depth_slice("dvs", 1371.0, step=30.0)
    assert v.shape == (lat.size, lon.size)
    assert v[4, 6] == pytest.approx(float(Field()._eval("dvs", np.array([lat[4]]), np.array([lon[6]]), np.array([1371.0]))[0]))
    p = 5000.0 * np.array([np.cos(np.radians(30.0)), 0.0, np.sin(np.radians(30.0))])
    assert hm.function()(*p) == pytest.approx(hm.evaluate(30.0, 0.0, 5000.0))


@pytest.fixture
def folder(tmp_path):
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    root = tmp_path / "data"
    vol, topo = root / "kernels/kernels-all_PREM-layers_Adrian", root / "kernels/kernels-topo-all_PREM-layers_Adrian"
    write(vol / "PREM_all-layers_radius",
          f"1 0.0 {nm.R_CMB}\n2 {nm.R_CMB} {nm.R_CMB}\n3 {nm.R_CMB} 5000.0\n4 5000.0 6000.0\n5 6371.0 0.0\n")
    write(vol / "kern_00s02_vs_PREM-layers-all_PREM-iso.kern", "2 1 9\n2 3 2.0\n2 4 3.0\n2 5 0\n")
    write(topo / "kern_00s02_topo-all_PREM-layers-all_PREM-iso.kern", "2 2 5.0\n")
    write(root / "observed/data_list_arwen-paula", "00s02\n")
    write(root / "observed/data_arwen-paula-fixed-anomalies/00s02.new", "2\n2 0 1 1\n2 1 1 1\n")
    write(root / "synthetics/raw_00s02_M.raw", "2 2 0 11.0\n2 2 1 21.0\n")
    write(root / "synthetics/raw_00s02_B.raw", "2 2 0 1.0\n2 2 1 1.0\n")
    return root


def test_predict_sums_the_layers_in_the_catalogue_basis(folder):
    data = nm.read_catalogue(folder, crust=False)
    K = nm.Kernels(folder)
    radii = K.midpoints()
    c = np.zeros((radii.size, 9))
    c[:, 4] = 0.25                                   # orthonormal (2, 0)
    c[:, 5] = 0.5                                    # orthonormal (2, 1)
    hm = nm.HarmonicModel("m", "vs", "fraction", 2, radii, c)
    # layer 1 is outside the model's radii only if it has no midpoint; here it has one, with K = 9
    assert nm.predict({"vs": hm}, data, K) == pytest.approx(np.array([1.0, np.sqrt(2.0)]) * 100 * (9 + 2.0 + 3.0) * np.array([0.25, 0.5]))
    assert K.layer_matrix("vs", 2, data.modes)[1] is K.layer_matrix("vs", 2, data.modes)[1]      # kept
    with pytest.raises(ValueError):
        K.layer_matrix("vs", 2, data.modes)[1][0, 0] = 1.0
    with pytest.raises(ValueError, match="no rows at s = 4"):
        K.layer_matrix("vs", 4, data.modes)
    with pytest.raises(ValueError, match="no row at s = 4"):
        K.boundary_vector("cmb", 4, data.modes)
    h = nm.HarmonicModel("h", "cmb", "km", 2, [nm.R_CMB], c[:1], basis="catalogue")
    assert nm.predict({"cmb": h}, data, K) == pytest.approx(5.0 * np.array([0.25, 0.5]))
    with pytest.raises(ValueError, match="unknown field or boundary"):
        nm.predict({"mantle": hm}, data, K)


def test_recover_topography(folder):
    K = nm.Kernels(folder)
    h = nm.recover_topography(K, folder / "synthetics", "M", "B", lmax=2)
    assert h.basis == "catalogue" and h(2, 0, np.array([nm.R_CMB]))[0] == pytest.approx(2.0) and h(2, 1, np.array([nm.R_CMB]))[0] == pytest.approx(4.0)
    data = nm.read_catalogue(folder, crust=False)
    assert nm.predict({"cmb": h}, data, K) == pytest.approx([10.0, 20.0])


# ---- optional, real models and data ----
DATA, MODELS = os.environ.get("PLI_NORMAL_MODE_DATA"), os.environ.get("GLOBALMODELS_ORIGINALS")
real = pytest.mark.skipif(not (DATA and MODELS), reason="PLI_NORMAL_MODE_DATA and GLOBALMODELS_ORIGINALS are not both set")


def closure(root, models, model):
    data = nm.read_splitting_data(root / "observed").select(degrees=range(2, 15, 2))
    syn = nm.synthetic_values(data, root / "synthetics", model)
    pred = nm.predict(models, data, nm.Kernels(root))
    slope = (pred @ syn) / (pred @ pred)
    return slope, np.linalg.norm(syn - pred) / np.linalg.norm(syn)


@real
def test_real_s20rts_closes_on_the_synthetics_without_any_factor():
    """The kernels at 100 K_l, the catalogue's basis, vp = 0.5 vs, rho = 0.3 vs: the mode code's own S20RTS synthetics."""
    root = Path(DATA)
    K = nm.Kernels(root)
    hm = nm.to_harmonics("S20RTS", "dvs", lmax=14, radii=K.midpoints(within=(3480.0, 6346.619)), n_lat=40)
    slope, residual = closure(root, {"vs": hm, "vp": hm.scaled(0.5), "rho": hm.scaled(0.3)}, "S20RTS")
    assert abs(slope - 1.0) < 1e-3 and residual < 2e-3
    # the other scalings do not close, so the check can tell
    assert closure(root, {"vs": hm, "vp": hm.scaled(0.5), "rho": hm.scaled(0.4)}, "S20RTS")[1] > 0.02
    assert closure(root, {"vs": hm}, "S20RTS")[1] > 0.1


@real
def test_real_s20rts_with_a_recovered_topography_closes_on_the_combined_synthetic():
    root = Path(DATA)
    K = nm.Kernels(root)
    hm = nm.to_harmonics("S20RTS", "dvs", lmax=14, radii=K.midpoints(within=(3480.0, 6346.619)), n_lat=40)
    modes = nm.read_mode_list(root / "observed" / "data_list_arwen-paula")
    h = nm.recover_topography(K, root / "synthetics", "S20RTS-Crust5.1_cmb_tanaka-2010", "S20RTS-Crust5.1", modes=modes)
    data = nm.read_splitting_data(root / "observed").select(degrees=range(2, 15, 2))
    diff = (nm.synthetic_values(data, root / "synthetics", "S20RTS-Crust5.1_cmb_tanaka-2010")
            - nm.synthetic_values(data, root / "synthetics", "S20RTS-Crust5.1"))
    pred = nm.predict({"cmb": h}, data, K)
    assert np.linalg.norm(diff - pred) <= 2e-3 * np.linalg.norm(diff)


@real
def test_real_forward_map_agrees_with_predict():
    """pg's G on S20RTS at degree 2 of 00s05. The model must span its coverage: on the layer midpoints alone it would
    stop half a layer short at each end (2e-2 here). The fit warns on the kinks of the linear radial interpolation."""
    root = Path(DATA)
    K = nm.Kernels(root)
    hm = nm.to_harmonics("S20RTS", "dvs", lmax=2, radii=np.linspace(3480.0, 6346.6, 400), n_lat=40).without_degree_zero()
    data = nm.read_splitting_data(root / "observed", modes="00s05").select(degrees=[2])
    M = pg.DirectSum({"m": pg.L2(pg.Shell(3480.0, 6346.6), realisation="radial_harmonics")}, weights={"m": 1.0})
    for rule in ("constant", "linear"):
        got = nm.forward(data, K, M, {"m": "vs"}, rule=rule)(M({"m": hm.function()})).values
        assert got == pytest.approx(nm.predict({"vs": hm}, data, K), rel=5e-3)


@real
def test_real_element_without_fitting_agrees_with_predict_at_full_size():
    root = Path(DATA)
    K = nm.Kernels(root)
    hm = nm.to_harmonics("S20RTS", "dvs", lmax=14, radii=np.linspace(3480.0, 6346.6, 400), n_lat=40).without_degree_zero()
    data = nm.read_catalogue(root)
    M = nm.model_space()
    m = M({"vs_m": hm.element(M.summand("vs_m")), "vp": hm.scaled(0.5).element(M.summand("vp")),
           "rho": hm.scaled(0.3).element(M.summand("rho"))})
    got = nm.forward(data, K)(m).values
    want = nm.predict({"vs": hm, "vp": hm.scaled(0.5), "rho": hm.scaled(0.3)}, data, K)
    assert np.linalg.norm(got - want) < 3e-3 * np.linalg.norm(want)


@real
def test_real_catalogue_lists_the_models():
    assert len(gm.model_names()) == 19
    assert "S20RTS" in gm.model_names() and gm.load("S20RTS").fields() == ["dvs"]
    for name, field in (("S40RTS", "dvs"), ("SP12RTS", "dvs"), ("SP12RTS", "dvp"), ("P12", "dvp")):
        m = gm.load(name)
        hm = nm.to_harmonics(m, field, lmax=8, radii=[5000.0])
        assert hm.coefficients[0] == pytest.approx(m.sh_coefficients(field, 1371.0, lmax=8))
