"""Readers on tiny synthetic files in the real formats (every number here is invented), and an optional check against
the real data, skipped unless PLI_NORMAL_MODE_DATA points at the data folder. No real data is committed."""
import math
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.special import sph_harm_y

import pygeoinf0 as pg
import pygeoinf0_normalmodes as nm


# Invented table with the real shape: row 3 is a zero-thickness boundary, row 4 the first shell above the same radius,
# row 6 the surface row with r_outer < r_inner.
TABLE = """1 0.0 10.5
2 10.5 20.25
3 20.25 20.25
4 20.25 30.0
5 30.0 40.0
6 40.0 0.0
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def prem(tmp_path):
    return nm.read_prem_layers(write(tmp_path / "table", TABLE))


def kernel(folder, mode, param, text):
    return write(folder / f"kern_{mode}_{param}_PREM-layers-all_PREM-iso.kern", text)


@pytest.fixture
def observed(tmp_path):
    root = tmp_path / "observed"
    write(root / "data_list_arwen-paula", "00s02\n01t03\n")
    write(root / "data_list_SP12RTS", "00s02\n")
    write(root / "data_arwen-paula" / "00s02.new", "2\n0 0 1.5 0.5\n2 0 -2e-1 0.25\n2 4 3.0 1\n")
    write(root / "data_arwen-paula" / "01t03.new", "4\n4 8 7.0 2.0\n")
    write(root / "data_arwen-paula-fixed-anomalies" / "00s02.new", "2\n0 0 9.0 0.5\n")
    write(root / "data_arwen-paula-fixed-anomalies" / "01t03.new", "4\n4 8 8.0 2.0\n")
    write(root / "data_arwen-paula-fixed-anomalies" / "notes.md", "not a data file\n")
    write(root / "data_SP12RTS" / "00s02.new", "0\n0 0 4.0 1.0\n")
    return root


def test_prem_layers(prem):
    assert len(prem) == 6
    assert prem.interval(2) == (10.5, 20.25)
    assert prem.layer[prem.zero_thickness].tolist() == [3]
    assert prem.layer[prem.inverted].tolist() == [6]


def test_prem_layers_may_skip_numbers(tmp_path):
    table = nm.read_prem_layers(write(tmp_path / "t", "1 0 1\n5 1 2\n"))
    assert table.interval(5) == (1.0, 2.0)
    with pytest.raises(ValueError, match="layer 2 is outside"):
        table.interval(2)


@pytest.mark.parametrize("text, message", [("1 0.0\n", "expected 3 columns"), ("x 0 1\n", "layer is not an integer"),
                                           ("1.5 0 1\n", "layer is not an integer"), ("1 a 1\n", "r_inner is not a number"),
                                           ("1 nan 1\n", "r_inner is not finite"), ("1 0 1\n1 1 2\n", "repeated"),
                                           ("\n", "no layers")])
def test_prem_layers_bad_rows(tmp_path, text, message):
    with pytest.raises(nm.FormatError, match=message):
        nm.read_prem_layers(write(tmp_path / "t", text))


def test_bad_row_names_file_and_line(tmp_path):
    path = write(tmp_path / "t", "1 0 1\n2 1\n")
    with pytest.raises(nm.FormatError, match=f"{path}:2"):
        nm.read_prem_layers(path)


def test_missing_files(tmp_path, prem):
    with pytest.raises(FileNotFoundError):
        nm.read_prem_layers(tmp_path / "none")
    with pytest.raises(FileNotFoundError):
        nm.read_mode_list(tmp_path / "none")
    with pytest.raises(FileNotFoundError):
        nm.read_kernel(tmp_path, "00s02", "vs", prem)
    with pytest.raises(FileNotFoundError):
        nm.list_kernel_modes(tmp_path / "none", "vs")
    with pytest.raises(FileNotFoundError):
        nm.read_splitting_data(tmp_path)
    with pytest.raises(FileNotFoundError):
        nm.read_synthetics(tmp_path, "00s02", "M")


def test_mode_list(tmp_path):
    assert nm.read_mode_list(write(tmp_path / "l", "00s02\n\n01t03\n")) == ("00s02", "01t03")
    with pytest.raises(nm.FormatError, match="not a mode id"):
        nm.read_mode_list(write(tmp_path / "l", "0s2\n"))
    with pytest.raises(nm.FormatError, match="repeated"):
        nm.read_mode_list(write(tmp_path / "l", "00s02\n00s02\n"))
    with pytest.raises(nm.FormatError, match="expected 1 columns"):
        nm.read_mode_list(write(tmp_path / "l", "00s02 01t03\n"))


def test_splitting_data(observed):
    d = nm.read_splitting_data(observed, "arwen-paula", superseded=True)
    assert d.modes == ("00s02", "01t03") and d.smax.tolist() == [2, 4] and len(d) == 4
    assert d.mode.tolist() == ["00s02"] * 3 + ["01t03"]
    assert d.s.tolist() == [0, 2, 2, 4] and d.t.tolist() == [0, 0, 4, 8]
    assert d.value.tolist() == [1.5, -0.2, 3.0, 7.0] and d.sigma.tolist() == [0.5, 0.25, 1.0, 2.0]
    assert d.smax_of("01t03") == 4
    with pytest.raises(ValueError, match="not in this selection"):
        d.smax_of("99s99")


def test_splitting_datasets_and_selection(observed):
    # the fixed-anomalies set shares the mode list and ignores other files in its folder
    assert nm.read_splitting_data(observed).value.tolist() == [9.0, 8.0]
    assert nm.read_splitting_data(observed, "SP12RTS", superseded=True).modes == ("00s02",)
    assert nm.read_splitting_data(observed, modes="01t03").value.tolist() == [8.0]
    both = nm.read_splitting_data(observed, "arwen-paula", modes=["01t03", "00s02"], superseded=True)
    assert both.mode.tolist() == ["01t03"] + ["00s02"] * 3
    # the SP12RTS list read from the corrected folder, not the delivered values
    assert nm.read_splitting_data(observed, "SP12RTS-fixed-anomalies").value.tolist() == [9.0]


@pytest.mark.parametrize("dataset", ["arwen-paula", "SP12RTS"])
def test_a_superseded_delivery_is_refused_unless_asked_for(observed, dataset):
    with pytest.raises(ValueError, match="uncorrected values for 00s03"):
        nm.read_splitting_data(observed, dataset)


def test_select_and_degree_zero(observed):
    d = nm.read_splitting_data(observed, "arwen-paula", superseded=True)
    no_zero = d.without_degree_zero()
    assert no_zero.s.tolist() == [2, 2, 4] and no_zero.modes == d.modes
    assert d.select(exclude=["01t03"]).modes == ("00s02",) and d.select(modes="01t03").value.tolist() == [7.0]
    gone = d.select(degrees=[4])                       # a mode with no row left is dropped, with its smax
    assert gone.modes == ("01t03",) and gone.smax.tolist() == [4] and not gone.value.flags.writeable


def test_splitting_refusals(observed):
    with pytest.raises(ValueError, match="unknown dataset 'other'"):
        nm.read_splitting_data(observed, "other")
    with pytest.raises(ValueError, match="not listed"):
        nm.read_splitting_data(observed, modes=["02s02"])
    (observed / "data_arwen-paula" / "01t03.new").unlink()
    with pytest.raises(FileNotFoundError, match="01t03.new"):
        nm.read_splitting_data(observed, "arwen-paula", superseded=True)


@pytest.mark.parametrize("text, message", [("", "first line must hold only smax"), ("2 0\n", "first line must hold only smax"),
                                           ("x\n", "smax is not an integer"), ("2\n0 0 1\n", "expected 4 columns"),
                                           ("2\n4 0 1 1\n", "outside 0..smax"), ("2\n2 5 1 1\n", "outside 0..2s"),
                                           ("2\n0 0 1 0\n", "not positive"), ("2\n0 0 inf 1\n", "not finite"),
                                           ("2\n0 0 x 1\n", "value is not a number"),
                                           ("2\n0 0 1 1\n0 0 2 1\n", "repeated")])
def test_splitting_bad_rows(observed, text, message):
    write(observed / "data_arwen-paula-fixed-anomalies" / "00s02.new", text)
    with pytest.raises(nm.FormatError, match=message):
        nm.read_splitting_data(observed, modes="00s02")


def test_kernel_gives_each_row_its_interval(tmp_path, prem):
    kernel(tmp_path, "00s02", "vs", "0 1 0.5\n0 2 1.5\n2 1 2.5\n")
    k = nm.read_kernel(tmp_path, "00s02", "vs", prem)
    assert k.s.tolist() == [0, 0, 2] and k.layer.tolist() == [1, 2, 1] and k.value.tolist() == [0.5, 1.5, 2.5]
    assert k.r_inner.tolist() == [0.0, 10.5, 0.0] and k.r_outer.tolist() == [10.5, 20.25, 10.5]


def test_kernel_row_order_does_not_matter_for_a_shuffled_table(tmp_path):
    table = nm.read_prem_layers(write(tmp_path / "t", "2 1 2\n1 0 1\n"))
    kernel(tmp_path, "00s02", "vs", "0 1 0.5\n0 2 1.5\n")
    assert nm.read_kernel(tmp_path, "00s02", "vs", table).r_inner.tolist() == [0.0, 1.0]


def test_the_layer_number_trap(tmp_path, prem):
    # The old topo file calls the boundary 4 (the shell above it), the new topo-all file 3 (the zero-thickness row).
    kernel(tmp_path, "00s02", "topo", "0 4 1.0\n2 4 2.0\n")
    kernel(tmp_path, "00s02", "topo-all", "0 3 1.0\n2 3 2.0\n0 6 5.0\n2 6 6.0\n")
    old = nm.read_kernel(tmp_path, "00s02", "topo", prem)
    new = nm.read_kernel(tmp_path, "00s02", "topo-all", prem)
    assert old.layer[0] != new.layer[0]
    at_old, at_new = old.at_boundary(20.25), new.at_boundary(20.25)
    assert at_old.value.tolist() == at_new.value.tolist() == [1.0, 2.0]
    assert at_new.s.tolist() == [0, 2]
    # one radius, two numbers in the table: asking by radius shows both
    assert prem.at_radius(20.25).tolist() == [3, 4]
    assert old.boundary_radii.tolist() == [20.25] and new.boundary_radii.tolist() == [20.25, 40.0]
    # the surface row has r_outer < r_inner, and r_inner is still its radius
    assert new.at_boundary(40.0).value.tolist() == [5.0, 6.0]
    with pytest.raises(ValueError, match="no boundary at radius 30"):
        new.at_boundary(30.0)


def test_at_boundary_is_for_topography(tmp_path, prem):
    kernel(tmp_path, "00s02", "vs", "0 1 0.5\n")
    k = nm.read_kernel(tmp_path, "00s02", "vs", prem)
    with pytest.raises(ValueError, match="volume kernel"):
        k.at_boundary(0.0)
    with pytest.raises(ValueError, match="volume kernel"):
        k.boundary_radii


def test_kernel_layer_outside_table(tmp_path, prem):
    path = kernel(tmp_path, "00s02", "vs", "0 1 0.5\n0 7 1.0\n")
    with pytest.raises(ValueError, match="layer 7 is outside") as e:
        nm.read_kernel(tmp_path, "00s02", "vs", prem)
    assert str(path) in str(e.value)
    kernel(tmp_path, "00s02", "vs", "0 0 0.5\n")
    with pytest.raises(ValueError, match="layer 0 is outside"):
        nm.read_kernel(tmp_path, "00s02", "vs", prem)


@pytest.mark.parametrize("text, message", [("0 1\n", "expected 3 columns"), ("x 1 1\n", "s is not an integer"),
                                           ("0 x 1\n", "layer is not an integer"), ("0 1 x\n", "value is not a number"),
                                           ("1 1 1\n", "not even"), ("-2 1 1\n", "not even"),
                                           ("0 1 1\n0 1 2\n", "repeated"), ("", "no rows")])
def test_kernel_bad_rows(tmp_path, prem, text, message):
    kernel(tmp_path, "00s02", "vs", text)
    with pytest.raises(nm.FormatError, match=message):
        nm.read_kernel(tmp_path, "00s02", "vs", prem)


def test_kernel_refusals(tmp_path, prem):
    with pytest.raises(ValueError, match="unknown kernel param 'q'"):
        nm.read_kernel(tmp_path, "00s02", "q", prem)
    with pytest.raises(nm.FormatError, match="not a mode id"):
        nm.read_kernel(tmp_path, "nope", "vs", prem)
    with pytest.raises(TypeError, match="PremLayers"):
        nm.read_kernel(tmp_path, "00s02", "vs", "table")
    with pytest.raises(ValueError, match="unknown kernel param"):
        nm.list_kernel_modes(tmp_path, "q")


def test_list_kernel_modes_keeps_params_apart(tmp_path):
    kernel(tmp_path, "01t03", "topo", "0 1 1\n")
    kernel(tmp_path, "00s02", "topo", "0 1 1\n")
    kernel(tmp_path, "02s02", "topo-all", "0 1 1\n")
    write(tmp_path / "README.md", "x\n")
    assert nm.list_kernel_modes(tmp_path, "topo") == ("00s02", "01t03")
    assert nm.list_kernel_modes(tmp_path, "topo-all") == ("02s02",)
    assert nm.list_kernel_modes(tmp_path, "vs") == ()


def test_synthetics(tmp_path):
    # model names may hold underscores
    write(tmp_path / "raw_00s02_A-B_c_d-1.raw", "2 0 0 1.5\n2 2 4 -2.5\n")
    d = nm.read_synthetics(tmp_path, "00s02", "A-B_c_d-1")
    assert d.l.tolist() == [2, 2] and d.s.tolist() == [0, 2] and d.t.tolist() == [0, 4] and d.value.tolist() == [1.5, -2.5]
    write(tmp_path / "raw_00s02_B.raw", "2 0 0\n")
    with pytest.raises(nm.FormatError, match="expected 4 columns"):
        nm.read_synthetics(tmp_path, "00s02", "B")
    write(tmp_path / "raw_00s02_C.raw", "")
    with pytest.raises(nm.FormatError, match="no rows"):
        nm.read_synthetics(tmp_path, "00s02", "C")


def test_records_are_read_only(tmp_path, prem, observed):
    kernel(tmp_path, "00s02", "topo", "0 4 1.0\n")
    k = nm.read_kernel(tmp_path, "00s02", "topo", prem)
    d = nm.read_splitting_data(observed)
    arrays = [prem.layer, prem.r_inner, prem.r_outer, prem.at_radius(20.25), d.smax, d.mode, d.s, d.t, d.value, d.sigma,
              k.s, k.layer, k.value, k.r_inner, k.r_outer, k.boundary_radii, k.at_boundary(20.25).value]
    for a in arrays:
        assert not a.flags.writeable
    with pytest.raises(ValueError):
        k.value[0] = 2.0
    with pytest.raises(AttributeError):
        k.value = None


# ---- catalogue, kernels as functions, forward map, on invented files ----
@pytest.fixture
def folder(tmp_path):
    """A tiny data folder: three shells and the surface row; one mode, degrees 0 and 2."""
    root = tmp_path / "data"
    vol, topo = root / "kernels/kernels-all_PREM-layers_Adrian", root / "kernels/kernels-topo-all_PREM-layers_Adrian"
    # invented radii on a body of radius 6371, with the real boundaries so that model_space() applies
    table = f"""1 0.0 {nm.R_ICB}
2 {nm.R_ICB} {nm.R_ICB}
3 {nm.R_ICB} {nm.R_CMB}
4 {nm.R_CMB} {nm.R_CMB}
5 {nm.R_CMB} 6371.0
6 6371.0 0.0
"""
    write(vol / "PREM_all-layers_radius", table)
    for field, scale in (("vp", 1.0), ("vs", 2.0), ("rho", 3.0)):
        rows = "".join(f"{s} {l} {scale * (s + 1) * l}\n" for s in (0, 2) for l in (1, 3, 5)) + "0 6 0\n2 6 0\n"
        kernel(vol, "00s02", field, rows)
    kernel(topo, "00s02", "topo-all", "".join(f"{s} {l} {(s + 1) * l * 0.5}\n" for s in (0, 2) for l in (2, 4, 6)))
    write(root / "observed/data_list_arwen-paula", "00s02\n")
    write(root / "observed/data_arwen-paula-fixed-anomalies/00s02.new", "2\n0 0 1.0 1\n2 0 2.0 1\n2 1 3.0 1\n2 2 4.0 1\n")
    write(root / "synthetics/raw_00s02_Crust5.1.raw", "2 0 0 0.5\n2 2 0 0.25\n2 2 1 0.5\n2 2 2 1.0\n2 2 3 9\n")
    return root


def test_read_catalogue_is_corrected_without_degree_zero_and_crust_subtracted(folder):
    d = nm.read_catalogue(folder)
    assert d.s.tolist() == [2, 2, 2] and d.value.tolist() == [1.75, 2.5, 3.0] and d.corrections == ("Crust5.1",)
    raw = nm.read_catalogue(folder, crust=False, degree_zero=True)
    assert raw.value.tolist() == [1.0, 2.0, 3.0, 4.0] and raw.corrections == ()
    with pytest.raises(ValueError, match="already been subtracted"):
        nm.crust_corrected(d, folder / "synthetics")
    assert len(nm.read_catalogue(folder, exclude=["00s02"])) == 0


def test_read_catalogue_needs_a_folder(monkeypatch):
    monkeypatch.delenv("PLI_NORMAL_MODE_DATA", raising=False)
    with pytest.raises(ValueError, match="PLI_NORMAL_MODE_DATA"):
        nm.read_catalogue()


def test_synthetic_values_refuse_a_missing_row(folder):
    d = nm.read_catalogue(folder, crust=False)
    write(folder / "synthetics/raw_00s02_Crust5.1.raw", "2 2 0 0.25\n")
    with pytest.raises(ValueError, match="no row"):
        nm.synthetic_values(d, folder / "synthetics", "Crust5.1")


def test_kernel_function_is_one_percent_per_layer(folder):
    K = nm.Kernels(folder)
    f = K.function("00s02", "vs", 2, rule="constant")
    # value = 2 * (s + 1) * layer on layer 3; the density is 100 value / thickness
    r = 0.5 * (nm.R_ICB + nm.R_CMB)
    assert f(np.array([r]))[0] == pytest.approx(100 * 2 * 3 * 3 / (nm.R_CMB - nm.R_ICB))
    g = K.function("00s02", "vs", 2)
    assert g(np.array([r]))[0] == pytest.approx(f(np.array([r]))[0])   # the shell's midpoint is a node of the linear rule
    assert g(np.array([6371.0]))[0] == 0.0                             # the surface row
    assert K.function("00s02", "vs", 2) is g                           # cached


def test_kernel_function_refusals(folder):
    K = nm.Kernels(folder)
    with pytest.raises(ValueError, match="unknown field"):
        K.function("00s02", "qq", 2)
    with pytest.raises(ValueError, match="no rows at s = 4"):
        K.function("00s02", "vs", 4)
    with pytest.raises(ValueError, match="rule is"):
        K.function("00s02", "vs", 2, rule="spline")
    kernel(K.volume, "00s02", "vp", "0 1 1\n0 6 2\n")
    with pytest.raises(ValueError, match="not a shell"):
        K.function("00s02", "vp", 0)


def test_boundary_by_name_and_radius(folder):
    K = nm.Kernels(folder)
    assert K.boundary("00s02", "cmb", 2) == K.boundary("00s02", nm.R_CMB, 2) == 3 * 4 * 0.5
    assert K.boundary("00s02", "surface", 0) == 6 * 0.5
    with pytest.raises(ValueError, match="unknown boundary"):
        K.boundary("00s02", "mantle", 2)
    with pytest.raises(ValueError, match="no boundary at radius"):
        K.boundary("00s02", 5000.0, 2)


def test_plot(folder):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = nm.Kernels(folder).plot("00s02", 2)
    assert [line.get_label() for line in ax.get_lines()[:3]] == list(nm.FIELDS)
    assert {t.get_text() for t in ax.texts} >= set(nm.FREE_BOUNDARIES)     # the discontinuities drawn and named
    matplotlib.pyplot.close("all")


def Y(l, j, x, y, z):
    """The package's real harmonic from scipy (Condon-Shortley removed): an independent reference."""
    r = np.sqrt(x * x + y * y + z * z)
    th = np.arccos(np.clip(z / np.where(r > 0, r, 1.0), -1, 1))
    ph = np.mod(np.arctan2(y, x), 2 * np.pi)
    m = (j + 1) // 2
    v = sph_harm_y(l, m, th, ph)
    out = v.real if j == 0 else math.sqrt(2) * (v.real if j % 2 else v.imag)
    return (-1.0) ** m * out


def test_forward_on_a_boundary_is_the_kernel_times_the_coefficient(folder):
    """A height equal to one orthonormal harmonic (1 km) gives K times the catalogue's factor for that row, 0 on the others."""
    data = nm.read_catalogue(folder)
    K = nm.Kernels(folder)
    M = pg.DirectSum({"cmb": pg.L2(pg.Sphere(1.0))}, weights={"cmb": 1 / (4 * np.pi * nm.R_CMB ** 2)})
    G = nm.forward(data, K, M, {"cmb": "cmb"})
    for j in data.t.tolist():
        out = G(M({"cmb": lambda x, y, z, j=j: Y(2, j, x, y, z)})).values
        factor = nm.CATALOGUE.factors(np.array([2]), np.array([j]))[0]
        expected = [K.boundary("00s02", "cmb", 2) * factor if i == j else 0.0 for i in data.t.tolist()]
        assert out == pytest.approx(expected, abs=1e-8)


def test_forward_on_a_field_is_one_percent_per_layer(folder):
    """vs = Y_2j on the mantle with radial factor 1: the layer's response to a fractional perturbation 1 is 100 K_l."""
    data = nm.read_catalogue(folder)
    K = nm.Kernels(folder)
    M = pg.DirectSum({"m": pg.L2(pg.Shell(nm.R_CMB, 6371.0), realisation="radial_harmonics")}, weights={"m": 1.0})
    G = nm.forward(data, K, M, {"m": "vs"}, rule="constant")
    j = 1
    out = G(M({"m": lambda x, y, z: Y(2, j, x, y, z)})).values
    factor = nm.CATALOGUE.factors(np.array([2]), np.array([j]))[0]
    # K_l = scale (s + 1) l on layer 5; the centre taper takes (eps / r)^2, about 2e-6 on this shell
    assert out[1] == pytest.approx(100 * (2 * 3 * 5) * factor, rel=1e-5)
    assert out[0] == pytest.approx(0.0, abs=1e-6) and out[2] == pytest.approx(0.0, abs=1e-6)


def test_forward_on_the_default_model_space(folder):
    data = nm.read_catalogue(folder)
    G = nm.forward(data, nm.Kernels(folder), nm.model_space(boundaries=("cmb", "icb")))
    assert G.structured_gram().matrix().shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(G.structured_gram().matrix()) > 0)


def test_forward_does_not_depend_on_how_the_space_is_split(folder):
    """The centre taper belongs to the field: a ball cut into two pieces has the same functionals, so the same Gram."""
    data = nm.read_catalogue(folder)
    K = nm.Kernels(folder)
    V = 4 * np.pi * 6371.0 ** 3 / 3
    whole = pg.DirectSum({"vp": pg.L2(pg.Ball(6371.0), realisation="radial_harmonics")}, weights={"vp": 1 / V})
    parts = pg.DirectSum({"a": pg.L2(pg.Ball(nm.R_CMB), realisation="radial_harmonics"),
                          "b": pg.L2(pg.Shell(nm.R_CMB, 6371.0), realisation="radial_harmonics")}, weights={"a": 1 / V, "b": 1 / V})
    A = nm.forward(data, K, whole, {"vp": "vp"}).structured_gram().matrix()
    B = nm.forward(data, K, parts, {"a": "vp", "b": "vp"}).structured_gram().matrix()
    assert B == pytest.approx(A, rel=1e-9)


def test_forward_refuses_a_row_without_a_kernel(folder):
    data = nm.read_catalogue(folder)
    K = nm.Kernels(folder)
    kernel(K.volume, "00s02", "vs", "0 1 1\n0 6 0\n")
    with pytest.raises(ValueError, match="no rows at s = 2"):
        nm.forward(data, K)


def test_plugin_is_registered():
    assert nm.PLUGIN.name == "normalmodes" and "forward" in nm.PLUGIN.objects
    assert "normalmodes" in pg.port.plugins.loaded()


# ---- optional, real data; nothing from it is written anywhere ----
DATA = os.environ.get("PLI_NORMAL_MODE_DATA")
real = pytest.mark.skipif(not DATA, reason="PLI_NORMAL_MODE_DATA is not set")
CMB, ICB, SURFACE = 3479.95776, 1221.49072, 6371.0
VOLUME = "kernels/kernels-all_PREM-layers_Adrian"
TOPO_ALL = "kernels/kernels-topo-all_PREM-layers_Adrian"


@pytest.fixture
def root():
    return Path(DATA)


@pytest.fixture
def table(root):
    return nm.read_prem_layers(root / VOLUME / "PREM_all-layers_radius")


@real
@pytest.mark.parametrize("dataset, modes, rows", [("arwen-paula", 186, 8210), ("arwen-paula-fixed-anomalies", 186, 8188),
                                                  ("SP12RTS", 143, 7324)])
def test_real_splitting_counts_and_values(root, dataset, modes, rows):
    d = nm.read_splitting_data(root / "observed", dataset, superseded=True)
    assert len(d.modes) == modes and len(d) == rows
    for i, mode in enumerate(d.modes):
        path = root / "observed" / f"data_{dataset}" / f"{mode}.new"
        assert int(path.read_text().split()[0]) == d.smax[i]
        expected = np.loadtxt(path, skiprows=1, ndmin=2)
        mine = d.mode == mode
        assert np.array_equal(np.column_stack([d.s[mine], d.t[mine], d.value[mine], d.sigma[mine]]), expected)


@real
def test_real_prem_tables(root, table):
    no_disc = nm.read_prem_layers(root / VOLUME / "PREM_all-layers_radius_no-disc")
    assert len(table) == 222 and len(no_disc) == 210
    assert np.array_equal(table.interval(55), (ICB, ICB)) and np.array_equal(table.interval(109), (CMB, CMB))
    assert table.interval(110)[0] == CMB
    assert table.at_radius(CMB).tolist() == [109, 110]
    # the legacy constant was wrong: layer 57 is not at the inner-core boundary
    assert 57 not in table.at_radius(ICB)
    assert table.layer[table.inverted].tolist() == [222] and table.r_inner[-1] == SURFACE


@real
def test_real_kernels_counts_and_values(root, table):
    folder = root / VOLUME
    for param in ("vp", "vs", "rho", "topo"):
        modes = nm.list_kernel_modes(folder, param)
        assert len(modes) == 190
        for mode in modes:
            k = nm.read_kernel(folder, mode, param, table)
            expected = np.loadtxt(folder / f"kern_{mode}_{param}_PREM-layers-all_PREM-iso.kern")
            assert np.array_equal(np.column_stack([k.s, k.layer, k.value]), expected.reshape(-1, 3))
            l = int(mode[3:])
            assert k.s.max() <= min(2 * l, 20)


@real
def test_real_topography_by_radius(root, table):
    folder = root / TOPO_ALL
    modes = nm.list_kernel_modes(folder, "topo-all")
    assert len(modes) == 186
    assert set(modes) == set(nm.read_mode_list(root / "observed" / "data_list_arwen-paula"))
    for mode in modes:
        new = nm.read_kernel(folder, mode, "topo-all", table)
        assert CMB in new.boundary_radii and ICB in new.boundary_radii and SURFACE in new.boundary_radii
        assert set(new.at_boundary(CMB).layer) == {109} and set(new.at_boundary(ICB).layer) == {55}
        # the older file numbers the same boundary differently; the radius finds it
        old = nm.read_kernel(root / VOLUME, mode, "topo", table)
        assert set(old.layer) == {110} and old.boundary_radii.tolist() == [CMB]
        assert len(old.at_boundary(CMB)) == len(new.at_boundary(CMB))


@real
def test_real_volume_kernels_and_the_two_tables(root, table):
    folder = root / VOLUME
    no_disc = nm.read_prem_layers(folder / "PREM_all-layers_radius_no-disc")
    mode = nm.list_kernel_modes(folder, "vs")[0]
    k = nm.read_kernel(folder, mode, "vs", table)
    # the volume kernels use the surface row (values zero there) and no zero-thickness row
    assert not table.zero_thickness[table.indices(k.layer)].any()
    assert (k.value[k.layer == 222] == 0).all()
    # the no-discontinuity table does not hold that row, so it cannot stand in for the full table
    with pytest.raises(ValueError, match="layer 222 is outside"):
        nm.read_kernel(folder, mode, "vs", no_disc)


@real
def test_real_synthetics_all_files_read(root):
    files = sorted((root / "synthetics").glob("raw_*.raw"))
    assert files
    for path in files:
        mode, model = path.stem[len("raw_"):].split("_", 1)
        d = nm.read_synthetics(root / "synthetics", mode, model)
        assert np.array_equal(np.column_stack([d.l, d.s, d.t, d.value]), np.loadtxt(path, ndmin=2))


@real
def test_real_catalogue_is_the_thesis_data(root):
    d = nm.read_catalogue(root)
    assert len(d.modes) == 186 and len(d) == 8002
    assert [int((d.s == s).sum()) for s in range(2, 15, 2)] == [930, 1521, 1924, 1734, 1239, 625, 29]
    assert d.smax_of("01s03") == 2
    assert set(nm.INNER_CORE_MODES) <= set(d.modes) and len(nm.INNER_CORE_MODES) == 38
    assert set(nm.INNER_CORE_EIGHT) <= set(nm.INNER_CORE_MODES)
    assert len(nm.read_catalogue(root, exclude=nm.INNER_CORE_MODES).modes) == 148
    # 01s03 has smax 2 once corrected, not 6: 22 rows fewer than the delivered SP12RTS
    assert len(nm.read_splitting_data(root / "observed", "SP12RTS-fixed-anomalies")) == 7324 - 22
    raw = nm.read_catalogue(root, crust=False)
    assert np.array_equal(raw.value - nm.synthetic_values(raw, root / "synthetics", "Crust5.1"), d.value)


@real
def test_real_corrected_sets_fit_the_synthetics_and_the_delivered_do_not(root):
    def ratios(dataset, modes):
        data = nm.read_splitting_data(root / "observed", dataset, modes=modes, superseded=True).select(degrees=[2])
        syn = nm.synthetic_values(data, root / "synthetics", "S20RTS-Crust5.1")
        return [np.sqrt((data.value[data.mode == m] ** 2).sum() / (syn[data.mode == m] ** 2).sum()) for m in data.modes]
    for dataset in ("arwen-paula", "SP12RTS"):
        assert max(ratios(dataset, nm.SUPERSEDED_MODES)) < 0.05
    for dataset in ("arwen-paula-fixed-anomalies", "SP12RTS-fixed-anomalies"):
        assert min(ratios(dataset, nm.SUPERSEDED_MODES)) > 0.3 and max(ratios(dataset, nm.SUPERSEDED_MODES)) < 3


@real
def test_real_corrected_differs_from_the_delivery_only_in_the_four_modes(root):
    for fixed, delivered in (("arwen-paula-fixed-anomalies", "arwen-paula"), ("SP12RTS-fixed-anomalies", "SP12RTS")):
        a = nm.read_splitting_data(root / "observed", fixed)
        b = nm.read_splitting_data(root / "observed", delivered, superseded=True)
        assert a.modes == b.modes
        for m in a.modes:
            same = np.array_equal(a.select(modes=m).value, b.select(modes=m).value)
            assert same == (m not in nm.SUPERSEDED_MODES)


@real
def test_real_kernel_function_uses_the_file_values(root):
    K = nm.Kernels(root)
    k = K.table("00s05", "vs")
    row = (k.s == 2) & (k.layer == 150)
    r = 0.5 * (k.r_inner[row] + k.r_outer[row])
    assert K.function("00s05", "vs", 2)(r)[0] == pytest.approx(100 * k.value[row][0] / k.thickness[row][0])
    assert K.boundary("00s05", "cmb", 2) == pytest.approx(K.table("00s05", "topo").at_boundary(nm.R_CMB).value[1], abs=1e-12)
    for name, radius in nm.BOUNDARIES.items():
        assert radius in K.table("00s05", "topo-all").boundary_radii


@real
def test_real_topography_kernels_explain_the_synthetics(root):
    """(cmb + base) - base = K_cmb(s) h_st for one number h_st across modes: the boundary kernels and the synthetics agree."""
    K = nm.Kernels(root)
    modes = [m for m in nm.read_mode_list(root / "observed" / "data_list_arwen-paula")][:40]
    d = nm.read_splitting_data(root / "observed", modes=modes).select(degrees=[2])
    model = "S20RTS-Crust5.1_cmb_tanaka-2010"
    diff = nm.synthetic_values(d, root / "synthetics", model) - nm.synthetic_values(d, root / "synthetics", "S20RTS-Crust5.1")
    kern = np.array([K.boundary(m, "cmb", 2) for m in d.mode.tolist()])
    for t in range(5):
        row = (d.t == t) & (kern != 0)
        assert np.all(diff[(d.t == t) & (kern == 0)] == 0)
        h = diff[row] / kern[row]
        assert np.std(h) <= 1e-3 * abs(np.mean(h))
