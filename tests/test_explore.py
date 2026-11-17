"""Selections, data plots and the window, headless on tiny invented files; a real-data smoke test only behind
PLI_NORMAL_MODE_DATA. No real data is committed."""
import os
import subprocess
import sys

import numpy as np
import pytest

import pygeoinf0_normalmodes as nm

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.backend_bases  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

TABLE = """1 0.0 1221.49072
2 1221.49072 1221.49072
3 1221.49072 3479.95776
4 3479.95776 3479.95776
5 3479.95776 6371.0
6 6371.0 0.0
"""
MODES = {"00s02": 2, "00s03": 2, "01s04": 4, "13s01": 2, "02t03": 2}      # 13s01 is an inner-core mode


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def root(tmp_path):
    """Every number invented: value = 1 + s + t, sigma = 0.5, plus a degree 0 row; Crust5.1 is 0.25 everywhere."""
    base = tmp_path / "data"
    write(base / "observed" / "data_list_arwen-paula", "".join(m + "\n" for m in MODES))
    write(base / "observed" / "data_list_SP12RTS", "00s02\n01s04\n")
    for mode, smax in MODES.items():
        scale = 10.0 if mode == "01s04" else 1.0
        rows = [(0, 0)] + [(s, t) for s in range(2, smax + 1, 2) for t in range(2 * s + 1)]
        write(base / "observed" / "data_arwen-paula-fixed-anomalies" / f"{mode}.new",
              f"{smax}\n" + "".join(f"{s} {t} {scale * (1 + s + t)} {scale * 0.5}\n" for s, t in rows))
        for model, value in (("Crust5.1", 0.25), ("S20RTS", 2.0), ("S20RTS-Crust5.1", 2.25), ("S20RTS-Crust5.1_cmb_x", 2.5)):
            write(base / "synthetics" / f"raw_{mode}_{model}.raw",
                  "".join(f"1 {s} {t} {value}\n" for s, t in rows))
        for param, folder in (("vp", "kernels/kernels-all_PREM-layers_Adrian"), ("vs", "kernels/kernels-all_PREM-layers_Adrian"),
                              ("rho", "kernels/kernels-all_PREM-layers_Adrian")):
            text = "".join(f"{s} {layer} {0.1 * (s + 1) * layer}\n" for s in range(0, smax + 1, 2) for layer in (1, 3, 5))
            write(base / folder / f"kern_{mode}_{param}_PREM-layers-all_PREM-iso.kern", text)
        write(base / "kernels/kernels-topo-all_PREM-layers_Adrian" / f"kern_{mode}_topo-all_PREM-layers-all_PREM-iso.kern",
              "".join(f"{s} {layer} {0.5 * (s + 1)}\n" for s in range(0, smax + 1, 2) for layer in (2, 4)))
    write(base / "kernels/kernels-all_PREM-layers_Adrian/PREM_all-layers_radius", TABLE)
    return base


@pytest.fixture(autouse=True)
def closed():
    yield
    plt.close("all")


@pytest.fixture
def data(root):
    return nm.read_catalogue(root)


# selection -------------------------------------------------------------------------------------------------------------
def test_each_selection_returns_a_new_catalogue_and_prints_what_it_did(data, capsys):
    out = data.select(n=0)
    assert out is not data and out.modes == ("00s02", "00s03") and len(data.modes) == 5
    assert capsys.readouterr().out.strip() == f"{data.size_text} -> {out.size_text}"
    data.select(n=0, quiet=True)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("kwargs, modes", [
    (dict(family="00s*"), ("00s02", "00s03")), (dict(kind="t"), ("02t03",)), (dict(n=[0, 1]), ("00s02", "00s03", "01s04")),
    (dict(n_range=(1, 2)), ("01s04", "02t03")), (dict(n_range=(2, None)), ("13s01", "02t03")), (dict(l=1), ("13s01",)),
    (dict(l_range=(3, 4)), ("00s03", "01s04", "02t03")), (dict(inner_core=True), ("13s01",)),
    (dict(inner_core=False), ("00s02", "00s03", "01s04", "02t03")), (dict(degrees=[4]), ("01s04",)),
    (dict(degree_range=(4, None)), ("01s04",)), (dict(modes="00s03"), ("00s03",)), (dict(exclude=["00s03"]), ("00s02", "01s04", "13s01", "02t03"))])
def test_mode_choices(data, kwargs, modes):
    assert data.select(quiet=True, **kwargs).modes == modes


def test_row_choices(data):
    assert set(data.select(t=[0], quiet=True).t.tolist()) == {0}
    assert np.all(data.select(sigma_max=1.0, quiet=True).sigma <= 1.0)
    assert data.select(sigma_min=1.0, quiet=True).modes == ("01s04",)
    picked = data.select(snr_min=10.0, quiet=True)
    assert np.all(np.abs(picked.value) / picked.sigma >= 10.0) and 0 < len(picked) < len(data)
    assert np.all(np.abs(data.select(snr_max=7.0, quiet=True).value) <= 7.0 * data.select(snr_max=7.0, quiet=True).sigma)


def test_selections_compose_and_order_does_not_matter(data):
    a = data.select(n=[0, 1], quiet=True).select(degrees=[2], quiet=True).select(t=[0, 1], quiet=True)
    b = data.select(t=[0, 1], degrees=[2], n=[0, 1], quiet=True)
    assert a.mode.tolist() == b.mode.tolist() and a.value.tolist() == b.value.tolist()
    assert "n=[0, 1]" in a.label and "degrees=[2]" in a.label and a.size_text in a.label


def test_select_refuses_a_kind_and_keeps_the_old_arguments(data):
    with pytest.raises(ValueError, match="kind is"):
        data.select(kind="q")
    assert data.select(["00s02"], ["00s03"], [2], quiet=True).modes == ("00s02",)


def test_an_empty_selection_is_allowed_and_a_plot_of_it_is_refused(data):
    empty = data.select(n=99, quiet=True)
    assert empty.modes == () and len(empty) == 0
    with pytest.raises(ValueError, match="no rows"):
        empty.plot.coefficients()


def test_switches_read_again_and_repeat_the_selections(data, root):
    narrowed = data.select(n=0, quiet=True)
    assert np.all(narrowed.select(crust=False, quiet=True).value == narrowed.value + 0.25)
    assert narrowed.select(crust=False, quiet=True).modes == narrowed.modes
    assert 0 in narrowed.select(degree_zero=True, quiet=True).s and 0 not in narrowed.s
    assert narrowed.select(dataset="SP12RTS-fixed-anomalies", quiet=True).modes == ("00s02",)
    assert narrowed.select(crust=False, quiet=True).select(crust=True, quiet=True).value.tolist() == narrowed.value.tolist()


def test_switches_need_a_catalogue_from_read_catalogue(root):
    d = nm.read_splitting_data(root / "observed")
    with pytest.raises(ValueError, match="read_catalogue"):
        d.select(crust=False)
    with pytest.raises(ValueError, match="synthetics_dir"):
        d.model_values("S20RTS")


def test_superseded_stays_refused_through_a_switch(data):
    with pytest.raises(ValueError, match="uncorrected"):
        data.select(dataset="arwen-paula")


# what a comparison compares -------------------------------------------------------------------------------------------
def test_the_model_is_the_same_quantity_as_the_data(data, root):
    assert np.all(data.model_values("S20RTS-Crust5.1") == 2.0)     # 2.25 minus Crust5.1, as the data has it subtracted
    assert np.all(data.model_values("S20RTS") == 2.0)
    assert data.model_label("S20RTS-Crust5.1") == "S20RTS-Crust5.1 minus Crust5.1"
    raw = nm.read_catalogue(root, crust=False)
    assert np.all(raw.model_values("S20RTS-Crust5.1") == 2.25) and raw.model_label("S20RTS-Crust5.1") == "S20RTS-Crust5.1"
    assert data.models() == ["Crust5.1", "S20RTS", "S20RTS-Crust5.1", "S20RTS-Crust5.1_cmb_x"]


def test_a_rewritten_file_is_read_again(data, root):
    data.model_values("S20RTS")
    path = root / "synthetics" / "raw_00s02_S20RTS.raw"
    rows = [line.split() for line in path.read_text().splitlines()]
    path.write_text("".join(f"{a} {b} {c} 3.5\n" for a, b, c, _ in rows))
    os.utime(path, ns=(10 ** 18, 10 ** 18))
    assert 3.5 in data.model_values("S20RTS").tolist()


# plots ----------------------------------------------------------------------------------------------------------------
PLOTS = [("coefficients", (), "as in the files"), ("snr", (), "dimensionless"), ("sigma", (), "rows"),
         ("mode", ("00s02",), "as in the files"), ("across", (2, 0), "as in the files"), ("overview", (), "mode")]


@pytest.mark.parametrize("name, args, ylabel", PLOTS)
def test_each_data_plot_uses_the_given_axes_or_makes_its_own(data, name, args, ylabel):
    _, given = plt.subplots()
    assert getattr(data.plot, name)(*args, ax=given) is given
    assert ylabel in given.get_ylabel() and given.get_xlabel() and data.dataset in given.get_title()
    before = plt.get_fignums()
    own = getattr(data.plot, name)(*args)
    assert own is not given and len(plt.get_fignums()) == len(before) + 1


def test_coefficients_plot_the_rows_of_the_selection(data):
    narrowed = data.select(n=0, quiet=True)
    ax = narrowed.plot.coefficients()
    assert ax.containers[0].lines[0].get_ydata().tolist() == narrowed.value.tolist()


def test_overview_is_the_mean_of_value_over_sigma_and_blank_without_rows(data):
    ax = data.select(modes=["00s02", "01s04"], quiet=True).plot.overview()
    grid = ax.images[0].get_array()
    assert grid[0, 0] == pytest.approx(np.mean([(3 + t - 0.25) / 0.5 for t in range(5)]))
    assert np.ma.is_masked(grid[0, 1]) or np.isnan(grid[0, 1])


def test_versus_has_a_residual_panel_and_states_the_subtraction(data):
    top, bottom = data.plot.versus("S20RTS-Crust5.1", mode="00s02")
    assert "minus Crust5.1" in top.get_title() and "sigma" in bottom.get_ylabel()
    d = data.select(modes=["00s02"], quiet=True)
    assert bottom.get_lines()[0].get_ydata().tolist() == pytest.approx(((d.value - 2.0) / d.sigma).tolist())
    all_rows, _ = data.plot.versus("S20RTS")
    assert "all rows" in all_rows.get_title()


def test_a_topography_model_says_its_closure_is_untested(data):
    top, _ = data.plot.versus("S20RTS-Crust5.1_cmb_x", mode="00s02")
    assert "not independently tested" in top.get_title()
    assert "not independently tested" in data.plot.mode("00s02", model="S20RTS-Crust5.1_cmb_x").get_title()


def test_versus_divides_a_given_axes(data):
    _, given = plt.subplots()
    top, bottom = data.plot.versus("S20RTS", ax=given)
    assert top is given and bottom.get_position().y1 <= top.get_position().y0


def test_kernel_plot_with_no_axes_makes_a_new_figure(root, data):
    """The bug found in use: K.plot drew on whatever axes were current."""
    K = nm.Kernels(root)
    earlier = data.plot.coefficients()
    lines = len(earlier.get_lines())
    ax = K.plot("00s02", 2)
    assert ax is not earlier and ax.figure is not earlier.figure and len(earlier.get_lines()) == lines


def test_kernel_plots_state_units_and_what_is_unconfirmed(root):
    K = nm.Kernels(root)
    for ax in (K.plot("00s02", 2), K.plot_modes(["00s02", "00s03"], "vs", 2), K.plot_map("vs", 2), K.plot_boundaries("00s02")):
        assert "not confirmed" in ax.get_title()
    assert "per km" in K.plot("00s02", 2).get_xlabel()
    assert "not independently tested" in K.plot_boundaries("00s02").get_title() and "1 km" in K.plot_boundaries("00s02").get_ylabel()


def test_kernel_map_values_and_a_mode_without_the_degree(root):
    K = nm.Kernels(root)
    ax = K.plot_map("vs", 4, normalise=False)
    assert "1 without" not in ax.get_title() and "4 without rows" in ax.get_title()
    values = np.asarray(ax.collections[0].get_array()).ravel()
    # value = 0.1 (s + 1) layer on layers 1, 3, 5: 01s04 at s = 4 is the only mode, its largest density on the thinnest of them
    thickness = np.array([1221.49072, 3479.95776 - 1221.49072, 6371.0 - 3479.95776])
    assert np.max(np.abs(values)) == pytest.approx(np.max(100 * 0.1 * 5 * np.array([1, 3, 5]) / thickness))
    with pytest.raises(ValueError, match="no mode"):
        K.plot_map("vs", 8)


def test_boundary_plot_by_name_and_radius(root):
    K = nm.Kernels(root)
    ax = K.plot_boundaries("00s02", boundaries=["cmb", 1221.49072])
    assert len(ax.get_xticks()) == 2
    assert [t.get_text().split("\n")[0] for t in ax.get_xticklabels()] == ["cmb", "1221 km"]
    with pytest.raises(ValueError, match="none of these"):
        K.plot_boundaries("00s02", boundaries=["d670"])


# the window -----------------------------------------------------------------------------------------------------------
def test_parse_numbers():
    from pygeoinf0_normalmodes.explorer import parse_numbers
    assert parse_numbers("") == (None, None) and parse_numbers("2,4") == ([2, 4], None) and parse_numbers("3") == ([3], None)
    assert parse_numbers("0-5") == (None, (0, 5)) and parse_numbers("3-") == (None, (3, None)) and parse_numbers("-5") == (None, (None, 5))
    with pytest.raises(ValueError):
        parse_numbers("a")


@pytest.fixture
def window(root):
    return nm.explore(root, show=False)


def plotted(ex):
    return [ax for ax in ex.fig.axes if ax not in ex.controls]


def said(ax):
    return " ".join(t.get_text() for t in ax.texts)


@pytest.mark.parametrize("view", ["coefficients", "value / sigma", "sigma", "one mode", "(s, t) across modes", "overview", "data vs synthetics",
                                  "volumetric kernels", "modes overlaid", "mode x layer map", "boundary kernels"])
def test_every_view_draws_without_a_message(window, view):
    window.set(view=view)
    for ax in plotted(window):
        assert "cannot draw" not in said(ax)
    assert plotted(window)[0].get_title()


def test_the_views_are_reached_by_the_controls(window):
    w = window.widgets
    labels = [t.get_text() for t in w["view"].labels]
    w["view"].set_active(labels.index("overview"))
    assert window.state["view"] == "overview" and plotted(window)[0].images
    w["mode_index"].set_val(2)
    assert window.state["mode"] == window.modes[2]
    w["next"].ax.figure.canvas.callbacks.process("key_press_event", matplotlib.backend_bases.KeyEvent("key_press_event", window.fig.canvas, "right"))
    assert window.state["mode"] == window.modes[3]
    assert w["mode"].text == window.state["mode"]
    w["s"].set_val(4)
    w["t"].set_val(8)
    assert (window.state["s"], window.state["t"]) == (4, 8)
    w["s"].set_val(2)
    assert window.state["t"] == 4     # t stays within 0..2s


def test_selection_controls_are_shared_by_all_views(window):
    w = window.widgets
    w["family"].begin_typing()
    w["family"].set_val("00s*")
    w["family"].stop_typing()
    assert window.modes == ("00s02", "00s03") and "->" in window.message
    assert window.state["mode"] in window.modes
    window.set(view="mode x layer map", s=2)
    assert "2 modes" in plotted(window)[0].get_title()
    window.set(family="", n="1-2")
    assert window.modes == ("01s04", "02t03")
    window.set(n="", inner_core="inner-core only")
    assert window.modes == ("13s01",)
    window.set(inner_core="all modes", snr_min=10.0, view="coefficients")
    assert len(window.selected) < len(window._catalogue())
    assert window.selected.size_text in window.status.get_text()


def test_switches_change_the_data(window):
    before = window.selected.value.copy()
    window.widgets["switches"].set_active(0)
    assert window.state["crust"] is False and np.all(window.selected.value == before + 0.25)
    window.widgets["switches"].set_active(1)
    assert 0 in window.selected.s
    window.set(dataset="SP12RTS-fixed-anomalies")
    assert window.modes == ("00s02", "01s04")


def test_what_cannot_be_drawn_is_said_and_the_window_stays(window):
    window.set(n="not numbers")
    assert "cannot draw" in said(plotted(window)[0])
    window.set(n="")
    assert "cannot draw" not in said(plotted(window)[0])
    window.set(n="99", view="coefficients")
    assert "no rows" in said(plotted(window)[0])
    window.set(n="", view="volumetric kernels", s=4, mode="00s02")
    assert "no rows at s = 4" in said(plotted(window)[0])
    window.set(view="boundary kernels", boundary="nowhere")
    assert "cannot draw" in said(plotted(window)[0])


def test_redrawing_does_not_accumulate_axes(window):
    window.set(view="overview")
    count = len(window.fig.axes)
    for view in ("data vs synthetics", "overview", "mode x layer map", "overview"):
        window.set(view=view)
    assert len(window.fig.axes) == count


def test_overlays_and_saving(window, tmp_path):
    window.set(view="one mode", show_model=True, model="S20RTS-Crust5.1_cmb_x")
    assert "cmb_x" in " ".join(t.get_text() for t in plotted(window)[0].get_legend().get_texts())
    window.set(view="modes overlaid", overlay="00s02, 00s03", field="rho")
    assert [t.get_text() for t in plotted(window)[0].get_legend().get_texts()] == ["00s02", "00s03"]
    window.save(tmp_path / "w.png")
    assert (tmp_path / "w.png").stat().st_size > 0


def test_a_folder_without_kernels_still_shows_the_data(root):
    import shutil
    shutil.rmtree(root / "kernels")
    ex = nm.explore(root, show=False, view="volumetric kernels")
    assert "no kernels" in said(plotted(ex)[0])
    ex.set(view="coefficients")
    assert "cannot draw" not in said(plotted(ex)[0])


def test_the_wrong_folder_is_refused_before_a_window_opens(tmp_path):
    with pytest.raises(ValueError, match="is not the data folder"):
        nm.explore(tmp_path, show=False)
    assert plt.get_fignums() == []
    r = run("-m", "pygeoinf0_normalmodes", "--root", str(tmp_path))
    assert r.returncode != 0 and "is not the data folder" in r.stderr


def test_a_missing_dataset_says_so_in_the_window(root):
    (root / "observed" / "data_list_arwen-paula").unlink()
    ex = nm.explore(root, show=False)
    assert "cannot draw" in said(plotted(ex)[0])


def test_unknown_settings_and_views_are_refused(window, root):
    with pytest.raises(ValueError, match="unknown settings"):
        window.set(colour="red")
    with pytest.raises(ValueError, match="unknown view"):
        nm.explore(root, show=False, view="other")


def test_inline_backend_says_how_to_get_a_window(root, capsys):
    nm.explore(root)      # Agg here: a picture
    assert "%matplotlib tk" in capsys.readouterr().out


def run(*args, env=None):
    src = os.path.dirname(os.path.dirname(nm.__file__))
    environ = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": os.pathsep.join([src] + sys.path), **(env or {})}
    environ.pop("PLI_NORMAL_MODE_DATA", None)
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=environ)


def test_the_command_line(root, tmp_path):
    r = run("-m", "pygeoinf0_normalmodes", "--root", str(root), "--list")
    assert r.returncode == 0 and "5 modes" in r.stdout and "S20RTS" in r.stdout
    image = tmp_path / "v.png"
    r = run("-m", "pygeoinf0_normalmodes", "--root", str(root), "--view", "overview", "--save", str(image), "--no-show")
    assert r.returncode == 0, r.stderr
    assert image.stat().st_size > 0
    assert run("-m", "pygeoinf0_normalmodes").returncode != 0


def test_the_loaders_import_without_matplotlib(root):
    code = ("import sys; sys.modules['matplotlib'] = None; sys.modules['matplotlib.pyplot'] = None\n"
            "import pygeoinf0_normalmodes as nm\n"
            f"d = nm.read_catalogue({str(root)!r}).select(n=0)\n"
            "try:\n    d.plot.coefficients()\nexcept ImportError as e:\n    print('refused:', e)\n"
            "try:\n    nm.explore(" + repr(str(root)) + ", show=False)\nexcept ImportError as e:\n    print('refused:', e)\n")
    r = run("-c", code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("refused: plots need matplotlib") == 2


# the real data, only behind the variable --------------------------------------------------------------------------------
DATA = os.environ.get("PLI_NORMAL_MODE_DATA")
real = pytest.mark.skipif(not DATA, reason="PLI_NORMAL_MODE_DATA is not set")


@real
def test_real_every_view_draws():
    ex = nm.explore(DATA, show=False)
    for view in ("coefficients", "value / sigma", "sigma", "one mode", "(s, t) across modes", "overview", "data vs synthetics",
                 "volumetric kernels", "modes overlaid", "mode x layer map", "boundary kernels"):
        ex.set(view=view, s=2, t=0)
        assert "cannot draw" not in said(plotted(ex)[0]), view
    ex.set(dataset="SP12RTS-fixed-anomalies", crust=False, degree_zero=True, view="data vs synthetics")
    assert "cannot draw" not in said(plotted(ex)[0])


@real
def test_real_selection_and_the_two_ways_to_compare():
    d = nm.read_catalogue(DATA)
    assert d.size_text == "186 modes, 8002 rows"
    assert d.select(family="00s*", quiet=True).modes == tuple(m for m in d.modes if m.startswith("00s"))
    assert len(d.select(inner_core=False, quiet=True).modes) == 186 - 38
    assert np.allclose(d.model_values("S20RTS-Crust5.1"), nm.synthetic_values(d, os.path.join(DATA, "synthetics"), "S20RTS"), rtol=1e-12, atol=1e-12)
