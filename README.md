# pygeoinf0-normalmodes

A pygeoinf0 plugin (plan 030): normal-mode splitting data and PREM-layer sensitivity kernels, as a clean catalogue, as
functions, and as forward maps.

```python
import pygeoinf0_normalmodes as nm

d = nm.read_catalogue(root)            # corrected, Crust5.1 subtracted, degree 0 removed: 186 modes, 8002 rows
K = nm.Kernels(root)
K.function("00s05", "vs", 2)           # a pg.Tabulated on [0, R]: the density against dr of a fractional perturbation
K.boundary("00s05", "cmb", 2)          # the response to 1 km of height; a name in BOUNDARIES or a radius, never a layer
K.plot("00s05", 2)                     # needs matplotlib
G = nm.forward(d, K)                   # pg.Functionals.separable on nm.model_space(), or on your own direct sum

# published Earth models, and moving them onto a harmonic basis
gm = nm.earthmodels
gm.list_models(); m = gm.load("S20RTS"); gm.plot_depth_slice(m, "dvs", 2800)
hm = nm.to_harmonics("S20RTS", "dvs", lmax=14, radii=K.midpoints(within=(3480, 6346.6)))
hm.in_basis("catalogue")               # also "orthonormal", "package" (pygeoinf0's), "4pi"; the same field
hm.evaluate(-20, 20, 3600)             # back to the sphere; hm.as_model() gives the slices, sections and plots
c = nm.predict({"vs": hm, "vp": hm.scaled(0.5), "rho": hm.scaled(0.3)}, d, K)
h = nm.recover_topography(K, root + "/synthetics", "S20RTS-Crust5.1_cmb_tanaka-2010", "S20RTS-Crust5.1")
M = nm.model_space(); m = M({"vs_m": hm.element(M.summand("vs_m")), ...}); G(m)   # give radii spanning the coverage
```

`root` is the data folder (`observed/`, `kernels/`, `synthetics/`), or set `PLI_NORMAL_MODE_DATA`. The files are never
changed.

**Seeing the data and the kernels.** Needs matplotlib (the `plot` extra), and no other package; the loaders import without it.

```python
nm.explore(root)                       # one window for all of it; or: python -m pygeoinf0_normalmodes, with --root DATA_FOLDER
                                       # if PLI_NORMAL_MODE_DATA is not set

small = d.select(family="00s*")        # a new catalogue; prints: 186 modes, 8002 rows -> 28 modes, 2007 rows (quiet=True: silent)
small = small.select(degree_range=(2, 8), snr_min=2)      # selections compose
d.select(n_range=(0, 5), l=[2, 3], t=0, sigma_max=1e-3, inner_core=False)
d.select(crust=False)                  # dataset, crust, degree_zero read the files again and repeat the selections so far

small.plot.coefficients()              # against degree; also snr, sigma, overview (heat map of |value| / sigma)
small.plot.mode("00s05", model="S20RTS")           # bars with error bars; across(s, t) follows one coefficient across modes
top, residual = d.plot.versus("S20RTS-Crust5.1_cmb_tanaka-2010", mode="00s05")     # mode=None: every row
K.plot("00s05", 2); K.plot_modes(["00s05", "01s03"], "vs", 2); K.plot_map("vs", 2); K.plot_boundaries("00s05", boundaries=["cmb", 5701.0])
```
Each plot takes `ax=` and returns its axes; with no `ax` it makes a new figure. On a catalogue with Crust5.1 subtracted,
`versus` and `model_values` subtract it from a model that holds it, and the label says so: S20RTS-Crust5.1 is S20RTS plus
Crust5.1 in the delivered files, so comparing it directly would count the crust twice.

The window's selections, plots and switches are these same calls. A plain script needs an interactive backend: `macosx` or
`TkAgg`, which come with matplotlib and Python on the Mac, and the call blocks until the window closes. A notebook needs
`%matplotlib tk` or `%matplotlib osx` before the call, which opens a separate window; the inline backend gives a picture whose
controls cannot respond, and `explore` says so. `ipympl` would allow `%matplotlib widget` in the notebook; it is not needed.

The coefficients are labelled "as in the files": the files do not state a unit. Kernel axes carry the thesis's reading, marked
as not confirmed by the mode code's author; boundary kernels and CMB-topography synthetics are marked as not independently tested.

**Shells, and the noise on D.**

```python
CUTS = ("icb", "cmb", "d670", "d400", "d220")          # names of BOUNDARIES; drop one by leaving it out
M, G, D = nm.problem(root, shells=CUTS)                # or from a selection: nm.problem(d.select(n=0), shells=CUTS)
M.names                                                # vp_centre_icb, ..., vs_cmb_d670, ..., rho_d220_surface, then the boundaries
pg.decompose(G, D.standard_gaussian())                          # D's standard Gaussian: coordinates N(0, diag(sigma^2))
pg.Covariance.glued(M, {("vs_d670_d400", "vs_d400_d220", "vs_d220_surface"): C, ...})      # a prior chooses which shells correlate
```
`nm.model_space(shells=CUTS)` is a flat direct sum of 17 shells plus five boundaries, with no vs in the outer core and the same
weights. The shells do not change G, only what a prior can say. The shell layout and `nm.problem` leave dpp out by default
(`PROBLEM_BOUNDARIES`; the author's word); `nm.model_space()` with no arguments keeps it, and `boundaries=` overrides either. `nm.problem(...)` also has `p.data`, `p.kernels` and `p.d`.

With shells cutting at both the ICB and the CMB, M is a `nm.ModeSpace`: a `pg.OnPieces` with these keys and the heights after it.
A target across shells is `M.functional("vp", bump.measure())`, with no hand cutting, and `M.split` and `M.glue` move a whole-ball
model in and out. They concern the volumetric fields; the heights are summands by name. Other cuts give a plain `pg.DirectSum`.
`M.model({"vs": hm, "cmb": h})` is the element for HarmonicModels, as `nm.predict` takes them. With pygeoinf0 plan 043 the restriction
to the pieces cuts the realisation exactly, in milliseconds.

A prior by degree needs no setup: `pg.decompose(G, D.standard_gaussian(), middle=C)`, and `G.adjoint((G @ G.adjoint).inv(y))` is the minimum-norm
solution through the blocks (pygeoinf0 plan 045; on an older pygeoinf0, `p.decompose(C)`). `p.plain()` is the same problem on the
unweighted D, and `p.noise` the noise as a measure. The Gram has a condition number near 1e15, so a minimum-norm fit depends on where
small directions are cut: state the cut, or use a prior.

A covariance on M is relative to M's weighted inner product. For a height with weight w = 1 / (4 pi r_b^2), coefficient variances
v_s in km^2 are `pg.Covariance.angular(pg.L2(pg.Sphere(1.0)), {s: w * v_s for s in degrees})`, not `{s: v_s}`.

D's noise assumes independent errors, as reported, and no error for the crustal correction. It changes the adjoint: G G* is
Gram W, W = diag(1 / sigma^2), so forming and inverting `G G* + I` densely on it is slow (33 s); `pg.decompose` is the route
(about 4 s or less on 8002 rows). `noise=False`, and `nm.forward` itself, keep the unweighted D. With `crust=False` the default
boundaries gain moho, ocean and surface; a free crustal boundary on data with the crust subtracted is refused.

The pairing on a height is `(1 / (4 pi r_b^2)) int_{S^2} h k dOmega`, an integral over the unit sphere: 1/r_b^2 times the area
average, not the area average. Sobolev spaces do not go through `forward`; use a Bessel covariance on these L2 spaces.

**Which coefficients.** `data_arwen-paula` and `data_SP12RTS` hold the same four files (`SUPERSEDED_MODES`: 00s03, 00s04,
00s05, 01s03) about 100-300 times too small, and are refused unless `superseded=True`. The default
`arwen-paula-fixed-anomalies` replaces those four; `SP12RTS-fixed-anomalies` reads the SP12RTS list from it. The separate
`fixed-data-files` delivery is not read: it lacks 22 modes.

**Choices `read_catalogue` makes, each with a switch.** `crust=False` keeps the Crust5.1 prediction in the data: use it if
Moho, ocean-floor or surface topography are free. `degree_zero=True` keeps degree 0. `exclude=nm.INNER_CORE_MODES` drops the
38 inner-core-sensitive modes, whose (2, 0) coefficient the isotropic relation does not describe.

**Units, as the thesis uses them, and checked.** A volumetric value is the response to 1 % over the layer, so the density is
100 K_l / dr_l. A boundary kernel keeps its scale and takes km. The basis is `CATALOGUE`. `forward` applies the centre taper
that keeps G bounded. With these rules and no other factor, S20RTS with vp = 0.5 vs and rho = 0.3 vs reproduces the delivered
S20RTS synthetics: slope 1.0002, residual 1e-3, over 17418 coefficients of degrees 2-14. The mode code's author has not
confirmed the unit input or those two scalings.

**Earth models.** `earthmodels` is the pack global-models-raw-2026-10-30 (commit ff7cb94): 19 published Vs and Vp models read
natively from their original files, with `CHECKS` there. The originals are not in this repository: set
`GLOBALMODELS_ORIGINALS`. Vp and Vs only; no density. Please cite each model's paper.

**Select boundaries by radius, never by layer number.** The same boundary has different numbers in different files, and
the tables hold rows that are not shells.

**No data here.** The data is confidential and this repository holds none of it: the tests write tiny invented files. Set
`PLI_NORMAL_MODE_DATA` to add the checks against the real files.

Status: 0.9.0; `hm.element` needs pygeoinf0 with `harmonic_element`. Released under the BSD 3-Clause licence (LICENSE);
data/coastlines.npz is third-party (THIRD_PARTY.md). No normal-mode data are included. Install pygeoinf0 from
github.com/Adrian-Mag/pygeoinf0-public first. Tests: `pytest tests`. Mentions of plans refer to the author's development
records, which are not included.
Checked against EXP-030's reproduction of the thesis (all 8002 rows): the data, sigma, every kernel and the Gram are identical. S20RTS as a
pg element through `forward` gives the synthetics to 3e-3 (slope 0.99994). Use `predict` for anything repeated; `hm.element` builds an element without fitting.
