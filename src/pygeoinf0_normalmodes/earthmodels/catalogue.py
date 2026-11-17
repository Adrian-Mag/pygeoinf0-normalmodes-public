"""What models there are, what each contains, and how to read it.

CATALOGUE[name] describes one model: citation, doi, the fields as stored in each source, the native
parametrisation, lateral and radial resolution, depth coverage, the reference 1-D model as its authors state it,
and the shipped original files (sizes are measured from originals/ when printed).
"""
import os

from . import _paths

CATALOGUE = {
    "DETOX-P2": dict(
        dir="DETOX-P2-P3",
        citation="Hosseini, K., Sigloch, K., Tsekhmistrenko, M., Zaheri, A., Nissen-Meyer, T. and Igel, H. "
                 "(2020). Global mantle structure from multifrequency tomography using P, PP and P-diffracted "
                 "waves. Geophys. J. Int. 220, 96-141. Data: Zenodo, doi:10.5281/zenodo.3993276.",
        doi="10.1093/gji/ggz394",
        sources={"grid": "author netCDF grids, 59 depths (0-2900 km, 50 km apart) x 0.5 deg nodes: dvp (percent)",
                 "mesh": "native tetrahedral mesh (396501 vertices, 2338710 tetrahedra), linear in each "
                         "tetrahedron: dvp (fraction)"},
        parametrisation="irregular tetrahedral mesh, vertex values, linear inside each tetrahedron; author "
                        "regular grids made from it with GMT surface",
        lateral="mesh adapted to the ray coverage (tens to hundreds of km); grids 0.5 deg",
        radial="mesh vertices at irregular radii; grids every 50 km",
        coverage="mesh: surface to the CMB (it continues through the core); grids 0-2900 km",
        reference="IASP91; degree 0 is the static offset from IASP91 (Hosseini et al. 2020)",
        notes="The archive DETOX_MODELS.zip holds DETOX-P1, -P2 and -P3; P1 is not listed here but can be "
              "read with globalmodels.readers.detox.load('DETOX-P1', source)."),
    "DETOX-P3": None,  # filled below (same archive as DETOX-P2)
    "GLAD-M25": dict(
        dir="GLAD-M25",
        citation="Lei, W., Ruan, Y., Bozdag, E., Peter, D., Lefebvre, M., Komatitsch, D., Tromp, J., Hill, J., "
                 "Podhorszki, N. and Pugmire, D. (2020). Global adjoint tomography - model GLAD-M25. Geophys. J. "
                 "Int. 223, 1-21.",
        doi="10.1093/gji/ggaa253",
        sources={"emc": "EMC netCDF grids: vsv, vsh, vpv, vph (km/s, absolute); derived on request vs_voigt, "
                        "vp_voigt"},
        parametrisation="spectral-element mesh model (adjoint tomography), distributed by the EMC regridded to "
                        "0.5 deg x 342 depths (-5 to 2891 km)",
        lateral="native: spectral-element mesh; grid 0.5 deg", radial="grid depths 1-25 km apart",
        coverage="-5 to 2891 km (grid)",
        reference="absolute speeds; 1-D starting/reference model STW105; radial anisotropy in the upper mantle",
        notes="Only the EMC regridding was available; the native mesh model was not obtained."),
    "GyPSuM": dict(
        dir="GyPSuM",
        citation="Simmons, N. A., Forte, A. M., Boschi, L. and Grand, S. P. (2010). GyPSuM: a joint tomographic "
                 "model of mantle density and seismic wave speeds. J. Geophys. Res. 115.",
        doi="10.1029/2010JB007631",
        sources={"emc": "EMC netCDF grids: dvs, dvp (percent) and vs, vp (km/s, absolute)"},
        parametrisation="native parameterisation: see Simmons et al. (2010); distributed by the EMC on 1 deg "
                        "nodes",
        lateral="grid 1 deg", radial="grid depths 25-100 km apart",
        coverage="0-2900 km (grid)",
        reference="dvs: average of the TNA/SNA models; dvp: 1-second isotropic equivalent PREM (netCDF "
                  "long_names)",
        notes="Density not obtained (out of scope here)."),
    "HMSL-P06": dict(
        dir="HMSL-P06",
        citation="Houser, C., Masters, G., Shearer, P. and Laske, G. (2008). Shear and compressional velocity "
                 "models of the mantle from cluster analysis of long-period waveforms. Geophys. J. Int. 174, "
                 "195-212.",
        doi="10.1111/j.1365-246X.2008.03763.x",
        sources={"author": "author block table hmsl.p.dat.gz: dvp (fraction), 18 layers x 4 deg cells",
                 "emc": "EMC netCDF regridding: dvp (percent, layer mean removed by the EMC)"},
        parametrisation="equal-area blocks 4 deg wide at the equator x 18 layers, piecewise constant",
        lateral="4 deg (~450 km)", radial="18 layers, 89-200 km thick",
        coverage="22-2886.7 km (layer bounds; centres 66.5-2798.35 km)",
        reference="no unique 1-D reference: rays in ak135, data mean removed",
        notes="Above 660 km the P model is tied to HMSL-S06 by a constraint (Houser et al. 2008)."),
    "HMSL-S06": None,
    "ME16": dict(
        dir="ME16",
        citation="Moulik, P. and Ekstrom, G. (2016). The relationships between large-scale variations in shear "
                 "velocity, density, and compressional velocity in the Earth's mantle. J. Geophys. Res. Solid "
                 "Earth 121, 2737-2771.",
        doi=None,
        sources={"nc4": "ME16.BOX25km_PIX1X1.avni.nc4: vs, vp, vsv, vsh (percent about STW105)",
                 "epix": "epix.tar.gz: the same boxes and pixels as text (7 significant digits)"},
        parametrisation="362 spherical splines x 16 radial splines per parameter (file ME16); distributed "
                        "evaluated on 115 boxes of 25 km x 1 x 1 deg pixels (read here)",
        lateral="362 spherical splines (knot spacing ~11 deg); pixels 1 deg", radial="16 radial splines; boxes 25 km",
        coverage="24.4-2891 km",
        reference="STW105 (anisotropic; vs, vp Voigt averages)",
        notes="Density (rho) and the other nc4 variables are in the file but not exposed as fields."),
    "MTT2022": dict(
        dir="MTT2022",
        citation="Muir, J. B., Tanaka, S. and Tkalcic, H. (2022). Long-wavelength topography and multiscale "
                 "velocity heterogeneity at the core-mantle boundary. Geophys. Res. Lett. 49. Data: Zenodo, "
                 "doi:10.5281/zenodo.6525050; code doi:10.5281/zenodo.6619144.",
        doi="10.1029/2022GL099943",
        sources={"author": "tomodata.dat: dvp, dvp_std (fraction), posterior mean and standard deviation"},
        parametrisation="real spherical harmonics, degrees 0-8, of slowness in one layer (the lowermost 300 km), "
                        "radially constant; jointly with CMB topography",
        lateral="degree 8 (the grid tomodata.dat is 2 deg)", radial="one layer, 300 km",
        coverage="2591.5-2891.5 km (the lowermost 300 km above ak135's CMB)",
        reference="13.61 km/s, the ak135 average vp of the lowermost 300 km",
        notes="A lowermost-mantle model only. The CMB topography and the coefficient summaries (m_ds.dat, "
              "m_dr.dat) are shipped too."),
    "P12": dict(
        dir="P12",
        citation="Ritsema's degree-12 P model as distributed with SPECFEM3D_GLOBE (model_s20rts.f90: 'P12 degree "
                 "12 P model from Ritsema'); compare Ritsema, J. and van Heijst, H. J. (2002), Geophys. J. Int. "
                 "149, 482-489.",
        doi=None,
        sources={"sph": "P12.sph: dvp (fraction)"},
        parametrisation="real spherical harmonics to degree 12 x 21 radial cubic splines (Ritsema format)",
        lateral="degree 12", radial="21 splines", coverage="24.4-2891 km (Moho radius 6346.619 km to the CMB)",
        reference="PREM (as used by SPECFEM3D_GLOBE); not stated in the file",
        notes=""),
    "S20RTS": dict(
        dir="S20RTS",
        citation="Ritsema, J., van Heijst, H. J. and Woodhouse, J. H. (1999). Complex shear wave velocity "
                 "structure imaged beneath Africa and Iceland. Science 286, 1925-1928.",
        doi=None,
        sources={"sph": "S20RTS.sph: dvs (fraction)"},
        parametrisation="real spherical harmonics to degree 20 x 21 radial cubic splines (Ritsema format)",
        lateral="degree 20", radial="21 splines", coverage="24.4-2891 km",
        reference="PREM (transversely isotropic, as in SPECFEM3D_GLOBE)", notes=""),
    "S362ANI": dict(
        dir="S362ANI",
        citation="Kustowski, B., Ekstrom, G. and Dziewonski, A. M. (2008). Anisotropic shear-wave velocity "
                 "structure of the Earth's mantle: a global model. J. Geophys. Res. 113.",
        doi="10.1029/2007JB005169",
        sources={"emc": "EMC netCDF grids: dvs, dvsv, dvsh (percent about STW105) and vs, vsv, vsh (km/s)"},
        parametrisation="362 spherical splines x radial splines in the upper and lower mantle (native SPECFEM "
                        "file S362ANI, shipped, not read here); EMC grids 2 deg x 25 depths",
        lateral="362 spherical splines (knot spacing ~11 deg); grid 2 deg", radial="grid depths 25-250 km apart",
        coverage="25-2890 km (grid)",
        reference="STW105 (netCDF long_names)",
        notes="The native SPECFEM files (S362ANI, Fortran readers) are shipped; three upstream reader files are "
              "zero bytes as distributed."),
    "S40RTS": dict(
        dir="S40RTS",
        citation="Ritsema, J., Deuss, A., van Heijst, H. J. and Woodhouse, J. H. (2011). S40RTS: a degree-40 "
                 "shear-velocity model for the mantle from new Rayleigh wave dispersion, teleseismic traveltime "
                 "and normal-mode splitting function measurements. Geophys. J. Int. 184, 1223-1236.",
        doi="10.1111/j.1365-246X.2010.04884.x",
        sources={"sph": "S40RTS.sph: dvs (fraction)"},
        parametrisation="real spherical harmonics to degree 40 x 21 radial cubic splines (Ritsema format)",
        lateral="degree 40", radial="21 splines", coverage="24.4-2891 km",
        reference="PREM (isotropic shear velocity w.r.t. PREM)", notes=""),
    "SAVANI": dict(
        dir="SAVANI",
        citation="Auer, L., Boschi, L., Becker, T. W., Nissen-Meyer, T. and Giardini, D. (2014). Savani: a "
                 "variable resolution whole-mantle model of anisotropic shear velocity variations based on "
                 "multiple data sets. J. Geophys. Res. Solid Earth 119, 3006-3034.",
        doi="10.1002/2013JB010773",
        sources={"sh": "shansyn .ab expansions to degree 60: dvs (percent), xi",
                 "grid": "author 1 x 1 deg grid: dvs (percent), vsv, vsh (m/s), xi; derived vs_voigt",
                 "voxels": "native variable-resolution voxels (grid.info): dvs, vsv, vsh, xi; derived vs_voigt"},
        parametrisation="variable-resolution voxels (1.25-5 deg in latitude) x 28 layers, piecewise constant; the authors "
                        "also give a 1 deg grid and a degree-60 expansion",
        lateral="voxels 1.25-5 deg in latitude (wider in longitude near the poles)", radial="28 layers, 25-150 km thick", coverage="0-2891 km",
        reference="PREM; dvs is the Voigt average sqrt((2vsv^2+vsh^2)/3) about PREM's", notes=""),
    "SAW24B16": dict(
        dir="SAW24B16",
        citation="Megnin, C. and Romanowicz, B. (2000). The three-dimensional shear velocity structure of the "
                 "mantle from the inversion of body, surface and higher-mode waveforms. Geophys. J. Int. 143, "
                 "709-728.",
        doi="10.1046/j.1365-246X.2000.00298.x",
        sources={"emc": "EMC netCDF grids: dvsh (percent), vsh (km/s)"},
        parametrisation="spherical harmonics to degree 24 x 16 radial splines (native file not obtained); "
                        "EMC grid 2 deg x 32 depths",
        lateral="degree 24; grid 2 deg", radial="16 splines; grid depths 50-100 km apart",
        coverage="50-2850 km (grid)", reference="anisotropic PREM (vsh)",
        notes="A VSH model: there is no isotropic vs."),
    "SEMUCB-WM1": dict(
        dir="SEMUCB-WM1",
        citation="French, S. W. and Romanowicz, B. A. (2014). Whole-mantle radially anisotropic shear velocity "
                 "structure from spectral-element waveform tomography. Geophys. J. Int. 199, 1303-1327.",
        doi="10.1093/gji/ggu334",
        sources={"native": "A3d model (authors' archive): dvs (Voigt d ln vs, fraction), dxi (d ln xi); "
                           "derived vs_abs (km/s)",
                 "emc": "EMC visualisation-only grids: vs (km/s), xi"},
        parametrisation="20 radial cubic B-splines x spherical splines (10242 knots, level 6, for vs; 642 for "
                        "xi); homogenised crust above a 30-60 km Moho",
        lateral="spherical splines ~2 deg (level 6)", radial="20 B-splines, knots 3480-6341 km radius",
        coverage="30-2891 km (nan inside the homogenised crust)",
        reference="model.ref (the authors' 1-D parameterisation reference, not PREM)",
        notes="The EMC grids are for visualisation only (the authors' warning)."),
    "SGLOBE-rani": dict(
        dir="SGLOBE-rani",
        citation="Chang, S.-J., Ferreira, A. M. G., Ritsema, J., van Heijst, H. J. and Woodhouse, J. H. (2015). "
                 "Joint inversion for global isotropic and radially anisotropic mantle structure including "
                 "crustal thickness perturbations. J. Geophys. Res. Solid Earth 120, 4278-4300.",
        doi="10.1002/2014JB011824",
        sources={"native": "SPECFEM files: dvs (Voigt, dvs_iso.dat), dvsv, dvsh (fraction); derived dvs_linear",
                 "emc": "EMC netCDF grids: dvs (percent), dxi (percent), xi"},
        parametrisation="real spherical harmonics to degree 35 x 21 radial cubic splines (Ritsema format)",
        lateral="degree 35", radial="21 splines", coverage="24.4-2891 km (native); 25-2800 km (EMC grid)",
        reference="PREM (transversely isotropic)",
        notes="SPECFEM scales vp (0.5 dvs) and density (0.4 dvs) from vs: no independent vp."),
    "SP12RTS": dict(
        dir="SP12RTS",
        citation="Koelemeijer, P., Ritsema, J., Deuss, A. and van Heijst, H.-J. (2016). SP12RTS: a degree-12 "
                 "model of shear- and compressional-wave velocity for Earth's mantle. Geophys. J. Int. 204, "
                 "1024-1039.",
        doi=None,
        sources={"sph": "SP12RTS..ES.sph: dvs; SP12RTS..EP.sph: dvp (fractions)"},
        parametrisation="real spherical harmonics to degree 12 x 21 radial cubic splines (Ritsema format)",
        lateral="degree 12", radial="21 splines", coverage="24.4-2891 km",
        reference="anisotropic PREM (isotropic velocity perturbations)",
        notes="The authors' archive also holds S20RTS, S40RTS, the ..EC.sph files and models with other "
              "damping and data weights (Models/); read any of them with readers.ritsema.read_sph."),
    "TX2011": dict(
        dir="TX2011",
        citation="Grand, S. P. (2002). Mantle shear-wave tomography and the fate of subducted slabs. Phil. "
                 "Trans. R. Soc. Lond. A 360, 2475-2491 (the paper the EMC file cites for TX2011).",
        doi="10.1098/rsta.2002.1077",
        sources={"emc": "EMC netCDF grid: dvs (percent)"},
        parametrisation="native parameterisation: see Grand (2002); EMC grid 2 deg x 100 depths",
        lateral="grid 2 deg", radial="grid depths 25-50 km apart", coverage="0-2890 km (grid)",
        reference="TX2011_ref (not in the file)",
        notes="The EMC labels the model TX2011 but cites Grand (2002); the version is as the EMC states."),
    "UU-P07": dict(
        dir="UU-P07",
        citation="Amaru, M. L. (2007). Global travel time tomography with 3-D reference models. PhD thesis, "
                 "Utrecht University (Geologica Ultraiectina 274). File from the Atlas of the Underworld; see "
                 "also Hall, R. and Spakman, W. (2015), Tectonophysics 658, 14-45.",
        doi=None,
        sources={"author": "author tabulation: dvp (percent), 29 layers x 0.5 deg cells"},
        parametrisation="irregular cells sampled by the authors on a 0.5 deg grid at 29 cell-midpoint depths",
        lateral="0.5 deg sampling of irregular cells", radial="29 layers, 10-180 km thick",
        coverage="0-2891 km", reference="not stated in the distributed files", notes=""),
}
CATALOGUE["DETOX-P3"] = dict(CATALOGUE["DETOX-P2"])
CATALOGUE["HMSL-S06"] = dict(CATALOGUE["HMSL-P06"],
                             sources={"author": "author block table hmsl.s.dat.gz: dvs (fraction), 18 layers x 4 "
                                                "deg cells",
                                      "emc": "EMC netCDF regridding: dvs (percent, layer mean removed by the EMC)"},
                             dir="HMSL-S06", notes="")


def model_names():
    return sorted(CATALOGUE)


def default_source(name):
    return next(iter(CATALOGUE[name]["sources"]))


def files(name):
    """[(relative path, bytes)] of the shipped originals of a model (DETOX-P2 and -P3 share one archive)."""
    d = os.path.join(_paths.originals_dir(), CATALOGUE[name]["dir"])
    out = []
    for root, _, fs in os.walk(d):
        for f in sorted(fs):
            p = os.path.join(root, f)
            out.append((os.path.relpath(p, _paths.originals_dir()), os.path.getsize(p)))
    return sorted(out)


def _size(n):
    for u in ("B", "kB", "MB", "GB"):
        if n < 1000 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1000.0


def list_models(out=print):
    """Per model: name, sources (default first), size of the originals, coverage, reference, the fields of
    each source, citation and doi. gm.info(name) prints the full record."""
    for n in model_names():
        c = CATALOGUE[n]
        try:
            size = _size(sum(b for _, b in files(n)))
        except OSError:
            size = "missing"
        out(f"{n}  [{size}; {c['coverage']}]")
        out(f"    reference: {c['reference']}")
        for s, d in c["sources"].items():
            out(f"    {s:<7} {d}")
        out(f"    {c['citation'][:110]}{'...' if len(c['citation']) > 110 else ''}"
            + (f" doi:{c['doi']}" if c["doi"] else ""))


def info(name, out=print):
    """Everything the catalogue knows about one model, and the fields of each source."""
    c = CATALOGUE[name]
    out(f"{name}")
    out(f"  citation:        {c['citation']}" + (f" doi:{c['doi']}" if c["doi"] else ""))
    out(f"  parametrisation: {c['parametrisation']}")
    out(f"  lateral:         {c['lateral']}")
    out(f"  radial:          {c['radial']}")
    out(f"  depth coverage:  {c['coverage']}")
    out(f"  reference:       {c['reference']}")
    out("  sources:")
    for s, d in c["sources"].items():
        out(f"    {s:<7} {d}")
    if c["notes"]:
        out(f"  notes:           {c['notes']}")
    fl = files(name)
    out(f"  original files ({_size(sum(b for _, b in fl))}, in {_paths.originals_dir()}):")
    for p, b in fl:
        out(f"    {_size(b):>9}  {p}")
