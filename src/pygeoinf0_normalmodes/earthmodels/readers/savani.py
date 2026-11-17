"""SAVANI (Auer et al. 2014), read from the authors' archive savani-<commit>.zip without unpacking it.

Three sources, all from the same archive:
  "sh"      shansyn/savani.dlnvs.60.m.ab and savani.xi.60.m.ab: per layer, real spherical harmonics to degree
            60 (Becker and Boschi 2002 format: n_layers; per layer: depth, lmax, then for l = 0..lmax,
            m = 0..l one line "A_lm B_lm"). Auer et al. 2014 eq. 6: dv = sum_l [A_l0 X_l0 + sqrt(2) sum_m X_lm
            (A_lm cos m phi + B_lm sin m phi)] with unit-norm real harmonics, i.e. (A, B) are the
            coefficients of the real orthonormal harmonics directly.
  "grid"    ascii_reg_grd_1deg/savani.{dlnvs,vsv,vsh,xi}.dat: the authors' expansion on a 1 x 1 degree node grid
            (lon 0..360, lat -90..90) at the 28 layer centres; bilinear between nodes here.
  "voxels"  ascii_var_grd/savani.{dlnvs,vsv,vsh,xi}.dat with grid.info: the native variable-resolution inversion
            grid, one value per voxel (lat_min..lat_max x lon_min..lon_max), piecewise constant.
Radially: 28 layers, constant in depth (grid.info gives each layer's bottom and centre; the first top is 0 km).
README: dlnvs is the Voigt average d ln(v_s) [%] w.r.t. PREM with v_s = sqrt((2 v_sv^2 + v_sh^2)/3); vsv, vsh
absolute speeds in m/s; xi the anisotropy parameter (README writes "xi=(v_sv/v_sv)^2"; read as
(v_sh/v_sv)^2, the usual definition; the values are about 1).
"""
import numpy as np

from .. import _paths, _sh
from .._model import Model, voigt_vs

ZIP = ("SAVANI", "raw", "savani-9348e270f28e5af9f9209c7eb8ccd9fcd552bb79.zip")
ROOT = "auerl-savani-9348e27/"
REF = "PREM (README: Voigt average d ln v_s w.r.t. PREM, v_s = sqrt((2 v_sv^2 + v_sh^2)/3))"


def _zip():
    return _paths.original(*ZIP)


def read_grid_info(text):
    """(layers (28, 4): id, n_voxels, bottom, centre; voxels (n, 5): id, lat_min, lat_max, lon_min, lon_max)."""
    lines = text.splitlines()
    i = 0
    while not lines[i].strip().startswith("Id"):
        i += 1
    i += 2
    layers = []
    while not lines[i].startswith("*"):
        p = lines[i].split()
        layers.append([float(x) for x in p[:4]])
        i += 1
    vox = [[float(x) for x in ln.split()] for ln in lines[i:] if len(ln.split()) == 5 and ln.split()[0].isdigit()]
    return np.array(layers), np.array(vox)


def read_ab(text):
    """{layer-centre depth: (A, B)} in the package convention (_sh), from a shansyn .ab file."""
    tok = text.split()
    pos, nl = 1, int(tok[0])
    out = {}
    for _ in range(nl):
        depth, lmax = float(tok[pos]), int(tok[pos + 1])
        pos += 2
        n = (lmax + 1) * (lmax + 2) // 2
        ab = np.array(tok[pos:pos + 2 * n], float).reshape(n, 2)
        pos += 2 * n
        A = np.zeros((lmax + 1, lmax + 1))
        B = np.zeros((lmax + 1, lmax + 1))
        k = 0
        for l in range(lmax + 1):
            for m in range(l + 1):
                f = 1.0 if m == 0 else np.sqrt(2.0)
                A[l, m], B[l, m] = f * ab[k, 0], f * ab[k, 1]
                k += 1
        out[depth] = (A, B)
    if pos != len(tok):
        raise ValueError(f"{len(tok) - pos} trailing tokens in .ab file")
    return out


class _Layered(Model):
    """Common layer handling: 28 constant layers, top of the first at 0 km."""

    def _setup_layers(self):
        layers, self._voxels = read_grid_info(_paths.read_text(_zip(), ROOT + "grid.info"))
        self.layer_bottom = layers[:, 2]
        self.layer_top = np.concatenate([[0.0], self.layer_bottom[:-1]])
        self.layer_centre = layers[:, 3]
        self.layer_nvox = layers[:, 1].astype(int)

    def layer_index(self, depth):
        """Index of the layer containing each depth (a depth on a boundary belongs to the layer below,
        except the bottom 2891 km); -1 outside."""
        d = np.asarray(depth, float)
        k = np.searchsorted(self.layer_bottom, d, side="right")
        k = np.where(np.isclose(d, self.layer_bottom[-1]), len(self.layer_bottom) - 1, k)
        return np.where((d >= 0.0) & (k < len(self.layer_bottom)), k, -1)

    def _info(self, unit, quantity, description):
        ref = REF if unit == "percent" else "none (absolute quantity)"
        return dict(unit=unit, quantity=quantity, reference=ref, coverage=(0.0, float(self.layer_bottom[-1])),
                    native_depths=self.layer_centre.copy(), description=description)


class SavaniSH(_Layered):
    name, source = "SAVANI", "sh"

    def __init__(self):
        super().__init__()
        self._setup_layers()
        self._ab = {}
        for f, fn, unit, q in (("dvs", "savani.dlnvs.60.m.ab", "percent", "d ln vs, Voigt average, degree 60"),
                               ("xi", "savani.xi.60.m.ab", "1", "xi = (vsh/vsv)^2 as expanded by the authors, "
                                                             "degree 60")):
            layers = read_ab(_paths.read_text(_zip(), ROOT + "shansyn/" + fn))
            keys = np.array(sorted(layers))
            if not np.allclose(keys, self.layer_centre):
                raise ValueError(f"{fn}: layer depths do not match grid.info")
            self._ab[f] = [layers[k] for k in keys]
            self._fields[f] = self._info(unit, q, f"zip member {ROOT}shansyn/{fn}; constant in each layer")

    def coefficients(self, field, depth):
        k = int(self.layer_index(depth))
        return None if k < 0 else self._ab[field][k]

    def _eval(self, field, lat, lon, depth):
        out = np.full(lat.shape, np.nan)
        k = self.layer_index(depth)
        for kk in np.unique(k[k >= 0]):
            sel = k == kk
            out[sel] = _sh.synth_points(*self._ab[field][kk], lat[sel], lon[sel])
        return out

    def _grid(self, field, depth, lat, lon):
        ab = self.coefficients(field, depth)
        return np.full((len(lat), len(lon)), np.nan) if ab is None else _sh.synth_grid(*ab, lat, lon)

    def _mean(self, field, depth):
        ab = self.coefficients(field, depth)
        return np.nan if ab is None else float(ab[0][0, 0] / np.sqrt(4.0 * np.pi))


_QUANT = {"dvs": ("dlnvs", "percent", "d ln vs, Voigt average"), "vsv": ("vsv", "m/s", "vsv (absolute)"),
          "vsh": ("vsh", "m/s", "vsh (absolute)"), "xi": ("xi", "1", "xi = (vsh/vsv)^2")}


def _derived_voigt(model):
    model._derived["vs_voigt"] = dict(inputs=["vsv", "vsh"], func=voigt_vs, unit="m/s",
                                      quantity="sqrt((2 vsv^2 + vsh^2)/3), computed on request",
                                      reference="none (absolute quantity)",
                                      description="Voigt average of the stored vsv and vsh")


class SavaniGrid(_Layered):
    """The authors' 1-degree node grid (bilinear between nodes, constant in each layer)."""
    name, source = "SAVANI", "grid"

    def __init__(self):
        super().__init__()
        self._setup_layers()
        self._data = {}
        for f, (stem, unit, q) in _QUANT.items():
            self._fields[f] = self._info(unit, q, f"zip member {ROOT}ascii_reg_grd_1deg/savani.{stem}.dat "
                                                  "(1 x 1 degree nodes; bilinear here)")
        _derived_voigt(self)

    def _load(self, field):
        if field not in self._data:
            stem = _QUANT[field][0]
            member = f"{ROOT}ascii_reg_grd_1deg/savani.{stem}.dat"

            def build():
                a = _paths.numbers(_paths.read_text(_zip(), member)).reshape(-1, 4)
                lons, lats, deps = np.unique(a[:, 0]), np.unique(a[:, 1]), np.unique(a[:, 2])
                v = np.full((deps.size, lats.size, lons.size), np.nan)
                v[np.searchsorted(deps, a[:, 2]), np.searchsorted(lats, a[:, 1]), np.searchsorted(lons, a[:, 0])] \
                    = a[:, 3]
                return {"lat": lats, "lon": lons, "depth": deps, "value": v}
            z = _paths.cached_arrays(f"savani_grid_{stem}", [_zip()], build)
            if not np.allclose(z["depth"], self.layer_centre):
                raise ValueError("SAVANI grid depths do not match grid.info layer centres")
            self._data[field] = z
        return self._data[field]

    def _eval(self, field, lat, lon, depth):
        z = self._load(field)
        out = np.full(lat.shape, np.nan)
        k = self.layer_index(depth)
        ok = k >= 0
        out[ok] = _bilinear(z["lat"], z["lon"], z["value"], k[ok], lat[ok], lon[ok])
        return out


def _bilinear(glat, glon, vals, k, lat, lon):
    """Bilinear interpolation on a node grid glat (ascending) x glon (ascending, spanning 360 degrees with
    or without a repeated meridian), vals (n_layer, n_lat, n_lon), layer index k per point."""
    span = glon[-1] - glon[0]
    if span < 359.999:          # no repeated meridian: append it
        glon = np.concatenate([glon, [glon[0] + 360.0]])
        vals = np.concatenate([vals, vals[:, :, :1]], axis=2)
    x = np.mod(lon - glon[0], 360.0) + glon[0]
    j = np.clip(np.searchsorted(glon, x, side="right") - 1, 0, glon.size - 2)
    tx = (x - glon[j]) / (glon[j + 1] - glon[j])
    y = np.clip(lat, glat[0], glat[-1])
    i = np.clip(np.searchsorted(glat, y, side="right") - 1, 0, glat.size - 2)
    ty = (y - glat[i]) / (glat[i + 1] - glat[i])
    return ((1 - ty) * (1 - tx) * vals[k, i, j] + (1 - ty) * tx * vals[k, i, j + 1]
            + ty * (1 - tx) * vals[k, i + 1, j] + ty * tx * vals[k, i + 1, j + 1])


class SavaniVoxels(_Layered):
    """The native variable-resolution voxel grid (grid.info + ascii_var_grd), piecewise constant."""
    name, source = "SAVANI", "voxels"

    def __init__(self):
        super().__init__()
        self._setup_layers()
        self._vals = {}
        for f, (stem, unit, q) in _QUANT.items():
            self._fields[f] = self._info(unit, q, f"zip member {ROOT}ascii_var_grd/savani.{stem}.dat on the "
                                                  "voxels of grid.info; piecewise constant")
        _derived_voigt(self)
        # Per layer: a rectilinear raster on the union of the voxel edges, holding the voxel number.
        self._raster = []
        off = 0
        for n in self.layer_nvox:
            v = self._voxels[off:off + n]
            la_e = np.unique(np.concatenate([v[:, 1], v[:, 2]]))
            lo_e = np.unique(np.concatenate([v[:, 3], v[:, 4]]))
            ids = np.full((la_e.size - 1, lo_e.size - 1), -1, dtype=np.int64)
            for q, row in enumerate(v):
                i0, i1 = np.searchsorted(la_e, row[1]), np.searchsorted(la_e, row[2])
                j0, j1 = np.searchsorted(lo_e, row[3]), np.searchsorted(lo_e, row[4])
                ids[i0:i1, j0:j1] = off + q
            if (ids < 0).any():
                raise ValueError("SAVANI voxels do not tile the sphere")
            self._raster.append((la_e, lo_e, ids))
            off += n

    def _load(self, field):
        if field not in self._vals:
            stem = _QUANT[field][0]
            a = _paths.numbers(_paths.read_text(_zip(), f"{ROOT}ascii_var_grd/savani.{stem}.dat")).reshape(-1, 4)
            if a.shape[0] != self._voxels.shape[0]:
                raise ValueError(f"savani.{stem}.dat has {a.shape[0]} rows for {self._voxels.shape[0]} voxels")
            self._vals[field] = a
        return self._vals[field]

    def voxel_index(self, lat, lon, depth):
        k = self.layer_index(depth)
        out = np.full(np.shape(lat), -1, dtype=np.int64)
        lo360 = np.mod(lon, 360.0)
        for kk in np.unique(k[k >= 0]):
            sel = k == kk
            la_e, lo_e, ids = self._raster[kk]
            i = np.clip(np.searchsorted(la_e, lat[sel], side="right") - 1, 0, ids.shape[0] - 1)
            j = np.clip(np.searchsorted(lo_e, lo360[sel], side="right") - 1, 0, ids.shape[1] - 1)
            out[sel] = ids[i, j]
        return out

    def _eval(self, field, lat, lon, depth):
        a = self._load(field)
        idx = self.voxel_index(lat, lon, depth)
        out = np.full(lat.shape, np.nan)
        ok = idx >= 0
        out[ok] = a[idx[ok], 3]
        return out


def load(source="sh"):
    return {"sh": SavaniSH, "grid": SavaniGrid, "voxels": SavaniVoxels}[source]()
