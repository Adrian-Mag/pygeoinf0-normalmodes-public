"""Models distributed as author tables: HMSL-S06 / HMSL-P06 (block tables), UU-P07 (cell tabulation) and the
lowermost-mantle Vp of MTT2022 (posterior-mean grid).

HMSL-S06, HMSL-P06 (Houser et al. 2008), hmsl.s.dat.gz / hmsl.p.dat.gz: rows "lon lat -depth value", 18 layers x
45 x 90 cells of 4 x 4 degrees (centres lon 2..358, lat -88..88), dV/V as a FRACTION. The model's equal-area
4-degree blocks are tabulated on these cells (near the poles neighbouring cells repeat one block value).
Piecewise constant laterally and in depth. The files give only the layer centres; the layer bounds used here are
22, 111, 200, 300, 400, 530, 660, 810, 960, 1110, 1310, 1510, 1710, 1910, 2110, 2310, 2510, 2710, 2886.7 km: the
ranges labelled in Houser et al. (2008) Fig. 8 and text (111-200, 300-400, 530-660, 660-810, 810-960, 1110-1310,
1310-1510, 1510-1710, 1710-1910, 1910-2110, 2110-2310, 2510-2710, 2710-2886), completed so that every file
centre is the midpoint of its layer (the top bound 22 km and the last bottom 2886.7 km follow from that).

UU-P07 (Amaru 2007), UU-P07_lon_lat_depth_dVp_cell_depth_midpoint.zip: rows "lon lat depth value", 29 depths x
360 x 720 values on a 0.5-degree cell-centre grid (lat 89.75..-89.75, lon -179.75..179.75; "latitudes are on
the sphere", info file), dVp/Vp in PERCENT at the cell-midpoint depths. The midpoints tile 0-2891 km as a chain
of layers (b_0 = 0, b_i+1 = 2 mid_i - b_i: 0, 10, 20, 35, 70, ..., 2560, 2740, 2891 km). Piecewise constant.
Reference model: not stated in the distributed files.

MTT2022 (Muir, Tanaka and Tkalcic 2022), tomodata.dat: rows "lat lon mean std" on a 2-degree node grid; the
posterior mean and standard deviation of d ln vp (FRACTION) in the lowermost 300 km of the mantle, radially
constant by construction, about 13.61 km/s (the ak135 average over that layer; data_explanation.txt and the
paper). The CMB is ak135's (radius 3479.5 km, depth 2891.5 km). Bilinear between nodes. The posterior summaries
of the spherical-harmonic coefficients (m_ds.dat: slowness, degrees 0-8) are shipped too but not read here.
"""
import numpy as np

from .. import _paths
from .._model import Model

HMSL_BOUNDS = np.array([22.0, 111.0, 200.0, 300.0, 400.0, 530.0, 660.0, 810.0, 960.0, 1110.0, 1310.0, 1510.0,
                        1710.0, 1910.0, 2110.0, 2310.0, 2510.0, 2710.0, 2886.7])


class _Cells(Model):
    """Piecewise-constant layers (bounds) x regular cells (lat edges ascending, lon edges ascending, 360 wide)."""

    def _setup(self, bounds, lat_edges, lon_edges, values):
        self.bounds = np.asarray(bounds, float)
        self.lat_edges, self.lon_edges = np.asarray(lat_edges, float), np.asarray(lon_edges, float)
        self.values = values                       # (n_layer, n_lat, n_lon), lat ascending, lon ascending

    def layer_index(self, depth):
        d = np.asarray(depth, float)
        k = np.searchsorted(self.bounds, d, side="right") - 1
        k = np.where(np.isclose(d, self.bounds[-1]), self.bounds.size - 2, k)
        return np.where((d >= self.bounds[0] - 1e-9) & (k >= 0) & (k < self.bounds.size - 1), k, -1)

    def _eval(self, field, lat, lon, depth):
        k = self.layer_index(depth)
        i = np.clip(np.searchsorted(self.lat_edges, lat, side="right") - 1, 0, self.lat_edges.size - 2)
        x = np.mod(lon - self.lon_edges[0], 360.0) + self.lon_edges[0]
        j = np.clip(np.searchsorted(self.lon_edges, x, side="right") - 1, 0, self.lon_edges.size - 2)
        out = np.full(lat.shape, np.nan)
        ok = k >= 0
        out[ok] = self.values[k[ok], i[ok], j[ok]]
        return out


class HMSL(_Cells):
    def __init__(self, model):
        super().__init__()
        self.name, self.source = model, "author"
        fn = {"HMSL-S06": "hmsl.s.dat.gz", "HMSL-P06": "hmsl.p.dat.gz"}[model]
        fld = {"HMSL-S06": "dvs", "HMSL-P06": "dvp"}[model]
        path = _paths.original(model, "raw", fn)
        a = _paths.numbers(_paths.read_text(path)).reshape(-1, 4)
        lon, lat, negd, v = a.T
        dep, la, lo = np.unique(-negd), np.unique(lat), np.unique(lon)
        if (dep.size, la.size, lo.size) != (18, 45, 90) or a.shape[0] != 18 * 45 * 90:
            raise ValueError(f"{fn}: unexpected table layout")
        val = np.full((18, 45, 90), np.nan)
        val[np.searchsorted(dep, -negd), np.searchsorted(la, lat), np.searchsorted(lo, lon)] = v
        mids = 0.5 * (HMSL_BOUNDS[:-1] + HMSL_BOUNDS[1:])
        if np.max(np.abs(mids - dep)) > 0.01:
            raise ValueError("HMSL layer bounds do not match the file's layer centres")
        self._setup(HMSL_BOUNDS, np.arange(-90.0, 90.1, 4.0), np.arange(0.0, 360.1, 4.0), val)
        self.layer_centres = dep
        self._fields[fld] = dict(
            unit="fraction", quantity=f"d ln {fld[1:]} (dV/V)",
            reference="none unique: rays in ak135, the data mean removed (Houser et al. 2008); layer mean of "
                      "the table not removed", coverage=(float(HMSL_BOUNDS[0]), float(HMSL_BOUNDS[-1])),
            native_depths=dep, description=f"{fn}: 18 layers x 4 x 4 degree cells, piecewise constant")


class UUP07(_Cells):
    name, source = "UU-P07", "author"

    def __init__(self):
        super().__init__()
        zp = _paths.original("UU-P07", "raw", "UU-P07_lon_lat_depth_dVp_cell_depth_midpoint.zip")
        info = _paths.read_text(_paths.original("UU-P07", "raw",
                                                "UU-P07_lon_lat_depth_dVp_cell_depth_midpoint_info.txt"))
        lines = info.splitlines()
        i = next(n for n, ln in enumerate(lines) if "LATITUDES ARE ON THE SPHERE" in ln)
        mids = _paths.numbers(" ".join(lines[i + 1:]))

        def build():
            raw = _paths.read_text(zp, "UU-P07_lon_lat_depth_%dVp_cell_depth_midpoint")
            a = _paths.numbers(raw)
            del raw
            if a.size != 29 * 360 * 720 * 4:
                raise ValueError("UU-P07 table has an unexpected number of values")
            a = a.reshape(29, 360, 720, 4)
            lon, lat, dep = a[0, 0, :, 0], a[0, :, 0, 1], a[:, 0, 0, 2]
            if not (np.allclose(lon, np.arange(-179.75, 180.0, 0.5)) and
                    np.allclose(lat, np.arange(89.75, -90.0, -0.5))):
                raise ValueError("UU-P07 grid is not the 0.5 degree cell-centre grid")
            if not (np.allclose(a[..., 0], lon) and np.allclose(a[..., 1], lat[None, :, None])
                    and np.allclose(a[..., 2], dep[:, None, None])):
                raise ValueError("UU-P07 rows are not a depth-major regular grid")
            return {"depth": dep, "value": a[:, ::-1, :, 3].astype(np.float32)}   # lat ascending
        z = _paths.cached_arrays("uu_p07", [zp], build)
        if not np.allclose(z["depth"], mids):
            raise ValueError("UU-P07 depths differ from the info file")
        b = [0.0]
        for m in mids:
            b.append(2.0 * m - b[-1])
        self._setup(b, np.arange(-90.0, 90.1, 0.5), np.arange(-180.0, 180.1, 0.5), z["value"])
        self._fields["dvp"] = dict(
            unit="percent", quantity="d ln vp (dVp/Vp)", reference="not stated in the distributed files",
            coverage=(0.0, float(b[-1])), native_depths=mids,
            description="zip member UU-P07_lon_lat_depth_%dVp_cell_depth_midpoint: 29 layers x 0.5 degree "
                        "cells, piecewise constant")


class MTT2022(Model):
    name, source = "MTT2022", "author"
    R_CMB_KM = 3479.5

    def __init__(self):
        super().__init__()
        a = _paths.numbers(_paths.read_text(_paths.original("MTT2022", "raw", "tomodata.dat"))).reshape(-1, 4)
        lat, lon = np.unique(a[:, 0]), np.unique(a[:, 1])
        v = np.full((2, lat.size, lon.size), np.nan)
        i, j = np.searchsorted(lat, a[:, 0]), np.searchsorted(lon, a[:, 1])
        v[0, i, j], v[1, i, j] = a[:, 2], a[:, 3]
        if np.isnan(v).any():
            raise ValueError("tomodata.dat is not a complete grid")
        self.glat, self.glon, self.v = lat, lon, v
        bot = 6371.0 - self.R_CMB_KM
        cov = (bot - 300.0, bot)
        ref = "13.61 km/s, the ak135 average vp of the lowermost 300 km (data_explanation.txt, Muir et al. 2022)"
        for f, q in (("dvp", "posterior mean of d ln vp"), ("dvp_std", "posterior standard deviation of d ln vp")):
            self._fields[f] = dict(unit="fraction", quantity=q + ", radially constant in the lowermost 300 km",
                                   reference=ref, coverage=cov, native_depths=None,
                                   description="tomodata.dat: 2-degree node grid; bilinear here")

    def _eval(self, field, lat, lon, depth):
        from .savani import _bilinear
        top, bot = self.coverage(field)
        k = np.where((depth >= top - 1e-9) & (depth <= bot + 1e-9), 0 if field == "dvp" else 1, -1)
        out = np.full(lat.shape, np.nan)
        ok = k >= 0
        out[ok] = _bilinear(self.glat, self.glon, self.v, k[ok], lat[ok], lon[ok])
        return out
