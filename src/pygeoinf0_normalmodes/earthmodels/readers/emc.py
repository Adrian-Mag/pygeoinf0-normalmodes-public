"""EarthScope EMC netCDF grids (Trabant et al. 2012, doi:10.17611/DP/EMC.1): variables on (depth, latitude,
longitude) node grids, as regridded by the model's authors or the EMC.

Evaluation: bilinear in latitude and longitude between the grid nodes (longitude periodic; a repeated meridian
is used once), linear in depth between the grid depths (a pair of close depths such as 400/401 km encodes a
discontinuity and is kept as given). Latitudes beyond the outermost grid rows (grids stopping at +-88 degrees)
take the value of the outermost row. Fill values are NaN.
"""
import numpy as np

from .. import _paths
from .._model import Model, voigt_vp, voigt_vs

STW105 = "STW105"


class EMCGrid(Model):
    """fields: {field: dict(file=(model_dir, *path), var, unit, quantity, reference, description)}."""

    def __init__(self, name, source, fields, derived=None):
        super().__init__()
        import netCDF4
        self.name, self.source = name, source
        self._spec, self._axes, self._full = {}, {}, {}
        for f, d in fields.items():
            path = _paths.original(*d.pop("file"))
            var = d.pop("var")
            with netCDF4.Dataset(path) as ds:
                v = ds[var]
                if v.dimensions != ("depth", "latitude", "longitude"):
                    raise ValueError(f"{path}:{var} has dimensions {v.dimensions}")
                dep = np.asarray(ds["depth"][:], float)
                lat = np.asarray(ds["latitude"][:], float)
                lon = np.asarray(ds["longitude"][:], float)
                long_name = getattr(v, "long_name", "")
                units = getattr(v, "units", "")
            if np.any(np.diff(dep) < 0) or np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
                raise ValueError(f"{path}: axes are not ascending")
            self._spec[f] = (path, var)
            self._axes[f] = (dep, lat, lon)
            d.setdefault("description", "")
            d["description"] = (f"{path.split('/originals/')[-1]} variable {var} ({long_name}; units {units}); "
                                f"{lat.size} x {lon.size} nodes, {dep.size} depths").strip()
            d["coverage"] = (float(dep[0]), float(dep[-1]))
            d["native_depths"] = dep
            self._fields[f] = d
        for f, d in (derived or {}).items():
            self._derived[f] = d

    # data access: each variable is read whole on first use and kept (float32). Several EMC files store a
    # variable as one compressed chunk, so reading a single depth costs as much as reading all of them
    # (GLAD-M25: about 1 s and 356 MB per variable).
    def _slice(self, field, k):
        return self._load_full(field)[int(k)]

    def _load_full(self, field):
        if field not in self._full:
            import netCDF4
            path, var = self._spec[field]
            with netCDF4.Dataset(path) as ds:
                v = ds[var]
                if hasattr(v, "scale_factor") or hasattr(v, "add_offset"):
                    a = np.ma.filled(v[:].astype(np.float32), np.nan)
                else:                                    # avoid masked-array copies of large variables
                    v.set_auto_maskandscale(False)
                    a = np.asarray(v[:], dtype=np.float32)
                    fills = [getattr(v, n) for n in ("_FillValue", "missing_value") if hasattr(v, n)]
                    if not hasattr(v, "_FillValue"):
                        fills.append(netCDF4.default_fillvals[v.dtype.str[1:]])
                    for fill in fills:
                        a[a == np.float32(fill)] = np.nan
            self._full[field] = a
        return self._full[field]

    def _depth_weights(self, field, depth):
        dep = self._axes[field][0]
        d = np.asarray(depth, float)
        ok = (d >= dep[0] - 1e-9) & (d <= dep[-1] + 1e-9)
        k = np.clip(np.searchsorted(dep, d, side="right") - 1, 0, dep.size - 2)
        span = dep[k + 1] - dep[k]
        t = np.where(span > 0, (d - dep[k]) / np.where(span > 0, span, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        return k, t, ok

    def _lateral(self, field, lat, lon):
        """Indices and weights for bilinear interpolation (periodic longitude, clamped latitude)."""
        _, glat, glon = self._axes[field]
        dup = abs((glon[-1] - glon[0]) - 360.0) < 1e-3
        nu = glon.size - 1 if dup else glon.size              # distinct meridians
        ext = np.concatenate([glon[:nu], [glon[0] + 360.0]])  # wrap-around node = column 0
        x = np.mod(np.asarray(lon, float) - glon[0], 360.0) + glon[0]
        j0 = np.clip(np.searchsorted(ext, x, side="right") - 1, 0, nu - 1)
        tx = (x - ext[j0]) / (ext[j0 + 1] - ext[j0])
        j1 = (j0 + 1) % nu
        y = np.clip(np.asarray(lat, float), glat[0], glat[-1])
        i0 = np.clip(np.searchsorted(glat, y, side="right") - 1, 0, glat.size - 2)
        ty = (y - glat[i0]) / (glat[i0 + 1] - glat[i0])
        return i0, ty, j0, j1, tx

    @staticmethod
    def _bilin(a, i0, ty, j0, j1, tx):
        return ((1 - ty) * ((1 - tx) * a[..., i0, j0] + tx * a[..., i0, j1])
                + ty * ((1 - tx) * a[..., i0 + 1, j0] + tx * a[..., i0 + 1, j1]))

    def _eval(self, field, lat, lon, depth):
        k, t, ok = self._depth_weights(field, depth)
        i0, ty, j0, j1, tx = self._lateral(field, lat, lon)
        out = np.full(lat.shape, np.nan)
        need = np.unique(np.concatenate([k[ok], k[ok] + 1]))
        if need.size > 8:
            a = self._load_full(field)
            lo =((1 - ty) * ((1 - tx) * a[k, i0, j0] + tx * a[k, i0, j1])
                  + ty * ((1 - tx) * a[k, i0 + 1, j0] + tx * a[k, i0 + 1, j1]))
            hi = ((1 - ty) * ((1 - tx) * a[k + 1, i0, j0] + tx * a[k + 1, i0, j1])
                  + ty * ((1 - tx) * a[k + 1, i0 + 1, j0] + tx * a[k + 1, i0 + 1, j1]))
            out = ((1 - t) * lo + t * hi).astype(float)
            out[~ok] = np.nan
            return out
        for kk in np.unique(k[ok]):
            sel = ok & (k == kk)
            s0 = self._bilin(self._slice(field, kk), i0[sel], ty[sel], j0[sel], j1[sel], tx[sel])
            if np.all(t[sel] == 0):
                out[sel] = s0
                continue
            s1 = self._bilin(self._slice(field, kk + 1), i0[sel], ty[sel], j0[sel], j1[sel], tx[sel])
            out[sel] = (1 - t[sel]) * s0 + t[sel] * s1
        return out


def _f(file, var, unit, quantity, reference):
    return dict(file=file, var=var, unit=unit, quantity=quantity, reference=reference)


def _voigt_s(ref):
    return {"vs_voigt": dict(inputs=["vsv", "vsh"], func=voigt_vs, unit="km/s", reference=ref,
                             quantity="sqrt((2 vsv^2 + vsh^2)/3), computed on request",
                             description="Voigt average of the stored vsv and vsh")}


def _voigt_p(ref):
    return {"vp_voigt": dict(inputs=["vpv", "vph"], func=voigt_vp, unit="km/s", reference=ref,
                             quantity="sqrt((vpv^2 + 4 vph^2)/5), computed on request (eta neglected)",
                             description="Voigt average of the stored vpv and vph")}


def glad_m25(source="emc"):
    ref = "absolute speeds; the model's 1-D reference is STW105 (Lei et al. 2020)"
    fs = {"vsv": _f(("GLAD-M25", "raw", "glad-m25-vs.r0.0-n4c.nc"), "vsv", "km/s", "vsv (absolute)", ref),
          "vsh": _f(("GLAD-M25", "raw", "glad-m25-vs.r0.0-n4c.nc"), "vsh", "km/s", "vsh (absolute)", ref),
          "vpv": _f(("GLAD-M25", "raw", "glad-m25-vp.r0.0-n4c.nc"), "vpv", "km/s", "vpv (absolute)", ref),
          "vph": _f(("GLAD-M25", "raw", "glad-m25-vp.r0.0-n4c.nc"), "vph", "km/s", "vph (absolute)", ref)}
    return EMCGrid("GLAD-M25", "emc", fs, {**_voigt_s(ref), **_voigt_p(ref)})


def gypsum(source="emc"):
    d = "GyPSuM"
    rs = "average of the TNA/SNA models (netCDF long_name)"
    rp = "1-second isotropic equivalent PREM (netCDF long_name)"
    fs = {"dvs": _f((d, "raw", "GYPSUM_percent.r0.0-n4c.nc"), "dvs", "percent", "d ln vs", rs),
          "dvp": _f((d, "raw", "GYPSUM_percent.r0.0-n4c.nc"), "dvp", "percent", "d ln vp", rp),
          "vs": _f((d, "raw", "GYPSUMS_kmps.r0.0-n4c.nc"), "vs", "km/s", "vs (absolute)", "absolute speed"),
          "vp": _f((d, "raw", "GYPSUMP_kmps.r0.0-n4c.nc"), "vp", "km/s", "vp (absolute)", "absolute speed")}
    return EMCGrid("GyPSuM", "emc", fs)


def s362ani(source="emc"):
    d = "S362ANI"
    r = "STW105 (netCDF long_names: perturbations w/t STW105)"
    fs = {"dvs": _f((d, "raw", "S362ANI_percent.r0.0-n4c.nc"), "dvs", "percent",
                    "d ln vs as given by the EMC file (definition of the isotropic average not stated)", r),
          "dvsv": _f((d, "raw", "S362ANI_percent.r0.0-n4c.nc"), "dvsv", "percent", "d ln vsv", r),
          "dvsh": _f((d, "raw", "S362ANI_percent.r0.0-n4c.nc"), "dvsh", "percent", "d ln vsh", r),
          "vs": _f((d, "raw", "S362ANI_kmps.r0.0-n4c.nc"), "vs", "km/s", "isotropic vs (absolute)", "absolute"),
          "vsv": _f((d, "raw", "S362ANI_kmps.r0.0-n4c.nc"), "vsv", "km/s", "vsv (absolute)", "absolute"),
          "vsh": _f((d, "raw", "S362ANI_kmps.r0.0-n4c.nc"), "vsh", "km/s", "vsh (absolute)", "absolute")}
    return EMCGrid("S362ANI", "emc", fs, _voigt_s("absolute"))


def saw24b16(source="emc"):
    d = "SAW24B16"
    fs = {"dvsh": _f((d, "raw", "SAW24B16_percent.r0.0-n4c.nc"), "dvsh", "percent",
                     "d ln vsh (SAW24B16 is a VSH model)", "anisotropic PREM vsh (netCDF long_name: w/t PREM)"),
          "vsh": _f((d, "raw", "SAW24B16_kmps.r0.0-n4c.nc"), "vsh", "km/s", "vsh (absolute)", "absolute")}
    return EMCGrid("SAW24B16", "emc", fs)


def tx2011(source="emc"):
    fs = {"dvs": _f(("TX2011", "raw", "TX2011_percent.r0.0-n4c.nc"), "dvs", "percent", "d ln vs",
                    "TX2011_ref (EMC file summary; the 1-D model itself is not in the file)")}
    return EMCGrid("TX2011", "emc", fs)


def hmsl_emc(model):
    fld = "dvs" if model == "HMSL-S06" else "dvp"
    fs = {fld: _f((model, "raw", f"{model}_percent.r0.0-n4c.nc"), fld, "percent",
                  f"d ln {fld[1:]} (EMC regridding of the author table; layer mean removed by the EMC)",
                  "none unique: ak135 for rays, data mean removed (EMC documentation)")}
    return EMCGrid(model, "emc", fs)


def sglobe_emc():
    d = "SGLOBE-rani"
    r = "PREM (EMC file summary)"
    fs = {"dvs": _f((d, "raw", "SGLOBE-rani-voigt_percent.r0.0-n4c.nc"), "dvs", "percent",
                    "d ln vs, Voigt average", r),
          "dxi": _f((d, "raw", "SGLOBE-rani-xi_percent.r0.0-n4c.nc"), "dxi", "percent",
                    "radial anisotropy in percent", r),
          "xi": _f((d, "raw", "SGLOBE-rani-true-xi.r0.0-n4c.nc"), "xi", "1", "xi = (vsh/vsv)^2", "absolute")}
    return EMCGrid("SGLOBE-rani", "emc", fs)


def semucb_emc():
    d = "SEMUCB-WM1"
    fs = {"vs": _f((d, "raw", "semucb-2014-ucb-vs-viz-only.r0.1-n4c.nc"), "vs", "km/s",
                   "Voigt average vs (absolute); EMC visualisation-only grid", "absolute"),
          "xi": _f((d, "raw", "semucb-2014-ucb-xi-viz-only.r0.1-n4c.nc"), "xi", "1",
                   "xi = (vsh/vsv)^2; EMC visualisation-only grid", "absolute")}
    return EMCGrid("SEMUCB-WM1", "emc", fs)
