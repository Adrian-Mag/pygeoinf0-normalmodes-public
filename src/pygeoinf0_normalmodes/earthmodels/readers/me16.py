"""ME16 (Moulik and Ekstrom 2016) in its distributed box/pixel form.

The inversion parameterisation (file ME16: 362 spherical splines x 16 radial splines per parameter, evaluated
by the authors' Fortran in PROGRAMS.tar.gz) is distributed together with its evaluation on 115 depth boxes
(25 km thick; 24.4-50, 50-75, ..., 2875-2891 km) x 64800 pixels of 1 x 1 degree:
  "nc4"   ME16.BOX25km_PIX1X1.avni.nc4 (default): variables vs, vp, vsv, vsh, ... (depth, latitude, longitude),
          pixel centres lat -89.5..89.5, lon 0.5..359.5; box limits in the start_depths / end_depths attributes.
  "epix"  epix.tar.gz: the same boxes as text files epix/<field>/ME16.<field>.<k>.epix (header lines
          "#KEY:value", then lat, lon, pixel width, value), 7 significant digits; the nc4 holds the same values
          rounded to 0.001 %. Read from the archive directly (one pass, cached).
Values: percent about STW105 (#REFMODEL, #UNIT:PERCENT; #IFREMAV:0, i.e. the layer mean is NOT removed),
piecewise constant over each pixel and box. vs and vp are Voigt averages (nc4 descriptions).
"""
import re
import tarfile

import numpy as np

from .. import _paths
from .._model import Model

NC4 = ("ME16", "ME16.BOX25km_PIX1X1.avni.nc4")
EPIX = ("ME16", "epix.tar.gz")
FIELDS = {"vs": "d ln vs, Voigt average", "vp": "d ln vp, Voigt average", "vsv": "d ln vsv", "vsh": "d ln vsh"}
REF = "STW105 (#REFMODEL; per-box reference speed in the nc4 refvalue attribute)"


class ME16(Model):
    name = "ME16"

    def __init__(self, source="nc4"):
        super().__init__()
        import netCDF4
        self.source = source
        self._nc_path = _paths.original(*NC4)
        with netCDF4.Dataset(self._nc_path) as ds:
            v = ds["vs"]
            self.top = np.asarray(v.getncattr("start_depths"), float)
            self.bottom = np.asarray(v.getncattr("end_depths"), float)
            self.centre = np.asarray(ds["depth"][:], float)
            self.refvalue = {f: np.asarray(ds[f].getncattr("refvalue"), float) for f in FIELDS}
        self._data = {}
        what = ("ME16.BOX25km_PIX1X1.avni.nc4" if source == "nc4" else "epix.tar.gz (epix/<field>/*.epix)")
        for f, q in FIELDS.items():
            self._fields[f] = dict(unit="percent", quantity=q, reference=REF,
                                   coverage=(float(self.top[0]), float(self.bottom[-1])),
                                   native_depths=self.centre.copy(),
                                   description=f"{what}: 115 boxes x 1 x 1 degree pixels, piecewise constant")

    def _load(self, field):
        if field in self._data:
            return self._data[field]
        if self.source == "nc4":
            import netCDF4
            with netCDF4.Dataset(self._nc_path) as ds:
                lat = np.asarray(ds["latitude"][:], float)
                lon = np.asarray(ds["longitude"][:], float)
                val = np.ma.filled(ds[field][:].astype(float), np.nan)
        else:
            z = _paths.cached_arrays(f"me16_epix_{field}", [_paths.original(*EPIX)],
                                     lambda: _read_epix_field(_paths.original(*EPIX), field))
            lat, lon, val = z["lat"], z["lon"], z["value"]
        if not (np.allclose(lat, np.arange(-89.5, 90.0)) and np.allclose(lon, np.arange(0.5, 360.0))):
            raise ValueError("ME16: unexpected pixel grid")
        self._data[field] = val
        return val

    def box_index(self, depth):
        d = np.asarray(depth, float)
        k = np.searchsorted(self.bottom, d, side="left")      # box whose bottom is the first >= d
        ok = (d >= self.top[0] - 1e-9) & (d <= self.bottom[-1] + 1e-9)
        return np.where(ok, np.clip(k, 0, self.bottom.size - 1), -1)

    def _eval(self, field, lat, lon, depth):
        val = self._load(field)
        k = self.box_index(depth)
        i = np.clip(np.floor(lat + 90.0).astype(int), 0, 179)
        j = np.floor(np.mod(lon, 360.0)).astype(int) % 360
        out = np.full(lat.shape, np.nan)
        ok = k >= 0
        out[ok] = val[k[ok], i[ok], j[ok]]
        return out


def _read_epix_field(tar_path, field):
    """{lat, lon, value (115, 180, 360)} from the epix files of one field, read from the tar.gz in one pass."""
    pat = re.compile(rf"epix/{re.escape(field)}/ME16\.{re.escape(field)}\.(\d+)\.epix$")
    layers = {}
    with tarfile.open(tar_path) as t:
        for m in t:
            g = pat.match(m.name)
            if not g:
                continue
            text = t.extractfile(m).read().decode("latin-1")
            body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
            a = _paths.numbers(body).reshape(-1, 4)
            layers[int(g.group(1))] = a
    if sorted(layers) != list(range(1, 116)):
        raise ValueError(f"epix: found layers {sorted(layers)[:5]}... for {field}, expected 1..115")
    a0 = layers[1]
    lat, lon = np.unique(a0[:, 0]), np.unique(a0[:, 1])
    val = np.full((115, lat.size, lon.size), np.nan)
    for k in range(1, 116):
        a = layers[k]
        val[k - 1, np.searchsorted(lat, a[:, 0]), np.searchsorted(lon, a[:, 1])] = a[:, 3]
    return {"lat": lat, "lon": lon, "value": val}


def load(source="nc4"):
    if source not in ("nc4", "epix"):
        raise ValueError("ME16 sources: nc4, epix")
    return ME16(source)
