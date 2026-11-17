"""The common interface of every model reader.

A Model holds one model read from one source (for example SAVANI from its spherical-harmonic expansion, or from
the authors' 1-degree grid). Its fields are the quantities AS STORED in that source, with their unit and their
reference stated in field_info(). Conversions happen only when asked for:

    unit="fraction" / "percent"  percent <-> fraction, for relative perturbations
    about_mean=True              perturbation about the area-weighted lateral mean at each depth
                                 (for absolute speeds: v / mean - 1, a fraction)
    derived fields               e.g. "vs_voigt" = sqrt((2 vsv^2 + vsh^2) / 3), listed by derived_fields()

Coordinates: latitude and longitude in degrees (geographic, as in the files), depth in km below the surface of
a 6371 km sphere. Arrays broadcast against each other. Outside a field's coverage the value is NaN.
"""
import numpy as np

from . import _sh

R_EARTH_KM = 6371.0
ABSOLUTE_UNITS = ("km/s", "m/s")


class Model:
    """Base class; readers set self._fields and implement _eval (and optionally _grid, _mean)."""

    name = "?"
    source = "?"

    def __init__(self):
        self._fields = {}       # field -> dict(unit, quantity, reference, coverage, native_depths, description)
        self._derived = {}      # field -> dict(inputs, func, unit, quantity, reference, description)
        self._mean_cache = {}

    # ------------------------------------------------------------------ what there is
    def fields(self):
        """Fields as stored in this source."""
        return list(self._fields)

    def derived_fields(self):
        """Fields that can be computed from the stored ones when asked for (never by default)."""
        return list(self._derived)

    def field_info(self, field):
        """unit, quantity, reference model, coverage (top, bottom km), native depth nodes, description."""
        if field in self._fields:
            return dict(self._fields[field])
        if field in self._derived:
            d = dict(self._derived[field])
            d.pop("func", None)
            d.setdefault("coverage", self._derived_coverage(field))
            return d
        raise KeyError(f"{self.name} ({self.source}) has no field {field!r}; fields: {self.fields()}"
                       + (f", derived: {self.derived_fields()}" if self._derived else ""))

    def coverage(self, field):
        """(top, bottom) depth in km where the field is defined."""
        return tuple(self.field_info(field)["coverage"])

    def native_depths(self, field):
        """The depths (km) of the native radial nodes, layer centres or knots inside the coverage, or None."""
        d = self.field_info(field).get("native_depths")
        return None if d is None else np.asarray(d, float)

    def __repr__(self):
        lines = [f"{self.name} (source: {self.source})"]
        for f in self.fields():
            i = self._fields[f]
            top, bot = i["coverage"]
            lines.append(f"  {f:<8} {i['unit']:<9} {top:7.1f}-{bot:7.1f} km  {i['quantity']}")
            lines.append(f"  {'':<8} reference: {i['reference']}")
        for f in self.derived_fields():
            lines.append(f"  {f:<8} (derived on request) {self._derived[f]['quantity']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ evaluation
    def _eval(self, field, lat, lon, depth):            # flat arrays in, flat array out
        raise NotImplementedError

    def _grid(self, field, depth, lat, lon):
        """Values on the grid lat (n_lat,) x lon (n_lon,) at one depth; readers may override for speed."""
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        return self._eval(field, LA.ravel(), LO.ravel(), np.full(LA.size, float(depth))).reshape(LA.shape)

    def _raw(self, field, lat, lon, depth):
        if field in self._fields:
            return self._eval(field, lat, lon, depth)
        d = self._derived[field]
        return d["func"](*[self._eval(f, lat, lon, depth) for f in d["inputs"]])

    def _raw_grid(self, field, depth, lat, lon):
        if field in self._fields:
            return self._grid(field, depth, lat, lon)
        d = self._derived[field]
        return d["func"](*[self._grid(f, depth, lat, lon) for f in d["inputs"]])

    def _derived_coverage(self, field):
        tops, bots = zip(*[self._fields[f]["coverage"] for f in self._derived[field]["inputs"]])
        return (max(tops), min(bots))

    def _unit(self, field):
        return self.field_info(field)["unit"]

    def _convert(self, field, values, unit, about_mean, mean):
        u = self._unit(field)
        if about_mean:
            if u in ABSOLUTE_UNITS:
                values, u = values / mean - 1.0, "fraction"
            else:
                values = values - mean
        if unit is None or unit == u:
            return values
        if u == "percent" and unit == "fraction":
            return values / 100.0
        if u == "fraction" and unit == "percent":
            return values * 100.0
        raise ValueError(f"cannot convert {field} from {u!r} to {unit!r}"
                         + (" (use about_mean=True for a relative perturbation)" if u in ABSOLUTE_UNITS else ""))

    def evaluate(self, field, lat, lon, depth, unit=None, about_mean=False):
        """The field at points (degrees, degrees, km; arrays broadcast), as stored unless unit/about_mean say
        otherwise. NaN outside the coverage."""
        self.field_info(field)
        lat, lon, depth = np.broadcast_arrays(*(np.asarray(x, float) for x in (lat, lon, depth)))
        shape = lat.shape
        la, lo, de = lat.ravel(), np.mod(lon.ravel() + 180.0, 360.0) - 180.0, depth.ravel()
        out = np.asarray(self._raw(field, la, lo, de), float)
        if about_mean or unit is not None:
            res = np.full(out.shape, np.nan)
            for d in np.unique(de):
                sel = de == d
                m = self.lateral_mean(field, float(d)) if about_mean else None
                res[sel] = self._convert(field, out[sel], unit, about_mean, m)
            out = res
        return out.reshape(shape)

    def unit_of(self, field, unit=None, about_mean=False):
        """The unit evaluate() returns for these options."""
        u = self._unit(field)
        if about_mean and u in ABSOLUTE_UNITS:
            u = "fraction"
        return unit or u

    # ------------------------------------------------------------------ slices, profiles, sections
    def depth_slice(self, field, depth, step=1.0, unit=None, about_mean=False, lat=None, lon=None):
        """(lat, lon, values) on a regular grid (default: step degrees, lat -90..90, lon -180..180);
        values has shape (n_lat, n_lon). NaN where the field is undefined."""
        top, bot = self.coverage(field)
        if not top - 1e-9 <= depth <= bot + 1e-9:
            raise ValueError(f"{self.name} {field} covers depths {top:g}-{bot:g} km, not {depth:g}")
        lat = np.arange(-90.0, 90.0 + step / 2, step) if lat is None else np.asarray(lat, float)
        lon = np.arange(-180.0, 180.0 + step / 2, step) if lon is None else np.asarray(lon, float)
        v = np.asarray(self._raw_grid(field, float(depth), lat, lon), float)
        m = self.lateral_mean(field, float(depth)) if about_mean else None
        return lat, lon, self._convert(field, v, unit, about_mean, m)

    def radial_profile(self, field, lat, lon, depths=None, unit=None, about_mean=False):
        """(depths, values) below one point; default depths: 2 km spacing over the coverage plus the native
        nodes (so steps of layered models and kinks of linear interpolation are resolved)."""
        if depths is None:
            top, bot = self.coverage(field)
            depths = np.arange(np.ceil(top), bot, 2.0)
            nd = self.native_depths(field)
            depths = np.unique(np.concatenate([[top, bot], depths, [] if nd is None else nd]))
        depths = np.asarray(depths, float)
        return depths, self.evaluate(field, lat, lon, depths, unit=unit, about_mean=about_mean)

    def cross_section(self, field, start, end, n=181, depths=None, unit=None, about_mean=False):
        """Values along the shorter great circle from start=(lat, lon) to end=(lat, lon): dict with
        distance (n,) in degrees, lat, lon (n,), depth (n_depth,), values (n_depth, n)."""
        glat, glon, dist = great_circle(start, end, n)
        if depths is None:
            top, bot = self.coverage(field)
            depths = np.linspace(top, bot, 120)
        depths = np.asarray(depths, float)
        D, LA = np.meshgrid(depths, glat, indexing="ij")
        _, LO = np.meshgrid(depths, glon, indexing="ij")
        v = self.evaluate(field, LA, LO, D, unit=unit, about_mean=about_mean)
        return {"distance": dist, "lat": glat, "lon": glon, "depth": depths, "values": v}

    # ------------------------------------------------------------------ lateral statistics
    def lateral_mean(self, field, depth, n_lat=180):
        """Area-weighted lateral mean at one depth, as stored (Gauss-Legendre quadrature with n_lat x 2 n_lat
        nodes unless the reader knows it exactly; points where the field is undefined are left out)."""
        key = (field, float(depth), n_lat)
        if key not in self._mean_cache:
            m = self._mean(field, float(depth)) if field in self._fields else None
            if m is None:
                lat, lon, w = _sh.gauss_grid(n_lat)
                v = self._raw_grid(field, float(depth), lat, lon)
                ok = np.isfinite(v)
                m = float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else np.nan
            self._mean_cache[key] = m
        return self._mean_cache[key]

    def _mean(self, field, depth):
        return None

    def rms_profile(self, field, depths=None, about_mean=True, unit=None, n_lat=90):
        """(depths, r.m.s.) of the area-weighted lateral r.m.s. at each depth (about the lateral mean by
        default), on a Gauss-Legendre grid of n_lat x 2 n_lat nodes."""
        if depths is None:
            top, bot = self.coverage(field)
            depths = np.linspace(top, bot, 100)
        lat, lon, w = _sh.gauss_grid(n_lat)
        out = []
        for d in np.asarray(depths, float):
            v = self._raw_grid(field, d, lat, lon)
            ok = np.isfinite(v)
            m = float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else np.nan
            v = self._convert(field, v, unit, about_mean, m)
            out.append(np.sqrt(np.sum(w[ok] * v[ok] ** 2) / np.sum(w[ok])) if ok.any() else np.nan)
        return np.asarray(depths, float), np.array(out)

    def sh_coefficients(self, field, depth, lmax=20, n_lat=None, unit=None):
        """Coefficients of the real orthonormal harmonics up to lmax (real_sh_indices order) of the field at
        one depth, by Gauss-Legendre quadrature of the native field (undefined points count as zero).
        A convenience for comparisons; the model itself is not truncated anywhere else."""
        n_lat = n_lat or max(2 * lmax + 2, 180)
        lat, lon, w = _sh.gauss_grid(n_lat)
        v = self._raw_grid(field, float(depth), lat, lon)
        v = self._convert(field, np.nan_to_num(v), unit, False, None)
        return _sh.project(v, lat, lon, w, lmax)


def great_circle(start, end, n=181):
    """(lat, lon, distance in degrees) of n points on the shorter great circle from start to end."""
    def xyz(p):
        la, lo = np.radians(p[0]), np.radians(p[1])
        return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    a, b = xyz(start), xyz(end)
    w = np.arccos(np.clip(a @ b, -1.0, 1.0))
    if w < 1e-12:
        raise ValueError("start and end coincide")
    f = np.linspace(0.0, 1.0, n)[:, None]
    p = (np.sin((1 - f) * w) * a + np.sin(f * w) * b) / np.sin(w)
    return (np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1))), np.degrees(np.arctan2(p[:, 1], p[:, 0])),
            np.degrees(f[:, 0] * w))


def voigt_vs(vsv, vsh):
    """Isotropic (Voigt) shear speed sqrt((2 vsv^2 + vsh^2) / 3)."""
    return np.sqrt((2.0 * vsv ** 2 + vsh ** 2) / 3.0)


def voigt_vp(vpv, vph):
    """Isotropic (Voigt) P speed sqrt((vpv^2 + 4 vph^2) / 5) (neglecting the eta term)."""
    return np.sqrt((vpv ** 2 + 4.0 * vph ** 2) / 5.0)


def voigt_linear(dvsv, dvsh):
    """First-order Voigt perturbation (2 dln vsv + dln vsh) / 3 from relative perturbations."""
    return (2.0 * dvsv + dvsh) / 3.0
