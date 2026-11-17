"""DETOX-P2 and DETOX-P3 (Hosseini et al. 2020), read from the Zenodo archive DETOX_MODELS.zip (which also holds
DETOX-P1) without unpacking it.

  "grid"  DETOX_MODELS/<model>/grid_nc4/<model>_<depth>.grd (default): netCDF grids at 59 depths 0, 50, ...,
          2900 km on a 0.5-degree node grid (lat -90..90, lon 0..360), made by the authors with GMT surface
          from per-depth slices of the mesh (history attribute). dVp/Vp in PERCENT (about 100 x the mesh
          values). Bilinear between nodes, linear in depth between the grids.
  "mesh"  the native model: DETOX_MODELS/<model>/vtk/<model>.vtk, an unstructured grid of 396501 vertices and
          2338710 tetrahedra with the vertex values (the same vertices and values as txt_tetrahedron/*.solx,
          "x y z dlnVp" in km). The field is linear inside each tetrahedron (barycentric interpolation of the
          vertex values); dlnVp as a FRACTION. The mesh extends above the surface and through the core; the
          coverage here is 0 km to IASP91's CMB (2889 km depth).
Reference model: IASP91 (Hosseini et al. 2020). Orientation of the mesh: lon = atan2(y, x), lat = asin(z/r).
"""
import io
import re
import zipfile

import numpy as np

from .. import _paths
from .._model import Model, R_EARTH_KM

ZIP = ("DETOX-P2-P3", "raw", "DETOX_MODELS.zip")
REF = "IASP91 (Hosseini et al. 2020)"
CMB_DEPTH_IASP91 = 6371.0 - 3482.0


class DetoxGrid(Model):
    source = "grid"

    def __init__(self, model):
        super().__init__()
        self.name = model
        zp = _paths.original(*ZIP)
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist()
                     if re.fullmatch(rf"DETOX_MODELS/{model}/grid_nc4/{model}_(\d+\.\d+)\.grd", n)]
        depths = np.array([float(re.search(r"_(\d+\.\d+)\.grd$", n).group(1)) for n in names])
        order = np.argsort(depths)
        self.names = [names[i] for i in order]
        self.depths = depths[order]

        def build():
            import netCDF4
            vals = []
            with zipfile.ZipFile(zp) as z:
                for n in self.names:
                    with netCDF4.Dataset("inmemory.nc", memory=z.read(n)) as ds:
                        lat = np.asarray(ds["lat"][:], float)
                        lon = np.asarray(ds["lon"][:], float)
                        vals.append(np.ma.filled(ds["z"][:].astype(np.float32), np.nan))
            return {"lat": lat, "lon": lon, "value": np.array(vals)}
        z = _paths.cached_arrays(f"detox_grid_{model}", [zp], build)
        self.glat, self.glon, self.v = z["lat"], z["lon"], z["value"]
        self._fields["dvp"] = dict(
            unit="percent", quantity="d ln vp (dVp/Vp)", reference=REF,
            coverage=(float(self.depths[0]), float(self.depths[-1])), native_depths=self.depths.copy(),
            description=f"zip members DETOX_MODELS/{model}/grid_nc4/*.grd: {self.depths.size} depths x "
                        f"{self.glat.size} x {self.glon.size} nodes (GMT surface grids of the mesh); "
                        "the 2900 km grid lies below IASP91's CMB")

    def _eval(self, field, lat, lon, depth):
        from .savani import _bilinear
        d = np.asarray(depth, float)
        ok = (d >= self.depths[0] - 1e-9) & (d <= self.depths[-1] + 1e-9)
        k = np.clip(np.searchsorted(self.depths, d, side="right") - 1, 0, self.depths.size - 2)
        t = np.clip((d - self.depths[k]) / (self.depths[k + 1] - self.depths[k]), 0.0, 1.0)
        out = np.full(lat.shape, np.nan)
        v0 = _bilinear(self.glat, self.glon, self.v, k[ok], lat[ok], lon[ok])
        v1 = _bilinear(self.glat, self.glon, self.v, k[ok] + 1, lat[ok], lon[ok])
        out[ok] = (1 - t[ok]) * v0 + t[ok] * v1
        return out


def _read_vtk(data):
    """(points (n, 3), tets (m, 4) int32, point values (n,)) from the ASCII VTK file bytes."""
    def section(key):
        m = re.search(rb"^" + key + rb"[^\n]*\n", data, re.M)
        if m is None:
            raise ValueError(f"VTK: no {key!r} section")
        return m
    mp = section(b"POINTS")
    npts = int(mp.group(0).split()[1])
    mc = section(b"CELLS")
    ncell = int(mc.group(0).split()[1])
    mt = section(b"CELL_TYPES")
    ml = section(b"LOOKUP_TABLE")
    pts = np.fromstring(data[mp.end():mc.start()].decode(), sep=" ")
    cells = np.fromstring(data[mc.end():mt.start()].decode(), sep=" ")
    tail = data[ml.end():]
    nxt = re.search(rb"^[A-Za-z]", tail, re.M)
    vals = np.fromstring(tail[:nxt.start() if nxt else len(tail)].decode(), sep=" ")
    types = np.fromstring(data[mt.end():data.index(b"POINT_DATA", mt.end())].decode(), sep=" ")
    if pts.size != 3 * npts or cells.size != 5 * ncell or vals.size != npts or types.size != ncell:
        raise ValueError("VTK: section sizes do not match their headers")
    cells = cells.reshape(ncell, 5)
    if not (np.all(cells[:, 0] == 4) and np.all(types == 10)):
        raise ValueError("VTK: cells are not all tetrahedra")
    return pts.reshape(npts, 3), cells[:, 1:].astype(np.int32), vals


class DetoxMesh(Model):
    source = "mesh"

    def __init__(self, model):
        super().__init__()
        self.name = model
        zp = _paths.original(*ZIP)

        def build():
            with zipfile.ZipFile(zp) as z:
                pts, tets, vals = _read_vtk(z.read(f"DETOX_MODELS/{model}/vtk/{model}.vtk"))
                solx = z.read(f"DETOX_MODELS/{model}/txt_tetrahedron/{model}.solx").decode()
            head, n, body = solx.split("\n", 2)
            if head.split()[-1] != "dlnVp" or int(n) != pts.shape[0]:
                raise ValueError("solx header does not match the VTK points")
            s = np.fromstring(body, sep=" ").reshape(-1, 4)
            return {"xyz": pts, "tets": tets, "vtk_value": vals, "solx": s}
        z = _paths.cached_arrays(f"detox_mesh_{model}", [zp], build)
        self.xyz, self.tets, self.value = z["xyz"], z["tets"], z["solx"][:, 3]
        self.check = {"max_abs_solx_minus_vtk_xyz_km": float(np.max(np.abs(z["solx"][:, :3] - self.xyz))),
                      "max_abs_solx_minus_vtk_value": float(np.max(np.abs(z["solx"][:, 3] - z["vtk_value"])))}
        self._tree = None
        self._fields["dvp"] = dict(
            unit="fraction", quantity="d ln vp (dlnVp)", reference=REF, coverage=(0.0, CMB_DEPTH_IASP91),
            native_depths=None,
            description=f"zip member DETOX_MODELS/{model}/vtk/{model}.vtk (+ .solx values): "
                        f"{self.xyz.shape[0]} vertices, {self.tets.shape[0]} tetrahedra, linear in each")

    def _locator(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            P = self.xyz[self.tets]                                   # (m, 4, 3)
            self._cent = P.mean(axis=1)
            T = np.transpose(P[:, 1:, :] - P[:, :1, :], (0, 2, 1))   # columns p1-p0, p2-p0, p3-p0
            self._tinv = np.linalg.inv(T)
            self._p0 = P[:, 0, :]
            self._tree = cKDTree(self._cent)
        return self._tree

    def _locate(self, q, k):
        """For points q (n, 3): (tet index or -1, barycentric weights (n, 4)) from the k nearest centroids."""
        tree = self._locator()
        _, cand = tree.query(q, k=k)
        best = np.full(q.shape[0], -1)
        bary = np.zeros((q.shape[0], 4))
        bestmin = np.full(q.shape[0], -np.inf)
        for c in range(k):
            t = cand[:, c]
            lam = np.einsum("nij,nj->ni", self._tinv[t], q - self._p0[t])
            l4 = np.concatenate([1.0 - lam.sum(axis=1, keepdims=True), lam], axis=1)
            mn = l4.min(axis=1)
            better = mn > bestmin                  # keep the most nearly containing tetrahedron
            best = np.where(better, t, best)
            bary = np.where(better[:, None], l4, bary)
            bestmin = np.where(better, mn, bestmin)
        return np.where(bestmin >= -1e-7, best, -1), bary

    def _eval(self, field, lat, lon, depth):
        out = np.full(lat.shape, np.nan)
        top, bot = self.coverage(field)
        ok = (depth >= top - 1e-9) & (depth <= bot + 1e-9)
        r = R_EARTH_KM - depth[ok]
        la, lo = np.radians(lat[ok]), np.radians(lon[ok])
        q = np.stack([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)], axis=1)
        res = np.full(q.shape[0], np.nan)
        todo = np.arange(q.shape[0])
        for k in (8, 32, 128):
            if todo.size == 0:
                break
            for s in range(0, todo.size, 200000):
                part = todo[s:s + 200000]
                t, b = self._locate(q[part], k)
                hit = t >= 0
                res[part[hit]] = np.einsum("ni,ni->n", b[hit], self.value[self.tets[t[hit]]])
            todo = todo[np.isnan(res[todo])]
        out[ok] = res
        return out


def load(model, source="grid"):
    if source == "grid":
        return DetoxGrid(model)
    if source == "mesh":
        return DetoxMesh(model)
    raise ValueError("DETOX sources: grid, mesh")
