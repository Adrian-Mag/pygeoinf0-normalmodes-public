"""Where the original files are, and small helpers to read them in place (including members of zip and tar
archives, which are never unpacked by hand).

The originals are not in this repository: set GLOBALMODELS_ORIGINALS to the folder holding <MODEL>/. Large members
that are slow to parse are cached (as .npz) in ~/.cache/pygeoinf0-normalmodes/, or in GLOBALMODELS_CACHE.
"""
import gzip
import hashlib
import io
import os
import tarfile
import zipfile

import numpy as np


def originals_dir():
    d = os.environ.get("GLOBALMODELS_ORIGINALS")
    if d is None:
        raise FileNotFoundError("set GLOBALMODELS_ORIGINALS to the folder holding the models' original files")
    return d


def cache_dir():
    d = os.environ.get("GLOBALMODELS_CACHE", os.path.expanduser("~/.cache/pygeoinf0-normalmodes"))
    os.makedirs(d, exist_ok=True)
    return d


def original(model_dir, *parts):
    """Path of an original file, e.g. original("S40RTS", "raw", "S40RTS.sph"); raises if it is missing."""
    p = os.path.join(originals_dir(), model_dir, *parts)
    if not os.path.exists(p):
        raise FileNotFoundError(f"original file not found: {p} (set GLOBALMODELS_ORIGINALS if the "
                                f"originals are elsewhere)")
    return p


def read_bytes(path, member=None):
    """Bytes of a file, of a member of a .zip / .tar(.gz) archive, or of a .gz file."""
    if member is None:
        if path.endswith(".gz") and not path.endswith(".tar.gz"):
            with gzip.open(path, "rb") as fh:
                return fh.read()
        with open(path, "rb") as fh:
            return fh.read()
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return z.read(member)
    with tarfile.open(path) as t:
        f = t.extractfile(member)
        if f is None:
            raise KeyError(f"{member} is not a regular file in {path}")
        return f.read()


def read_text(path, member=None):
    return read_bytes(path, member).decode("latin-1")


def open_member(path, member):
    """A binary file object for a member of a zip archive (streamed, not read into memory)."""
    return zipfile.ZipFile(path).open(member)


def numbers(text, dtype=float):
    """All whitespace-separated numbers of a text, as a 1-D array (Fortran D exponents accepted)."""
    if "D" in text or "d" in text:
        text = text.replace("D", "E").replace("d", "e")
    a = np.fromstring(text, sep=" ")             # text mode: any whitespace separates
    n_tok = len(text.split()) if len(text) < 1_000_000 else None
    if n_tok is not None and a.size != n_tok:
        raise ValueError(f"could not parse all {n_tok} numbers ({a.size} read)")
    return a.astype(dtype, copy=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cached_arrays(name, sources, builder):
    """Arrays built once by builder() and cached in cache_dir()/<name>.npz, keyed by the sources' sizes and
    modification times (rebuilt if a source changes)."""
    key = ";".join(f"{os.path.basename(s)}:{os.path.getsize(s)}:{int(os.path.getmtime(s))}" for s in sources)
    path = os.path.join(cache_dir(), name + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            if "_key" in z.files and str(z["_key"]) == key:
                return {k: z[k] for k in z.files if k != "_key"}
    arrays = builder()
    tmp = path + ".tmp.npz"
    np.savez(tmp, _key=np.array(key), **arrays)
    os.replace(tmp, path)
    return arrays


def text_io(path, member=None):
    return io.StringIO(read_text(path, member))
