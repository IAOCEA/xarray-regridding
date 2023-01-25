from importlib.metadata import version

try:
    __version__ = version("xarray_healpy")
except Exception:
    __version__ = "999"
