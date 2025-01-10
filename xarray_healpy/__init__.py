from importlib.metadata import version

from xarray_healpy.grid import HealpyGridInfo  # noqa: F401
from xarray_healpy.regridder import HealpyRegridder  # noqa: F401

try:
    __version__ = version("xarray_healpy")
except Exception:
    __version__ = "999"
