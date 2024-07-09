from importlib.metadata import version

from xarray_regridding.grid import HealpyGridInfo  # noqa: F401
from xarray_regridding.regridder import HealpyRegridder  # noqa: F401

try:
    __version__ = version("xarray_regridding")
except Exception:
    __version__ = "999"
