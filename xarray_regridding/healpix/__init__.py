try:
    from xarray_regridding.healpix import cdshealpix
except ImportError:
    cdshealpix = None

try:
    from xarray_regridding.healpix import healpy
except ImportError:
    healpy = None


backends = {
    "cdshealpix": cdshealpix,
    "healpy": healpy,
}


def use_backend(name):
    if name not in backends:
        raise ValueError("unknown healpix backend")

    backend = backends[name]
    if backend is None:
        raise ValueError(
            "healpix backend not available. Install the appropriate packages."
        )

    return backend


def lonlat_to_healpix(lon, lat, level, backend="cdshealpix"):
    return use_backend(backend).lonlat_to_healpix(lon, lat, level)
