import healpy as hp
import numba
import numpy as np


@numba.jit(nopython=True, parallel=True)
def _compute_indices(nside):
    nstep = int(np.log2(nside))

    lidx = np.arange(nside**2)
    xx = np.zeros_like(lidx)
    yy = np.zeros_like(lidx)

    for i in range(nstep):
        p1 = 2**i
        p2 = (lidx // 4**i) % 4

        xx = xx + p1 * (p2 % 2)
        yy = yy + p1 * (p2 // 2)

    return xx, yy


def _compute_coords(nside, sector, xx, yy):
    lidx = sector * nside**2 + np.arange(nside**2)
    theta, phi = hp.pix2ang(nside, lidx, nest=True)

    lat_ = 90.0 - np.rad2deg(theta)
    lon_ = -np.rad2deg(phi)

    lat = np.full([nside, nside], fill_value=np.nan)
    lon = np.full([nside, nside], fill_value=np.nan)

    lat[xx, yy] = lat_
    lon[xx, yy] = lon_

    return lat, lon


def _fit_weights(nside, source_lat, source_lon, rot={"lat": 0, "lon": 0}):
    theta = np.deg2rad(90.0 - (source_lat.flatten() - rot["lat"]))
    phi = -np.deg2rad(source_lon.flatten() - rot["lon"])

    pix, weights = hp.get_interp_weights(nside, theta, phi, nest=True)
    pix -= nside**2 * np.min(pix) // nside**2

    return pix, weights
