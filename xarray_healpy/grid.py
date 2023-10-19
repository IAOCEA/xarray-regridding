from dataclasses import dataclass, field

import healpy as hp
import numba
import numpy as np
import xarray as xr


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


def _compute_coords(nside):
    lidx = np.arange(nside**2)
    theta, phi = hp.pix2ang(nside, lidx, nest=True)

    lat = 90.0 - np.rad2deg(theta)
    lon = -np.rad2deg(phi)

    return lat, lon, lidx


@dataclass
class HealpyGridInfo:
    """class representing a HealPix grid

    Attributes
    ----------
    nside : int
        HealPix grid resolution
    rot : dict of str to float
        Rotation of the healpix sphere.
    coords : xr.Dataset
        Unstructured grid coordinates: latitude, longitude, cell ids.
    indices : xr.DataArray
        Indices that can be used to reorder to a flattened 2D healpy grid
    """

    nside: int

    rot: dict[str, float]

    indices: xr.DataArray = field(repr=False)
    coords: xr.Dataset = field(repr=False)

    def unstructured_to_2d(
        self, unstructured, dim="cells", keep_attrs="drop_conflicts"
    ):
        def _unstructured_to_2d(unstructured, indices, new_shape):
            if unstructured.size in (0, 1):
                return np.ones((1, 1), dtype=unstructured.dtype)

            return unstructured[..., indices].reshape(new_shape)

        new_sizes = {"x": self.nside, "y": self.nside}

        return xr.apply_ufunc(
            _unstructured_to_2d,
            unstructured,
            self.indices,
            input_core_dims=[[dim], ["cells"]],
            output_core_dims=[["x", "y"]],
            kwargs={"new_shape": tuple(new_sizes.values())},
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": new_sizes},
            vectorize=True,
            keep_attrs=keep_attrs,
        )

    def to_xarray(self):
        attrs = {"nside": self.nside} | {f"rot_{k}": v for k, v in self.rot.items()}

        return self.coords.assign_attrs(attrs)


def create_grid(nside, rot={"lat": 0, "lon": 0}):
    xx, yy = _compute_indices(nside)

    raw_indices = np.full((nside, nside), fill_value=-1, dtype=int)
    raw_indices[xx, yy] = np.arange(nside**2)
    indices = xr.DataArray(np.ravel(raw_indices), dims="cells").chunk()

    lat_, lon_, cell_ids = _compute_coords(nside)
    lat = lat_ - rot["lat"]
    lon = lon_ + rot["lon"]

    resolution = hp.nside2resol(nside)
    coords = xr.Dataset(
        {
            "latitude": (["cells"], lat, {"units": "deg"}),
            "longitude": (["cells"], lon, {"units": "deg"}),
            "cell_ids": (["cells"], cell_ids),
        },
        coords={"resolution": ((), resolution, {"units": "rad"})},
    )

    return HealpyGridInfo(nside=nside, rot=rot, indices=indices, coords=coords)
