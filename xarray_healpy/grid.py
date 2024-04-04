from dataclasses import dataclass,field

import cdshealpix
import dask
import healpy as hp
import numba
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from astropy.coordinates import Latitude, Longitude

from xarray_healpy.conversions import base_pixel


@numba.jit(nopython=True, parallel=True)
def _compute_indices(level):
    lidx = np.arange(4**level)
    xx = np.zeros_like(lidx)
    yy = np.zeros_like(lidx)

    for i in range(level):
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


def _find_grid_indices(grid_values, values):
    index = pd.Index(values)

    return index.get_indexer(grid_values)


def _to_2d(data, indices, new_shape):
    if data.size in (0, 1):
        return np.ones((1, 1), dtype=data.dtype)

    return data[..., indices].reshape(new_shape)


@dataclass
class HealpyGridInfo:
    """class representing a HealPix grid

    Attributes
    ----------
    level : int
        HealPix grid resolution level
    rot : dict of str to float
        Rotation of the healpix sphere.
    """

    level: int

    rot: dict[str, float] = field(default_factory=lambda: {"lat": 0, "lon": 0})

    @property
    def nside(self):
        return 2**self.level

    def rotate(self, grid, *, direction="rotated"):
        if direction == "rotated":
            return grid.assign_coords(
                {
                    "latitude": grid["latitude"] - self.rot["lat"],
                    "longitude": grid["longitude"] - self.rot["lon"],
                }
            )
        elif direction == "global":
            return grid.assign_coords(
                {
                    "latitude": grid["latitude"] + self.rot["lat"],
                    "longitude": grid["longitude"] + self.rot["lon"],
                }
            )

    def target_grid(self, source_grid):
        if self.rot:
            source_grid = self.rotate(source_grid, direction="rotated")

        lon, lat = dask.compute(source_grid["longitude"], source_grid["latitude"])

        bbox = shapely.box(
            lon.min().item(), lat.min().item(), lon.max().item(), lat.max().item()
        )

        # TODO: compute from lon / lat
        segment_length = 0.5

        outline = shapely.segmentize(bbox, max_segment_length=segment_length)
        outline_coords = shapely.get_coordinates(outline)

        pixel_indices_, _, fully_covered = cdshealpix.nested.polygon_search(
            Longitude(outline_coords[:, 0], unit="deg"),
            Latitude(outline_coords[:, 1], unit="deg"),
            depth=self.level,
            flat=True,
        )
        pixel_indices = pixel_indices_[fully_covered.astype(bool)]

        target_lon, target_lat = map(
            lambda x: np.asarray(np.rad2deg(x)),
            cdshealpix.nested.healpix_to_lonlat(pixel_indices, depth=self.level),
        )

        grid_params = (
            {
                "grid_type": "healpix",
                "level": self.level,
                "nside": self.nside,
            }
            | self.rot
            | {f"rot_{k}": v for k, v in self.rot.items()}
        )

        target_grid = xr.Dataset(
            coords={
                "cell_ids": ("cells", pixel_indices, grid_params),
                "latitude": ("cells", target_lat, {"units": "deg"}),
                "longitude": ("cells", target_lon, {"units": "deg"}),
                "resolution": ((), hp.nside2resol(self.nside), {"units": "rad"}),
            },
            attrs=grid_params,  # for compat, delete later
        )

        return self.rotate(target_grid, direction="global")

    def to_2d(self, ds, dim="cells"):
        # rotate if necessary
        if self.rot:
            ds = self.rotate(ds, direction="rotated")

        cell_ids = ds["cell_ids"].compute()

        base_pixels = np.unique(base_pixel(self.level, cell_ids.data))
        if base_pixels.size != 1:
            raise ValueError(
                "can only reshape for a single base pixel for now."
                f" Data covers base pixels {base_pixels.tolist()}."
            )

        # extract base pixel
        unique_base_pixel = base_pixels[0]

        # compute 2D pixel index
        xx, yy = _compute_indices(self.level)
        all_pixels = np.full((self.nside, self.nside), fill_value=-1, dtype=int)
        all_pixels[xx, yy] = unique_base_pixel * 4**self.level + np.arange(
            4**self.level
        )

        # filter out rows and columns that are all not in the data
        mask = np.isin(all_pixels, cell_ids)
        rows_to_keep = np.squeeze(np.argwhere(np.any(mask, axis=1)))
        columns_to_keep = np.squeeze(np.argwhere(np.any(mask, axis=0)))

        filtered_pixels = all_pixels[rows_to_keep, :][:, columns_to_keep]
        filtered_mask = mask[rows_to_keep, :][:, columns_to_keep]

        # find the indices of the input cells in the flattened 2d grid
        indices = _find_grid_indices(np.ravel(filtered_pixels), cell_ids.data)

        # generate new coordinates
        new_lon, new_lat = hp.pix2ang(
            self.nside, filtered_pixels, nest=True, lonlat=True
        )

        new_dims = ["y", "x"]
        new_sizes = dict(zip(new_dims, filtered_pixels.shape))

        reshaped_mask = xr.DataArray(filtered_mask, dims=new_dims)
        new_coords = xr.Dataset(
            coords={
                "cell_ids": (new_dims, filtered_pixels, cell_ids.attrs),
                "latitude": (new_dims, new_lat, ds["latitude"].attrs),
                "longitude": (new_dims, new_lon, ds["longitude"].attrs),
            }
        )

        # apply the reshaping
        coords_to_drop = ["cell_ids", "latitude", "longitude"]
        reshaped = (
            xr.apply_ufunc(
                _to_2d,
                ds.drop_vars(coords_to_drop),
                xr.DataArray(indices, dims="new_cells"),
                input_core_dims=[[dim], ["new_cells"]],
                output_core_dims=[new_dims],
                kwargs={"new_shape": tuple(new_sizes.values())},
                dask="parallelized",
                dask_gufunc_kwargs={"output_sizes": new_sizes},
                vectorize=True,
                keep_attrs=True,
            )
            .where(reshaped_mask)
            .assign_coords(new_coords.coords)
        )

        # rotate if necessary
        return self.rotate(reshaped, direction="global")

def create_grid(nside, rot=None):
    if rot is None:
        rot = {"lat": 0, "lon": 0}
    return HealpyGridInfo(nside=nside, rot=rot)
