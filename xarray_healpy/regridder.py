import itertools
import json
import os
from dataclasses import dataclass, field
from functools import partial

import healpy as hp
import numpy as np
import sparse
import xarray as xr

from xarray_healpy.grid import HealpyGridInfo


def concat(iterable):
    return itertools.chain.from_iterable(iterable)


def unique(iterable):
    return list(dict.fromkeys(iterable))


def _compute_weights(source_lat, source_lon, *, nside, rot={"lat": 0, "lon": 0}):
    theta = (90.0 - (source_lat - rot["lat"])) / 180.0 * np.pi
    phi = -(source_lon - rot["lon"]) / 180.0 * np.pi

    new_dim = "neighbors"
    input_core_dims = unique(concat([source_lat.dims, source_lon.dims]))
    output_core_dims = [new_dim] + input_core_dims

    pix, weights = xr.apply_ufunc(
        partial(hp.get_interp_weights, nside, nest=True),
        theta,
        phi,
        input_core_dims=[input_core_dims, input_core_dims],
        output_core_dims=[output_core_dims, output_core_dims],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {new_dim: 4}},
    )
    pix -= nside**2 * (np.min(pix) // nside**2)

    return pix, weights


def _weights_to_sparse(weights):
    rows = weights["dst_cell_ids"]
    cols = weights["src_cell_ids"]

    # TODO: reshape such that we get a matrix of (dst_cells, nj, ni)
    sizes = {"dst_cells": weights.attrs["n_dst_cells"]} | json.loads(
        weights.attrs["src_grid_dims"]
    )

    coo = sparse.COO(
        coords=np.stack([rows.data, cols.data]),
        data=weights.weights.data,
        fill_value=0,
        shape=(weights.attrs["n_dst_cells"], weights.attrs["n_src_cells"]),
    ).reshape(tuple(sizes.values()))

    sparse_weights = xr.DataArray(
        coo,
        dims=list(sizes),
        attrs=weights.attrs,
    )

    return sparse_weights


@dataclass(repr=False)
class HealpyRegridder:
    """regrid a dataset to healpy face 0

    Parameters
    ----------
    input_grid : xr.Dataset
        The input dataset. For now, it has to have the `"latitude"` and `"longitude"` coordinates.
    output_grid : HealpyGridInfo
        The target grid, containing healpix parameters like `nside` and `rot`.
    """

    input_grid: xr.Dataset
    output_grid: HealpyGridInfo

    weights_path: str | os.PathLike | None = None

    weights: xr.Dataset = field(init=False)

    def __post_init__(self):
        coords = ["latitude", "longitude"]

        stacked_dim = "src_grid_cells"

        src_grid = self.input_grid
        src_grid_dims = unique(
            concat(src_grid.variables[coord].dims for coord in coords)
        )
        src_grid_sizes = {name: src_grid.sizes[name] for name in src_grid_dims}

        stacked_src_grid = src_grid.stack({stacked_dim: src_grid_dims}).reset_index(
            stacked_dim
        )

        dst_grid = self.output_grid.to_xarray()
        dst_grid_dims = unique(concat(dst_grid.variables[name].dims for name in coords))
        dst_grid_sizes = {name: dst_grid.sizes[name] for name in dst_grid_dims}

        src_cell_ids = xr.DataArray(
            np.arange(stacked_src_grid.sizes[stacked_dim]), dims=stacked_dim
        )

        stacked_variables = [stacked_src_grid[coord] for coord in coords]
        pix_, weights_ = _compute_weights(
            *stacked_variables,
            nside=self.output_grid.nside,
            rot=self.output_grid.rot,
        )
        _, aligned = xr.broadcast(pix_, src_cell_ids)

        self.weights = (
            xr.Dataset(
                {"dst_cell_ids": pix_, "weights": weights_, "src_cell_ids": aligned}
            )
            .set_coords(["dst_cell_ids", "src_cell_ids"])
            .stack({"ncol": ["neighbors", "src_grid_cells"]})
            .reset_index("ncol")
            .drop_vars(coords)
            .merge(dst_grid, combine_attrs="drop_conflicts")
            .assign_attrs(
                {
                    "src_grid_dims": json.dumps(src_grid_sizes),
                    "dst_grid_dims": json.dumps(dst_grid_sizes),
                    "n_src_cells": stacked_src_grid.sizes[stacked_dim],
                    "n_dst_cells": self.output_grid.nside**2,
                }
            )
        )

    def regrid_ds(self, ds):
        """regrid a dataset on the same grid as the input grid

        The regridding method is restricted to linear interpolation so far.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset.

        Returns
        -------
        regridded : xr.Dataset
            The regridded dataset
        """
        # based on https://github.com/pangeo-data/xESMF/issues/222#issuecomment-1524041837

        def _apply(da, weights, normalization):
            import opt_einsum

            # 🐵 🔧
            xr.core.duck_array_ops.einsum = opt_einsum.contract

            ans = xr.dot(
                # drop all input coords, as those would most likely be broadcast
                da.variable,
                weights,
                # This dimension will be "contracted"
                # or summmed over after multiplying by the weights
                dims=src_dims,
            )

            # 🐵 🔧 : restore back to original
            xr.core.duck_array_ops.einsum = np.einsum

            normalized = ans / normalization

            return normalized

        # construct the sparse weights matrix and pre-compute normalization factors
        weights = _weights_to_sparse(self.weights)

        src_dims = list(json.loads(weights.attrs["src_grid_dims"]))
        normalization = weights.sum(src_dims).as_numpy()

        # regrid only those variables with the source dims
        vars_with_src_dims = [
            name
            for name, array in ds.variables.items()
            if set(src_dims).issubset(array.dims) and name not in weights.coords
        ]
        regridded = ds[vars_with_src_dims].map(
            _apply,
            weights=weights.chunk(),
            normalization=normalization,
        )

        # reshape to a healpy 2d grid and assign coordinates
        coords = (
            self.output_grid.to_xarray().rename_dims({"cells": "dst_cells"}).chunk()
        )
        reshaped = (
            regridded.merge(coords, combine_attrs="drop_conflicts")
            .pipe(self.output_grid.unstructured_to_2d, dim="dst_cells")
            .set_coords(["latitude", "longitude", "cell_ids"])
        )

        # merge in other variables, but skip those that are already set
        to_drop = set(reshaped.variables) & set(ds.variables)
        merged = xr.merge(
            [ds.drop_vars(to_drop), reshaped],
            combine_attrs="drop_conflicts",
        ).drop_dims(src_dims)

        # crop all-missing rows and columns
        cropped = merged.dropna(dim="x", how="all", subset=["H0"]).dropna(
            dim="y", how="all", subset=["H0"]
        )

        return cropped
