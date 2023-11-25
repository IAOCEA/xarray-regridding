import os
from dataclasses import dataclass, field

import xarray as xr

from xarray_healpy.interpolations.bilinear import bilinear_interpolation_weights

interpolation_methods = {
    "bilinear": bilinear_interpolation_weights,
}


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
    output_grid: xr.Dataset
    method: str = "bilinear"
    interpolation_kwargs: dict = field(default_factory=dict)

    weights_path: str | os.PathLike | None = None

    weights: xr.Dataset = field(init=False)

    def __post_init__(self):
        interpolator = interpolation_methods.get(self.method)
        if interpolator is None:
            raise ValueError(f"unknown interpolation method: {self.method}")

        self.weights = interpolator(
            self.input_grid, self.output_grid, **self.interpolation_kwargs
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
        def _apply_weights(arr, weights):
            src_dims = weights.attrs["sum_dims"]

            if not set(src_dims).issubset(arr.dims):
                return arr

            return xr.dot(
                # drop all input coords, as those would most likely be broadcast
                arr.variable,
                weights,
                # This dimension will be "contracted"
                # or summed over after multiplying by the weights
                dims=src_dims,
            )

        return ds.map(_apply_weights, weights=self.weights.chunk())
