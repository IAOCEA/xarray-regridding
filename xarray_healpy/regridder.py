import os
from dataclasses import dataclass, field

import xarray as xr

from xarray_healpy.interpolation.bilinear import bilinear_interpolation_weights

interpolation_methods = {
    "bilinear": bilinear_interpolation_weights,
}


@dataclass(repr=True)
class HealpyRegridder:
    """regrid the given dataset to a healpix grid

    Parameters
    ----------
    source_grid : xr.Dataset
        The source dataset. Has to have ``"latitude"`` and ``"longitude"`` coordinates.
    target_grid : xr.Dataset
        The target grid. Has to have ``"latitude"`` and ``"longitude"`` coordinates.
    method : str, default: "bilinear"
        The interpolation method. For now, only bilinear exists.
    interpolation_kwargs : dict, optional
        Additional parameters for the interpolation method.

    Warnings
    --------
    At the moment, none of the interpolation methods can deal with the nature of spherical
    coordinates on the plane. This means that global interpolation will fail in regions
    close to the poles and the ante-meridian. For regional interpolation make sure that
    the ante-meridian is far from the interpolation domain (for example by choosing the
    coordinate range – 0° to 360° or -180° to 180° – appropriately). Regions close to the
    poles will still fail to interpolate.
    """

    input_grid: xr.Dataset = field(repr=False)
    output_grid: xr.Dataset = field(repr=False)
    method: str = "bilinear"
    interpolation_kwargs: dict = field(default_factory=dict)

    weights_path: str | os.PathLike | None = field(default=None, repr=False)

    weights: xr.Dataset = field(init=False, repr=False)

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
