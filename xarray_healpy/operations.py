import healpy as hp
import numpy as np
import xarray as xr


def buffer_points(
    cell_ids,
    positions,
    *,
    buffer_size,
    nside,
    sphere_radius=6371e3,
    factor=4,
    intersect=False,
):
    """select the cells within a circular buffer around the given positions

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids within the given grid.
    positions : xarray.DataArray
        The positions of the points in cartesian coordinates.
    buffer_size : float
        The size of the circular buffer.
    nside : int
        The resolution of the healpix grid.
    sphere_radius : float, default: 6371000
        The radius of the underlying sphere, used to convert `radius` to radians. By
        default, this is the standard earth's radius in meters.
    factor : int, default: 4
        The increased resolution for the buffer search.
    intersect : bool, default: False
        If `False`, select all cells where the center is within the buffer. If `True`,
        select cells which intersect the buffer.

    Returns
    -------
    masks : xarray.DataArray
        The masks for each position. The cells within the buffer are `True`, every other
        cell is set to `False`.

    See Also
    --------
    pangeo_fish.healpy.geographic_to_astronomic
    pangeo_fish.healpy.astronomic_to_cartesian
    """

    def _buffer_masks(cell_ids, vector, nside, radius, factor=4, intersect=False):
        selected_cells = hp.query_disc(
            nside, vector, radius, nest=True, fact=factor, inclusive=intersect
        )
        return np.isin(cell_ids, selected_cells, assume_unique=True)

    radius_ = buffer_size / sphere_radius

    masks = xr.apply_ufunc(
        _buffer_masks,
        cell_ids,
        positions,
        input_core_dims=[["x", "y"], ["cartesian"]],
        kwargs={
            "radius": radius_,
            "nside": nside,
            "factor": factor,
            "intersect": intersect,
        },
        output_core_dims=[["x", "y"]],
        vectorize=True,
    )

    return masks.assign_coords(cell_ids=cell_ids)
