import dask
import healpy as hp
import numpy as np
import xarray as xr


def geographic_to_cartesian(lon, lat, rot, dim=None):
    lon_ = lon - rot["lon"]
    lat_ = lat - rot["lat"]

    if dim is None:
        dims = list(lon.dims)
    elif isinstance(dim, str):
        dims = [dim]
    else:
        dims = dim

    return xr.apply_ufunc(
        hp.ang2vec,
        lon_,
        lat_,
        input_core_dims=[dims, dims],
        output_core_dims=[[*dims, "cartesian"]],
        kwargs={"lonlat": True},
    )


def geographic_to_astronomic(lat, lon, rot):
    """transform geographic coordinates to astronomic coordinates

    Parameters
    ----------
    lat : array-like
        geographic latitude, in degrees
    lon : array-like
        geographic longitude, in degrees
    rot : list-like
        Two element list with the rotation transformation (shift?) used by the grid, in
        degrees

    Returns
    -------
    theta : array-like
        Colatitude in degrees
    phi : array-like
        Astronomic longitude in degrees
    """
    theta = 90.0 - lat - rot["lat"]
    phi = -lon + rot["lon"]

    return theta, phi


def astronomic_to_cartesian(theta, phi, dim="receiver_id"):
    """transform astronomic coordinates to cartesian coordinates

    Parameters
    ----------
    theta : DataArray
        astronomic colatitude, in degrees
    phi : DataArray
        astronomic longitude, in degrees
    dim : hashable
        Name of the dimension

    Returns
    -------
    cartesian : Dataset
        Cartesian coordinates

    See Also
    --------
    healpy.ang2vec
    """
    # TODO: try to determine `dim` automatically
    cartesian = xr.apply_ufunc(
        hp.ang2vec,
        np.deg2rad(theta),
        np.deg2rad(phi),
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim, "cartesian"]],
    )

    return cartesian.assign_coords(cartesian=["x", "y", "z"])


def astronomic_to_cell_ids(nside, phi, theta):
    """Compute cell ids from astronomic coordinates

    Parameters
    ----------
    nside : int
        Healpix resolution level
    phi, theta : xr.DataArray
        astronomic longitude and colatitude, in degrees

    Returns
    -------
    cell_ids : xr.DataArray
        The computed cell ids
    """
    phi_, theta_ = dask.compute(phi, theta)

    cell_ids = xr.apply_ufunc(
        hp.ang2pix,
        nside,
        np.deg2rad(theta_),
        np.deg2rad(phi_),
        kwargs={"nest": True},
        input_core_dims=[[], ["x", "y"], ["x", "y"]],
        output_core_dims=[["x", "y"]],
    )

    return cell_ids


def base_pixel(level, indices):
    return indices >> (level * 2)
