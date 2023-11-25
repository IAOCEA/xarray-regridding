import itertools

import numba
import numpy as np
import sparse
import xarray as xr

# TODO: replace with a scipy KDTree
from sklearn.neighbors import BallTree

from xarray_healpy.interpolation.mask import mask_weights


@numba.njit
def norm(array, axis=-1):
    """axis-aware version of numpy.linalg.norm

    Todo: figure out how to replace this with numpy.linalg.norm or any other numpy function
    """
    return np.sqrt(np.sum(array**2, axis=axis))


@numba.njit
def _compute_bilinear_interpolation_weights(
    target_coords, source_coords, neighbor_indices
):
    """compute bilinear interpolation weights for target point

    Parameters
    ----------
    target_coords : array-like
        Source coordinates of the target points. Has to have a shape of ``(n_target_points, 2)``
    source_coords : array-like
        Source coordinates of the source points. Has to have a shape of ``(n_source_points, 2)``
    neighbor_indices : array-like
        Indices of neighbors in the source grid for each target grid. Has to have a shape
        of ``(n_target_points, n_neighbors)``, where the number of neighbors has to be at
        least 4 (better 6).

    Returns
    -------
    weights : array-like
        The computed weights for each of the four surrounding vertices of each target point.
    surrounding_vertex_indices : array-like
        The indices of the surrounding vertices determined from the neighbors.

    Warnings
    --------
    Note that the interpolation function doesn't support curvilinear source grids yet.
    """
    # todo: use `guvectorize` instead of `njit` and the manual loop, as that would allow
    # parallelizing the loop
    n_points = target_coords.shape[0]
    weights = np.zeros((n_points, 4), dtype=source_coords.dtype)
    cell_indices = np.zeros((n_points, 4), dtype=neighbor_indices.dtype)
    for index in range(n_points):
        # unpack
        P = target_coords[index, :]
        current_indices = neighbor_indices[index, :]
        neighbors = source_coords[current_indices, :]

        # create local index array
        cell_vertex_indices = np.zeros(4, dtype="int64")

        # A is always the closest neighbor
        index_A = 0
        cell_vertex_indices[0] = index_A
        A = neighbors[index_A, :]

        # B is always the second-closest neighbor
        index_B = 1
        cell_vertex_indices[1] = index_B
        B = neighbors[index_B, :]

        # determine C
        edges = neighbors - A
        edge_lengths = norm(edges)

        AB = edges[index_B, :] / edge_lengths[index_B]

        dot_product = edges @ AB / edge_lengths
        non_parallel_edge_indices = np.argwhere(np.abs(dot_product) < 0.99)

        index_C = non_parallel_edge_indices[0].item()
        cell_vertex_indices[2] = index_C
        C = neighbors[index_C, :]

        # determine D
        BC = C - B

        mid_point = B + BC / 2
        distances_mid = norm(neighbors - mid_point)
        distances_mid[cell_vertex_indices[:3]] = np.inf  # mask the already taken points

        index_D = np.argmin(distances_mid)
        cell_vertex_indices[3] = index_D

        # construct vectors
        AP = P - A
        AB = edges[index_B, :]
        AC = edges[index_C, :]

        # compute vector lengths
        lAB = edge_lengths[index_B]
        lAC = edge_lengths[index_C]

        # compute orthogonal distances
        s1 = abs(AB @ AP) / lAB
        s2 = abs(AC @ AP) / lAC

        s3 = lAB - s1
        s4 = lAC - s2

        # compute areas
        full_area = lAB * lAC
        # s1*s2 → D
        # s2*s3 → C
        # s1*s4 → B
        # s3*s4 → A
        small_areas = np.array([s3 * s4, s1 * s4, s2 * s3, s1 * s2])

        # weights
        current_weights = small_areas / full_area

        weights[index, :] = current_weights
        cell_indices[index, :] = current_indices[cell_vertex_indices]

    return weights, cell_indices


def determine_stack_dims(grid, variables):
    all_dims = (tuple(grid[var].dims) for var in variables)

    return tuple(dict.fromkeys(itertools.chain.from_iterable(all_dims)))


def prepare_coords(grid, coords, stacked_dim, stacked_dims):
    stacked = grid[coords].stack({stacked_dim: stacked_dims})

    return np.stack([stacked[coord].data for coord in coords], axis=-1)


def bilinear_interpolation_weights(
    source_grid,
    target_grid,
    *,
    n_neighbors=6,
    metric="euclidean",
    coords=["longitude", "latitude"],
    mask=None,
    min_vertices=3,
):
    """xarray-aware bilinear interpolation weights computation

    Parameters
    ----------
    source_grid : xarray.Dataset
        The source grid. Has to have the coordinates specified by ``coords``.
    target_grid : xarray.Dataset
        The target grid. Has to have the coordinates specified by ``coords``.
    n_neighbors : int, default: 6
        How many neighbors the to search for each target point. Minimum is 4, but with 6
        the surrounding vertices are found more accurately.
    metric : str, default: "euclidean"
        The metric to use when find the nearest neighbors. Look at the value of
        ``BallTree.valid_metrics`` for the full list of metrics. Note that choosing
        metrics other than ``"euclidean"`` affects the neighbors search, but so far the
        weights computation itself happens in a euclidean space.
    coords : list of str, default: ["longitude", "latitude"]
        The names of the spatial coordinates in both the source and target grids.
    mask : str or xarray.DataArray, optional
        If given, set the weight of input grid cells where the mask is ``False`` to 0. If
        a ``str``, the variable of that name will be pulled from the source grid. If a
        :py:class:`xarray.DataArray`, has to have the same dimensions as the source grid.

    Returns
    -------
    weights : xarray.DataArray
        The computed weights as a sparse matrix.
    """
    # TODO: how do we detect the variables and dims to stack?
    # For now, just use the coords directly
    source_stacked_dim = "_source_cells"
    target_stacked_dim = "_target_cells"

    # prepare the grids
    source_stack_dims = determine_stack_dims(source_grid, coords)
    source_coords = prepare_coords(
        source_grid, coords, source_stacked_dim, source_stack_dims
    )

    target_stack_dims = determine_stack_dims(target_grid, coords)
    target_coords = prepare_coords(
        target_grid, coords, target_stacked_dim, target_stack_dims
    )

    common_dtype = np.common_type(source_coords, target_coords)

    # use a tree index to find the n closest neighbors
    tree = BallTree(source_coords, metric=metric)
    _, neighbor_indices = tree.query(target_coords, k=n_neighbors)

    # compute the weights
    raw_weights, surrounding_vertex_indices = _compute_bilinear_interpolation_weights(
        target_coords.astype(common_dtype),
        source_coords.astype(common_dtype),
        neighbor_indices,
    )

    if mask is not None:
        if isinstance(mask, str):
            mask_ = source_grid[mask]
        else:
            mask_ = mask

        stacked_mask = mask_.stack({source_stacked_dim: source_stack_dims})

        raw_weights = mask_weights(
            raw_weights,
            surrounding_vertex_indices,
            stacked_mask.data,
            min_vertices=min_vertices,
        )

    # arrange as a sparse matrix
    n_target = target_coords.shape[0]
    n_source = source_coords.shape[0]

    target_indices = np.broadcast_to(
        np.arange(n_target)[:, None], surrounding_vertex_indices.shape
    )
    sparse_coords = np.stack([target_indices, surrounding_vertex_indices], axis=0)

    source_shape = tuple([source_grid.sizes[dim] for dim in source_stack_dims])
    target_shape = tuple([target_grid.sizes[dim] for dim in target_stack_dims])

    reshaped_sparse_coords = np.reshape(sparse_coords, (2, -1))
    reshaped_raw_weights = np.reshape(raw_weights, -1)

    if mask is not None:
        not_ignored = np.reshape(np.argwhere(reshaped_raw_weights != 0.0), -1)

        coords = reshaped_sparse_coords[:, not_ignored]
        data = reshaped_raw_weights[not_ignored]
    else:
        coords = reshaped_sparse_coords
        data = reshaped_raw_weights

    raw_weights_matrix = sparse.COO(
        coords=coords, data=data, shape=(n_target, n_source), fill_value=0.0
    )
    weights_matrix = np.reshape(raw_weights_matrix, target_shape + source_shape)

    # put into a DataArray
    weights = xr.DataArray(
        weights_matrix,
        dims=target_stack_dims + source_stack_dims,
        coords=target_grid.coords,
        attrs={"sum_dims": source_stack_dims},
    )

    return weights
