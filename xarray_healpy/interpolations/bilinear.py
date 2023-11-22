import numba
import numpy as np


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
