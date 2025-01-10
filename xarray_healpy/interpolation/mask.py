import numpy as np


def mask_weights(weights, indices, mask, min_vertices=3):
    """mask weights

    Parameters
    ----------
    weights : array-like
        The raw weights from any of the interpolation methods
    indices : array-like
        The indices of the surrounding vertices into the source grid
    mask : array-like
        The mask to apply. Has to be ``True`` for data pixels and ``False`` for missing values.
    """
    neighbors_masked = np.asarray(mask)[indices]
    n_non_missing = np.sum(neighbors_masked, axis=-1)
    masked_weights = np.where(
        neighbors_masked & (n_non_missing[:, None] >= min_vertices), weights, 0
    )

    return masked_weights / np.sum(masked_weights, axis=-1)[:, None]
