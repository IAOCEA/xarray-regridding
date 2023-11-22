import numba
import numpy as np


@numba.njit
def norm(array, axis=-1):
    """axis-aware version of numpy.linalg.norm

    Todo: figure out how to replace this with numpy.linalg.norm or any other numpy function
    """
    return np.sqrt(np.sum(array**2, axis=axis))
