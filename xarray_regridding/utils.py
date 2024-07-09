import importlib


def module_available(module: str) -> bool:
    """Checks whether a module is installed without importing it.
    Use this for a lightweight check and lazy imports.
    Parameters
    ----------
    module : str
        Name of the module.
    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    return importlib.util.find_spec(module) is not None


def is_dask_collection(x):
    if module_available("dask"):
        from dask.base import is_dask_collection

        return is_dask_collection(x)
    else:
        return False
