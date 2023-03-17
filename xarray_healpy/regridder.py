import attrs
import numpy as np
import dask

from . import _healpy
import xarray as xr
import cf_xarray  # noqa: F401


def _compute_indices(nside, sector):
    xx, yy = _healpy._compute_indices(nside)

    return xr.Dataset(coords={
        "cell_x": ("cell_id", xx),
        "cell_y": ("cell_id", yy),
    })


def _compute_coords(nside, sector, cell_indices):
    lat, lon = xr.apply_ufunc(
        _healpy._compute_coords,
        nside,
        sector,
        cell_indices["cell_x"],
        cell_indices["cell_y"],
        input_core_dims=[[], [], ["cell_id"], ["cell_id"]],
        output_core_dims=[["x", "y"], ["x", "y"]],
    )
    coords = xr.Dataset(coords={"lat": lat, "lon": lon})

    return coords


def _fit_weights(nside, lat, lon, rotation):
    axis_order = ["X", "Y"]
    dims = {axis: lat.cf.axes[axis][0] for axis in axis_order}

    order = list(dims.values())

    # compute all dask arrays at once
    lat_, lon_ = dask.compute(lat, lon)

    pix, weights = xr.apply_ufunc(
        _healpy._fit_weights,
        nside,
        lat_,
        lon_,
        kwargs={"rot": rotation},
        dask="forbidden",
        input_core_dims=[[], order, order],
        output_core_dims=[["neighbor", "source_cell"], ["neighbor", "source_cell"]],
    )

    return pix, weights


def create_grid(nside, sector):
    cell_indices = _compute_indices(nside, sector)
    coords = _compute_coords(nside, sector, cell_indices)

    return Grid(nside=nside, sector=sector, coords=coords, indices=cell_indices)


@attrs.define
class Grid:
    nside: int = attrs.field()

    # todo: allow multiple sectors
    sector: int = attrs.field(validator=attrs.validators.in_(range(12)))

    indices: xr.Dataset = attrs.field(repr=False)
    coords: xr.Dataset = attrs.field(repr=False)

    def fit_weights(self, obj, rotation={"lat": 0, "lon": 0}):
        if isinstance(obj, xr.DataArray):
            if obj.name is None:
                obj = obj.rename("<temporary name>")
            obj = obj.to_dataset()

        source_grid = obj.reset_coords().cf[["latitude", "longitude"]].compute()
        
        pix, weights = _fit_weights(self.nside, source_grid.cf["latitude"], source_grid.cf["longitude"], rotation=rotation)

        return Regridder(grid=self, pixels=pix, weights=weights, source_grid=source_grid.reset_coords())


@attrs.define
class Regridder:
    grid: Grid = attrs.field(repr=False)
    source_grid: xr.Dataset = attrs.field(repr=False)

    pixels = attrs.field(repr=False)
    weights = attrs.field(repr=False)

    sector: int = attrs.field(init=False)
    nside: int = attrs.field(init=False)
    source: dict[str, tuple[float]] = attrs.field(init=False, factory=dict)


    def __attrs_post_init__(self):
        self.sector = self.grid.sector
        self.nside = self.grid.nside

        min_ = self.source_grid.min()
        max_ = self.source_grid.max()

        self.source = {
            "latitude": [min_.latitude.item(), max_.latitude.item()],
            "longitude": [min_.longitude.item(), max_.longitude.item()],
        }
