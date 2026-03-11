from __future__ import annotations

import os
import re

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import param

from ..transforms.sql import SQLCount, SQLFilter, SQLLimit, SQLSelectFrom
from ..util import get_dataframe_schema
from .base import BaseSQLSource, Source, cached, cached_schema

if TYPE_CHECKING:
    pass

try:
    import xarray as xr
    from xarray_sql import XarrayContext
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


class XArraySource(BaseSQLSource):
    """
    XArraySource provides a SQL interface to xarray datasets (NetCDF, Zarr, HDF5, GRIB).

    It uses `xarray-sql <https://github.com/alxmrs/xarray-sql>`_ to register
    xarray datasets with a DataFusion-based SQL engine, allowing standard SQL
    queries against multidimensional, labeled scientific data.

    Each table maps to an xarray Dataset loaded from a file path or an in-memory
    dataset. Coordinates become queryable columns alongside data variables.

    Supported formats (auto-detected from file extension):

    - **NetCDF** (``.nc``, ``.nc4``, ``.netcdf``) — engine: ``netcdf4``
    - **Zarr** (``.zarr``) — engine: ``zarr``
    - **HDF5** (``.h5``, ``.hdf5``, ``.he5``) — engine: ``h5netcdf``
    - **GRIB** (``.grib``, ``.grib2``, ``.grb``) — engine: ``cfgrib``
    - **OpenDAP URLs** (``http://...``) — auto-detected by xarray
    - **In-memory** ``xarray.Dataset`` objects

    Parameters
    ----------
    tables : dict
        Dictionary mapping table names to file paths, remote URLs,
        or ``xarray.Dataset`` objects.
    chunks : dict
        Dask chunking to apply when opening datasets. Required for lazy loading
        of large files. Example: ``{'time': 24}``.
    engine : str
        Override the xarray backend engine. If None, auto-detected from
        file extension.

    Examples
    --------
    From file paths:

    >>> source = XArraySource(
    ...     tables={'weather': '/path/to/weather.nc'},
    ...     chunks={'time': 24}
    ... )
    >>> source.get('weather')  # returns a pandas DataFrame

    From an in-memory dataset:

    >>> import xarray as xr
    >>> ds = xr.tutorial.open_dataset('air_temperature')
    >>> source = XArraySource(tables={'air': ds}, chunks={'time': 24})

    From a remote OpenDAP URL:

    >>> source = XArraySource(
    ...     tables={'sst': 'https://opendap.example.com/sst.nc'},
    ... )
    """

    chunks = param.Dict(default={'time': 24}, doc="""
        Dask chunking dimensions to apply when opening datasets.
        Keys are dimension names, values are chunk sizes.
        Example: ``{'time': 24, 'lat': 10}``.""")

    engine = param.String(default=None, allow_None=True, doc="""
        xarray backend engine. One of 'netcdf4', 'h5netcdf', 'scipy',
        'zarr', etc. If None, xarray auto-detects the engine.""")

    filter_in_sql = param.Boolean(default=True, doc="""
        Whether to apply filters in SQL or in-memory.""")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The default SQL expression template for tables.""")

    tables = param.ClassSelector(class_=(dict,), default={}, doc="""
        Dictionary mapping table names to file paths or xarray.Dataset objects.
        Supported formats: .nc/.nc4/.netcdf (NetCDF), .zarr (Zarr),
        .h5/.hdf5/.he5 (HDF5), .grib/.grib2/.grb (GRIB), or OpenDAP URLs.""")

    source_type: ClassVar[str] = 'xarray'

    dialect: str = 'duckdb'

    _supports_sql: ClassVar[bool] = True

    def __init__(self, **params):
        if not XARRAY_AVAILABLE:
            raise ImportError(
                "XArraySource requires 'xarray' and 'xarray-sql'. "
                "Install them with: pip install xarray xarray-sql"
            )
        super().__init__(**params)
        self._ctx = XarrayContext()
        self._datasets: dict[str, xr.Dataset] = {}
        self._sql_expressions: dict[str, str] = {}
        self._register_tables()

    def _register_tables(self):
        """Open datasets from file paths and register them with the SQL context."""
        for table_name, source in self.tables.items():
            ds = self._open_dataset(source)
            self._datasets[table_name] = ds
            # xarray-sql requires chunked datasets
            if not ds.chunks:
                ds = ds.chunk(self.chunks)
            self._ctx.from_dataset(table_name, ds, chunks=self.chunks)

    def _open_dataset(self, source) -> xr.Dataset:
        """Open a dataset from a file path or return an existing Dataset."""
        if isinstance(source, xr.Dataset):
            return source
        # source is a file path string or remote URL
        path = str(source)
        kwargs = {}
        if self.engine:
            kwargs['engine'] = self.engine
        elif path.endswith('.zarr') or path.endswith('.zarr/'):
            kwargs['engine'] = 'zarr'
        elif path.endswith(('.h5', '.hdf5', '.he5')):
            kwargs['engine'] = 'h5netcdf'
        elif path.endswith(('.grib', '.grib2', '.grb')):
            # Import cfgrib to register it as an xarray backend plugin
            try:
                import cfgrib  # noqa: F401
            except ImportError:
                raise ImportError(
                    "cfgrib is required to read GRIB files. "
                    "Install it with: pip install cfgrib or pixi install"
                )
            kwargs['engine'] = 'cfgrib'
        elif path.endswith(('.nc4', '.netcdf')):
            kwargs['engine'] = 'netcdf4'
        # For .nc and remote OpenDAP URLs, let xarray auto-detect

        ds = xr.open_dataset(path, chunks=self.chunks or None, **kwargs)
        return ds

    @property
    def _reload_params(self) -> list[str]:
        return ['tables', 'chunks', 'engine']

    def get_tables(self) -> list[str]:
        tables = [t for t in list(self.tables) if not self._is_table_excluded(t)]
        tables.extend(
            t for t in self._sql_expressions if not self._is_table_excluded(t)
        )
        return tables

    def get_sql_expr(self, table: str | dict) -> str:
        """Returns the SQL expression for a table."""
        if table in self._sql_expressions:
            return self._sql_expressions[table]
        if isinstance(self.tables, dict):
            if table not in self.tables:
                raise KeyError(f"Table {table!r} not found in {list(self.tables.keys())}")
        return SQLSelectFrom(sql_expr=self.sql_expr).apply(table)

    @cached
    def get(self, table: str, **query) -> pd.DataFrame:
        query.pop('__dask', None)
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = list(query.items())

        if self.filter_in_sql and conditions:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms

        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)

        df = self.execute(sql_expr)

        if not self.filter_in_sql and conditions:
            from ..transforms import Filter
            df = Filter.apply_to(df, conditions=conditions)

        return df

    def execute(self, sql_query: str, params: list | dict | None = None, *args, **kwargs) -> pd.DataFrame:
        """Execute a SQL query against the registered xarray datasets."""
        try:
            result = self._ctx.sql(sql_query)
            return result.to_pandas()
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute query on XArraySource: {e}\n"
                f"Query: {sql_query}"
            ) from e

    @cached_schema
    def get_schema(
        self, table: str | None = None, limit: int | None = None,
        shuffle: bool = False
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        schemas = {}
        tables = self.get_tables() if table is None else [table]
        for name in tables:
            if name in self._sql_expressions:
                # For SQL expression tables, infer schema from query result
                try:
                    sql_expr = self._sql_expressions[name]
                    limit_expr = SQLLimit(limit=1).apply(sql_expr)
                    df = self.execute(limit_expr)
                    schema = get_dataframe_schema(df)
                    prop_schema = schema.get('items', {}).get('properties', {})
                except Exception:
                    prop_schema = {}
            else:
                ds = self._datasets[name]
                prop_schema = {}
                # Coordinates (filterable dimensions)
                for coord_name, coord in ds.coords.items():
                    prop_schema[str(coord_name)] = self._coord_schema(coord)
                # Data variables
                for var_name, var in ds.data_vars.items():
                    prop_schema[str(var_name)] = self._var_schema(var)
            schemas[name] = prop_schema
        return schemas if table is None else schemas[table]

    def _coord_schema(self, coord: xr.DataArray) -> dict:
        """Build JSON schema entry for a coordinate."""
        dtype = coord.dtype
        schema: dict[str, Any] = {'type': self._numpy_type_to_json(dtype)}
        try:
            values = coord.values
            if np.issubdtype(dtype, np.datetime64):
                schema['format'] = 'datetime'
                schema['inclusiveMinimum'] = pd.Timestamp(values.min())
                schema['inclusiveMaximum'] = pd.Timestamp(values.max())
            elif np.issubdtype(dtype, np.number):
                schema['inclusiveMinimum'] = float(values.min())
                schema['inclusiveMaximum'] = float(values.max())
            elif np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
                schema['enum'] = sorted(set(str(v) for v in values))
        except Exception:
            pass
        return schema

    def _var_schema(self, var: xr.DataArray) -> dict:
        """Build JSON schema entry for a data variable."""
        return {'type': self._numpy_type_to_json(var.dtype)}

    @staticmethod
    def _numpy_type_to_json(dtype) -> str:
        """Map numpy dtype to JSON schema type string."""
        if np.issubdtype(dtype, np.integer):
            return 'integer'
        elif np.issubdtype(dtype, np.floating):
            return 'number'
        elif np.issubdtype(dtype, np.bool_):
            return 'boolean'
        elif np.issubdtype(dtype, np.datetime64):
            return 'string'
        else:
            return 'string'

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """Generate metadata from xarray dataset attributes and structure."""
        metadata = {}
        for table_name in tables:
            ds = self._datasets[table_name]
            # Build column metadata from coords + data_vars
            columns = {}
            for coord_name, coord in ds.coords.items():
                columns[str(coord_name)] = {
                    'data_type': str(coord.dtype),
                    'description': coord.attrs.get('long_name', coord.attrs.get('standard_name', '')),
                    'units': coord.attrs.get('units', ''),
                    'kind': 'coordinate',
                }
            for var_name, var in ds.data_vars.items():
                columns[str(var_name)] = {
                    'data_type': str(var.dtype),
                    'description': var.attrs.get('long_name', var.attrs.get('standard_name', '')),
                    'units': var.attrs.get('units', ''),
                    'kind': 'data_variable',
                    'dimensions': list(var.dims),
                }

            # Count rows via SQL
            try:
                count_expr = SQLCount().apply(self.get_sql_expr(table_name))
                count = self.execute(count_expr).iloc[0, 0]
            except Exception:
                count = None

            table_metadata = {
                'description': ds.attrs.get('title', ds.attrs.get('description', '')),
                'columns': columns,
                'rows': count,
                'dimensions': dict(ds.dims),
                'global_attrs': dict(ds.attrs),
                'updated_at': None,
                'created_at': None,
            }
            metadata[table_name] = table_metadata
        return metadata

    def create_sql_expr_source(
        self, tables: dict[str, str],
        params: dict[str, list | dict] | None = None,
        **kwargs
    ):
        """
        Creates a new XArraySource with derived SQL expressions.

        Reuses the already-loaded xarray datasets for existing tables and
        stores new SQL expressions (e.g., from AI-generated queries) as
        virtual tables that are executed against the xarray SQL context.
        """
        # Separate real dataset tables from SQL expression tables
        real_tables = {}
        sql_expressions = {}

        for name, value in tables.items():
            if name in self._datasets:
                # Pass the already-loaded Dataset to avoid re-opening files
                real_tables[name] = self._datasets[name]
            else:
                # This is a SQL expression for a virtual table
                sql_expressions[name] = value

        # Include all original tables using their loaded datasets
        all_real_tables = {name: ds for name, ds in self._datasets.items()}
        all_real_tables.update(real_tables)

        source_params = {
            'tables': all_real_tables,
            'chunks': self.chunks,
            'engine': self.engine,
        }
        source_params.update(kwargs)
        source_params.pop('name', None)

        source = type(self)(**source_params)
        source._sql_expressions = sql_expressions
        return source

    def close(self):
        """Close all open datasets."""
        for ds in self._datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        self._datasets.clear()
