import numpy as np
import pandas as pd
import pytest

try:
    import xarray as xr
    import xarray_sql  # noqa: F401

    from lumen.sources.xarray_sql import XArraySource
    pytestmark = pytest.mark.xdist_group("xarray")
except ImportError:
    pytestmark = pytest.mark.skip(reason="xarray or xarray-sql is not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def weather_ds():
    """3-D atmospheric dataset with time, lat, lon and a data variable."""
    times = pd.date_range("2020-01-01", periods=5, freq="6h")
    lats = np.array([10.0, 20.0, 30.0])
    lons = np.array([100.0, 110.0])
    data = np.arange(5 * 3 * 2, dtype=float).reshape(5, 3, 2)
    return xr.Dataset(
        {"air": (("time", "lat", "lon"), data)},
        coords={
            "time": times,
            "lat": xr.DataArray(lats, dims=["lat"], attrs={"standard_name": "latitude", "units": "degrees_north"}),
            "lon": xr.DataArray(lons, dims=["lon"], attrs={"standard_name": "longitude", "units": "degrees_east"}),
        },
    )


@pytest.fixture
def depth_ds():
    """Dataset with NO time/lat/lon — numeric depth + species_id dimensions.

    String coordinates are not supported by xarray-sql (it calls .min() on
    all coords internally), so species is represented as integer IDs.
    """
    depths = np.array([0.0, 10.0, 50.0, 200.0])
    species_ids = np.array([1, 2, 3])
    data = np.random.default_rng(0).integers(0, 100, size=(4, 3)).astype(float)
    return xr.Dataset(
        {"count": (("depth", "species"), data)},
        coords={"depth": depths, "species": species_ids},
    )


@pytest.fixture
def weather_source(weather_ds):
    src = XArraySource(
        tables={"weather": weather_ds},
        chunks={"time": 2},
    )
    yield src
    src.close()


@pytest.fixture
def depth_source(depth_ds):
    src = XArraySource(
        tables={"ocean": depth_ds},
        chunks={"depth": 2},
    )
    yield src
    src.close()


@pytest.fixture
def multi_source(weather_ds, depth_ds):
    src = XArraySource(
        tables={"weather": weather_ds, "ocean": depth_ds},
        chunks={"time": 2},
    )
    yield src
    src.close()


# ---------------------------------------------------------------------------
# Basic source behaviour
# ---------------------------------------------------------------------------

def test_source_type():
    assert XArraySource.source_type == "xarray"


def test_get_tables_single(weather_source):
    assert weather_source.get_tables() == ["weather"]


def test_get_tables_multiple(multi_source):
    assert set(multi_source.get_tables()) == {"weather", "ocean"}


def test_get_returns_dataframe(weather_source):
    df = weather_source.get("weather")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"air", "time", "lat", "lon"}


def test_get_correct_row_count(weather_source):
    # 5 times × 3 lats × 2 lons = 30 rows
    df = weather_source.get("weather")
    assert len(df) == 30


def test_get_dataset_depth(depth_source):
    # 4 depths × 3 species = 12 rows
    df = depth_source.get("ocean")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert set(df.columns) >= {"count", "depth", "species"}


# ---------------------------------------------------------------------------
# get_schema
# ---------------------------------------------------------------------------

def test_get_schema_columns_weather(weather_source):
    schema = weather_source.get_schema("weather")
    assert "air" in schema
    assert "time" in schema
    assert "lat" in schema
    assert "lon" in schema


def test_get_schema_types(weather_source):
    schema = weather_source.get_schema("weather")
    assert schema["air"]["type"] == "number"
    assert schema["time"]["type"] == "string"
    assert schema["time"]["format"] == "datetime"
    assert schema["lat"]["type"] == "number"
    assert schema["lon"]["type"] == "number"


def test_get_schema_minmax_lat(weather_source):
    schema = weather_source.get_schema("weather")
    assert schema["lat"]["inclusiveMinimum"] == pytest.approx(10.0)
    assert schema["lat"]["inclusiveMaximum"] == pytest.approx(30.0)


def test_get_schema_minmax_lon(weather_source):
    schema = weather_source.get_schema("weather")
    assert schema["lon"]["inclusiveMinimum"] == pytest.approx(100.0)
    assert schema["lon"]["inclusiveMaximum"] == pytest.approx(110.0)


def test_get_schema_no_latlon(depth_source):
    """Dataset without lat/lon/time still produces a valid schema."""
    schema = depth_source.get_schema("ocean")
    assert "count" in schema
    assert "depth" in schema
    assert "species" in schema
    assert schema["count"]["type"] == "number"
    assert schema["depth"]["type"] == "number"


def test_coord_schema_string_enum(weather_source):
    """_coord_schema produces an enum for object/string dtype coordinates.

    xarray-sql cannot register datasets with string-dtype coordinates (it
    calls .min() internally), so we test _coord_schema directly here.
    """
    labels = np.array(["north", "south", "east"], dtype=object)
    coord = xr.DataArray(labels, dims=["region"])
    schema = weather_source._coord_schema(coord)
    assert "enum" in schema
    assert set(schema["enum"]) == {"north", "south", "east"}


def test_get_schema_all_tables(multi_source):
    """Calling get_schema() without a table returns schema for every table."""
    schema = multi_source.get_schema()
    assert "weather" in schema
    assert "ocean" in schema


# ---------------------------------------------------------------------------
# execute — raw SQL
# ---------------------------------------------------------------------------

def test_execute_select_all(weather_source):
    df = weather_source.execute('SELECT "air" FROM weather')
    assert "air" in df.columns
    assert len(df) == 30


def test_execute_limit(weather_source):
    df = weather_source.execute('SELECT "air" FROM weather LIMIT 5')
    assert len(df) == 5


def test_execute_aggregation(weather_source):
    df = weather_source.execute('SELECT AVG("air") AS avg_air FROM weather')
    assert "avg_air" in df.columns
    assert len(df) == 1


def test_execute_filter(weather_source):
    df = weather_source.execute('SELECT "lat", "air" FROM weather WHERE "lat" = 10.0')
    assert set(df["lat"].unique()) == {10.0}
    # 5 times × 1 lat × 2 lons = 10 rows
    assert len(df) == 10


# ---------------------------------------------------------------------------
# _strip_schema_prefix — LLM generates SourceName.table references
# ---------------------------------------------------------------------------

def test_strip_schema_prefix_single(weather_source):
    """Schema qualifier is removed before DataFusion sees the query."""
    sql = f'SELECT "air" FROM {weather_source.name}.weather LIMIT 3'
    df = weather_source.execute(sql)
    assert len(df) == 3


def test_strip_schema_prefix_no_schema(weather_source):
    """Query without a schema prefix still works."""
    df = weather_source.execute('SELECT "air" FROM weather LIMIT 2')
    assert len(df) == 2


def test_strip_schema_prefix_catalog(weather_source):
    """Three-part catalog.schema.table is also stripped."""
    sql = f'SELECT "air" FROM datafusion.{weather_source.name}.weather LIMIT 1'
    df = weather_source.execute(sql)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# CF role detection
# ---------------------------------------------------------------------------

def test_cf_role_time(weather_source):
    schema = weather_source.get_schema("weather")
    assert schema["time"].get("format") == "datetime"


def test_cf_role_lat(weather_source):
    """lat coordinate has a cf_role of latitude (detected via CF standard_name)."""
    schema = weather_source.get_schema("weather")
    assert schema["lat"].get("cf_role") == "latitude"


def test_cf_role_lon(weather_source):
    schema = weather_source.get_schema("weather")
    assert schema["lon"].get("cf_role") == "longitude"


def test_no_cf_role_for_depth(depth_source):
    """depth and species are numeric coordinates with no lat/lon/time CF role."""
    schema = depth_source.get_schema("ocean")
    assert schema["depth"].get("cf_role") is None
    assert schema["species"].get("cf_role") is None


# ---------------------------------------------------------------------------
# Metadata (_get_table_metadata)
# ---------------------------------------------------------------------------

def test_metadata_has_spatial_bounds(weather_source):
    meta = weather_source._get_table_metadata(["weather"])
    bounds = meta["weather"].get("spatial_bounds")
    assert bounds is not None
    assert bounds["lat_min"] == pytest.approx(10.0)
    assert bounds["lat_max"] == pytest.approx(30.0)
    assert bounds["lon_min"] == pytest.approx(100.0)
    assert bounds["lon_max"] == pytest.approx(110.0)


def test_metadata_has_temporal_range(weather_source):
    meta = weather_source._get_table_metadata(["weather"])
    tr = meta["weather"].get("temporal_range")
    assert tr is not None
    assert tr["steps"] == 5
    assert "start" in tr
    assert "end" in tr
    assert "resolution" in tr


def test_metadata_no_spatial_bounds_for_depth(depth_source):
    """Depth dataset has no lat/lon so spatial_bounds should be absent."""
    meta = depth_source._get_table_metadata(["ocean"])
    assert "spatial_bounds" not in meta["ocean"]


def test_metadata_no_temporal_range_for_depth(depth_source):
    """Depth dataset has no time coordinate so temporal_range should be absent."""
    meta = depth_source._get_table_metadata(["ocean"])
    assert "temporal_range" not in meta["ocean"]


def test_metadata_row_count(weather_source):
    meta = weather_source._get_table_metadata(["weather"])
    assert meta["weather"]["rows"] == 30


def test_metadata_dimensions(weather_source):
    meta = weather_source._get_table_metadata(["weather"])
    dims = meta["weather"]["dimensions"]
    assert dims == {"time": 5, "lat": 3, "lon": 2}


def test_metadata_chunk_info(weather_source):
    meta = weather_source._get_table_metadata(["weather"])
    chunk_info = meta["weather"].get("chunk_info")
    assert chunk_info is not None
    assert "time" in chunk_info
    assert "_total_size_bytes" in chunk_info
    assert "_total_size_human" in chunk_info


# ---------------------------------------------------------------------------
# create_sql_expr_source
# ---------------------------------------------------------------------------

def test_create_sql_expr_source_tables(weather_source):
    """create_sql_expr_source propagates all original tables."""
    derived = weather_source.create_sql_expr_source(
        {"weather": "weather", "avg_air": 'SELECT AVG("air") AS avg_air FROM weather'}
    )
    assert "weather" in derived.get_tables()


def test_create_sql_expr_source_execute(weather_source):
    """Derived SQL expression tables can be executed."""
    derived = weather_source.create_sql_expr_source(
        {"weather": "weather", "avg_air": 'SELECT AVG("air") AS avg_air FROM weather'}
    )
    df = derived.get("avg_air")
    assert "avg_air" in df.columns
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Dataset with time dimension — chunking behaviour
# ---------------------------------------------------------------------------

def test_chunked_dataset_is_dask(weather_ds):
    """Dataset opened with chunks should be a Dask-backed xarray Dataset."""
    src = XArraySource(tables={"weather": weather_ds}, chunks={"time": 2})
    ds = src.get_dataset("weather")
    assert ds.chunks, "Expected a chunked (Dask) dataset"
    src.close()


def test_no_time_dim_chunks_filtered():
    """If chunks={'time':24} but dataset has no time dim, unknown keys are
    filtered out and loading still works without raising."""
    ds = xr.Dataset(
        {"value": (("x", "y"), np.ones((4, 4)))},
        coords={"x": np.arange(4), "y": np.arange(4)},
    )
    # Should not raise even though 'time' chunk key doesn't exist in this dataset
    src = XArraySource(tables={"grid": ds}, chunks={"time": 24})
    df = src.get("grid")
    assert len(df) == 16
    src.close()


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------

def test_get_dataset_returns_xarray(weather_source):
    ds = weather_source.get_dataset("weather")
    assert isinstance(ds, xr.Dataset)


def test_get_dataset_missing_table_raises(weather_source):
    with pytest.raises(KeyError, match="not found"):
        weather_source.get_dataset("nonexistent")


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

def test_close_clears_datasets(weather_source):
    weather_source.close()
    assert weather_source._datasets == {}
