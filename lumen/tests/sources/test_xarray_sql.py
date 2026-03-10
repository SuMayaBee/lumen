"""Tests for XArraySource."""
import numpy as np
import pytest
import xarray as xr

try:
    from lumen.sources.xarray_sql import XArraySource
    XARRAY_SQL_AVAILABLE = True
except ImportError:
    XARRAY_SQL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not XARRAY_SQL_AVAILABLE,
    reason="xarray-sql not available"
)


@pytest.fixture
def sample_dataset():
    """Create a simple xarray dataset for testing."""
    np.random.seed(42)
    return xr.Dataset(
        {
            'temperature': (['time', 'lat', 'lon'], np.random.rand(10, 3, 4) * 30 + 260),
            'pressure': (['time', 'lat', 'lon'], np.random.rand(10, 3, 4) * 100 + 900),
        },
        coords={
            'time': range(10),
            'lat': [10.0, 20.0, 30.0],
            'lon': [100.0, 110.0, 120.0, 130.0],
        },
        attrs={
            'title': 'Test Weather Dataset',
            'description': 'Synthetic data for testing',
        },
    )


@pytest.fixture
def xarray_source(sample_dataset):
    """Create an XArraySource from an in-memory dataset."""
    return XArraySource(
        tables={'weather': sample_dataset},
        chunks={'time': 5},
    )


class TestXArraySourceBasics:

    def test_get_tables(self, xarray_source):
        tables = xarray_source.get_tables()
        assert tables == ['weather']

    def test_get_returns_dataframe(self, xarray_source):
        df = xarray_source.get('weather')
        assert isinstance(df, type(df))  # is a DataFrame
        assert len(df) > 0
        assert 'temperature' in df.columns
        assert 'pressure' in df.columns
        assert 'time' in df.columns
        assert 'lat' in df.columns
        assert 'lon' in df.columns

    def test_get_correct_row_count(self, xarray_source):
        """10 time × 3 lat × 4 lon = 120 rows."""
        df = xarray_source.get('weather')
        assert len(df) == 120

    def test_get_invalid_table(self, xarray_source):
        with pytest.raises(KeyError):
            xarray_source.get('nonexistent')


class TestXArraySourceSchema:

    def test_get_schema_single_table(self, xarray_source):
        schema = xarray_source.get_schema('weather')
        assert 'temperature' in schema
        assert 'pressure' in schema
        assert 'time' in schema
        assert 'lat' in schema
        assert 'lon' in schema

    def test_schema_types(self, xarray_source):
        schema = xarray_source.get_schema('weather')
        # Data variables should be numeric
        assert schema['temperature']['type'] in ('number', 'integer')
        assert schema['pressure']['type'] in ('number', 'integer')

    def test_coord_schema_has_bounds(self, xarray_source):
        schema = xarray_source.get_schema('weather')
        # lat should have inclusiveMinimum/Maximum
        assert 'inclusiveMinimum' in schema['lat']
        assert 'inclusiveMaximum' in schema['lat']
        assert schema['lat']['inclusiveMinimum'] == 10.0
        assert schema['lat']['inclusiveMaximum'] == 30.0


class TestXArraySourceMetadata:

    def test_get_metadata(self, xarray_source):
        metadata = xarray_source.get_metadata('weather')
        assert 'columns' in metadata
        assert 'description' in metadata
        assert metadata['description'] == 'Test Weather Dataset'
        assert 'dimensions' in metadata

    def test_metadata_columns(self, xarray_source):
        metadata = xarray_source.get_metadata('weather')
        columns = metadata['columns']
        assert 'temperature' in columns
        assert columns['temperature']['kind'] == 'data_variable'
        assert 'lat' in columns
        assert columns['lat']['kind'] == 'coordinate'

    def test_metadata_dimensions(self, xarray_source):
        metadata = xarray_source.get_metadata('weather')
        dims = metadata['dimensions']
        assert dims['time'] == 10
        assert dims['lat'] == 3
        assert dims['lon'] == 4


class TestXArraySourceSQL:

    def test_execute_sql(self, xarray_source):
        df = xarray_source.execute("SELECT * FROM weather LIMIT 5")
        assert len(df) == 5

    def test_execute_sql_with_filter(self, xarray_source):
        df = xarray_source.execute("SELECT * FROM weather WHERE lat = 20.0")
        assert len(df) == 40  # 10 time × 1 lat × 4 lon
        assert all(df['lat'] == 20.0)

    def test_execute_sql_aggregation(self, xarray_source):
        df = xarray_source.execute(
            "SELECT lat, AVG(temperature) as avg_temp FROM weather GROUP BY lat"
        )
        assert len(df) == 3  # 3 lat values
        assert 'avg_temp' in df.columns

    def test_sql_transforms(self, xarray_source):
        """Test that SQLTransforms work through the get() interface."""
        from lumen.transforms.sql import SQLLimit
        df = xarray_source.get('weather', sql_transforms=[SQLLimit(limit=7)])
        assert len(df) == 7


class TestXArraySourceFromFile:

    def test_from_netcdf(self, sample_dataset, tmp_path):
        """Test loading from a .nc file."""
        nc_path = str(tmp_path / 'test.nc')
        sample_dataset.to_netcdf(nc_path)

        source = XArraySource(
            tables={'weather': nc_path},
            chunks={'time': 5},
        )
        df = source.get('weather')
        assert len(df) == 120
        assert 'temperature' in df.columns

    def test_from_zarr(self, sample_dataset, tmp_path):
        """Test loading from a .zarr store."""
        zarr_path = str(tmp_path / 'test.zarr')
        sample_dataset.to_zarr(zarr_path)

        source = XArraySource(
            tables={'weather': zarr_path},
            chunks={'time': 5},
        )
        df = source.get('weather')
        assert len(df) == 120

    def test_close(self, xarray_source):
        xarray_source.close()
        assert len(xarray_source._datasets) == 0
