"""
Pytest configuration and fixtures for Carbon Tracker tests
"""
import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, Generator
import os
import tempfile
import json

# Test configuration
TEST_CONFIG = {
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "topic": "test-carbon-emissions"
    },
    "influxdb": {
        "url": "http://localhost:8086",
        "token": "test-token",
        "org": "test-org",
        "bucket": "test-bucket"
    },
    "redis": {
        "url": "redis://localhost:6379/1"  # Use DB 1 for tests
    },
    "postgres": {
        "url": "postgresql://test_user:test_password@localhost:5432/test_db"
    }
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG

@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing."""
    producer = MagicMock()
    producer.send = AsyncMock()
    producer.flush = AsyncMock()
    producer.close = AsyncMock()
    return producer

@pytest.fixture
def mock_influxdb_client():
    """Mock InfluxDB client for testing."""
    client = MagicMock()
    write_api = MagicMock()
    query_api = MagicMock()
    
    client.write_api.return_value = write_api
    client.query_api.return_value = query_api
    
    write_api.write = AsyncMock()
    query_api.query = AsyncMock()
    
    return client

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.delete = AsyncMock()
    client.exists = AsyncMock()
    client.expire = AsyncMock()
    return client

@pytest.fixture
def sample_carbon_data():
    """Sample carbon emission data for testing."""
    return {
        "device_id": "test-device-001",
        "timestamp": "2024-01-15T10:30:00Z",
        "carbon_emission": 1.25,
        "energy_consumption": 5.0,
        "emission_factor": 0.25,
        "data_source": "smart_meter",
        "location": "Building A, Floor 2",
        "device_type": "electricity_meter"
    }

@pytest.fixture
def sample_device_data():
    """Sample device registration data for testing."""
    return {
        "device_id": "test-device-001",
        "device_name": "Smart Meter - Building A",
        "device_type": "electricity_meter",
        "location": "Building A, Floor 2",
        "metadata": {
            "manufacturer": "SmartTech",
            "model": "ST-2024",
            "firmware_version": "1.2.3"
        }
    }

@pytest.fixture
def sample_ml_features():
    """Sample ML features for testing."""
    return {
        "hour_of_day": 10,
        "day_of_week": 1,
        "month": 1,
        "temperature": 22.5,
        "humidity": 65.0,
        "occupancy": 85,
        "historical_avg": 1.1,
        "weather_condition": "sunny"
    }

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_model_registry():
    """Mock ML model registry for testing."""
    registry = MagicMock()
    registry.get_model = MagicMock()
    registry.register_model = MagicMock()
    registry.list_models = MagicMock()
    return registry

@pytest.fixture
def sample_prediction_result():
    """Sample ML prediction result for testing."""
    return {
        "prediction_id": "pred-001",
        "device_id": "test-device-001",
        "predicted_value": 1.35,
        "confidence_score": 0.92,
        "features": {
            "hour_of_day": 10,
            "day_of_week": 1,
            "temperature": 22.5
        },
        "model_version": "v1.0.0"
    }

@pytest.fixture
def mock_api_client():
    """Mock API client for testing external services."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client

@pytest.fixture
def sample_alert_data():
    """Sample alert data for testing."""
    return {
        "alert_id": "alert-001",
        "device_id": "test-device-001",
        "alert_type": "high_emission",
        "severity": "high",
        "message": "Carbon emission exceeded threshold",
        "threshold_value": 2.0,
        "actual_value": 2.5,
        "triggered_at": "2024-01-15T10:30:00Z"
    }

@pytest.fixture
def mock_notification_service():
    """Mock notification service for testing."""
    service = AsyncMock()
    service.send_email = AsyncMock()
    service.send_sms = AsyncMock()
    service.send_webhook = AsyncMock()
    return service

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "KAFKA_BOOTSTRAP_SERVERS": TEST_CONFIG["kafka"]["bootstrap_servers"],
        "INFLUXDB_URL": TEST_CONFIG["influxdb"]["url"],
        "INFLUXDB_TOKEN": TEST_CONFIG["influxdb"]["token"],
        "REDIS_URL": TEST_CONFIG["redis"]["url"],
        "POSTGRES_URL": TEST_CONFIG["postgres"]["url"]
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

@pytest.fixture
def mock_flink_environment():
    """Mock Flink environment for stream processing tests."""
    env = MagicMock()
    env.add_source = MagicMock()
    env.map = MagicMock()
    env.filter = MagicMock()
    env.add_sink = MagicMock()
    env.execute = MagicMock()
    return env

@pytest.fixture
def sample_time_series_data():
    """Sample time series data for testing aggregations."""
    return [
        {"timestamp": "2024-01-15T10:00:00Z", "value": 1.2},
        {"timestamp": "2024-01-15T10:15:00Z", "value": 1.5},
        {"timestamp": "2024-01-15T10:30:00Z", "value": 1.1},
        {"timestamp": "2024-01-15T10:45:00Z", "value": 1.8},
        {"timestamp": "2024-01-15T11:00:00Z", "value": 1.3}
    ]

@pytest.fixture
def mock_database_session():
    """Mock database session for testing."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session

# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Generate performance test data."""
    return {
        "concurrent_users": 100,
        "test_duration": "30s",
        "ramp_up_time": "10s",
        "target_rps": 1000,
        "endpoints": [
            "/api/carbon/readings",
            "/api/carbon/devices",
            "/api/carbon/predictions"
        ]
    }

# Integration test fixtures
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "test_timeout": 30,
        "retry_attempts": 3,
        "cleanup_after_test": True
    }

# Async test helpers
@pytest_asyncio.fixture
async def async_test_client():
    """Async test client for API testing."""
    from fastapi.testclient import TestClient
    # This would be replaced with actual async client setup
    yield None

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup test data after each test."""
    yield
    # Cleanup logic would go here
    # e.g., clear test database, reset mocks, etc.

# Parametrized fixtures for different test scenarios
@pytest.fixture(params=["electricity", "gas", "fuel", "transport"])
def emission_source_type(request):
    """Parametrized fixture for different emission source types."""
    return request.param

@pytest.fixture(params=["hour", "day", "week", "month"])
def aggregation_period(request):
    """Parametrized fixture for different aggregation periods."""
    return request.param

@pytest.fixture(params=["low", "medium", "high", "critical"])
def alert_severity(request):
    """Parametrized fixture for different alert severity levels."""
    return request.param