"""
AVA Test Configuration
======================

Shared pytest fixtures and configuration for all tests.
"""

import asyncio
import os
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src and legacy to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "legacy"))


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


# ============================================================================
# Event Loop Fixture
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def simulation_mode():
    """Enable simulation mode for testing."""
    original = os.environ.get("AVA_SIMULATION_MODE")
    os.environ["AVA_SIMULATION_MODE"] = "true"
    yield
    if original is None:
        del os.environ["AVA_SIMULATION_MODE"]
    else:
        os.environ["AVA_SIMULATION_MODE"] = original


@pytest.fixture
def test_config_path(tmp_path: Path) -> Path:
    """Create a temporary test config file."""
    config_content = """
development:
  simulation_mode: true

medulla:
  low_surprise_threshold: 0.3
  high_surprise_threshold: 0.7
  ollama_url: "http://localhost:11434"
  model_name: "gemma3:4b"

cortex:
  enabled: false

thermal:
  enabled: false
  max_gpu_power_percent: 15

search_first:
  enabled: false
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing without actual LLM."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a mock response from Ollama.",
            "done": True,
            "model": "gemma3:4b",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None

        mock_client.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def mock_thermal_monitor():
    """Mock thermal monitor for testing without GPU."""
    mock = MagicMock()
    mock.get_status.return_value = MagicMock(
        temperature=45.0,
        power_draw_watts=10.0,
        power_percent=10.0,
        is_throttled=False,
    )
    mock.should_throttle.return_value = False
    mock.should_pause.return_value = False
    mock.is_power_exceeded.return_value = False
    return mock


# ============================================================================
# AVA Instance Fixtures
# ============================================================================


@pytest.fixture
async def ava_instance(simulation_mode, mock_ollama):
    """Create an AVA instance in simulation mode for testing."""
    from ava import AVA

    ava = AVA()
    await ava.start()
    yield ava
    await ava.stop()


@pytest.fixture
async def ava_engine(simulation_mode, mock_ollama):
    """Create an AVA engine instance for testing."""
    from ava.engine import AVAEngine, EngineConfig

    config = EngineConfig(simulation_mode=True)
    engine = AVAEngine(config)
    await engine.start()
    yield engine
    await engine.stop()


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample messages for testing chat functionality."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is Python?"},
    ]


@pytest.fixture
def sample_queries() -> list[str]:
    """Sample queries for testing different routing behaviors."""
    return [
        "Hello",  # Simple greeting - Medulla
        "What is 2+2?",  # Simple math - Medulla
        "Explain quantum mechanics in detail",  # Complex - Cortex
        "What is the capital of France?",  # Factual - may trigger search
    ]


# ============================================================================
# Server Fixtures
# ============================================================================


@pytest.fixture
async def test_server(simulation_mode, mock_ollama, tmp_path):
    """Start a test server instance."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    # Import server routes
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import server

    # Create app
    app = web.Application()
    app.router.add_get("/health", server.health_handler)
    app.router.add_get("/status", server.status_handler)
    app.router.add_post("/chat", server.chat_handler)
    app.router.add_post("/think", server.think_handler)
    app.router.add_get("/tools", server.tools_handler)

    test_server = TestServer(app)
    await test_server.start_server()

    yield test_server

    await test_server.close()


@pytest.fixture
async def test_client(test_server):
    """Create a test client for the server."""
    from aiohttp.test_utils import TestClient

    client = TestClient(test_server)
    await client.start_server()
    yield client
    await client.close()


# ============================================================================
# Utility Functions
# ============================================================================


def assert_response_valid(response: dict):
    """Assert that a chat response has required fields."""
    assert "text" in response, "Response must have 'text' field"
    assert isinstance(response["text"], str), "'text' must be a string"
    assert len(response["text"]) > 0, "'text' must not be empty"


def assert_error_response(response: dict):
    """Assert that an error response has required fields."""
    assert "error" in response or "message" in response, "Error response must have error field"
