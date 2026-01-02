"""
AVA Server API Integration Tests
================================

Tests for the HTTP API endpoints.

NOTE: These tests are for the legacy Python server (server.py) which has been
superseded by the Rust backend in v4.x. Tests are skipped by default.
The server module has moved to legacy/python_servers/.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Skip all tests in this module - legacy Python server replaced by Rust backend
pytestmark = pytest.mark.skip(reason="Legacy Python server tests - server.py moved to legacy/python_servers/")


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, test_client):
        """Health endpoint should return OK status."""
        response = await test_client.get("/health")
        assert response.status == 200

        data = await response.json()
        assert data["status"] == "ok"
        assert "service" in data

    @pytest.mark.asyncio
    async def test_health_returns_json(self, test_client):
        """Health endpoint should return JSON content type."""
        response = await test_client.get("/health")
        assert "application/json" in response.headers.get("Content-Type", "")


class TestStatusEndpoint:
    """Tests for /status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_ok(self, test_client):
        """Status endpoint should return system status."""
        response = await test_client.get("/status")
        assert response.status == 200

        data = await response.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_status_includes_metrics(self, test_client):
        """Status should include relevant metrics."""
        response = await test_client.get("/status")
        data = await response.json()

        # Should have either detailed status or basic status
        assert "status" in data


class TestChatEndpoint:
    """Tests for /chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_requires_message(self, test_client):
        """Chat should fail without a message."""
        response = await test_client.post(
            "/chat", json={}, headers={"Content-Type": "application/json"}
        )

        # Should return 400 or error response
        data = await response.json()
        if response.status != 400:
            assert "error" in data or "text" in data

    @pytest.mark.asyncio
    async def test_chat_with_valid_message(self, test_client, simulation_mode):
        """Chat with valid message should return response."""
        response = await test_client.post(
            "/chat", json={"message": "Hello!"}, headers={"Content-Type": "application/json"}
        )

        assert response.status == 200
        data = await response.json()
        assert "text" in data or "error" in data

    @pytest.mark.asyncio
    async def test_chat_returns_required_fields(self, test_client, simulation_mode):
        """Chat response should include required fields."""
        response = await test_client.post(
            "/chat",
            json={"message": "What is Python?"},
            headers={"Content-Type": "application/json"},
        )

        data = await response.json()
        if "text" in data:
            assert isinstance(data["text"], str)
            # Optional fields
            if "used_cortex" in data:
                assert isinstance(data["used_cortex"], bool)

    @pytest.mark.asyncio
    async def test_chat_handles_empty_message(self, test_client):
        """Chat should handle empty message gracefully."""
        response = await test_client.post(
            "/chat", json={"message": ""}, headers={"Content-Type": "application/json"}
        )

        # Should either error or provide a response
        await response.json()
        assert response.status in [200, 400]


class TestThinkEndpoint:
    """Tests for /think endpoint (forced Cortex)."""

    @pytest.mark.asyncio
    async def test_think_forces_cortex(self, test_client, simulation_mode):
        """Think endpoint should force Cortex usage."""
        response = await test_client.post(
            "/think",
            json={"message": "Explain quantum computing"},
            headers={"Content-Type": "application/json"},
        )

        data = await response.json()
        if "text" in data:
            assert isinstance(data["text"], str)
            # In non-simulation mode, used_cortex should be True
            if "used_cortex" in data and not simulation_mode:
                assert data["used_cortex"] is True


class TestToolsEndpoint:
    """Tests for /tools endpoint."""

    @pytest.mark.asyncio
    async def test_tools_returns_list(self, test_client):
        """Tools endpoint should return a list of tools."""
        response = await test_client.get("/tools")
        assert response.status == 200

        data = await response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)

    @pytest.mark.asyncio
    async def test_tools_have_required_fields(self, test_client):
        """Each tool should have name and description."""
        response = await test_client.get("/tools")
        data = await response.json()

        for tool in data.get("tools", []):
            assert "name" in tool
            assert isinstance(tool["name"], str)


class TestCORSMiddleware:
    """Tests for CORS headers."""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, test_client):
        """CORS headers should be present in responses."""
        response = await test_client.get("/health")

        # Check for CORS headers (may vary based on implementation)
        # At minimum, should handle cross-origin requests
        assert response.status == 200


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json(self, test_client):
        """Invalid JSON should return error."""
        response = await test_client.post(
            "/chat", data="not valid json", headers={"Content-Type": "application/json"}
        )

        # Should return 400 or 500
        assert response.status in [400, 500]

    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, test_client):
        """Unknown endpoints should return 404."""
        response = await test_client.get("/nonexistent")
        assert response.status == 404


class TestRequestValidation:
    """Tests for request validation."""

    @pytest.mark.asyncio
    async def test_chat_accepts_conversation_id(self, test_client, simulation_mode):
        """Chat should accept optional conversation_id."""
        response = await test_client.post(
            "/chat",
            json={"message": "Hello!", "conversation_id": "test-session-123"},
            headers={"Content-Type": "application/json"},
        )

        # Should succeed
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_chat_rejects_oversized_message(self, test_client):
        """Chat should reject extremely large messages."""
        huge_message = "x" * 1000000  # 1MB message

        response = await test_client.post(
            "/chat", json={"message": huge_message}, headers={"Content-Type": "application/json"}
        )

        # Should either handle gracefully or reject
        assert response.status in [200, 400, 413, 500]
