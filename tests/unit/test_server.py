"""
Unit Tests for AVA Server
=========================

Tests for the HTTP API server endpoints and handlers.

NOTE: These tests are for the legacy Python server (server.py) which has been
superseded by the Rust backend in v4.x. Tests are skipped by default.
The server module has moved to legacy/python_servers/.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip all tests in this module - legacy Python server replaced by Rust backend
pytestmark = pytest.mark.skip(reason="Legacy Python server tests - server.py moved to legacy/python_servers/")


class TestHealthEndpointUnit:
    """Unit tests for health endpoint handler."""

    @pytest.mark.asyncio
    async def test_health_returns_json_response(self):
        """Health endpoint should return JSON with status ok."""
        from aiohttp import web

        from server import health_handler

        # Create mock request
        request = MagicMock(spec=web.Request)

        response = await health_handler(request)

        assert response.status == 200
        assert response.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_health_response_body(self):
        """Health response should have status field."""
        import json

        from aiohttp import web

        from server import health_handler

        request = MagicMock(spec=web.Request)
        response = await health_handler(request)

        body = json.loads(response.body)
        assert "status" in body
        assert body["status"] == "ok"


class TestStatusEndpointUnit:
    """Unit tests for status endpoint handler."""

    @pytest.mark.asyncio
    async def test_status_returns_json(self):
        """Status endpoint should return JSON."""
        from aiohttp import web

        from server import status_handler

        request = MagicMock(spec=web.Request)

        # Create mock AVA that will be returned by get_ava
        mock_ava = MagicMock()
        mock_ava._engine = MagicMock()
        mock_ava._engine.get_stats.return_value = {
            "total_requests": 10,
            "cortex_requests": 2,
        }

        # Patch get_ava as an async function that returns our mock
        async def mock_get_ava():
            return mock_ava

        with patch("server.get_ava", mock_get_ava):
            response = await status_handler(request)

            assert response.status == 200
            assert response.content_type == "application/json"


class TestChatEndpointUnit:
    """Unit tests for chat endpoint handler."""

    @pytest.mark.asyncio
    async def test_chat_requires_message(self):
        """Chat endpoint should require message field."""
        from aiohttp import web

        from server import chat_handler

        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(return_value={})  # No message

        response = await chat_handler(request)

        # Should return error or handle gracefully
        assert response.status in [200, 400]

    @pytest.mark.asyncio
    async def test_chat_with_valid_message(self):
        """Chat endpoint should process valid messages."""
        import json

        from aiohttp import web

        from server import chat_handler

        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(return_value={"message": "Hello!"})

        with patch("server.get_ava") as mock_get_ava:
            mock_ava = AsyncMock()
            mock_ava.chat.return_value = MagicMock(
                text="Hello! How can I help?",
                used_cortex=False,
                cognitive_state="FLOW",
                confidence=0.9,
                tools_used=[],
            )
            mock_get_ava.return_value = mock_ava

            response = await chat_handler(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert "text" in body or "error" not in body

    @pytest.mark.asyncio
    async def test_chat_handles_empty_message(self):
        """Chat should handle empty message strings."""

        from aiohttp import web

        from server import chat_handler

        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(return_value={"message": ""})

        response = await chat_handler(request)

        # Should return a response (may be error or success)
        assert response.status in [200, 400]


class TestToolsEndpointUnit:
    """Unit tests for tools endpoint handler."""

    @pytest.mark.asyncio
    async def test_tools_returns_list(self):
        """Tools endpoint should return list of tools."""
        import json

        from aiohttp import web

        from server import tools_handler

        request = MagicMock(spec=web.Request)

        # Create mock AVA
        mock_ava = MagicMock()
        mock_ava.list_tools.return_value = [
            {"name": "search", "description": "Web search"},
            {"name": "browse", "description": "Browse URL"},
        ]

        # Patch get_ava as an async function
        async def mock_get_ava():
            return mock_ava

        with patch("server.get_ava", mock_get_ava):
            response = await tools_handler(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert "tools" in body
            assert isinstance(body["tools"], list)


class TestThinkEndpointUnit:
    """Unit tests for think endpoint handler."""

    @pytest.mark.asyncio
    async def test_think_requires_message(self):
        """Think endpoint should require message field."""
        from aiohttp import web

        from server import think_handler

        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(return_value={})

        response = await think_handler(request)

        assert response.status in [200, 400]

    @pytest.mark.asyncio
    async def test_think_forces_cortex(self):
        """Think endpoint should force cortex processing."""
        import json

        from aiohttp import web

        from server import think_handler

        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(return_value={"message": "Complex question"})

        with patch("server.get_ava") as mock_get_ava:
            mock_ava = AsyncMock()
            mock_ava.think.return_value = MagicMock(
                text="Deep analysis...",
                used_cortex=True,  # Should be True for think
                cognitive_state="VERIFYING",
            )
            mock_get_ava.return_value = mock_ava

            response = await think_handler(request)

            body = json.loads(response.body)
            # Think should use cortex
            if "used_cortex" in body:
                assert body["used_cortex"] is True


class TestCORSMiddlewareUnit:
    """Unit tests for CORS middleware."""

    @pytest.mark.asyncio
    async def test_cors_adds_headers(self):
        """CORS middleware should add access control headers."""
        from aiohttp import web

        from server import cors_middleware

        # Create mock request and handler
        request = MagicMock(spec=web.Request)
        request.method = "GET"

        async def mock_handler(request):
            return web.json_response({"test": "data"})

        response = await cors_middleware(request, mock_handler)

        # Check CORS headers are present
        assert "Access-Control-Allow-Origin" in response.headers

    @pytest.mark.asyncio
    async def test_cors_handles_options(self):
        """CORS should handle OPTIONS preflight requests."""
        from aiohttp import web

        from server import cors_middleware

        request = MagicMock(spec=web.Request)
        request.method = "OPTIONS"

        async def mock_handler(request):
            return web.json_response({})

        response = await cors_middleware(request, mock_handler)

        # OPTIONS should return 200
        assert response.status == 200


class TestResponseModels:
    """Test response dataclasses."""

    def test_chat_response_defaults(self):
        """ChatResponse should have sensible defaults."""
        from server import ChatResponse

        response = ChatResponse()
        assert response.text == ""
        assert response.used_cortex is False
        assert response.cognitive_state == "FLOW"
        assert response.confidence == 0.8

    def test_status_response_defaults(self):
        """StatusResponse should have sensible defaults."""
        from server import StatusResponse

        response = StatusResponse()
        assert response.status == "ok"
        assert response.uptime_seconds == 0
        assert response.total_requests == 0


class TestCreateApp:
    """Test application factory."""

    def test_create_app_returns_application(self):
        """create_app should return an aiohttp Application."""
        from aiohttp.web import Application

        from server import create_app

        app = create_app()
        assert isinstance(app, Application)

    def test_app_has_routes(self):
        """Application should have routes defined."""
        from server import create_app

        app = create_app()
        # Check that routes exist
        assert len(app.router.routes()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
