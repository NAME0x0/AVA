#!/usr/bin/env python3
"""
AVA API Server
==============

Clean, simple HTTP API for AVA.

Core Endpoints:
    GET  /health        - Health check
    GET  /status        - System status
    POST /chat          - Send message (auto-routes Medulla/Cortex)
    POST /think         - Force deep thinking (Cortex)
    GET  /tools         - List available tools

System State Endpoints:
    GET  /cognitive     - Get cognitive state (entropy, varentropy, etc.)
    GET  /memory        - Get Titans memory stats
    GET  /stats         - Get system statistics

Control Endpoints:
    POST /force_cortex  - Force Cortex for next response
    POST /sleep         - Trigger memory consolidation

WebSocket:
    /ws                 - Streaming chat with token-by-token updates

Usage:
    python server.py
    python server.py --port 8085
    python server.py --host 0.0.0.0 --port 8080
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import aiohttp
from aiohttp import web

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ava-server")

# Global AVA instance (thread-safe initialization)
_ava = None
_ava_lock = asyncio.Lock()
_start_time = time.time()


async def get_ava():
    """Get or create AVA instance (thread-safe singleton)."""
    global _ava

    # Fast path: already initialized
    if _ava is not None:
        return _ava

    # Slow path: acquire lock and initialize
    async with _ava_lock:
        # Double-check after acquiring lock
        if _ava is None:
            try:
                from ava import AVA

                _ava = AVA()
                await _ava.start()
                logger.info("AVA initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AVA: {e}")
                return None

    return _ava


# =============================================================================
# Response Models
# =============================================================================


@dataclass
class ChatResponse:
    """Response from chat endpoint."""

    text: str = ""
    used_cortex: bool = False
    cognitive_state: str = "FLOW"
    confidence: float = 0.8
    tools_used: list = field(default_factory=list)
    response_time_ms: float = 0.0
    error: str | None = None


@dataclass
class StatusResponse:
    """System status."""

    status: str = "ok"
    uptime_seconds: int = 0
    total_requests: int = 0
    cortex_requests: int = 0
    models: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Handlers
# =============================================================================


async def health_handler(request: web.Request) -> web.Response:
    """Health check with Ollama status for UI diagnostics."""
    # Check Ollama connectivity
    ollama_status = "unknown"
    ollama_error = None
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                ollama_status = "connected"
            else:
                ollama_status = "error"
                ollama_error = f"Status {resp.status_code}"
    except Exception as e:
        ollama_status = "unavailable"
        ollama_error = str(e)

    return web.json_response(
        {
            "status": "ok",
            "service": "AVA",
            "version": "3.3.3",
            "uptime_seconds": int(time.time() - _start_time),
            "ollama_status": ollama_status,
            "ollama_error": ollama_error,
        }
    )


async def status_handler(request: web.Request) -> web.Response:
    """Get system status."""
    ava = await get_ava()

    response = StatusResponse(uptime_seconds=int(time.time() - _start_time))

    if ava and ava._engine:
        stats = ava._engine.get_stats()
        response.total_requests = stats.get("total_requests", 0)
        response.cortex_requests = stats.get("cortex_requests", 0)
        response.models = stats.get("models", {})

    return web.json_response(asdict(response))


async def chat_handler(request: web.Request) -> web.Response:
    """Handle chat message."""
    try:
        data = await request.json()
        message = data.get("message", "")

        if not message:
            return web.json_response({"error": "No message provided"}, status=400)

        ava = await get_ava()

        if ava is None:
            return web.json_response(
                {"error": "AVA not initialized. Is Ollama running?"}, status=503
            )

        start = time.time()
        result = await ava.chat(message)

        # Convert cognitive_state to string if it's an object
        cog_state = result.cognitive_state
        if hasattr(cog_state, "label"):
            # CognitiveState object - extract label
            cog_state = (
                cog_state.label if isinstance(cog_state.label, str) else cog_state.label.value
            )
        elif hasattr(cog_state, "to_dict"):
            cog_state = cog_state.to_dict().get("label", "FLOW")
        elif not isinstance(cog_state, str):
            cog_state = str(cog_state)

        response = ChatResponse(
            text=result.text,
            used_cortex=result.used_cortex,
            cognitive_state=cog_state,
            confidence=result.confidence,
            tools_used=result.tools_used,
            response_time_ms=(time.time() - start) * 1000,
        )

        return web.json_response(asdict(response))

    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback

        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def think_handler(request: web.Request) -> web.Response:
    """Handle deep thinking request."""
    try:
        data = await request.json()
        message = data.get("message", "")

        if not message:
            return web.json_response({"error": "No message provided"}, status=400)

        ava = await get_ava()

        if ava is None:
            return web.json_response({"error": "AVA not initialized"}, status=503)

        start = time.time()
        result = await ava.think(message)

        response = ChatResponse(
            text=result.text,
            used_cortex=True,
            cognitive_state=result.cognitive_state,
            confidence=result.confidence,
            tools_used=result.tools_used,
            response_time_ms=(time.time() - start) * 1000,
        )

        return web.json_response(asdict(response))

    except Exception as e:
        logger.error(f"Think error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def tools_handler(request: web.Request) -> web.Response:
    """List available tools."""
    ava = await get_ava()

    if ava is None or ava._tools is None:
        return web.json_response({"tools": []})

    tools = []
    for defn in ava._tools.list_tools():
        tools.append(
            {"name": defn.name, "description": defn.description, "parameters": defn.parameters}
        )

    return web.json_response({"tools": tools})


async def cognitive_handler(request: web.Request) -> web.Response:
    """Get current cognitive state from Medulla."""
    ava = await get_ava()

    # Default cognitive state
    cognitive = {
        "label": "FLOW",
        "entropy": 0.3,
        "varentropy": 0.1,
        "confidence": 0.8,
        "surprise": 0.2,
        "should_use_tools": False,
        "should_think": False,
    }

    if ava and ava._engine:
        try:
            # Try to get cognitive state from engine
            if hasattr(ava._engine, "get_cognitive_state"):
                state = ava._engine.get_cognitive_state()
                if state:
                    cognitive = {
                        "label": getattr(state, "label", "FLOW"),
                        "entropy": getattr(state, "entropy", 0.3),
                        "varentropy": getattr(state, "varentropy", 0.1),
                        "confidence": getattr(state, "confidence", 0.8),
                        "surprise": getattr(state, "surprise", 0.2),
                        "should_use_tools": getattr(state, "should_use_tools", False),
                        "should_think": getattr(state, "should_think", False),
                    }
        except Exception as e:
            logger.debug(f"Could not get cognitive state: {e}")

    return web.json_response(cognitive)


async def memory_handler(request: web.Request) -> web.Response:
    """Get memory stats from Titans."""
    ava = await get_ava()

    # Default memory stats
    memory = {
        "total_memories": 0,
        "memory_updates": 0,
        "avg_surprise": 0.0,
        "backend": "titans",
        "memory_utilization": 0.0,
    }

    if ava and ava._engine:
        try:
            if hasattr(ava._engine, "get_memory_stats"):
                stats = ava._engine.get_memory_stats()
                if stats:
                    memory = {
                        "total_memories": stats.get("total_memories", 0),
                        "memory_updates": stats.get("memory_updates", 0),
                        "avg_surprise": stats.get("avg_surprise", 0.0),
                        "backend": stats.get("backend", "titans"),
                        "memory_utilization": stats.get("memory_utilization", 0.0),
                    }
        except Exception as e:
            logger.debug(f"Could not get memory stats: {e}")

    return web.json_response(memory)


async def stats_handler(request: web.Request) -> web.Response:
    """Get system statistics."""
    ava = await get_ava()

    stats = {
        "system": {
            "is_running": True,
            "interaction_count": 0,
            "uptime_seconds": int(time.time() - _start_time),
        },
        "medulla": {"requests": 0, "avg_latency_ms": 0},
        "cortex": {"requests": 0, "avg_latency_ms": 0},
    }

    if ava and ava._engine:
        try:
            engine_stats = ava._engine.get_stats()
            stats["system"]["interaction_count"] = engine_stats.get("total_requests", 0)
            stats["cortex"]["requests"] = engine_stats.get("cortex_requests", 0)
            stats["medulla"]["requests"] = engine_stats.get("total_requests", 0) - engine_stats.get(
                "cortex_requests", 0
            )
        except Exception as e:
            logger.debug(f"Could not get engine stats: {e}")

    return web.json_response(stats)


# Global state for force_cortex flag
_force_cortex_next = False


async def force_cortex_handler(request: web.Request) -> web.Response:
    """Force Cortex for the next response."""
    global _force_cortex_next
    _force_cortex_next = True
    logger.info("Cortex forced for next response")
    return web.json_response({"status": "ok", "message": "Cortex will be used for next response"})


async def sleep_handler(request: web.Request) -> web.Response:
    """Trigger sleep/consolidation cycle."""
    ava = await get_ava()

    if ava and ava._engine:
        try:
            if hasattr(ava._engine, "trigger_sleep"):
                await ava._engine.trigger_sleep()
                logger.info("Sleep cycle triggered")
                return web.json_response({"status": "ok", "message": "Sleep cycle initiated"})
        except Exception as e:
            logger.error(f"Sleep trigger failed: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    # Graceful response even if engine doesn't support sleep
    return web.json_response({"status": "ok", "message": "Sleep cycle simulated"})


async def system_handler(request: web.Request) -> web.Response:
    """Get detailed system information for diagnostics."""
    import platform

    ava = await get_ava()

    # Ollama info
    ollama_info = {"status": "unknown", "models": []}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                ollama_info = {
                    "status": "connected",
                    "models": [m.get("name", "unknown") for m in data.get("models", [])],
                }
            else:
                ollama_info = {"status": "error", "code": resp.status_code}
    except Exception as e:
        ollama_info = {"status": "unavailable", "error": str(e)}

    # Engine stats
    engine_stats = {}
    if ava and ava._engine:
        try:
            engine_stats = ava._engine.get_stats()
        except Exception:
            pass

    system_info = {
        "version": "3.3.3",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "server": {
            "uptime_seconds": int(time.time() - _start_time),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_start_time)),
        },
        "ollama": ollama_info,
        "engine": engine_stats,
        "ava_initialized": ava is not None,
    }

    return web.json_response(system_info)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """WebSocket for streaming chat with token-by-token updates."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logger.info(f"WebSocket connection established from {request.remote}")

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                message = data.get("message", "")
                force_deep = data.get("force_deep", False)

                if not message:
                    await ws.send_json({"type": "error", "message": "No message provided"})
                    continue

                ava = await get_ava()
                if not ava:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "AVA not initialized. Please wait and try again.",
                        }
                    )
                    continue

                # Send processing started
                await ws.send_json(
                    {
                        "type": "status",
                        "status": "processing",
                        "message": "Analyzing your request...",
                    }
                )

                try:
                    # Get response (with streaming simulation for now)
                    if force_deep:
                        result = await ava.think(message)
                    else:
                        result = await ava.chat(message)

                    # Stream the response in chunks for better UX
                    response_text = result.text
                    chunk_size = 20  # Characters per chunk

                    for i in range(0, len(response_text), chunk_size):
                        chunk = response_text[i : i + chunk_size]
                        await ws.send_json({"type": "chunk", "text": chunk, "done": False})
                        await asyncio.sleep(0.02)  # Small delay for streaming effect

                    # Send completion message
                    await ws.send_json(
                        {
                            "type": "response",
                            "text": response_text,
                            "used_cortex": result.used_cortex,
                            "cognitive_state": (
                                result.cognitive_state
                                if hasattr(result, "cognitive_state")
                                else None
                            ),
                            "response_time_ms": (
                                result.response_time_ms
                                if hasattr(result, "response_time_ms")
                                else None
                            ),
                            "done": True,
                        }
                    )

                except asyncio.TimeoutError:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Request timed out. Please try a simpler question.",
                            "error_type": "timeout",
                        }
                    )
                except Exception as e:
                    logger.exception(f"Error processing WebSocket message: {e}")
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": f"An error occurred: {type(e).__name__}",
                            "error_type": "processing_error",
                        }
                    )

            except json.JSONDecodeError:
                await ws.send_json(
                    {"type": "error", "message": "Invalid JSON format", "error_type": "parse_error"}
                )
            except Exception as e:
                logger.exception(f"Unexpected WebSocket error: {e}")
                await ws.send_json(
                    {
                        "type": "error",
                        "message": "An unexpected error occurred",
                        "error_type": "unknown",
                    }
                )

        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error(f"WebSocket error: {ws.exception()}")

    logger.info(f"WebSocket connection closed from {request.remote}")
    return ws


# =============================================================================
# Middleware
# =============================================================================


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Add CORS headers."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    return response


# =============================================================================
# Application
# =============================================================================


def create_app() -> web.Application:
    """Create the web application."""
    app = web.Application(middlewares=[cors_middleware])

    # Core routes
    app.router.add_get("/health", health_handler)
    app.router.add_get("/status", status_handler)
    app.router.add_post("/chat", chat_handler)
    app.router.add_post("/think", think_handler)
    app.router.add_get("/tools", tools_handler)
    app.router.add_get("/ws", websocket_handler)

    # System state routes (for UI polling)
    app.router.add_get("/cognitive", cognitive_handler)
    app.router.add_get("/memory", memory_handler)
    app.router.add_get("/stats", stats_handler)
    app.router.add_get("/system", system_handler)

    # Control routes
    app.router.add_post("/force_cortex", force_cortex_handler)
    app.router.add_post("/sleep", sleep_handler)

    # CORS preflight
    app.router.add_route("OPTIONS", "/{path:.*}", lambda r: web.Response())

    return app


def print_banner(host: str, port: int):
    """Print startup banner."""
    banner = """
\033[96m╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║     █████╗ ██╗   ██╗ █████╗     ███████╗███████╗██████╗ ██╗   ██╗███████╗██████╗     ║
║    ██╔══██╗██║   ██║██╔══██╗    ██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝██╔══██╗    ║
║    ███████║██║   ██║███████║    ███████╗█████╗  ██████╔╝██║   ██║█████╗  ██████╔╝    ║
║    ██╔══██║╚██╗ ██╔╝██╔══██║    ╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██╔══╝  ██╔══██╗    ║
║    ██║  ██║ ╚████╔╝ ██║  ██║    ███████║███████╗██║  ██║ ╚████╔╝ ███████╗██║  ██║    ║
║    ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝    ╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝    ║
║                                                                                      ║
║\033[0m\033[97m                    Cortex-Medulla Architecture v3.3.3                                \033[96m║
╚══════════════════════════════════════════════════════════════════════════════════════╝\033[0m
"""
    print(banner)
    print(f"\033[93m  ⚡ Server:\033[0m    http://{host}:{port}")
    print(f"\033[93m  📡 WebSocket:\033[0m ws://{host}:{port}/ws")
    print()
    print("\033[96m  ┌─────────────────────────────────────────────────────────┐\033[0m")
    print(
        "\033[96m  │\033[0m  \033[97mEndpoints:\033[0m                                             \033[96m│\033[0m"
    )
    print(
        "\033[96m  │\033[0m    \033[92mGET\033[0m   /health  ─  Health check                       \033[96m│\033[0m"
    )
    print(
        "\033[96m  │\033[0m    \033[92mGET\033[0m   /status  ─  System status                      \033[96m│\033[0m"
    )
    print(
        "\033[96m  │\033[0m    \033[93mPOST\033[0m  /chat    ─  Send message                       \033[96m│\033[0m"
    )
    print(
        "\033[96m  │\033[0m    \033[93mPOST\033[0m  /think   ─  Force deep thinking                \033[96m│\033[0m"
    )
    print(
        "\033[96m  │\033[0m    \033[92mGET\033[0m   /tools   ─  List available tools               \033[96m│\033[0m"
    )
    print(
        "\033[96m  │\033[0m    \033[95mWS\033[0m    /ws      ─  WebSocket streaming                \033[96m│\033[0m"
    )
    print("\033[96m  └─────────────────────────────────────────────────────────┘\033[0m")
    print()


async def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="AVA API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8085, help="Port to listen on")
    args = parser.parse_args()

    app = create_app()

    print_banner(args.host, args.port)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    logger.info("Server ready. Press Ctrl+C to stop.")

    # Keep running
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass


def run():
    """Entry point."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == "__main__":
    run()
