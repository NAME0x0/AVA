#!/usr/bin/env python3
"""
AVA API Server
==============

Clean, simple HTTP API for AVA.

Endpoints:
    GET  /              - Web UI (if available)
    GET  /health        - Health check
    GET  /status        - System status
    POST /chat          - Send message
    POST /think         - Force deep thinking
    GET  /tools         - List available tools
    
WebSocket:
    /ws                 - Streaming chat

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiohttp import web
import aiohttp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ava-server")

# Global AVA instance
_ava = None
_start_time = time.time()


async def get_ava():
    """Get or create AVA instance."""
    global _ava
    
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
    error: Optional[str] = None


@dataclass
class StatusResponse:
    """System status."""
    status: str = "ok"
    uptime_seconds: int = 0
    total_requests: int = 0
    cortex_requests: int = 0
    models: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Handlers
# =============================================================================

async def health_handler(request: web.Request) -> web.Response:
    """Health check."""
    return web.json_response({
        "status": "ok",
        "service": "AVA",
        "version": "3.1.0",
        "uptime_seconds": int(time.time() - _start_time)
    })


async def status_handler(request: web.Request) -> web.Response:
    """Get system status."""
    ava = await get_ava()
    
    response = StatusResponse(
        uptime_seconds=int(time.time() - _start_time)
    )
    
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
                {"error": "AVA not initialized. Is Ollama running?"},
                status=503
            )
        
        start = time.time()
        result = await ava.chat(message)
        
        response = ChatResponse(
            text=result.text,
            used_cortex=result.used_cortex,
            cognitive_state=result.cognitive_state,
            confidence=result.confidence,
            tools_used=result.tools_used,
            response_time_ms=(time.time() - start) * 1000
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
            return web.json_response(
                {"error": "AVA not initialized"},
                status=503
            )
        
        start = time.time()
        result = await ava.think(message)
        
        response = ChatResponse(
            text=result.text,
            used_cortex=True,
            cognitive_state=result.cognitive_state,
            confidence=result.confidence,
            tools_used=result.tools_used,
            response_time_ms=(time.time() - start) * 1000
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
        tools.append({
            "name": defn.name,
            "description": defn.description,
            "parameters": defn.parameters
        })
    
    return web.json_response({"tools": tools})


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
                    await ws.send_json({
                        "type": "error",
                        "message": "No message provided"
                    })
                    continue

                ava = await get_ava()
                if not ava:
                    await ws.send_json({
                        "type": "error",
                        "message": "AVA not initialized. Please wait and try again."
                    })
                    continue

                # Send processing started
                await ws.send_json({
                    "type": "status",
                    "status": "processing",
                    "message": "Analyzing your request..."
                })

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
                        chunk = response_text[i:i + chunk_size]
                        await ws.send_json({
                            "type": "chunk",
                            "text": chunk,
                            "done": False
                        })
                        await asyncio.sleep(0.02)  # Small delay for streaming effect

                    # Send completion message
                    await ws.send_json({
                        "type": "response",
                        "text": response_text,
                        "used_cortex": result.used_cortex,
                        "cognitive_state": result.cognitive_state if hasattr(result, 'cognitive_state') else None,
                        "response_time_ms": result.response_time_ms if hasattr(result, 'response_time_ms') else None,
                        "done": True
                    })

                except asyncio.TimeoutError:
                    await ws.send_json({
                        "type": "error",
                        "message": "Request timed out. Please try a simpler question.",
                        "error_type": "timeout"
                    })
                except Exception as e:
                    logger.exception(f"Error processing WebSocket message: {e}")
                    await ws.send_json({
                        "type": "error",
                        "message": f"An error occurred: {type(e).__name__}",
                        "error_type": "processing_error"
                    })

            except json.JSONDecodeError:
                await ws.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "error_type": "parse_error"
                })
            except Exception as e:
                logger.exception(f"Unexpected WebSocket error: {e}")
                await ws.send_json({
                    "type": "error",
                    "message": "An unexpected error occurred",
                    "error_type": "unknown"
                })

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
    
    # Routes
    app.router.add_get("/health", health_handler)
    app.router.add_get("/status", status_handler)
    app.router.add_post("/chat", chat_handler)
    app.router.add_post("/think", think_handler)
    app.router.add_get("/tools", tools_handler)
    app.router.add_get("/ws", websocket_handler)
    
    # CORS preflight
    app.router.add_route("OPTIONS", "/{path:.*}", lambda r: web.Response())
    
    return app


async def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="AVA API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8085, help="Port to listen on")
    args = parser.parse_args()
    
    app = create_app()
    
    logger.info("=" * 60)
    logger.info("AVA API SERVER")
    logger.info("=" * 60)
    logger.info(f"Starting on http://{args.host}:{args.port}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /health  - Health check")
    logger.info("  GET  /status  - System status")
    logger.info("  POST /chat    - Send message")
    logger.info("  POST /think   - Force deep thinking")
    logger.info("  GET  /tools   - List tools")
    logger.info("  WS   /ws      - WebSocket streaming")
    logger.info("=" * 60)
    
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
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == "__main__":
    run()
