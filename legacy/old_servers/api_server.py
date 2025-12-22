#!/usr/bin/env python3
"""
AVA API Server - HTTP interface for the Rust UI
================================================

Provides a lightweight HTTP API for the AVA UI to communicate with
the Frankensystem backend.

Usage:
    python api_server.py
    python api_server.py --port 8080
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiohttp import web

logger = logging.getLogger("AVA-API")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Global Frankensystem instance
_system = None
_initialized = False


async def get_system():
    """Get or create the Frankensystem instance."""
    global _system, _initialized
    
    if _system is None:
        from run_frankensystem import Frankensystem, FrankenConfig
        
        config = FrankenConfig(
            enable_sleep=True,
        )
        _system = Frankensystem(config)
    
    if not _initialized:
        await _system.initialize()
        _initialized = True
    
    return _system


# ============================================================================
# API Handlers
# ============================================================================

async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "ok",
        "service": "AVA Frankensystem",
        "version": "2.0.0",
    })


async def get_cognitive_state(request: web.Request) -> web.Response:
    """Get current Entropix cognitive state."""
    try:
        system = await get_system()
        
        if system._entropix and system._last_cognitive_state:
            state = system._last_cognitive_state
            return web.json_response({
                "label": state.label.name if hasattr(state.label, 'name') else str(state.label),
                "entropy": float(state.entropy),
                "varentropy": float(state.varentropy),
                "confidence": float(getattr(state, 'confidence', 0.0)),
                "surprise": float(getattr(state, 'surprise', 0.0)),
                "should_use_tools": getattr(state, 'should_use_tools', False),
                "should_think": getattr(state, 'should_think', False),
            })
        else:
            return web.json_response({
                "label": "UNKNOWN",
                "entropy": 0.0,
                "varentropy": 0.0,
                "confidence": 0.0,
                "surprise": 0.0,
                "should_use_tools": False,
                "should_think": False,
            })
    except Exception as e:
        logger.error(f"Error getting cognitive state: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_memory_stats(request: web.Request) -> web.Response:
    """Get Titans neural memory statistics."""
    try:
        system = await get_system()
        
        if system._titans_sidecar:
            stats = system._titans_sidecar.get_stats()
            return web.json_response({
                "total_memories": stats.get("memory_updates", 0),
                "memory_updates": stats.get("memory_updates", 0),
                "avg_surprise": stats.get("avg_surprise", 0.0),
                "backend": system._titans_sidecar.backend,
            })
        else:
            return web.json_response({
                "total_memories": 0,
                "memory_updates": 0,
                "avg_surprise": 0.0,
                "backend": "none",
            })
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_buffer_stats(request: web.Request) -> web.Response:
    """Get episodic buffer statistics."""
    try:
        system = await get_system()
        
        if system._episodic_buffer:
            stats = system._episodic_buffer.get_stats()
            return web.json_response({
                "total_episodes": stats.get("total_episodes", 0),
                "avg_surprise": stats.get("avg_surprise", 0.0),
                "avg_quality": stats.get("avg_quality", 0.0),
                "high_priority_count": stats.get("high_priority_count", 0),
            })
        else:
            return web.json_response({
                "total_episodes": 0,
                "avg_surprise": 0.0,
                "avg_quality": 0.0,
                "high_priority_count": 0,
            })
    except Exception as e:
        logger.error(f"Error getting buffer stats: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_sleep_state(request: web.Request) -> web.Response:
    """Get Nightmare Engine sleep state."""
    try:
        system = await get_system()
        
        if system._nightmare_engine:
            return web.json_response({
                "phase": system._nightmare_engine.current_phase.name,
                "is_sleeping": system._nightmare_engine._is_sleeping.is_set(),
                "progress": 0.0,  # TODO: Calculate actual progress
                "episodes_processed": 0,
            })
        else:
            return web.json_response({
                "phase": "AWAKE",
                "is_sleeping": False,
                "progress": 0.0,
                "episodes_processed": 0,
            })
    except Exception as e:
        logger.error(f"Error getting sleep state: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def chat(request: web.Request) -> web.Response:
    """Process a chat message through the Frankensystem."""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return web.json_response({"error": "No message provided"}, status=400)
        
        system = await get_system()
        result = await system.process(message)
        
        # Extract cognitive state
        cognitive_state = None
        if "cognitive_state" in result and result["cognitive_state"]:
            state = result["cognitive_state"]
            cognitive_state = {
                "label": state.label.name if hasattr(state.label, 'name') else str(state.label),
                "entropy": float(state.entropy),
                "varentropy": float(state.varentropy),
                "confidence": float(getattr(state, 'confidence', 0.0)),
                "surprise": float(getattr(state, 'surprise', 0.0)),
                "should_use_tools": getattr(state, 'should_use_tools', False),
                "should_think": getattr(state, 'should_think', False),
            }
        
        return web.json_response({
            "response": result.get("response", ""),
            "cognitive_state": cognitive_state,
            "surprise": result.get("surprise", 0.0),
            "tokens_generated": result.get("tokens", 0),
        })
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def force_sleep(request: web.Request) -> web.Response:
    """Force the Nightmare Engine into a sleep cycle."""
    try:
        system = await get_system()
        
        if system._nightmare_engine:
            # Trigger sleep in background
            asyncio.create_task(asyncio.to_thread(system._nightmare_engine.dream))
            return web.json_response({"status": "sleep_initiated"})
        else:
            return web.json_response({"error": "Nightmare Engine not available"}, status=503)
            
    except Exception as e:
        logger.error(f"Error forcing sleep: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_full_stats(request: web.Request) -> web.Response:
    """Get comprehensive system statistics."""
    try:
        system = await get_system()
        
        stats = {
            "cognitive": {},
            "memory": {},
            "buffer": {},
            "sleep": {},
            "system": {
                "interaction_count": system._interaction_count,
                "total_surprise": system._total_surprise,
                "is_running": system.is_running,
            }
        }
        
        # Cognitive
        if system._entropix and system._last_cognitive_state:
            state = system._last_cognitive_state
            stats["cognitive"] = {
                "label": state.label.name if hasattr(state.label, 'name') else str(state.label),
                "entropy": float(state.entropy),
                "varentropy": float(state.varentropy),
            }
        
        # Memory
        if system._titans_sidecar:
            stats["memory"] = system._titans_sidecar.get_stats()
        
        # Buffer
        if system._episodic_buffer:
            stats["buffer"] = system._episodic_buffer.get_stats()
        
        # Sleep
        if system._nightmare_engine:
            stats["sleep"] = {
                "phase": system._nightmare_engine.current_phase.name,
                "is_sleeping": system._nightmare_engine._is_sleeping.is_set(),
            }
        
        return web.json_response(stats)
        
    except Exception as e:
        logger.error(f"Error getting full stats: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ============================================================================
# CORS Middleware
# ============================================================================

@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Add CORS headers to all responses."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    
    return response


# ============================================================================
# Application Setup
# ============================================================================

def create_app() -> web.Application:
    """Create and configure the web application."""
    app = web.Application(middlewares=[cors_middleware])
    
    # Routes
    app.router.add_get("/health", health_check)
    app.router.add_get("/cognitive", get_cognitive_state)
    app.router.add_get("/memory", get_memory_stats)
    app.router.add_get("/buffer", get_buffer_stats)
    app.router.add_get("/sleep", get_sleep_state)
    app.router.add_get("/stats", get_full_stats)
    app.router.add_post("/chat", chat)
    app.router.add_post("/sleep", force_sleep)
    
    # OPTIONS for CORS preflight
    app.router.add_route("OPTIONS", "/{path:.*}", lambda r: web.Response())
    
    return app


async def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AVA API Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    app = create_app()
    
    logger.info(f"Starting AVA API Server on http://{args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info("  GET  /health    - Health check")
    logger.info("  GET  /cognitive - Entropix cognitive state")
    logger.info("  GET  /memory    - Titans memory stats")
    logger.info("  GET  /buffer    - Episodic buffer stats")
    logger.info("  GET  /sleep     - Nightmare Engine state")
    logger.info("  GET  /stats     - Full system statistics")
    logger.info("  POST /chat      - Send message to AVA")
    logger.info("  POST /sleep     - Force sleep cycle")
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    
    logger.info("Server running. Press Ctrl+C to stop.")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped.")
