#!/usr/bin/env python3
"""
AVA API Server v3 - Cortex-Medulla HTTP Interface
==================================================

Provides a WebSocket + HTTP API for the new Next.js/Tauri UI to communicate
with the Cortex-Medulla backend. Supports real-time streaming responses.

Endpoints:
----------
GET  /health           - Health check
GET  /system/state     - Full system state
GET  /cognitive        - Medulla cognitive state
GET  /memory           - Titans memory stats
GET  /belief           - Active Inference belief state
POST /chat             - Send message (returns full response)
POST /chat/stream      - Send message (WebSocket streaming)
POST /force_cortex     - Force next response via Cortex
POST /sleep            - Trigger sleep cycle

Usage:
------
    python api_server_v3.py
    python api_server_v3.py --port 8085
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aiohttp import web
import aiohttp

logger = logging.getLogger("AVA-API-v3")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Global system instance
_system = None
_initialized = False
_force_cortex = False
_start_time = time.time()
_interaction_count = 0
_cortex_count = 0
_total_response_time = 0.0


async def get_system():
    """Get or create the AVA Core System instance."""
    global _system, _initialized
    
    if _system is None:
        try:
            # V2: Use the new Ollama-backed core
            from src.core.core_v2 import get_core
            _system = await get_core()
            _initialized = True
            logger.info("Initialized AVA Core V2 (Ollama backend)")
            return _system
        except Exception as e:
            logger.warning(f"Failed to load Core V2: {e}")
            
        # Fallback: Try old core system
        try:
            from src.core.core_loop import AVACoreSystem, CoreConfig
            config = CoreConfig()
            _system = AVACoreSystem(config)
        except ImportError:
            # Fallback to Frankensystem if core not available
            try:
                from run_frankensystem import Frankensystem, FrankenConfig
                config = FrankenConfig(enable_sleep=True)
                _system = Frankensystem(config)
            except ImportError:
                logger.warning("No backend system available, running in demo mode")
                _system = None
                return None
    
    if _system and not _initialized:
        try:
            await _system.initialize()
            _initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
    
    return _system


# =============================================================================
# Response Models
# =============================================================================

@dataclass
class CognitiveStateResponse:
    """Cognitive state from the Medulla/Entropix."""
    label: str = "UNKNOWN"
    entropy: float = 0.0
    varentropy: float = 0.0
    confidence: float = 0.0
    surprise: float = 0.0
    should_use_tools: bool = False
    should_think: bool = False


@dataclass
class SystemStateResponse:
    """Full system state."""
    connected: bool = True
    system_state: str = "running"  # initializing, running, paused, sleeping, error
    active_component: str = "idle"  # medulla, cortex, bridge, idle
    uptime_seconds: int = 0
    total_interactions: int = 0
    cortex_invocations: int = 0
    avg_response_time_ms: float = 0.0


@dataclass
class MemoryStatsResponse:
    """Titans neural memory statistics."""
    total_memories: int = 0
    memory_updates: int = 0
    avg_surprise: float = 0.0
    backend: str = "numpy"
    memory_utilization: float = 0.0


@dataclass
class BeliefStateResponse:
    """Active Inference belief state."""
    current_state: str = "IDLE"
    state_distribution: Dict[str, float] = field(default_factory=dict)
    policy_distribution: Dict[str, float] = field(default_factory=dict)
    free_energy: float = 0.0


@dataclass
class ChatResponse:
    """Response from chat endpoint."""
    response: str = ""
    cognitive_state: Optional[Dict[str, Any]] = None
    surprise: float = 0.0
    tokens_generated: int = 0
    used_cortex: bool = False
    policy_selected: str = "REFLEX_REPLY"
    response_time_ms: float = 0.0


# =============================================================================
# API Handlers
# =============================================================================

async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "ok",
        "service": "AVA Cortex-Medulla",
        "version": "3.0.0",
        "uptime_seconds": int(time.time() - _start_time),
    })


async def get_system_state(request: web.Request) -> web.Response:
    """Get full system state."""
    global _interaction_count, _cortex_count, _total_response_time
    
    system = await get_system()
    
    active_component = "idle"
    system_state_str = "running"
    
    if system:
        # Check if we have core system
        if hasattr(system, 'state'):
            system_state_str = system.state.value if hasattr(system.state, 'value') else str(system.state)
        if hasattr(system, '_active_component'):
            active_component = system._active_component
    
    avg_response = (_total_response_time / _interaction_count) if _interaction_count > 0 else 0.0
    
    response = SystemStateResponse(
        connected=True,
        system_state=system_state_str,
        active_component=active_component,
        uptime_seconds=int(time.time() - _start_time),
        total_interactions=_interaction_count,
        cortex_invocations=_cortex_count,
        avg_response_time_ms=avg_response,
    )
    
    return web.json_response(asdict(response))


async def get_cognitive_state(request: web.Request) -> web.Response:
    """Get Medulla cognitive state (Entropix)."""
    system = await get_system()
    
    response = CognitiveStateResponse()
    
    if system:
        # V2 Core System
        if hasattr(system, 'get_cognitive_state'):
            try:
                state = await system.get_cognitive_state()
                response.label = state.get("label", "UNKNOWN")
                response.entropy = state.get("entropy", 0.0)
                response.varentropy = state.get("varentropy", 0.0)
                response.confidence = state.get("confidence", 0.0)
                response.surprise = state.get("surprise", 0.0)
                response.should_use_tools = state.get("should_use_tools", False)
                response.should_think = state.get("should_think", False)
            except Exception as e:
                logger.warning(f"Error getting cognitive state: {e}")
        # Try Core system with medulla
        elif hasattr(system, 'medulla') and system.medulla:
            medulla = system.medulla
            if hasattr(medulla, '_last_surprise'):
                surprise = medulla._last_surprise
                response.surprise = float(surprise.value) if surprise else 0.0
                response.entropy = float(surprise.value * 2) if surprise else 0.0  # Approximation
                response.varentropy = float(surprise.value * 1.5) if surprise else 0.0
                response.label = "FLOW" if response.entropy < 2.0 else "HESITATION" if response.entropy < 3.5 else "CONFUSION"
        # Try Frankensystem
        elif hasattr(system, '_entropix') and hasattr(system, '_last_cognitive_state'):
            state = system._last_cognitive_state
            if state:
                response.label = state.label.name if hasattr(state.label, 'name') else str(state.label)
                response.entropy = float(state.entropy)
                response.varentropy = float(state.varentropy)
                response.confidence = float(getattr(state, 'confidence', 0.0))
                response.surprise = float(getattr(state, 'surprise', 0.0))
                response.should_use_tools = getattr(state, 'should_use_tools', False)
                response.should_think = getattr(state, 'should_think', False)
    
    return web.json_response(asdict(response))


async def get_memory_stats(request: web.Request) -> web.Response:
    """Get Titans neural memory statistics."""
    system = await get_system()
    
    response = MemoryStatsResponse()
    
    if system:
        # V2 Core System
        if hasattr(system, 'get_memory_stats'):
            try:
                stats = await system.get_memory_stats()
                response.total_memories = stats.get("total_memories", 0)
                response.memory_updates = stats.get("memory_updates", 0)
                response.avg_surprise = stats.get("avg_surprise", 0.0)
                response.backend = stats.get("backend", "embedding")
                response.memory_utilization = stats.get("memory_utilization", 0.0)
            except Exception as e:
                logger.warning(f"Error getting memory stats: {e}")
        # Try Core system
        elif hasattr(system, 'medulla') and system.medulla and hasattr(system.medulla, '_titans'):
            titans = system.medulla._titans
            if titans:
                stats = titans.get_stats() if hasattr(titans, 'get_stats') else {}
                response.total_memories = stats.get("memory_updates", 0)
                response.memory_updates = stats.get("memory_updates", 0)
                response.avg_surprise = stats.get("avg_surprise", 0.0)
                response.backend = "numpy"
        # Try Frankensystem
        elif hasattr(system, '_titans_sidecar') and system._titans_sidecar:
            stats = system._titans_sidecar.get_stats()
            response.total_memories = stats.get("memory_updates", 0)
            response.memory_updates = stats.get("memory_updates", 0)
            response.avg_surprise = stats.get("avg_surprise", 0.0)
            response.backend = system._titans_sidecar.backend
    
    return web.json_response(asdict(response))


async def get_belief_state(request: web.Request) -> web.Response:
    """Get Active Inference belief state."""
    system = await get_system()
    
    response = BeliefStateResponse()
    
    if system and hasattr(system, 'agency') and system.agency:
        agency = system.agency
        if hasattr(agency, '_current_belief'):
            belief = agency._current_belief
            response.current_state = belief.preferred_state.name if hasattr(belief.preferred_state, 'name') else str(belief.preferred_state)
            response.free_energy = float(belief.free_energy) if hasattr(belief, 'free_energy') else 0.0
            
            # Convert distributions to dict
            if hasattr(belief, 'state_distribution'):
                for state, prob in enumerate(belief.state_distribution):
                    response.state_distribution[f"STATE_{state}"] = float(prob)
        
        if hasattr(agency, '_last_efe'):
            for policy, efe in agency._last_efe.items():
                policy_name = policy.name if hasattr(policy, 'name') else str(policy)
                response.policy_distribution[policy_name] = float(1.0 / (1.0 + efe))  # Convert EFE to probability-like
    
    return web.json_response(asdict(response))


async def chat(request: web.Request) -> web.Response:
    """Process a chat message through the Cortex-Medulla system."""
    global _force_cortex, _interaction_count, _cortex_count, _total_response_time
    
    try:
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return web.json_response({"error": "No message provided"}, status=400)
        
        start_time = time.time()
        system = await get_system()
        
        response = ChatResponse()
        
        if system:
            # Determine if we should use Cortex
            use_cortex = _force_cortex
            _force_cortex = False  # Reset flag
            
            # V2 Core System - returns ProcessResult
            if hasattr(system, 'process_input'):
                result = await system.process_input(message, force_cortex=use_cortex)
                
                # V2 ProcessResult has structured fields
                if hasattr(result, 'response'):
                    # ProcessResult from core_v2
                    response.response = result.response
                    response.used_cortex = result.used_cortex
                    response.policy_selected = result.policy.value if hasattr(result.policy, 'value') else str(result.policy)
                    response.surprise = result.surprise
                    response.tokens_generated = len(result.response.split())  # Approximate token count
                    
                    # Get cognitive state from result or from system
                    if result.cognitive_state:
                        response.cognitive_state = {
                            "label": result.cognitive_state.label.value if hasattr(result.cognitive_state.label, 'value') else str(result.cognitive_state.label),
                            "entropy": result.cognitive_state.entropy,
                            "varentropy": result.cognitive_state.varentropy,
                            "confidence": result.cognitive_state.confidence,
                            "surprise": result.cognitive_state.surprise,
                            "should_use_tools": False,
                            "should_think": result.used_cortex,
                        }
                    elif hasattr(system, 'get_cognitive_state'):
                        cog_state = await system.get_cognitive_state()
                        response.cognitive_state = cog_state
                elif isinstance(result, str):
                    response.response = result
                    response.used_cortex = use_cortex
                    response.policy_selected = "DEEP_THOUGHT" if use_cortex else "REFLEX_REPLY"
                    # Get surprise from medulla if available
                    if hasattr(system, '_medulla') and hasattr(system._medulla, '_last_surprise'):
                        response.surprise = float(system._medulla._last_surprise)
                elif isinstance(result, dict):
                    response.response = result.get("response", "")
                    response.used_cortex = result.get("used_cortex", False)
                    response.policy_selected = result.get("policy", "REFLEX_REPLY")
                    response.surprise = result.get("surprise", 0.0)
                    if "cognitive_state" in result:
                        response.cognitive_state = result["cognitive_state"]
                        
            elif hasattr(system, 'process'):
                # Frankensystem - returns string or dict
                result = await system.process(message)
                
                if isinstance(result, str):
                    response.response = result
                else:
                    response.response = result.get("response", "")
                    response.surprise = result.get("surprise", 0.0)
                    
                    if "cognitive_state" in result and result["cognitive_state"]:
                        state = result["cognitive_state"]
                        response.cognitive_state = {
                            "label": state.label.name if hasattr(state.label, 'name') else str(state.label),
                            "entropy": float(state.entropy),
                            "varentropy": float(state.varentropy),
                            "confidence": float(getattr(state, 'confidence', 0.0)),
                            "surprise": float(getattr(state, 'surprise', 0.0)),
                            "should_use_tools": getattr(state, 'should_use_tools', False),
                            "should_think": getattr(state, 'should_think', False),
                        }
        else:
            # Demo mode response
            response.response = f"[Demo Mode] I received your message: '{message[:50]}...'. The backend is not connected, but the UI is working correctly!"
            response.cognitive_state = {
                "label": "FLOW",
                "entropy": 1.5,
                "varentropy": 1.0,
                "confidence": 0.8,
                "surprise": 0.5,
                "should_use_tools": False,
                "should_think": False,
            }
        
        elapsed = (time.time() - start_time) * 1000
        response.response_time_ms = elapsed
        
        # Update stats
        _interaction_count += 1
        _total_response_time += elapsed
        if response.used_cortex:
            _cortex_count += 1
        
        return web.json_response(asdict(response))
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def force_cortex_handler(request: web.Request) -> web.Response:
    """Force the next response to use Cortex."""
    global _force_cortex
    _force_cortex = True
    logger.info("Cortex forced for next response")
    return web.json_response({"status": "cortex_forced"})


async def force_sleep(request: web.Request) -> web.Response:
    """Trigger a sleep cycle."""
    system = await get_system()
    
    if system:
        if hasattr(system, '_nightmare_engine') and system._nightmare_engine:
            asyncio.create_task(asyncio.to_thread(system._nightmare_engine.dream))
            return web.json_response({"status": "sleep_initiated"})
    
    return web.json_response({"status": "sleep_not_available"})


async def get_full_stats(request: web.Request) -> web.Response:
    """Get comprehensive system statistics (compatibility endpoint)."""
    system = await get_system()
    
    stats = {
        "cognitive": {},
        "memory": {},
        "buffer": {},
        "sleep": {},
        "system": {
            "interaction_count": _interaction_count,
            "total_surprise": 0.0,
            "is_running": True,
        }
    }
    
    if system:
        # Get cognitive state
        if hasattr(system, '_entropix') and hasattr(system, '_last_cognitive_state'):
            state = system._last_cognitive_state
            if state:
                stats["cognitive"] = {
                    "label": state.label.name if hasattr(state.label, 'name') else str(state.label),
                    "entropy": float(state.entropy),
                    "varentropy": float(state.varentropy),
                }
        
        # Get memory stats
        if hasattr(system, '_titans_sidecar') and system._titans_sidecar:
            stats["memory"] = system._titans_sidecar.get_stats()
        
        # Get buffer stats
        if hasattr(system, '_episodic_buffer') and system._episodic_buffer:
            stats["buffer"] = system._episodic_buffer.get_stats()
        
        # Get sleep state
        if hasattr(system, '_nightmare_engine') and system._nightmare_engine:
            stats["sleep"] = {
                "phase": system._nightmare_engine.current_phase.name,
                "is_sleeping": system._nightmare_engine._is_sleeping.is_set(),
            }
    
    return web.json_response(stats)


# =============================================================================
# WebSocket Handler for Streaming
# =============================================================================

async def chat_stream(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint for streaming chat responses."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                message = data.get("message", "")
                
                if message:
                    # Send start event
                    await ws.send_json({"type": "start"})
                    
                    system = await get_system()
                    
                    if system and hasattr(system, 'process_input_streaming'):
                        # Stream from core system
                        async for chunk in system.process_input_streaming(message):
                            await ws.send_json({
                                "type": "chunk",
                                "content": chunk.get("content", ""),
                                "done": chunk.get("done", False),
                            })
                    else:
                        # Fallback to non-streaming
                        response = await chat_handler_internal(message)
                        await ws.send_json({
                            "type": "chunk",
                            "content": response["response"],
                            "done": True,
                        })
                    
                    # Send end event
                    await ws.send_json({"type": "end"})
                    
            except Exception as e:
                await ws.send_json({"type": "error", "error": str(e)})
        
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error(f"WebSocket error: {ws.exception()}")
    
    return ws


async def chat_handler_internal(message: str) -> dict:
    """Internal chat handler for WebSocket fallback."""
    system = await get_system()
    
    if system:
        if hasattr(system, 'process_input'):
            result = await system.process_input(message)
            return {"response": result.get("response", "")}
        elif hasattr(system, 'process'):
            result = await system.process(message)
            return {"response": result.get("response", "")}
    
    return {"response": f"[Demo] Received: {message}"}


# =============================================================================
# CORS Middleware
# =============================================================================

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


# =============================================================================
# Application Setup
# =============================================================================

def create_app() -> web.Application:
    """Create and configure the web application."""
    app = web.Application(middlewares=[cors_middleware])
    
    # Health & System
    app.router.add_get("/health", health_check)
    app.router.add_get("/system/state", get_system_state)
    
    # Cognitive/Memory/Belief State
    app.router.add_get("/cognitive", get_cognitive_state)
    app.router.add_get("/memory", get_memory_stats)
    app.router.add_get("/belief", get_belief_state)
    
    # Chat
    app.router.add_post("/chat", chat)
    app.router.add_get("/chat/stream", chat_stream)
    
    # Actions
    app.router.add_post("/force_cortex", force_cortex_handler)
    app.router.add_post("/sleep", force_sleep)
    
    # Legacy compatibility
    app.router.add_get("/stats", get_full_stats)
    app.router.add_get("/buffer", get_memory_stats)  # Alias
    
    # OPTIONS for CORS preflight
    app.router.add_route("OPTIONS", "/{path:.*}", lambda r: web.Response())
    
    return app


async def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AVA API Server v3")
    parser.add_argument("--port", type=int, default=8085, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    app = create_app()
    
    logger.info(f"Starting AVA API Server v3 on http://{args.host}:{args.port}")
    logger.info("=" * 60)
    logger.info("Cortex-Medulla API Endpoints:")
    logger.info("-" * 60)
    logger.info("  GET  /health         - Health check")
    logger.info("  GET  /system/state   - Full system state")
    logger.info("  GET  /cognitive      - Medulla cognitive state")
    logger.info("  GET  /memory         - Titans memory stats")
    logger.info("  GET  /belief         - Active Inference belief state")
    logger.info("  POST /chat           - Send message")
    logger.info("  WS   /chat/stream    - Streaming chat (WebSocket)")
    logger.info("  POST /force_cortex   - Force Cortex for next response")
    logger.info("  POST /sleep          - Trigger sleep cycle")
    logger.info("=" * 60)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    
    logger.info("Server running. Press Ctrl+C to stop.")
    
    # Use an event that we never set - this keeps the server running
    stop_event = asyncio.Event()
    
    try:
        await stop_event.wait()  # This will never complete unless cancelled
    except asyncio.CancelledError:
        logger.info("Server shutting down...")
    finally:
        logger.info("Cleanup complete.")


def run_server():
    """Run the server with proper Windows event loop handling."""
    import sys
    
    if sys.platform == 'win32':
        # Use selector event loop policy for better Windows compatibility
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    finally:
        # Clean shutdown
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


if __name__ == "__main__":
    try:
        run_server()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
