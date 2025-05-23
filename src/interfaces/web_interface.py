#!/usr/bin/env python3
"""
AVA Web Interface Module

This module provides a comprehensive web-based interface for AVA (Afsah's Virtual Assistant)
integrating with Open WebUI, providing real-time streaming, chat interface, system monitoring,
and agentic task visualization. Optimized for NVIDIA RTX A2000 4GB VRAM constraints.

Author: Assistant
Date: 2024
"""

import asyncio
import json
import uuid
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
import traceback
import subprocess
import os
import signal
import atexit

try:
    from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn
    from starlette.websockets import WebSocketDisconnect
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Import AVA core components
from ..core.config import get_config_manager, AVAConfig, InterfaceMode
from ..core.logger import get_logger, LogCategory, log_performance, AVALogger
from ..core.assistant import get_assistant, AssistantRequest, AssistantResponse, ResponseType
from ..core.command_handler import get_command_handler, CommandContext, CommandSource


class WebInterfaceMode(Enum):
    """Web interface operational modes."""
    EMBEDDED = "embedded"       # Embedded FastAPI server
    OPEN_WEBUI = "open_webui"   # Integration with Open WebUI
    HYBRID = "hybrid"           # Both embedded and Open WebUI
    PROXY = "proxy"             # Proxy to external interface


class ConnectionType(Enum):
    """Types of client connections."""
    WEBSOCKET = "websocket"
    HTTP = "http"
    SSE = "sse"
    API = "api"


@dataclass
class ClientConnection:
    """Client connection information."""
    connection_id: str
    connection_type: ConnectionType
    session_id: str
    user_id: Optional[str] = None
    websocket: Optional[WebSocket] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebMessage:
    """Web interface message structure."""
    message_id: str
    message_type: str
    content: Any
    session_id: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:
    class ChatMessage(BaseModel):
        """Chat message model."""
        content: str = Field(..., description="Message content")
        session_id: Optional[str] = Field(None, description="Session ID")
        user_id: Optional[str] = Field(None, description="User ID")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class ChatResponse(BaseModel):
        """Chat response model."""
        response: str = Field(..., description="Assistant response")
        session_id: str = Field(..., description="Session ID")
        processing_time_ms: float = Field(..., description="Processing time in milliseconds")
        function_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Function calls made")
        reasoning_trace: Optional[List[str]] = Field(None, description="Reasoning steps")

    class SystemStatus(BaseModel):
        """System status model."""
        status: str = Field(..., description="System status")
        uptime_seconds: float = Field(..., description="System uptime")
        active_connections: int = Field(..., description="Active connections")
        memory_usage_mb: float = Field(..., description="Memory usage in MB")
        gpu_memory_mb: float = Field(..., description="GPU memory usage in MB")

    class CommandRequest(BaseModel):
        """Command execution request."""
        command: str = Field(..., description="Command to execute")
        session_id: Optional[str] = Field(None, description="Session ID")
        arguments: Dict[str, Any] = Field(default_factory=dict, description="Command arguments")


class OpenWebUIIntegration:
    """Integration with Open WebUI for enhanced GUI capabilities."""
    
    def __init__(self, config: AVAConfig):
        """Initialize Open WebUI integration."""
        self.config = config
        self.logger = get_logger("ava.webui.integration")
        self.docker_client = None
        self.container = None
        self.is_running = False
        
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                self.logger.warning(f"Docker not available: {e}", LogCategory.INTERFACE)
    
    async def start_open_webui(self) -> bool:
        """Start Open WebUI container."""
        try:
            if not self.docker_client:
                self.logger.error("Docker client not available", LogCategory.INTERFACE)
                return False
            
            # Check if container already exists
            try:
                self.container = self.docker_client.containers.get("ava-open-webui")
                if self.container.status == 'running':
                    self.logger.info("Open WebUI container already running", LogCategory.INTERFACE)
                    self.is_running = True
                    return True
                else:
                    self.logger.info("Starting existing Open WebUI container", LogCategory.INTERFACE)
                    self.container.start()
            except docker.errors.NotFound:
                # Create new container
                self.logger.info("Creating new Open WebUI container", LogCategory.INTERFACE)
                
                # Docker configuration for Open WebUI
                container_config = {
                    'image': 'ghcr.io/open-webui/open-webui:main',
                    'name': 'ava-open-webui',
                    'ports': {
                        '8080/tcp': ('127.0.0.1', self.config.interface.web_port)
                    },
                    'environment': {
                        'OLLAMA_BASE_URL': f'http://host.docker.internal:11434',
                        'WEBUI_NAME': 'AVA Assistant',
                        'WEBUI_URL': f'http://localhost:{self.config.interface.web_port}',
                        'ENABLE_SIGNUP': 'false',
                        'DEFAULT_USER_ROLE': 'user'
                    },
                    'volumes': {
                        'open-webui': {'bind': '/app/backend/data', 'mode': 'rw'}
                    },
                    'detach': True,
                    'auto_remove': False
                }
                
                # Add GPU support if available
                if self.config.hardware.enable_gpu:
                    container_config['runtime'] = 'nvidia'
                    container_config['environment']['NVIDIA_VISIBLE_DEVICES'] = 'all'
                
                self.container = self.docker_client.containers.run(**container_config)
            
            # Wait for container to be ready
            await self._wait_for_container_ready()
            self.is_running = True
            
            self.logger.info(f"Open WebUI started successfully at http://localhost:{self.config.interface.web_port}",
                           LogCategory.INTERFACE)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Open WebUI: {e}", LogCategory.INTERFACE)
            return False
    
    async def _wait_for_container_ready(self, timeout: int = 60):
        """Wait for Open WebUI container to be ready."""
        import aiohttp
        
        start_time = time.time()
        url = f"http://localhost:{self.config.interface.web_port}"
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            return
            except:
                pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Open WebUI container did not become ready in time")
    
    async def stop_open_webui(self):
        """Stop Open WebUI container."""
        try:
            if self.container:
                self.container.stop()
                self.is_running = False
                self.logger.info("Open WebUI container stopped", LogCategory.INTERFACE)
        except Exception as e:
            self.logger.error(f"Error stopping Open WebUI: {e}", LogCategory.INTERFACE)
    
    def get_webui_url(self) -> str:
        """Get Open WebUI URL."""
        return f"http://localhost:{self.config.interface.web_port}"


class WebSocketManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: Dict[str, ClientConnection] = {}
        self.logger = get_logger("ava.websocket.manager")
    
    async def connect(self, websocket: WebSocket, connection_id: str, session_id: str) -> ClientConnection:
        """Accept new WebSocket connection."""
        try:
            await websocket.accept()
            
            connection = ClientConnection(
                connection_id=connection_id,
                connection_type=ConnectionType.WEBSOCKET,
                session_id=session_id,
                websocket=websocket
            )
            
            self.active_connections[connection_id] = connection
            
            self.logger.info(f"WebSocket connected: {connection_id}", LogCategory.INTERFACE,
                           connection_id=connection_id, session_id=session_id)
            
            return connection
            
        except Exception as e:
            self.logger.error(f"Error connecting WebSocket: {e}", LogCategory.INTERFACE)
            raise
    
    async def disconnect(self, connection_id: str):
        """Disconnect WebSocket."""
        try:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
                self.logger.info(f"WebSocket disconnected: {connection_id}", LogCategory.INTERFACE)
        except Exception as e:
            self.logger.error(f"Error disconnecting WebSocket: {e}", LogCategory.INTERFACE)
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection."""
        try:
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                if connection.websocket:
                    await connection.websocket.send_json(message)
                    connection.last_activity = datetime.now()
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {e}", LogCategory.INTERFACE)
            # Remove failed connection
            await self.disconnect(connection_id)
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_connection: Optional[str] = None):
        """Broadcast message to all connections."""
        try:
            for connection_id, connection in list(self.active_connections.items()):
                if connection_id != exclude_connection and connection.websocket:
                    try:
                        await connection.websocket.send_json(message)
                        connection.last_activity = datetime.now()
                    except:
                        await self.disconnect(connection_id)
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}", LogCategory.INTERFACE)
    
    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active connections."""
        return [
            {
                'connection_id': conn.connection_id,
                'session_id': conn.session_id,
                'user_id': conn.user_id,
                'connected_at': conn.connected_at.isoformat(),
                'last_activity': conn.last_activity.isoformat()
            }
            for conn in self.active_connections.values()
        ]


class AVAWebInterface:
    """Main web interface for AVA."""
    
    def __init__(self, config: Optional[AVAConfig] = None):
        """Initialize web interface."""
        self.config = config or get_config_manager().get_config()
        self.logger = get_logger("ava.web.interface", {
            'log_level': self.config.logging.level.value
        })
        
        # Core components
        self.assistant = None
        self.command_handler = get_command_handler(self.config)
        
        # Web components
        self.app = None
        self.websocket_manager = WebSocketManager()
        self.open_webui = OpenWebUIIntegration(self.config)
        
        # State management
        self.is_running = False
        self.server_task: Optional[asyncio.Task] = None
        self._shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'websocket_connections': 0,
            'start_time': datetime.now()
        }
        
        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self._setup_fastapi_app()
        
        self.logger.info("Web interface initialized", LogCategory.INTERFACE)
    
    def _setup_fastapi_app(self):
        """Setup FastAPI application."""
        self.app = FastAPI(
            title="AVA Web Interface",
            description="Afsah's Virtual Assistant Web API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security
        security = HTTPBearer(auto_error=False)
        
        # Setup routes
        self._setup_routes()
        
        # Setup WebSocket routes
        self._setup_websocket_routes()
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Root endpoint with basic web interface."""
            return self._get_basic_html_interface()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/status", response_model=SystemStatus)
        async def get_status():
            """Get system status."""
            try:
                if self.assistant:
                    assistant_status = self.assistant.get_status()
                    return SystemStatus(
                        status=assistant_status.state.value,
                        uptime_seconds=assistant_status.uptime_seconds,
                        active_connections=len(self.websocket_manager.active_connections),
                        memory_usage_mb=assistant_status.memory_usage_mb,
                        gpu_memory_mb=assistant_status.gpu_memory_usage_mb
                    )
                else:
                    uptime = (datetime.now() - self.stats['start_time']).total_seconds()
                    return SystemStatus(
                        status="initializing",
                        uptime_seconds=uptime,
                        active_connections=len(self.websocket_manager.active_connections),
                        memory_usage_mb=0.0,
                        gpu_memory_mb=0.0
                    )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(message: ChatMessage):
            """Chat endpoint for direct API access."""
            try:
                start_time = time.time()
                self.stats['requests_total'] += 1
                
                if not self.assistant:
                    raise HTTPException(status_code=503, detail="Assistant not available")
                
                # Create assistant request
                request = AssistantRequest(
                    request_id=str(uuid.uuid4()),
                    session_id=message.session_id or str(uuid.uuid4()),
                    user_input=message.content,
                    metadata=message.metadata
                )
                
                # Process request
                response = await self.assistant.process_request(request)
                
                processing_time = (time.time() - start_time) * 1000
                self.stats['requests_successful'] += 1
                
                return ChatResponse(
                    response=response.content,
                    session_id=response.session_id,
                    processing_time_ms=processing_time,
                    function_calls=response.function_calls,
                    reasoning_trace=response.reasoning_trace
                )
                
            except Exception as e:
                self.stats['requests_failed'] += 1
                self.logger.error(f"Error in chat endpoint: {e}", LogCategory.INTERFACE)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/command")
        async def execute_command(command_request: CommandRequest):
            """Execute command endpoint."""
            try:
                context = CommandContext(
                    source=CommandSource.API,
                    session_id=command_request.session_id or str(uuid.uuid4())
                )
                
                result = await self.command_handler.execute_command(
                    command_request.command, context
                )
                
                return {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms
                }
                
            except Exception as e:
                self.logger.error(f"Error executing command: {e}", LogCategory.INTERFACE)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stream/{session_id}")
        async def stream_chat(session_id: str, query: str):
            """Server-Sent Events endpoint for streaming responses."""
            async def generate():
                try:
                    if not self.assistant:
                        yield f"data: {json.dumps({'error': 'Assistant not available'})}\n\n"
                        return
                    
                    request = AssistantRequest(
                        request_id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_input=query
                    )
                    
                    # Stream response
                    async for token in self.assistant.stream_response(request):
                        data = json.dumps({"token": token, "type": "token"})
                        yield f"data: {data}\n\n"
                        await asyncio.sleep(0.01)
                    
                    # Send completion signal
                    completion_data = json.dumps({"type": "complete"})
                    yield f"data: {completion_data}\n\n"
                    
                except Exception as e:
                    error_data = json.dumps({"error": str(e), "type": "error"})
                    yield f"data: {error_data}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
        
        @self.app.get("/sessions")
        async def list_sessions():
            """List active sessions."""
            try:
                if self.assistant:
                    return self.assistant.list_active_sessions()
                else:
                    return []
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/connections")
        async def list_connections():
            """List active WebSocket connections."""
            return self.websocket_manager.get_active_connections()
        
        @self.app.get("/stats")
        async def get_stats():
            """Get interface statistics."""
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
            return {
                **self.stats,
                'uptime_seconds': uptime,
                'active_connections': len(self.websocket_manager.active_connections)
            }
    
    def _setup_websocket_routes(self):
        """Setup WebSocket routes."""
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for real-time communication."""
            connection_id = str(uuid.uuid4())
            connection = None
            
            try:
                # Connect WebSocket
                connection = await self.websocket_manager.connect(
                    websocket, connection_id, session_id
                )
                self.stats['websocket_connections'] += 1
                
                # Send welcome message
                await self.websocket_manager.send_message(connection_id, {
                    "type": "connected",
                    "connection_id": connection_id,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Message loop
                while True:
                    try:
                        # Receive message
                        data = await websocket.receive_json()
                        
                        # Process message
                        await self._process_websocket_message(
                            connection_id, session_id, data
                        )
                        
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        self.logger.error(f"Error processing WebSocket message: {e}",
                                        LogCategory.INTERFACE)
                        await self.websocket_manager.send_message(connection_id, {
                            "type": "error",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
            
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}", LogCategory.INTERFACE)
            
            finally:
                if connection:
                    await self.websocket_manager.disconnect(connection_id)
    
    async def _process_websocket_message(self, connection_id: str, session_id: str, data: Dict[str, Any]):
        """Process incoming WebSocket message."""
        try:
            message_type = data.get("type", "unknown")
            
            if message_type == "chat":
                # Handle chat message
                content = data.get("content", "")
                if not content:
                    return
                
                if self.assistant:
                    # Create assistant request
                    request = AssistantRequest(
                        request_id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_input=content
                    )
                    
                    # Send typing indicator
                    await self.websocket_manager.send_message(connection_id, {
                        "type": "typing",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Check if streaming is requested
                    if data.get("stream", False):
                        # Stream response
                        await self._stream_websocket_response(
                            connection_id, session_id, request
                        )
                    else:
                        # Regular response
                        response = await self.assistant.process_request(request)
                        
                        await self.websocket_manager.send_message(connection_id, {
                            "type": "response",
                            "content": response.content,
                            "processing_time_ms": response.processing_time_ms,
                            "function_calls": response.function_calls,
                            "reasoning_trace": response.reasoning_trace,
                            "timestamp": datetime.now().isoformat()
                        })
                else:
                    await self.websocket_manager.send_message(connection_id, {
                        "type": "error",
                        "error": "Assistant not available",
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == "command":
                # Handle command execution
                command = data.get("command", "")
                if command:
                    context = CommandContext(
                        source=CommandSource.API,
                        session_id=session_id
                    )
                    
                    result = await self.command_handler.execute_command(command, context)
                    
                    await self.websocket_manager.send_message(connection_id, {
                        "type": "command_result",
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                        "execution_time_ms": result.execution_time_ms,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == "ping":
                # Handle ping/pong
                await self.websocket_manager.send_message(connection_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            else:
                self.logger.warning(f"Unknown WebSocket message type: {message_type}",
                                  LogCategory.INTERFACE)
                
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}", LogCategory.INTERFACE)
            await self.websocket_manager.send_message(connection_id, {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def _stream_websocket_response(self, connection_id: str, session_id: str, request: AssistantRequest):
        """Stream assistant response via WebSocket."""
        try:
            async for token in self.assistant.stream_response(request):
                await self.websocket_manager.send_message(connection_id, {
                    "type": "stream_token",
                    "token": token,
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(0.01)
            
            # Send completion signal
            await self.websocket_manager.send_message(connection_id, {
                "type": "stream_complete",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            await self.websocket_manager.send_message(connection_id, {
                "type": "stream_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    def _get_basic_html_interface(self) -> str:
        """Get basic HTML interface for embedded mode."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AVA - Afsah's Virtual Assistant</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #764ba2;
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            color: #666;
            margin: 10px 0;
        }
        .chat-container {
            border: 1px solid #ddd;
            border-radius: 8px;
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            margin-bottom: 20px;
            background: #f9f9f9;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        .user-message {
            background: #e3f2fd;
            text-align: right;
        }
        .assistant-message {
            background: #f3e5f5;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
        }
        .input-area button {
            padding: 12px 24px;
            background: #764ba2;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        }
        .input-area button:hover {
            background: #5a3780;
        }
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            background: #e8f5e8;
            border-radius: 6px;
            color: #2e7d32;
        }
        .links {
            text-align: center;
            margin-top: 30px;
        }
        .links a {
            color: #764ba2;
            text-decoration: none;
            margin: 0 15px;
            font-weight: 500;
        }
        .links a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AVA</h1>
            <p>Afsah's Virtual Assistant</p>
            <p>Local Agentic AI - Optimized for RTX A2000 4GB</p>
        </div>
        
        <div class="status" id="status">
            🟢 System Ready - WebSocket Connected
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message assistant-message">
                <strong>AVA:</strong> Hello! I'm AVA, your local AI assistant. How can I help you today?
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <div class="links">
            <a href="/docs">📚 API Documentation</a>
            <a href="/status">📊 System Status</a>
            <a href="/stats">📈 Statistics</a>
        </div>
    </div>

    <script>
        let websocket = null;
        let sessionId = 'web_' + Date.now();
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
            
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function(event) {
                updateStatus('🟢 Connected to AVA', 'success');
            };
            
            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            websocket.onclose = function(event) {
                updateStatus('🔴 Disconnected - Attempting to reconnect...', 'error');
                setTimeout(connectWebSocket, 3000);
            };
            
            websocket.onerror = function(error) {
                updateStatus('⚠️ Connection error', 'error');
            };
        }
        
        function handleWebSocketMessage(data) {
            const chatContainer = document.getElementById('chatContainer');
            
            if (data.type === 'response') {
                addMessage('AVA', data.content, 'assistant');
            } else if (data.type === 'error') {
                addMessage('System', `Error: ${data.error}`, 'error');
            } else if (data.type === 'typing') {
                updateStatus('🤔 AVA is thinking...', 'thinking');
            }
        }
        
        function addMessage(sender, content, type) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type === 'assistant' ? 'assistant-message' : 'user-message'}`;
            messageDiv.innerHTML = `<strong>${sender}:</strong> ${content}`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            updateStatus('🟢 Ready', 'success');
        }
        
        function updateStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (message && websocket && websocket.readyState === WebSocket.OPEN) {
                addMessage('You', message, 'user');
                
                websocket.send(JSON.stringify({
                    type: 'chat',
                    content: message,
                    stream: false
                }));
                
                input.value = '';
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Initialize connection
        connectWebSocket();
    </script>
</body>
</html>
        """
    
    async def initialize(self) -> bool:
        """Initialize web interface components."""
        try:
            self.logger.info("Initializing web interface", LogCategory.INTERFACE)
            
            # Initialize assistant
            self.assistant = await get_assistant()
            
            # Initialize Open WebUI if configured
            if (self.config.interface.mode in [InterfaceMode.GUI, InterfaceMode.BOTH] and 
                self.config.interface.enable_web_ui):
                
                if await self.open_webui.start_open_webui():
                    self.logger.info("Open WebUI integration initialized", LogCategory.INTERFACE)
                else:
                    self.logger.warning("Failed to initialize Open WebUI", LogCategory.INTERFACE)
            
            self.logger.info("Web interface initialized successfully", LogCategory.INTERFACE)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize web interface: {e}", LogCategory.INTERFACE)
            return False
    
    async def start_server(self) -> bool:
        """Start the web server."""
        try:
            if not FASTAPI_AVAILABLE:
                self.logger.error("FastAPI not available - cannot start web server", LogCategory.INTERFACE)
                return False
            
            self.logger.info(f"Starting web server on port {self.config.interface.web_port}", 
                           LogCategory.INTERFACE)
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=self.config.interface.web_port,
                log_level="info",
                access_log=False  # Disable access log to avoid conflicts with our logger
            )
            
            server = uvicorn.Server(config)
            
            # Start server in background task
            self.server_task = asyncio.create_task(server.serve())
            self.is_running = True
            
            self.logger.info(f"Web server started at http://localhost:{self.config.interface.web_port}",
                           LogCategory.INTERFACE)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}", LogCategory.INTERFACE)
            return False
    
    async def stop_server(self):
        """Stop the web server."""
        try:
            self.logger.info("Stopping web server", LogCategory.INTERFACE)
            
            if self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
            
            # Stop Open WebUI
            await self.open_webui.stop_open_webui()
            
            self.is_running = False
            self.logger.info("Web server stopped", LogCategory.INTERFACE)
            
        except Exception as e:
            self.logger.error(f"Error stopping web server: {e}", LogCategory.INTERFACE)
    
    async def shutdown(self):
        """Shutdown the web interface."""
        try:
            self.logger.info("Shutting down web interface", LogCategory.INTERFACE)
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop server
            await self.stop_server()
            
            # Shutdown assistant
            if self.assistant:
                await self.assistant.shutdown()
            
            # Shutdown command handler
            if self.command_handler:
                await self.command_handler.shutdown()
            
            self.logger.info("Web interface shutdown completed", LogCategory.INTERFACE)
            
        except Exception as e:
            self.logger.error(f"Error during web interface shutdown: {e}", LogCategory.INTERFACE)
    
    def get_interface_stats(self) -> Dict[str, Any]:
        """Get web interface statistics."""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'active_connections': len(self.websocket_manager.active_connections),
            'open_webui_running': self.open_webui.is_running,
            'server_running': self.is_running
        }


# Global web interface instance
_global_web_interface: Optional[AVAWebInterface] = None


async def get_web_interface(config: Optional[AVAConfig] = None) -> AVAWebInterface:
    """Get global web interface instance."""
    global _global_web_interface
    if _global_web_interface is None:
        _global_web_interface = AVAWebInterface(config)
        await _global_web_interface.initialize()
    return _global_web_interface


async def start_web_interface(config: Optional[AVAConfig] = None, port: Optional[int] = None) -> AVAWebInterface:
    """Start web interface with optional port override."""
    web_interface = await get_web_interface(config)
    
    if port:
        web_interface.config.interface.web_port = port
    
    await web_interface.start_server()
    return web_interface


# Context manager for web interface lifecycle
from contextlib import asynccontextmanager

@asynccontextmanager
async def web_interface_context(config: Optional[AVAConfig] = None):
    """Context manager for web interface lifecycle."""
    web_interface = None
    try:
        web_interface = AVAWebInterface(config)
        await web_interface.initialize()
        await web_interface.start_server()
        yield web_interface
    finally:
        if web_interface:
            await web_interface.shutdown()


# Tunnel integration for remote access
class TunnelManager:
    """Manages secure tunneling for remote access."""
    
    def __init__(self, config: AVAConfig):
        """Initialize tunnel manager."""
        self.config = config
        self.logger = get_logger("ava.tunnel.manager")
        self.tunnel_process = None
        self.tunnel_url = None
    
    async def start_tunnel(self, service: str = "localtonet") -> Optional[str]:
        """Start secure tunnel."""
        try:
            if service == "localtonet":
                return await self._start_localtonet()
            elif service == "ngrok":
                return await self._start_ngrok()
            else:
                self.logger.error(f"Unknown tunnel service: {service}", LogCategory.INTERFACE)
                return None
        except Exception as e:
            self.logger.error(f"Failed to start tunnel: {e}", LogCategory.INTERFACE)
            return None
    
    async def _start_localtonet(self) -> Optional[str]:
        """Start LocalToNet tunnel."""
        try:
            cmd = [
                "localtonet",
                "http",
                str(self.config.interface.web_port),
                "--subdomain", "ava-assistant"
            ]
            
            self.tunnel_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for tunnel URL (simplified - would need proper parsing)
            await asyncio.sleep(5)
            
            # Mock URL for example
            self.tunnel_url = f"https://ava-assistant.localtonet.com"
            self.logger.info(f"LocalToNet tunnel started: {self.tunnel_url}", LogCategory.INTERFACE)
            
            return self.tunnel_url
            
        except Exception as e:
            self.logger.error(f"LocalToNet tunnel error: {e}", LogCategory.INTERFACE)
            return None
    
    async def _start_ngrok(self) -> Optional[str]:
        """Start ngrok tunnel."""
        try:
            cmd = [
                "ngrok",
                "http",
                str(self.config.interface.web_port),
                "--log", "stdout"
            ]
            
            self.tunnel_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait and parse ngrok output for URL
            await asyncio.sleep(5)
            
            # Mock URL for example
            self.tunnel_url = f"https://random-id.ngrok.io"
            self.logger.info(f"Ngrok tunnel started: {self.tunnel_url}", LogCategory.INTERFACE)
            
            return self.tunnel_url
            
        except Exception as e:
            self.logger.error(f"Ngrok tunnel error: {e}", LogCategory.INTERFACE)
            return None
    
    async def stop_tunnel(self):
        """Stop the tunnel."""
        try:
            if self.tunnel_process:
                self.tunnel_process.terminate()
                await self.tunnel_process.wait()
                self.tunnel_process = None
                self.tunnel_url = None
                self.logger.info("Tunnel stopped", LogCategory.INTERFACE)
        except Exception as e:
            self.logger.error(f"Error stopping tunnel: {e}", LogCategory.INTERFACE)


# Testing functions
async def test_web_interface():
    """Test web interface functionality."""
    print("Testing AVA Web Interface...")
    
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available - cannot test web interface")
        return
    
    try:
        async with web_interface_context() as web_interface:
            # Test basic functionality
            print(f"✅ Web interface created")
            
            # Test stats
            stats = web_interface.get_interface_stats()
            print(f"✅ Stats retrieved: {stats['uptime_seconds']:.2f}s uptime")
            
            # Test WebSocket manager
            connections = web_interface.websocket_manager.get_active_connections()
            print(f"✅ WebSocket manager working: {len(connections)} connections")
            
            print("✅ Web interface test completed successfully!")
            
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        traceback.print_exc()


# Main function for standalone execution
async def main():
    """Main function for running web interface standalone."""
    try:
        print("🚀 Starting AVA Web Interface...")
        
        # Get configuration
        config = get_config_manager().get_config()
        
        # Create and start web interface
        web_interface = await start_web_interface(config)
        
        print(f"🌐 Web interface running at http://localhost:{config.interface.web_port}")
        if web_interface.open_webui.is_running:
            print(f"🎨 Open WebUI available at {web_interface.open_webui.get_webui_url()}")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler():
            print("\n🛑 Shutting down...")
            asyncio.create_task(web_interface.shutdown())
        
        import signal
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda s, f: signal_handler())
        
        # Keep running until shutdown
        try:
            while web_interface.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        
        await web_interface.shutdown()
        print("👋 AVA Web Interface stopped")
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--test":
        asyncio.run(test_web_interface())
    else:
        exit_code = asyncio.run(main())
        os.sys.exit(exit_code)
