#!/usr/bin/env python3
"""
AVA Core Assistant Module

This module provides the main AVA Assistant orchestrator that integrates all core components
including configuration, logging, scheduling, dialogue management, function calling, reasoning,
and MCP integration. Optimized for NVIDIA RTX A2000 4GB VRAM constraints.

Author: Assistant
Date: 2024
"""

import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
from contextlib import asynccontextmanager
import queue
import traceback

# Import AVA core components
from .config import get_config_manager, AVAConfig, ModelBackend, InterfaceMode
from .logger import get_logger, LogCategory, log_performance, AVALogger
from .scheduler import TaskScheduler, TaskPriority, get_scheduler

# Import AVA modules (these would be from other completed modules)
# from ..ava_core.agent import AVAAgent
# from ..ava_core.dialogue_manager import DialogueManager
# from ..ava_core.function_calling import FunctionCallingEngine
# from ..ava_core.reasoning import ReasoningEngine
# from ..mcp_integration.mcp_host import MCPHost


class AssistantState(Enum):
    """Assistant operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class ResponseType(Enum):
    """Types of assistant responses."""
    TEXT = "text"
    STRUCTURED = "structured"
    FUNCTION_CALL = "function_call"
    ERROR = "error"
    STREAM = "stream"


@dataclass
class ConversationContext:
    """Conversation context tracking."""
    session_id: str
    user_id: Optional[str] = None
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, Any]] = field(default_factory=list)
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    total_tokens: int = 0
    total_function_calls: int = 0


@dataclass
class AssistantRequest:
    """Request structure for assistant processing."""
    request_id: str
    session_id: str
    user_input: str
    context: Optional[ConversationContext] = None
    options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AssistantResponse:
    """Response structure from assistant."""
    request_id: str
    session_id: str
    response_type: ResponseType
    content: Any
    reasoning_trace: Optional[List[str]] = None
    function_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    token_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemStatus:
    """System status monitoring."""
    state: AssistantState
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    active_sessions: int
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    avg_response_time_ms: float
    model_loaded: bool
    scheduler_running: bool
    mcp_connected: bool
    last_update: datetime = field(default_factory=datetime.now)


class AVAAssistant:
    """Main AVA Assistant orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize AVA Assistant."""
        self.assistant_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # Core components
        self.config_manager = get_config_manager(config_path)
        self.config = self.config_manager.get_config()
        self.logger = get_logger("ava.assistant", {
            'log_level': self.config.logging.level.value,
            'enable_file_logging': self.config.logging.enable_file_logging,
            'enable_console_logging': self.config.logging.enable_console_logging
        })
        
        # State management
        self.state = AssistantState.INITIALIZING
        self._shutdown_event = threading.Event()
        self._request_queue = asyncio.Queue()
        self._active_sessions: Dict[str, ConversationContext] = {}
        self._request_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'response_times': []
        }
        
        # Task scheduling
        self.scheduler = get_scheduler(
            max_workers=self.config.performance.max_worker_threads,
            gpu_memory_limit_mb=self.config.performance.memory_threshold_mb
        )
        
        # Component placeholders (to be initialized)
        self.agent = None
        self.dialogue_manager = None
        self.function_engine = None
        self.reasoning_engine = None
        self.mcp_host = None
        self.model_backend = None
        
        # Threading
        self._processing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("AVA Assistant initialized", LogCategory.SYSTEM, 
                        assistant_id=self.assistant_id)
    
    async def initialize(self) -> bool:
        """Initialize all assistant components."""
        try:
            self.logger.info("Starting AVA Assistant initialization", LogCategory.SYSTEM)
            
            with log_performance(self.logger, "assistant_initialization"):
                # Initialize model backend
                await self._initialize_model_backend()
                
                # Initialize core engines (placeholder implementations)
                await self._initialize_engines()
                
                # Initialize MCP if enabled
                if self.config.agent.enable_mcp:
                    await self._initialize_mcp()
                
                # Start background tasks
                await self._start_background_tasks()
                
                # Schedule maintenance tasks
                self._schedule_maintenance_tasks()
                
            self.state = AssistantState.READY
            self.logger.info("AVA Assistant initialization completed", LogCategory.SYSTEM)
            return True
            
        except Exception as e:
            self.state = AssistantState.ERROR
            self.logger.error(f"Failed to initialize AVA Assistant: {e}", LogCategory.SYSTEM)
            return False
    
    async def _initialize_model_backend(self):
        """Initialize the model backend (Ollama/Transformers/Custom)."""
        try:
            if self.config.model.backend == ModelBackend.OLLAMA:
                # Initialize Ollama backend
                self.logger.info("Initializing Ollama backend", LogCategory.MODEL)
                # Placeholder for Ollama initialization
                self.model_backend = "ollama_initialized"
                
            elif self.config.model.backend == ModelBackend.TRANSFORMERS:
                # Initialize Transformers backend
                self.logger.info("Initializing Transformers backend", LogCategory.MODEL)
                # Placeholder for Transformers initialization
                self.model_backend = "transformers_initialized"
                
            else:
                # Custom backend
                self.logger.info("Initializing custom backend", LogCategory.MODEL)
                self.model_backend = "custom_initialized"
            
            self.logger.info(f"Model backend initialized: {self.config.model.backend.value}", 
                           LogCategory.MODEL)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model backend: {e}", LogCategory.MODEL)
            raise
    
    async def _initialize_engines(self):
        """Initialize core processing engines."""
        try:
            # Initialize dialogue manager
            self.logger.info("Initializing dialogue manager", LogCategory.AGENT)
            # self.dialogue_manager = DialogueManager(self.config)
            self.dialogue_manager = "dialogue_manager_placeholder"
            
            # Initialize function calling engine
            if self.config.agent.enable_tool_use:
                self.logger.info("Initializing function calling engine", LogCategory.TOOL)
                # self.function_engine = FunctionCallingEngine(self.config)
                self.function_engine = "function_engine_placeholder"
            
            # Initialize reasoning engine
            if self.config.agent.enable_reasoning_trace:
                self.logger.info("Initializing reasoning engine", LogCategory.AGENT)
                # self.reasoning_engine = ReasoningEngine(self.config)
                self.reasoning_engine = "reasoning_engine_placeholder"
            
            # Initialize main agent
            self.logger.info("Initializing main agent", LogCategory.AGENT)
            # self.agent = AVAAgent(self.config, self.dialogue_manager, 
            #                       self.function_engine, self.reasoning_engine)
            self.agent = "agent_placeholder"
            
            self.logger.info("Core engines initialized successfully", LogCategory.AGENT)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize engines: {e}", LogCategory.AGENT)
            raise
    
    async def _initialize_mcp(self):
        """Initialize Model Context Protocol host."""
        try:
            self.logger.info("Initializing MCP host", LogCategory.SYSTEM)
            # self.mcp_host = MCPHost(self.config)
            # await self.mcp_host.initialize()
            self.mcp_host = "mcp_host_placeholder"
            
            self.logger.info("MCP host initialized successfully", LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP host: {e}", LogCategory.SYSTEM)
            raise
    
    async def _start_background_tasks(self):
        """Start background processing tasks."""
        try:
            # Start request processing task
            self._processing_task = asyncio.create_task(self._process_requests())
            
            # Start monitoring task
            self._monitoring_task = asyncio.create_task(self._monitor_system())
            
            self.logger.info("Background tasks started", LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}", LogCategory.SYSTEM)
            raise
    
    def _schedule_maintenance_tasks(self):
        """Schedule periodic maintenance tasks."""
        try:
            # Schedule session cleanup
            self.scheduler.schedule_task(
                name="session_cleanup",
                func=self._cleanup_expired_sessions,
                interval_seconds=300,  # Every 5 minutes
                priority=TaskPriority.LOW
            )
            
            # Schedule memory monitoring
            self.scheduler.schedule_task(
                name="memory_monitor",
                func=self._monitor_memory_usage,
                interval_seconds=60,  # Every minute
                priority=TaskPriority.LOW
            )
            
            # Schedule stats logging
            self.scheduler.schedule_task(
                name="stats_logging",
                func=self._log_system_stats,
                interval_seconds=600,  # Every 10 minutes
                priority=TaskPriority.LOW
            )
            
            self.logger.info("Maintenance tasks scheduled", LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Failed to schedule maintenance tasks: {e}", LogCategory.SYSTEM)
    
    async def process_request(self, request: AssistantRequest) -> AssistantResponse:
        """Process a user request."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing request {request.request_id}", LogCategory.AGENT,
                           request_id=request.request_id, session_id=request.session_id)
            
            # Update request stats
            self._request_stats['total'] += 1
            
            # Get or create conversation context
            context = await self._get_or_create_context(request)
            
            # Process the request through the agent pipeline
            response = await self._process_with_agent(request, context)
            
            # Update context and session
            await self._update_context(context, request, response)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time
            self._request_stats['response_times'].append(processing_time)
            self._request_stats['successful'] += 1
            
            self.logger.info(f"Request {request.request_id} processed successfully", 
                           LogCategory.AGENT, request_id=request.request_id,
                           duration_ms=processing_time)
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._request_stats['failed'] += 1
            
            error_response = AssistantResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                response_type=ResponseType.ERROR,
                content=f"Error processing request: {str(e)}",
                error=str(e),
                processing_time_ms=processing_time
            )
            
            self.logger.error(f"Error processing request {request.request_id}: {e}", 
                            LogCategory.AGENT, request_id=request.request_id,
                            duration_ms=processing_time, exc_info=True)
            
            return error_response
    
    async def _process_with_agent(self, request: AssistantRequest, 
                                 context: ConversationContext) -> AssistantResponse:
        """Process request through the agent pipeline."""
        try:
            # For now, return a placeholder response
            # In the full implementation, this would:
            # 1. Use dialogue manager to understand intent
            # 2. Apply reasoning if needed
            # 3. Execute function calls if required
            # 4. Generate final response
            
            response_content = f"AVA Assistant processed: {request.user_input}"
            
            return AssistantResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                response_type=ResponseType.TEXT,
                content=response_content,
                reasoning_trace=["Placeholder reasoning step"],
                metadata={
                    'model_used': self.config.model.model_name,
                    'backend': self.config.model.backend.value,
                    'context_length': len(context.messages)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in agent processing: {e}", LogCategory.AGENT)
            raise
    
    async def _get_or_create_context(self, request: AssistantRequest) -> ConversationContext:
        """Get existing or create new conversation context."""
        if request.session_id in self._active_sessions:
            context = self._active_sessions[request.session_id]
            context.last_activity = datetime.now()
            return context
        
        context = ConversationContext(
            session_id=request.session_id,
            user_id=request.metadata.get('user_id')
        )
        
        self._active_sessions[request.session_id] = context
        
        self.logger.info(f"Created new conversation context for session {request.session_id}",
                        LogCategory.AGENT, session_id=request.session_id)
        
        return context
    
    async def _update_context(self, context: ConversationContext, 
                            request: AssistantRequest, response: AssistantResponse):
        """Update conversation context with request and response."""
        try:
            # Add user message
            context.messages.append({
                'role': 'user',
                'content': request.user_input,
                'timestamp': request.timestamp.isoformat()
            })
            
            # Add assistant response
            context.messages.append({
                'role': 'assistant',
                'content': response.content,
                'timestamp': response.timestamp.isoformat()
            })
            
            # Update function calls if any
            if response.function_calls:
                context.function_calls.extend(response.function_calls)
                context.total_function_calls += len(response.function_calls)
            
            # Update reasoning trace if any
            if response.reasoning_trace:
                context.reasoning_trace.extend(response.reasoning_trace)
            
            # Trim context if it exceeds limits
            max_messages = self.config.agent.conversation_context_limit
            if len(context.messages) > max_messages:
                # Keep system messages and recent messages
                context.messages = context.messages[-max_messages:]
            
            context.last_activity = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating context: {e}", LogCategory.AGENT)
    
    async def stream_response(self, request: AssistantRequest) -> AsyncGenerator[str, None]:
        """Stream response tokens for real-time interaction."""
        try:
            self.logger.info(f"Starting stream response for request {request.request_id}",
                           LogCategory.AGENT, request_id=request.request_id)
            
            # Get context
            context = await self._get_or_create_context(request)
            
            # Simulate streaming response (in real implementation, this would
            # stream from the actual model)
            response_text = f"AVA is processing your request: {request.user_input}"
            words = response_text.split()
            
            for i, word in enumerate(words):
                if self._shutdown_event.is_set():
                    break
                
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.1)  # Simulate processing time
            
            # Update context after streaming
            response = AssistantResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                response_type=ResponseType.STREAM,
                content=response_text
            )
            await self._update_context(context, request, response)
            
        except Exception as e:
            self.logger.error(f"Error in stream response: {e}", LogCategory.AGENT)
            yield f"Error: {str(e)}"
    
    async def _process_requests(self):
        """Background task to process queued requests."""
        while not self._shutdown_event.is_set():
            try:
                # This would be used for background request processing
                # if requests were queued for batch processing
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in request processing task: {e}", LogCategory.SYSTEM)
                await asyncio.sleep(1.0)
    
    async def _monitor_system(self):
        """Background task to monitor system health."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_system_health()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}", LogCategory.SYSTEM)
                await asyncio.sleep(5.0)
    
    async def _check_system_health(self):
        """Check overall system health."""
        try:
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / 1024 / 1024
            
            # Get GPU memory usage if available
            gpu_memory_usage_mb = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
            
            # Check if memory usage is too high
            if memory_usage_mb > self.config.performance.memory_threshold_mb:
                self.logger.warning(f"High memory usage detected: {memory_usage_mb:.1f}MB",
                                  LogCategory.PERFORMANCE, memory_usage_mb=memory_usage_mb)
            
            # Log performance metrics
            self.logger.performance("System health check",
                                   memory_usage_mb=memory_usage_mb,
                                   gpu_memory_mb=gpu_memory_usage_mb,
                                   active_sessions=len(self._active_sessions))
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}", LogCategory.SYSTEM)
    
    def _cleanup_expired_sessions(self):
        """Clean up expired conversation sessions."""
        try:
            current_time = datetime.now()
            timeout_minutes = self.config.security.session_timeout_minutes
            expired_sessions = []
            
            for session_id, context in self._active_sessions.items():
                if (current_time - context.last_activity).total_seconds() > timeout_minutes * 60:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._active_sessions[session_id]
                self.logger.info(f"Cleaned up expired session {session_id}", LogCategory.SYSTEM)
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions",
                               LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {e}", LogCategory.SYSTEM)
    
    def _monitor_memory_usage(self):
        """Monitor and log memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / 1024 / 1024
            memory_percent = memory.percent
            
            # Get GPU memory if available
            gpu_memory_usage_mb = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
            
            self.logger.performance("Memory usage monitor",
                                   memory_usage_mb=memory_usage_mb,
                                   memory_percent=memory_percent,
                                   gpu_memory_mb=gpu_memory_usage_mb)
            
            # Trigger garbage collection if memory usage is high
            if memory_percent > 85:
                import gc
                gc.collect()
                self.logger.info("Triggered garbage collection due to high memory usage",
                               LogCategory.PERFORMANCE)
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {e}", LogCategory.PERFORMANCE)
    
    def _log_system_stats(self):
        """Log comprehensive system statistics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            avg_response_time = (
                sum(self._request_stats['response_times']) / 
                len(self._request_stats['response_times'])
                if self._request_stats['response_times'] else 0
            )
            
            stats = {
                'uptime_seconds': uptime,
                'total_requests': self._request_stats['total'],
                'successful_requests': self._request_stats['successful'],
                'failed_requests': self._request_stats['failed'],
                'success_rate': (self._request_stats['successful'] / 
                               max(1, self._request_stats['total'])),
                'avg_response_time_ms': avg_response_time,
                'active_sessions': len(self._active_sessions),
                'scheduler_stats': self.scheduler.get_scheduler_stats()
            }
            
            self.logger.info("System statistics", LogCategory.PERFORMANCE, metadata=stats)
            
            # Reset response times to prevent memory growth
            if len(self._request_stats['response_times']) > 1000:
                self._request_stats['response_times'] = (
                    self._request_stats['response_times'][-100:]
                )
            
        except Exception as e:
            self.logger.error(f"Error logging system stats: {e}", LogCategory.PERFORMANCE)
    
    def get_status(self) -> SystemStatus:
        """Get current system status."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            avg_response_time = (
                sum(self._request_stats['response_times']) / 
                len(self._request_stats['response_times'])
                if self._request_stats['response_times'] else 0
            )
            
            # Get memory usage
            import psutil
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / 1024 / 1024
            
            gpu_memory_usage_mb = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
            
            return SystemStatus(
                state=self.state,
                uptime_seconds=uptime,
                total_requests=self._request_stats['total'],
                successful_requests=self._request_stats['successful'],
                failed_requests=self._request_stats['failed'],
                active_sessions=len(self._active_sessions),
                memory_usage_mb=memory_usage_mb,
                gpu_memory_usage_mb=gpu_memory_usage_mb,
                avg_response_time_ms=avg_response_time,
                model_loaded=self.model_backend is not None,
                scheduler_running=not self._shutdown_event.is_set(),
                mcp_connected=self.mcp_host is not None
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}", LogCategory.SYSTEM)
            return SystemStatus(
                state=AssistantState.ERROR,
                uptime_seconds=0,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                active_sessions=0,
                memory_usage_mb=0,
                gpu_memory_usage_mb=0,
                avg_response_time_ms=0,
                model_loaded=False,
                scheduler_running=False,
                mcp_connected=False
            )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        if session_id not in self._active_sessions:
            return None
        
        context = self._active_sessions[session_id]
        return {
            'session_id': context.session_id,
            'user_id': context.user_id,
            'conversation_id': context.conversation_id,
            'message_count': len(context.messages),
            'function_calls': context.total_function_calls,
            'created_at': context.created_at.isoformat(),
            'last_activity': context.last_activity.isoformat(),
            'total_tokens': context.total_tokens
        }
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            self.get_session_info(session_id) 
            for session_id in self._active_sessions.keys()
        ]
    
    async def shutdown(self):
        """Shutdown the assistant gracefully."""
        try:
            self.logger.info("Shutting down AVA Assistant", LogCategory.SYSTEM)
            self.state = AssistantState.SHUTDOWN
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown scheduler
            if self.scheduler:
                self.scheduler.shutdown()
            
            # Close MCP connections
            if self.mcp_host:
                # await self.mcp_host.shutdown()
                pass
            
            # Save session data if needed
            await self._save_session_data()
            
            self.logger.info("AVA Assistant shutdown completed", LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", LogCategory.SYSTEM)
    
    async def _save_session_data(self):
        """Save session data before shutdown."""
        try:
            # Placeholder for saving session data
            # In a full implementation, this would persist conversation contexts
            session_count = len(self._active_sessions)
            if session_count > 0:
                self.logger.info(f"Saving data for {session_count} active sessions",
                               LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Error saving session data: {e}", LogCategory.SYSTEM)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AVAAssistant(id={self.assistant_id}, state={self.state.value}, sessions={len(self._active_sessions)})"


# Convenience functions for assistant management
_global_assistant: Optional[AVAAssistant] = None


async def get_assistant(config_path: Optional[str] = None) -> AVAAssistant:
    """Get global assistant instance."""
    global _global_assistant
    if _global_assistant is None:
        _global_assistant = AVAAssistant(config_path)
        await _global_assistant.initialize()
    return _global_assistant


async def shutdown_assistant():
    """Shutdown global assistant."""
    global _global_assistant
    if _global_assistant:
        await _global_assistant.shutdown()
        _global_assistant = None


# Context manager for assistant lifecycle
@asynccontextmanager
async def assistant_context(config_path: Optional[str] = None):
    """Context manager for assistant lifecycle."""
    assistant = None
    try:
        assistant = AVAAssistant(config_path)
        await assistant.initialize()
        yield assistant
    finally:
        if assistant:
            await assistant.shutdown()


# Testing and validation functions
async def test_assistant():
    """Test assistant functionality."""
    print("Testing AVA Assistant...")
    
    async with assistant_context() as assistant:
        # Test basic request processing
        request = AssistantRequest(
            request_id=str(uuid.uuid4()),
            session_id="test_session",
            user_input="Hello, AVA! How are you today?"
        )
        
        response = await assistant.process_request(request)
        print(f"Response: {response.content}")
        print(f"Processing time: {response.processing_time_ms:.2f}ms")
        
        # Test streaming response
        print("\nTesting streaming response:")
        async for token in assistant.stream_response(request):
            print(token, end="", flush=True)
        print()
        
        # Test system status
        status = assistant.get_status()
        print(f"\nSystem status: {status.state.value}")
        print(f"Active sessions: {status.active_sessions}")
        print(f"Total requests: {status.total_requests}")
        
        # Test session info
        session_info = assistant.get_session_info("test_session")
        print(f"\nSession info: {session_info}")
    
    print("\nAVA Assistant test completed!")


if __name__ == "__main__":
    asyncio.run(test_assistant())
