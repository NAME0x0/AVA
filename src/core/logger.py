#!/usr/bin/env python3
"""
AVA Core Logging Module

This module provides comprehensive logging functionality for AVA (Afsah's Virtual Assistant),
including structured logging, performance monitoring, error tracking, and audit trails.
Optimized for production environments with configurable log levels and output formats.

Author: Assistant
Date: 2024
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback
import threading
import time
from datetime import datetime, timedelta
import uuid
import psutil
import contextlib
from functools import wraps, lru_cache
import queue
import atexit


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    MODEL = "model"
    AGENT = "agent"
    INTERFACE = "interface"
    TOOL = "tool"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DEBUG = "debug"
    AUDIT = "audit"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    category: LogCategory
    module: str
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    active_threads: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_metadata: bool = True):
        super().__init__()
        self.include_metadata = include_metadata
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        try:
            # Extract custom attributes
            category = getattr(record, 'category', LogCategory.SYSTEM)
            session_id = getattr(record, 'session_id', None)
            user_id = getattr(record, 'user_id', None)
            request_id = getattr(record, 'request_id', None)
            duration_ms = getattr(record, 'duration_ms', None)
            memory_usage_mb = getattr(record, 'memory_usage_mb', None)
            gpu_memory_mb = getattr(record, 'gpu_memory_mb', None)
            metadata = getattr(record, 'metadata', {})
            
            # Create structured entry
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created).isoformat(),
                level=record.levelname,
                category=category,
                module=record.name,
                message=record.getMessage(),
                session_id=session_id,
                user_id=user_id,
                request_id=request_id,
                function_name=record.funcName,
                line_number=record.lineno,
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage_mb,
                gpu_memory_mb=gpu_memory_mb,
                metadata=metadata if self.include_metadata else {}
            )
            
            # Add exception info if present
            if record.exc_info:
                entry.error_type = record.exc_info[0].__name__ if record.exc_info[0] else None
                entry.stack_trace = self.formatException(record.exc_info)
            
            return json.dumps(asdict(entry), default=str, ensure_ascii=False)
            
        except Exception as e:
            # Fallback to standard formatting
            return f"LOGGING_ERROR: {e} | {record.getMessage()}"


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        try:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            
            # Extract category and session info
            category = getattr(record, 'category', LogCategory.SYSTEM)
            session_id = getattr(record, 'session_id', None)
            duration_ms = getattr(record, 'duration_ms', None)
            
            # Build formatted message
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            category_str = f"[{category.value.upper()}]" if isinstance(category, LogCategory) else f"[{category}]"
            
            # Add performance info if available
            perf_info = ""
            if duration_ms is not None:
                perf_info = f" ({duration_ms:.1f}ms)"
            
            # Add session info if available
            session_info = ""
            if session_id:
                session_info = f" [{session_id[:8]}]"
            
            formatted = (
                f"{color}{timestamp}{reset} "
                f"{color}[{record.levelname}]{reset} "
                f"{category_str} "
                f"{record.name}:{record.funcName}:{record.lineno} "
                f"- {record.getMessage()}"
                f"{perf_info}{session_info}"
            )
            
            return formatted
            
        except Exception as e:
            return f"FORMATTING_ERROR: {e} | {record.getMessage()}"


class PerformanceMonitor:
    """Performance monitoring for logging."""
    
    def __init__(self):
        self._start_time = time.time()
        self._last_check = time.time()
        self._metrics_history: List[PerformanceMetrics] = []
        self._max_history = 1000
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        try:
            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU memory info (if available)
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            except:
                pass
            
            metrics = PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                gpu_memory_used_mb=gpu_memory_used,
                gpu_memory_total_mb=gpu_memory_total,
                active_threads=threading.active_count()
            )
            
            # Store in history
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            return PerformanceMetrics(metadata={'error': str(e)})
    
    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get performance metrics summary for the last N minutes."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_metrics = [
                m for m in self._metrics_history
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            return {
                'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                'max_cpu_percent': max(m.cpu_percent for m in recent_metrics),
                'avg_memory_mb': sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics),
                'max_memory_mb': max(m.memory_used_mb for m in recent_metrics),
                'avg_gpu_memory_mb': sum(m.gpu_memory_used_mb for m in recent_metrics) / len(recent_metrics),
                'max_gpu_memory_mb': max(m.gpu_memory_used_mb for m in recent_metrics),
                'sample_count': len(recent_metrics),
                'time_period_minutes': minutes
            }
            
        except Exception as e:
            return {'error': str(e)}


class AVALogger:
    """Enhanced logger for AVA system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize AVA logger."""
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.session_id = str(uuid.uuid4())
        self.performance_monitor = PerformanceMonitor()
        self._setup_logger()
        
        # Threading for async logging
        self._log_queue = queue.Queue()
        self._log_thread = None
        self._shutdown_event = threading.Event()
        
        # Start async logging if enabled
        if self.config.get('enable_async_logging', True):
            self._start_async_logging()
        
        # Register cleanup
        atexit.register(self.shutdown)
    
    def _setup_logger(self):
        """Setup logger configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        log_level = self.config.get('log_level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler with colored output
        if self.config.get('enable_console_logging', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredConsoleFormatter())
            console_handler.setLevel(getattr(logging, log_level.upper()))
            self.logger.addHandler(console_handler)
        
        # File handler with structured output
        if self.config.get('enable_file_logging', True):
            log_dir = Path(self.config.get('log_directory', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"ava_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_log_size_mb', 100) * 1024 * 1024,
                backupCount=self.config.get('log_rotation_count', 5),
                encoding='utf-8'
            )
            file_handler.setFormatter(StructuredFormatter())
            file_handler.setLevel(logging.DEBUG)  # File gets all levels
            self.logger.addHandler(file_handler)
        
        # Performance log handler
        if self.config.get('enable_performance_logging', True):
            perf_log_dir = Path(self.config.get('log_directory', 'logs')) / 'performance'
            perf_log_dir.mkdir(parents=True, exist_ok=True)
            
            perf_file = perf_log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
            
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=3,
                encoding='utf-8'
            )
            perf_handler.setFormatter(StructuredFormatter())
            perf_handler.addFilter(lambda record: getattr(record, 'category', None) == LogCategory.PERFORMANCE)
            self.logger.addHandler(perf_handler)
    
    def _start_async_logging(self):
        """Start async logging thread."""
        self._log_thread = threading.Thread(target=self._async_log_worker, daemon=True)
        self._log_thread.start()
    
    def _async_log_worker(self):
        """Async logging worker thread."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for log entry with timeout
                log_entry = self._log_queue.get(timeout=1.0)
                if log_entry is None:  # Shutdown signal
                    break
                
                # Process log entry
                level, category, message, metadata, extra = log_entry
                self._log_sync(level, category, message, metadata, extra)
                
            except queue.Empty:
                continue
            except Exception as e:
                # Emergency logging to console
                print(f"ASYNC_LOG_ERROR: {e}", file=sys.stderr)
    
    def _log_sync(self, level: LogLevel, category: LogCategory, message: str, 
                  metadata: Dict[str, Any], extra: Dict[str, Any]):
        """Synchronous logging implementation."""
        try:
            # Prepare log record attributes
            record_extra = {
                'category': category,
                'session_id': self.session_id,
                'metadata': metadata,
                **extra
            }
            
            # Add performance metrics if enabled
            if self.config.get('include_performance_metrics', False):
                metrics = self.performance_monitor.get_current_metrics()
                record_extra.update({
                    'memory_usage_mb': metrics.memory_used_mb,
                    'gpu_memory_mb': metrics.gpu_memory_used_mb
                })
            
            # Log the message
            self.logger.log(level.value, message, extra=record_extra)
            
        except Exception as e:
            # Emergency fallback logging
            print(f"LOG_ERROR: {e} | {message}", file=sys.stderr)
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Log a message with specified level and category."""
        metadata = metadata or {}
        extra = kwargs
        
        if self.config.get('enable_async_logging', True) and self._log_thread and self._log_thread.is_alive():
            # Queue for async processing
            try:
                self._log_queue.put((level, category, message, metadata, extra), block=False)
            except queue.Full:
                # Fall back to sync logging if queue is full
                self._log_sync(level, category, message, metadata, extra)
        else:
            # Synchronous logging
            self._log_sync(level, category, message, metadata, extra)
    
    def debug(self, message: str, category: LogCategory = LogCategory.DEBUG, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, message, **kwargs)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, category, message, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, category, message, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              exc_info: bool = True, **kwargs):
        """Log error message."""
        if exc_info:
            kwargs['exc_info'] = True
        self.log(LogLevel.ERROR, category, message, **kwargs)
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, category, message, **kwargs)
    
    def performance(self, message: str, duration_ms: Optional[float] = None, **kwargs):
        """Log performance metrics."""
        extra_data = {'duration_ms': duration_ms} if duration_ms else {}
        extra_data.update(kwargs)
        self.log(LogLevel.INFO, LogCategory.PERFORMANCE, message, **extra_data)
    
    def audit(self, message: str, user_id: Optional[str] = None, **kwargs):
        """Log audit event."""
        extra_data = {'user_id': user_id} if user_id else {}
        extra_data.update(kwargs)
        self.log(LogLevel.INFO, LogCategory.AUDIT, message, **extra_data)
    
    def get_performance_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get performance summary for logging."""
        return self.performance_monitor.get_metrics_summary(minutes)
    
    def shutdown(self):
        """Shutdown logger and cleanup resources."""
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop async logging
            if self._log_thread and self._log_thread.is_alive():
                self._log_queue.put(None)  # Shutdown signal
                self._log_thread.join(timeout=5.0)
            
            # Close all handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
            
        except Exception as e:
            print(f"LOGGER_SHUTDOWN_ERROR: {e}", file=sys.stderr)


class LoggerDecorator:
    """Decorator for automatic function logging."""
    
    def __init__(self, logger: AVALogger, category: LogCategory = LogCategory.SYSTEM,
                 log_args: bool = False, log_result: bool = False):
        self.logger = logger
        self.category = category
        self.log_args = log_args
        self.log_result = log_result
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            request_id = str(uuid.uuid4())
            
            # Log function entry
            entry_msg = f"Entering {func_name}"
            if self.log_args:
                entry_msg += f" with args={args}, kwargs={kwargs}"
            
            self.logger.debug(entry_msg, self.category, 
                            request_id=request_id, function_name=func_name)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful completion
                duration_ms = (time.time() - start_time) * 1000
                exit_msg = f"Completed {func_name}"
                if self.log_result:
                    exit_msg += f" with result={result}"
                
                self.logger.debug(exit_msg, self.category,
                                request_id=request_id, function_name=func_name,
                                duration_ms=duration_ms)
                
                return result
                
            except Exception as e:
                # Log error
                duration_ms = (time.time() - start_time) * 1000
                self.logger.error(f"Error in {func_name}: {e}", self.category,
                                request_id=request_id, function_name=func_name,
                                duration_ms=duration_ms, exc_info=True)
                raise
        
        return wrapper


@contextlib.contextmanager
def log_performance(logger: AVALogger, operation: str, category: LogCategory = LogCategory.PERFORMANCE):
    """Context manager for performance logging."""
    start_time = time.time()
    start_metrics = logger.performance_monitor.get_current_metrics()
    
    logger.debug(f"Starting {operation}", category)
    
    try:
        yield
        
        # Log successful completion
        duration_ms = (time.time() - start_time) * 1000
        end_metrics = logger.performance_monitor.get_current_metrics()
        
        memory_delta = end_metrics.memory_used_mb - start_metrics.memory_used_mb
        gpu_memory_delta = end_metrics.gpu_memory_used_mb - start_metrics.gpu_memory_used_mb
        
        logger.performance(
            f"Completed {operation}",
            duration_ms=duration_ms,
            memory_delta_mb=memory_delta,
            gpu_memory_delta_mb=gpu_memory_delta
        )
        
    except Exception as e:
        # Log error with performance data
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed {operation}: {e}", category, 
                   duration_ms=duration_ms, exc_info=True)
        raise


# Global logger registry
_loggers: Dict[str, AVALogger] = {}
_default_config: Dict[str, Any] = {}


def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> AVALogger:
    """Get or create a logger instance."""
    global _loggers, _default_config
    
    if name not in _loggers:
        logger_config = {**_default_config, **(config or {})}
        _loggers[name] = AVALogger(name, logger_config)
    
    return _loggers[name]


def set_default_config(config: Dict[str, Any]):
    """Set default configuration for all loggers."""
    global _default_config
    _default_config = config


def shutdown_all_loggers():
    """Shutdown all active loggers."""
    global _loggers
    for logger in _loggers.values():
        logger.shutdown()
    _loggers.clear()


# Testing and validation functions
def test_logger():
    """Test logger functionality."""
    print("Testing AVA Logger...")
    
    # Test basic logging
    logger = get_logger("test_logger", {
        'log_level': 'DEBUG',
        'enable_console_logging': True,
        'enable_file_logging': False,
        'enable_async_logging': False
    })
    
    logger.info("Test info message", LogCategory.SYSTEM)
    logger.debug("Test debug message", LogCategory.DEBUG)
    logger.warning("Test warning message", LogCategory.AGENT)
    logger.error("Test error message", LogCategory.MODEL, exc_info=False)
    
    # Test performance logging
    with log_performance(logger, "test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Test decorator
    @LoggerDecorator(logger, LogCategory.TOOL, log_args=True, log_result=True)
    def test_function(x, y):
        return x + y
    
    result = test_function(1, 2)
    print(f"Function result: {result}")
    
    # Test performance summary
    summary = logger.get_performance_summary(1)
    print(f"Performance summary: {summary}")
    
    logger.shutdown()
    print("Logger test completed!")


if __name__ == "__main__":
    test_logger()
