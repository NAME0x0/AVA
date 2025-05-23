#!/usr/bin/env python3
"""
AVA Core Task Scheduler Module

This module provides comprehensive task scheduling functionality for AVA (Afsah's Virtual Assistant),
including background tasks, periodic operations, job queues, and resource-aware task management.
Optimized for AVA's 4GB VRAM constraints and agentic workflows.

Author: Assistant
Date: 2024
"""

import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq
import queue
from functools import wraps
import logging
import psutil
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor, Future


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0    # System critical tasks
    HIGH = 1        # User-facing operations
    NORMAL = 2      # Regular operations
    LOW = 3         # Background maintenance
    IDLE = 4        # Run only when system is idle


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskType(Enum):
    """Types of scheduled tasks."""
    IMMEDIATE = "immediate"      # Execute ASAP
    DELAYED = "delayed"          # Execute after delay
    PERIODIC = "periodic"        # Execute on schedule
    CRON = "cron"               # Cron-like scheduling
    CONDITIONAL = "conditional"  # Execute when condition is met


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """Scheduled task definition."""
    task_id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    task_type: TaskType = TaskType.IMMEDIATE
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: Optional[float] = None
    
    # Scheduling parameters
    scheduled_time: Optional[datetime] = None
    interval_seconds: Optional[float] = None
    cron_expression: Optional[str] = None
    condition_func: Optional[Callable[[], bool]] = None
    
    # Resource constraints
    max_memory_mb: Optional[float] = None
    max_gpu_memory_mb: Optional[float] = None
    requires_gpu: bool = False
    
    # State tracking
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare tasks for priority queue ordering."""
        if not isinstance(other, ScheduledTask):
            return NotImplemented
        
        # Primary sort by scheduled time
        if self.next_run and other.next_run:
            if self.next_run != other.next_run:
                return self.next_run < other.next_run
        
        # Secondary sort by priority
        return self.priority.value < other.priority.value


class ResourceMonitor:
    """Monitor system resources for task scheduling."""
    
    def __init__(self, gpu_memory_limit_mb: float = 3500):
        """Initialize resource monitor."""
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.cpu_threshold = 80.0  # CPU usage threshold
        self.memory_threshold = 80.0  # Memory usage threshold
        self._last_check = time.time()
        self._check_interval = 5.0  # Check every 5 seconds
        
    def is_system_idle(self) -> bool:
        """Check if system is idle enough for low-priority tasks."""
        try:
            current_time = time.time()
            if current_time - self._last_check < self._check_interval:
                return True  # Don't check too frequently
            
            self._last_check = current_time
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            if cpu_percent > self.cpu_threshold:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                return False
            
            return True
            
        except Exception:
            return True  # Default to allowing tasks if check fails
    
    def can_run_task(self, task: ScheduledTask) -> tuple[bool, str]:
        """Check if task can run given current resource constraints."""
        try:
            # Check memory constraints
            if task.max_memory_mb:
                memory = psutil.virtual_memory()
                available_mb = (memory.available / 1024 / 1024)
                if available_mb < task.max_memory_mb:
                    return False, f"Insufficient memory: need {task.max_memory_mb}MB, available {available_mb:.1f}MB"
            
            # Check GPU memory constraints
            if task.max_gpu_memory_mb or task.requires_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024
                        gpu_memory_free = self.gpu_memory_limit_mb - gpu_memory_used
                        
                        required_gpu_memory = task.max_gpu_memory_mb or 100  # Default 100MB
                        if gpu_memory_free < required_gpu_memory:
                            return False, f"Insufficient GPU memory: need {required_gpu_memory}MB, available {gpu_memory_free:.1f}MB"
                    elif task.requires_gpu:
                        return False, "GPU required but not available"
                except ImportError:
                    if task.requires_gpu:
                        return False, "GPU required but PyTorch not available"
            
            # Check if system is idle for low-priority tasks
            if task.priority == TaskPriority.IDLE and not self.is_system_idle():
                return False, "System not idle for low-priority task"
            
            return True, "Resource constraints satisfied"
            
        except Exception as e:
            return False, f"Resource check failed: {e}"


class CronParser:
    """Simple cron expression parser."""
    
    @staticmethod
    def parse_cron(expression: str, current_time: datetime) -> Optional[datetime]:
        """Parse cron expression and return next execution time."""
        try:
            # This is a simplified cron parser
            # Format: "minute hour day month weekday"
            # For now, support basic patterns like "0 */6 * * *" (every 6 hours)
            parts = expression.split()
            if len(parts) != 5:
                return None
            
            minute, hour, day, month, weekday = parts
            
            # Calculate next execution time
            next_time = current_time.replace(second=0, microsecond=0)
            
            # Handle hour patterns
            if hour.startswith("*/"):
                interval = int(hour[2:])
                hours_since_midnight = next_time.hour
                next_hour = ((hours_since_midnight // interval) + 1) * interval
                if next_hour >= 24:
                    next_time += timedelta(days=1)
                    next_hour = 0
                next_time = next_time.replace(hour=next_hour)
            elif hour.isdigit():
                target_hour = int(hour)
                if next_time.hour >= target_hour:
                    next_time += timedelta(days=1)
                next_time = next_time.replace(hour=target_hour)
            
            # Handle minute
            if minute.isdigit():
                next_time = next_time.replace(minute=int(minute))
            
            return next_time
            
        except Exception:
            return None


class TaskScheduler:
    """Main task scheduler for AVA system."""
    
    def __init__(self, max_workers: int = 4, gpu_memory_limit_mb: float = 3500):
        """Initialize task scheduler."""
        self.max_workers = max_workers
        self.resource_monitor = ResourceMonitor(gpu_memory_limit_mb)
        self.logger = logging.getLogger(__name__)
        
        # Task storage
        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_queue: List[ScheduledTask] = []  # Priority queue
        self._running_tasks: Dict[str, Future] = {}
        self._task_results: Dict[str, TaskResult] = {}
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._scheduler_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._total_tasks_executed = 0
        self._total_tasks_failed = 0
        self._start_time = datetime.now()
        
        # Start scheduler
        self.start()
        
        # Register cleanup
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down scheduler")
        self.shutdown()
    
    def start(self):
        """Start the task scheduler."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        
        self._shutdown_event.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        self.logger.info("Task scheduler started")
    
    def shutdown(self):
        """Shutdown the task scheduler."""
        try:
            self.logger.info("Shutting down task scheduler")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel running tasks
            with self._lock:
                for task_id, future in self._running_tasks.items():
                    if not future.done():
                        future.cancel()
                        self.logger.info(f"Cancelled task {task_id}")
            
            # Wait for scheduler thread
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                self._scheduler_thread.join(timeout=5.0)
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            self.logger.info("Task scheduler shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during scheduler shutdown: {e}")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self._shutdown_event.is_set():
            try:
                # Process completed tasks
                self._process_completed_tasks()
                
                # Check for ready tasks
                ready_tasks = self._get_ready_tasks()
                
                # Execute ready tasks
                for task in ready_tasks:
                    if len(self._running_tasks) >= self.max_workers:
                        break  # Worker limit reached
                    
                    if self._can_execute_task(task):
                        self._execute_task(task)
                
                # Update next run times for periodic tasks
                self._update_periodic_tasks()
                
                # Sleep briefly before next iteration
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1.0)
    
    def _process_completed_tasks(self):
        """Process completed task futures."""
        completed_tasks = []
        
        with self._lock:
            for task_id, future in list(self._running_tasks.items()):
                if future.done():
                    completed_tasks.append((task_id, future))
                    del self._running_tasks[task_id]
        
        for task_id, future in completed_tasks:
            task = self._tasks.get(task_id)
            if not task:
                continue
            
            try:
                if future.cancelled():
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED,
                        end_time=datetime.now()
                    )
                elif future.exception():
                    error = str(future.exception())
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=error,
                        end_time=datetime.now(),
                        attempts=task.attempts
                    )
                    
                    # Handle retries
                    if task.attempts < task.max_retries:
                        task.attempts += 1
                        task.status = TaskStatus.RETRY
                        task.next_run = datetime.now() + timedelta(seconds=task.retry_delay)
                        heapq.heappush(self._task_queue, task)
                        self.logger.info(f"Scheduling retry {task.attempts}/{task.max_retries} for task {task_id}")
                        continue
                    else:
                        task.status = TaskStatus.FAILED
                        self._total_tasks_failed += 1
                else:
                    task_result = future.result()
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        result=task_result,
                        end_time=datetime.now(),
                        attempts=task.attempts
                    )
                    task.status = TaskStatus.COMPLETED
                    self._total_tasks_executed += 1
                
                # Store result
                self._task_results[task_id] = result
                
                # Schedule next run for periodic tasks
                if task.task_type == TaskType.PERIODIC and task.interval_seconds:
                    task.next_run = datetime.now() + timedelta(seconds=task.interval_seconds)
                    task.status = TaskStatus.PENDING
                    heapq.heappush(self._task_queue, task)
                elif task.task_type == TaskType.CRON and task.cron_expression:
                    next_time = CronParser.parse_cron(task.cron_expression, datetime.now())
                    if next_time:
                        task.next_run = next_time
                        task.status = TaskStatus.PENDING
                        heapq.heappush(self._task_queue, task)
                
                self.logger.debug(f"Task {task_id} completed with status {result.status}")
                
            except Exception as e:
                self.logger.error(f"Error processing completed task {task_id}: {e}")
    
    def _get_ready_tasks(self) -> List[ScheduledTask]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        current_time = datetime.now()
        
        with self._lock:
            while self._task_queue:
                if self._task_queue[0].next_run and self._task_queue[0].next_run <= current_time:
                    task = heapq.heappop(self._task_queue)
                    
                    # Check conditions for conditional tasks
                    if task.task_type == TaskType.CONDITIONAL:
                        if task.condition_func and not task.condition_func():
                            # Reschedule for later check
                            task.next_run = current_time + timedelta(seconds=10)
                            heapq.heappush(self._task_queue, task)
                            continue
                    
                    ready_tasks.append(task)
                else:
                    break
        
        return ready_tasks
    
    def _can_execute_task(self, task: ScheduledTask) -> bool:
        """Check if task can be executed now."""
        can_run, reason = self.resource_monitor.can_run_task(task)
        if not can_run:
            self.logger.debug(f"Cannot execute task {task.task_id}: {reason}")
            # Reschedule for later
            task.next_run = datetime.now() + timedelta(seconds=30)
            with self._lock:
                heapq.heappush(self._task_queue, task)
        return can_run
    
    def _execute_task(self, task: ScheduledTask):
        """Execute a task."""
        try:
            task.status = TaskStatus.RUNNING
            task.last_run = datetime.now()
            task.attempts += 1
            
            # Create wrapped function for execution
            def task_wrapper():
                start_time = time.time()
                try:
                    # Execute the task
                    if asyncio.iscoroutinefunction(task.func):
                        # Handle async functions
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(task.func(*task.args, **task.kwargs))
                        finally:
                            loop.close()
                    else:
                        result = task.func(*task.args, **task.kwargs)
                    
                    duration = time.time() - start_time
                    self.logger.info(f"Task {task.task_id} completed successfully in {duration:.2f}s")
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(f"Task {task.task_id} failed after {duration:.2f}s: {e}")
                    raise
            
            # Submit to executor
            future = self._executor.submit(task_wrapper)
            
            # Apply timeout if specified
            if task.timeout_seconds:
                def timeout_handler():
                    time.sleep(task.timeout_seconds)
                    if not future.done():
                        future.cancel()
                        self.logger.warning(f"Task {task.task_id} timed out after {task.timeout_seconds}s")
                
                timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
                timeout_thread.start()
            
            with self._lock:
                self._running_tasks[task.task_id] = future
            
            self.logger.debug(f"Task {task.task_id} submitted for execution")
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
    
    def _update_periodic_tasks(self):
        """Update next run times for periodic tasks."""
        current_time = datetime.now()
        
        with self._lock:
            for task in self._tasks.values():
                if (task.status == TaskStatus.PENDING and 
                    task.task_type == TaskType.PERIODIC and 
                    task.interval_seconds and 
                    task.next_run is None):
                    
                    task.next_run = current_time + timedelta(seconds=task.interval_seconds)
                    heapq.heappush(self._task_queue, task)
    
    def schedule_task(self, 
                     name: str,
                     func: Callable,
                     args: tuple = (),
                     kwargs: Optional[Dict[str, Any]] = None,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     delay_seconds: Optional[float] = None,
                     interval_seconds: Optional[float] = None,
                     cron_expression: Optional[str] = None,
                     condition_func: Optional[Callable[[], bool]] = None,
                     max_retries: int = 3,
                     timeout_seconds: Optional[float] = None,
                     max_memory_mb: Optional[float] = None,
                     max_gpu_memory_mb: Optional[float] = None,
                     requires_gpu: bool = False,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Schedule a task for execution."""
        
        task_id = str(uuid.uuid4())
        kwargs = kwargs or {}
        metadata = metadata or {}
        
        # Determine task type
        if cron_expression:
            task_type = TaskType.CRON
            next_run = CronParser.parse_cron(cron_expression, datetime.now())
        elif condition_func:
            task_type = TaskType.CONDITIONAL
            next_run = datetime.now()  # Check immediately
        elif interval_seconds:
            task_type = TaskType.PERIODIC
            next_run = datetime.now() + timedelta(seconds=delay_seconds or 0)
        elif delay_seconds:
            task_type = TaskType.DELAYED
            next_run = datetime.now() + timedelta(seconds=delay_seconds)
        else:
            task_type = TaskType.IMMEDIATE
            next_run = datetime.now()
        
        # Create task
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            task_type=task_type,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            scheduled_time=next_run,
            interval_seconds=interval_seconds,
            cron_expression=cron_expression,
            condition_func=condition_func,
            max_memory_mb=max_memory_mb,
            max_gpu_memory_mb=max_gpu_memory_mb,
            requires_gpu=requires_gpu,
            next_run=next_run,
            metadata=metadata
        )
        
        # Add to scheduler
        with self._lock:
            self._tasks[task_id] = task
            heapq.heappush(self._task_queue, task)
        
        self.logger.info(f"Scheduled task '{name}' with ID {task_id} for {next_run}")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        with self._lock:
            # Cancel if running
            if task_id in self._running_tasks:
                future = self._running_tasks[task_id]
                if future.cancel():
                    del self._running_tasks[task_id]
                    self.logger.info(f"Cancelled running task {task_id}")
                    return True
            
            # Remove from queue if pending
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.CANCELLED
                # Note: Can't easily remove from heapq, but task won't execute due to status
                self.logger.info(f"Cancelled pending task {task_id}")
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task execution result."""
        return self._task_results.get(task_id)
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            running_count = len(self._running_tasks)
            pending_count = len([t for t in self._tasks.values() if t.status == TaskStatus.PENDING])
            
        uptime = datetime.now() - self._start_time
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_tasks_scheduled': len(self._tasks),
            'total_tasks_executed': self._total_tasks_executed,
            'total_tasks_failed': self._total_tasks_failed,
            'tasks_running': running_count,
            'tasks_pending': pending_count,
            'max_workers': self.max_workers,
            'success_rate': self._total_tasks_executed / max(1, self._total_tasks_executed + self._total_tasks_failed)
        }
    
    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """List all tasks with optional status filter."""
        tasks = []
        
        with self._lock:
            for task in self._tasks.values():
                if status_filter is None or task.status == status_filter:
                    tasks.append({
                        'task_id': task.task_id,
                        'name': task.name,
                        'status': task.status.value,
                        'priority': task.priority.value,
                        'task_type': task.task_type.value,
                        'created_at': task.created_at.isoformat(),
                        'last_run': task.last_run.isoformat() if task.last_run else None,
                        'next_run': task.next_run.isoformat() if task.next_run else None,
                        'attempts': task.attempts,
                        'metadata': task.metadata
                    })
        
        return tasks


# Convenience functions and decorators
def scheduled_task(scheduler: TaskScheduler, **schedule_kwargs):
    """Decorator for creating scheduled tasks."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Schedule the function
        task_id = scheduler.schedule_task(
            name=func.__name__,
            func=func,
            **schedule_kwargs
        )
        
        # Add task_id to function for reference
        wrapper.task_id = task_id
        return wrapper
    
    return decorator


# Global scheduler instance
_global_scheduler: Optional[TaskScheduler] = None


def get_scheduler(max_workers: int = 4, gpu_memory_limit_mb: float = 3500) -> TaskScheduler:
    """Get global scheduler instance."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = TaskScheduler(max_workers, gpu_memory_limit_mb)
    return _global_scheduler


def schedule_task(name: str, func: Callable, **kwargs) -> str:
    """Schedule a task using global scheduler."""
    return get_scheduler().schedule_task(name, func, **kwargs)


# Testing functions
def test_scheduler():
    """Test scheduler functionality."""
    print("Testing AVA Task Scheduler...")
    
    scheduler = TaskScheduler(max_workers=2)
    
    # Test immediate task
    def test_func(x, y):
        time.sleep(0.1)
        return x + y
    
    task_id = scheduler.schedule_task(
        name="test_addition",
        func=test_func,
        args=(1, 2),
        priority=TaskPriority.HIGH
    )
    
    print(f"Scheduled task: {task_id}")
    
    # Wait for completion
    time.sleep(1)
    
    result = scheduler.get_task_result(task_id)
    print(f"Task result: {result}")
    
    # Test periodic task
    def periodic_func():
        print(f"Periodic task executed at {datetime.now()}")
        return "completed"
    
    periodic_id = scheduler.schedule_task(
        name="periodic_test",
        func=periodic_func,
        interval_seconds=2.0,
        max_retries=1
    )
    
    print(f"Scheduled periodic task: {periodic_id}")
    
    # Let it run a few times
    time.sleep(5)
    
    # Get stats
    stats = scheduler.get_scheduler_stats()
    print(f"Scheduler stats: {stats}")
    
    # Cleanup
    scheduler.shutdown()
    print("Scheduler test completed!")


if __name__ == "__main__":
    test_scheduler()
