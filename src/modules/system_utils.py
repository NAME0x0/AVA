#!/usr/bin/env python3
"""
Enhanced System Utilities Module for AVA
Production-Ready System Monitoring and Resource Management
"""

import asyncio
import json
import logging
import os
import platform
import psutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

# Optional imports with fallbacks
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    warnings.warn("GPUtil not available - GPU monitoring limited")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - CUDA monitoring limited")

try:
    import nvidia_ml_py3 as nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    warnings.warn("nvidia-ml-py3 not available - advanced GPU monitoring unavailable")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"


class AlertLevel(Enum):
    """Alert levels for resource monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ProcessPriority(Enum):
    """Process priority levels."""
    IDLE = "idle"
    BELOW_NORMAL = "below_normal"
    NORMAL = "normal"
    ABOVE_NORMAL = "above_normal"
    HIGH = "high"
    REALTIME = "realtime"


@dataclass
class SystemInfo:
    """System information structure."""
    platform: str = ""
    architecture: str = ""
    processor: str = ""
    cores_physical: int = 0
    cores_logical: int = 0
    memory_total_gb: float = 0.0
    python_version: str = ""
    hostname: str = ""
    uptime_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Resource usage information."""
    resource_type: ResourceType
    current_usage: float = 0.0
    peak_usage: float = 0.0
    available: float = 0.0
    total: float = 0.0
    usage_percentage: float = 0.0
    timestamp: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUInfo:
    """GPU information structure."""
    gpu_id: int = 0
    name: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_free_mb: float = 0.0
    memory_usage_percent: float = 0.0
    temperature_c: float = 0.0
    power_usage_w: float = 0.0
    utilization_percent: float = 0.0
    is_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessInfo:
    """Process information structure."""
    pid: int = 0
    name: str = ""
    status: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0
    created_time: float = 0.0
    command_line: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemAlert:
    """System alert structure."""
    level: AlertLevel
    resource_type: ResourceType
    message: str
    threshold: float = 0.0
    current_value: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """Advanced system monitoring and resource management."""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0,
            "disk_percent": 90.0,
            "temperature_c": 80.0
        }
        self.monitoring_active = False
        self.resource_history: Dict[ResourceType, List[ResourceUsage]] = {
            resource_type: [] for resource_type in ResourceType
        }
        self.alerts: List[SystemAlert] = []
        self._nvml_initialized = False
        
        # Initialize NVML if available
        if HAS_NVML:
            try:
                nvml.nvmlInit()
                self._nvml_initialized = True
                logger.info("NVIDIA ML initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA ML: {e}")
    
    async def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information."""
        try:
            uname = platform.uname()
            boot_time = psutil.boot_time()
            current_time = time.time()
            
            return SystemInfo(
                platform=f"{uname.system} {uname.release}",
                architecture=uname.machine,
                processor=uname.processor or platform.processor(),
                cores_physical=psutil.cpu_count(logical=False) or 0,
                cores_logical=psutil.cpu_count(logical=True) or 0,
                memory_total_gb=psutil.virtual_memory().total / (1024**3),
                python_version=platform.python_version(),
                hostname=platform.node(),
                uptime_seconds=current_time - boot_time,
                metadata={
                    "platform_version": uname.version,
                    "boot_time": boot_time,
                    "current_time": current_time
                }
            )
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return SystemInfo()
    
    async def get_cpu_usage(self) -> ResourceUsage:
        """Get CPU usage information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_times = psutil.cpu_times()
            
            return ResourceUsage(
                resource_type=ResourceType.CPU,
                current_usage=cpu_percent,
                usage_percentage=cpu_percent,
                timestamp=time.time(),
                details={
                    "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                    "frequency_max_mhz": cpu_freq.max if cpu_freq else 0,
                    "per_core_usage": psutil.cpu_percent(percpu=True, interval=0.1),
                    "times": {
                        "user": cpu_times.user,
                        "system": cpu_times.system,
                        "idle": cpu_times.idle
                    }
                }
            )
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return ResourceUsage(resource_type=ResourceType.CPU)
    
    async def get_memory_usage(self) -> ResourceUsage:
        """Get memory usage information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return ResourceUsage(
                resource_type=ResourceType.MEMORY,
                current_usage=memory.used / (1024**3),  # GB
                total=memory.total / (1024**3),  # GB
                available=memory.available / (1024**3),  # GB
                usage_percentage=memory.percent,
                timestamp=time.time(),
                details={
                    "used_gb": memory.used / (1024**3),
                    "free_gb": memory.free / (1024**3),
                    "buffers_gb": memory.buffers / (1024**3),
                    "cached_gb": memory.cached / (1024**3),
                    "swap_total_gb": swap.total / (1024**3),
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_percent": swap.percent
                }
            )
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return ResourceUsage(resource_type=ResourceType.MEMORY)
    
    async def get_gpu_info(self) -> List[GPUInfo]:
        """Get comprehensive GPU information."""
        gpu_info_list = []
        
        try:
            # Try NVML first (most comprehensive)
            if self._nvml_initialized:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                        
                        # Memory info
                        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_total = mem_info.total / (1024**2)  # MB
                        memory_used = mem_info.used / (1024**2)  # MB
                        memory_free = mem_info.free / (1024**2)  # MB
                        
                        # Temperature
                        try:
                            temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        except:
                            temperature = 0.0
                        
                        # Power usage
                        try:
                            power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                        except:
                            power_usage = 0.0
                        
                        # Utilization
                        try:
                            util = nvml.nvmlDeviceGetUtilizationRates(handle)
                            utilization = util.gpu
                        except:
                            utilization = 0.0
                        
                        gpu_info = GPUInfo(
                            gpu_id=i,
                            name=name,
                            memory_total_mb=memory_total,
                            memory_used_mb=memory_used,
                            memory_free_mb=memory_free,
                            memory_usage_percent=(memory_used / memory_total) * 100 if memory_total > 0 else 0,
                            temperature_c=temperature,
                            power_usage_w=power_usage,
                            utilization_percent=utilization,
                            is_available=True
                        )
                        
                        # Add CUDA info if PyTorch is available
                        if HAS_TORCH and torch.cuda.is_available():
                            gpu_info.cuda_version = torch.version.cuda
                            gpu_info.metadata["torch_device"] = f"cuda:{i}"
                        
                        gpu_info_list.append(gpu_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to get info for GPU {i}: {e}")
            
            # Fallback to GPUtil if NVML unavailable
            elif HAS_GPUTIL:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_info = GPUInfo(
                        gpu_id=i,
                        name=gpu.name,
                        memory_total_mb=gpu.memoryTotal,
                        memory_used_mb=gpu.memoryUsed,
                        memory_free_mb=gpu.memoryFree,
                        memory_usage_percent=gpu.memoryUtil * 100,
                        temperature_c=gpu.temperature,
                        utilization_percent=gpu.load * 100,
                        is_available=True
                    )
                    gpu_info_list.append(gpu_info)
            
            # PyTorch-only fallback
            elif HAS_TORCH and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    try:
                        name = torch.cuda.get_device_name(i)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)  # MB
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                        
                        gpu_info = GPUInfo(
                            gpu_id=i,
                            name=name,
                            memory_used_mb=memory_allocated,
                            cuda_version=torch.version.cuda,
                            is_available=True,
                            metadata={
                                "memory_reserved_mb": memory_reserved,
                                "torch_device": f"cuda:{i}"
                            }
                        )
                        gpu_info_list.append(gpu_info)
                    except Exception as e:
                        logger.warning(f"Failed to get PyTorch GPU info for device {i}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to get GPU information: {e}")
        
        return gpu_info_list
    
    async def get_disk_usage(self, path: str = "/") -> ResourceUsage:
        """Get disk usage information for a specific path."""
        try:
            usage = psutil.disk_usage(path)
            
            return ResourceUsage(
                resource_type=ResourceType.DISK,
                current_usage=usage.used / (1024**3),  # GB
                total=usage.total / (1024**3),  # GB
                available=usage.free / (1024**3),  # GB
                usage_percentage=(usage.used / usage.total) * 100,
                timestamp=time.time(),
                details={
                    "path": path,
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3)
                }
            )
        except Exception as e:
            logger.error(f"Failed to get disk usage for {path}: {e}")
            return ResourceUsage(resource_type=ResourceType.DISK)
    
    async def get_network_usage(self) -> ResourceUsage:
        """Get network usage information."""
        try:
            net_io = psutil.net_io_counters()
            
            return ResourceUsage(
                resource_type=ResourceType.NETWORK,
                timestamp=time.time(),
                details={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "errin": net_io.errin,
                    "errout": net_io.errout,
                    "dropin": net_io.dropin,
                    "dropout": net_io.dropout
                }
            )
        except Exception as e:
            logger.error(f"Failed to get network usage: {e}")
            return ResourceUsage(resource_type=ResourceType.NETWORK)
    
    async def get_process_info(self, pid: Optional[int] = None) -> Union[ProcessInfo, List[ProcessInfo]]:
        """Get process information for a specific PID or all processes."""
        try:
            if pid is not None:
                # Get info for specific process
                process = psutil.Process(pid)
                return self._build_process_info(process)
            else:
                # Get info for all processes
                process_list = []
                for process in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                    try:
                        process_info = self._build_process_info(process)
                        process_list.append(process_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                return process_list
        except Exception as e:
            logger.error(f"Failed to get process info: {e}")
            return ProcessInfo() if pid is not None else []
    
    def _build_process_info(self, process) -> ProcessInfo:
        """Build ProcessInfo from psutil Process object."""
        try:
            memory_info = process.memory_info()
            
            return ProcessInfo(
                pid=process.pid,
                name=process.name(),
                status=process.status(),
                cpu_percent=process.cpu_percent(),
                memory_percent=process.memory_percent(),
                memory_rss_mb=memory_info.rss / (1024**2),
                memory_vms_mb=memory_info.vms / (1024**2),
                created_time=process.create_time(),
                command_line=process.cmdline(),
                metadata={
                    "parent_pid": process.ppid(),
                    "num_threads": process.num_threads(),
                    "username": process.username() if hasattr(process, 'username') else None
                }
            )
        except Exception as e:
            logger.warning(f"Failed to build process info: {e}")
            return ProcessInfo()
    
    async def optimize_for_ava(self) -> Dict[str, Any]:
        """Optimize system settings for AVA's 4GB VRAM constraints."""
        optimizations = {
            "applied": [],
            "failed": [],
            "recommendations": []
        }
        
        try:
            # Clear GPU memory cache if PyTorch is available
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimizations["applied"].append("Cleared CUDA memory cache")
            
            # Set process priority to high for current process
            try:
                current_process = psutil.Process()
                current_process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
                optimizations["applied"].append("Set high process priority")
            except (psutil.AccessDenied, AttributeError):
                optimizations["failed"].append("Failed to set high process priority")
            
            # Check memory availability
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                optimizations["recommendations"].append(
                    "Consider closing unnecessary applications - high memory usage detected"
                )
            
            # Check GPU memory availability
            gpus = await self.get_gpu_info()
            for gpu in gpus:
                if gpu.memory_usage_percent > 50:
                    optimizations["recommendations"].append(
                        f"GPU {gpu.gpu_id} has {gpu.memory_usage_percent:.1f}% memory usage - "
                        f"consider reducing other GPU workloads for optimal AVA performance"
                    )
            
            # Check thermal throttling
            for gpu in gpus:
                if gpu.temperature_c > 75:
                    optimizations["recommendations"].append(
                        f"GPU {gpu.gpu_id} temperature is {gpu.temperature_c}°C - "
                        f"ensure adequate cooling for sustained performance"
                    )
        
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            optimizations["failed"].append(f"Optimization error: {str(e)}")
        
        return optimizations
    
    async def check_alerts(self) -> List[SystemAlert]:
        """Check for system alerts based on current resource usage."""
        current_alerts = []
        timestamp = time.time()
        
        try:
            # CPU alerts
            cpu_usage = await self.get_cpu_usage()
            if cpu_usage.usage_percentage > self.alert_thresholds["cpu_percent"]:
                current_alerts.append(SystemAlert(
                    level=AlertLevel.WARNING,
                    resource_type=ResourceType.CPU,
                    message=f"High CPU usage: {cpu_usage.usage_percentage:.1f}%",
                    threshold=self.alert_thresholds["cpu_percent"],
                    current_value=cpu_usage.usage_percentage,
                    timestamp=timestamp
                ))
            
            # Memory alerts
            memory_usage = await self.get_memory_usage()
            if memory_usage.usage_percentage > self.alert_thresholds["memory_percent"]:
                current_alerts.append(SystemAlert(
                    level=AlertLevel.WARNING,
                    resource_type=ResourceType.MEMORY,
                    message=f"High memory usage: {memory_usage.usage_percentage:.1f}%",
                    threshold=self.alert_thresholds["memory_percent"],
                    current_value=memory_usage.usage_percentage,
                    timestamp=timestamp
                ))
            
            # GPU alerts
            gpus = await self.get_gpu_info()
            for gpu in gpus:
                if gpu.memory_usage_percent > self.alert_thresholds["gpu_memory_percent"]:
                    current_alerts.append(SystemAlert(
                        level=AlertLevel.CRITICAL,
                        resource_type=ResourceType.GPU,
                        message=f"GPU {gpu.gpu_id} high memory usage: {gpu.memory_usage_percent:.1f}%",
                        threshold=self.alert_thresholds["gpu_memory_percent"],
                        current_value=gpu.memory_usage_percent,
                        timestamp=timestamp
                    ))
                
                if gpu.temperature_c > self.alert_thresholds["temperature_c"]:
                    current_alerts.append(SystemAlert(
                        level=AlertLevel.WARNING,
                        resource_type=ResourceType.GPU,
                        message=f"GPU {gpu.gpu_id} high temperature: {gpu.temperature_c}°C",
                        threshold=self.alert_thresholds["temperature_c"],
                        current_value=gpu.temperature_c,
                        timestamp=timestamp
                    ))
        
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
        
        return current_alerts
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Synchronous interface for function calling compatibility.
        
        Args:
            operation: The operation to perform
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with operation result or error information
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if operation == "system_info":
                result = loop.run_until_complete(self.get_system_info())
                return {
                    "platform": result.platform,
                    "architecture": result.architecture,
                    "processor": result.processor,
                    "cores_physical": result.cores_physical,
                    "cores_logical": result.cores_logical,
                    "memory_total_gb": result.memory_total_gb,
                    "python_version": result.python_version,
                    "hostname": result.hostname,
                    "uptime_seconds": result.uptime_seconds
                }
            
            elif operation == "resource_usage":
                cpu = loop.run_until_complete(self.get_cpu_usage())
                memory = loop.run_until_complete(self.get_memory_usage())
                gpus = loop.run_until_complete(self.get_gpu_info())
                
                return {
                    "cpu_percent": cpu.usage_percentage,
                    "memory_percent": memory.usage_percentage,
                    "memory_used_gb": memory.current_usage,
                    "memory_available_gb": memory.available,
                    "gpus": [
                        {
                            "gpu_id": gpu.gpu_id,
                            "name": gpu.name,
                            "memory_usage_percent": gpu.memory_usage_percent,
                            "memory_used_mb": gpu.memory_used_mb,
                            "memory_total_mb": gpu.memory_total_mb,
                            "temperature_c": gpu.temperature_c,
                            "utilization_percent": gpu.utilization_percent
                        }
                        for gpu in gpus
                    ]
                }
            
            elif operation == "optimize":
                result = loop.run_until_complete(self.optimize_for_ava())
                return result
            
            elif operation == "alerts":
                alerts = loop.run_until_complete(self.check_alerts())
                return {
                    "alerts": [
                        {
                            "level": alert.level.value,
                            "resource_type": alert.resource_type.value,
                            "message": alert.message,
                            "threshold": alert.threshold,
                            "current_value": alert.current_value,
                            "timestamp": alert.timestamp
                        }
                        for alert in alerts
                    ]
                }
            
            else:
                return {"error": f"Unknown operation: {operation}"}
        
        except Exception as e:
            return {"error": f"Operation failed: {str(e)}"}
        
        finally:
            loop.close()
    
    def __del__(self):
        """Cleanup NVML if initialized."""
        if self._nvml_initialized:
            try:
                nvml.nvmlShutdown()
            except:
                pass


def test_system_utils():
    """Test the enhanced system utilities module."""
    logger.info("=== Testing Enhanced System Utilities ===")
    
    # Initialize system monitor
    monitor = SystemMonitor()
    
    test_operations = [
        "system_info",
        "resource_usage",
        "optimize",
        "alerts"
    ]
    
    for operation in test_operations:
        logger.info(f"\nTesting operation: {operation}")
        result = monitor.run(operation)
        
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            if operation == "system_info":
                logger.info(f"Platform: {result['platform']}")
                logger.info(f"CPU: {result['processor']} ({result['cores_physical']} physical, {result['cores_logical']} logical)")
                logger.info(f"Memory: {result['memory_total_gb']:.1f} GB")
            
            elif operation == "resource_usage":
                logger.info(f"CPU: {result['cpu_percent']:.1f}%")
                logger.info(f"Memory: {result['memory_percent']:.1f}% ({result['memory_used_gb']:.1f}GB used)")
                logger.info(f"GPUs found: {len(result['gpus'])}")
                for gpu in result['gpus']:
                    logger.info(f"  GPU {gpu['gpu_id']}: {gpu['name']} - {gpu['memory_usage_percent']:.1f}% memory used")
            
            elif operation == "optimize":
                logger.info(f"Applied optimizations: {len(result['applied'])}")
                for opt in result['applied']:
                    logger.info(f"  ✓ {opt}")
                if result['recommendations']:
                    logger.info("Recommendations:")
                    for rec in result['recommendations']:
                        logger.info(f"  → {rec}")
            
            elif operation == "alerts":
                logger.info(f"Active alerts: {len(result['alerts'])}")
                for alert in result['alerts']:
                    logger.warning(f"  {alert['level'].upper()}: {alert['message']}")


async def main():
    """Main function for standalone testing."""
    test_system_utils()


if __name__ == "__main__":
    asyncio.run(main())
