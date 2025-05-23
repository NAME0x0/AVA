#!/usr/bin/env python3
"""
Enhanced MCP Host for AVA - Model Context Protocol Implementation
Enables AVA to discover, manage, and communicate with MCP servers for direct data access.
Optimized for local agentic AI on RTX A2000 4GB VRAM constraints.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import uuid
import hashlib
import os

# Configure logging
logger = logging.getLogger(__name__)


class MCPServerStatus(Enum):
    """Status of MCP server connections."""
    UNKNOWN = "unknown"
    DISCOVERING = "discovering"
    AVAILABLE = "available"
    CONNECTED = "connected"
    ERROR = "error"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"


class MCPCapabilityType(Enum):
    """Types of MCP capabilities."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    DIRECTORY_LIST = "directory_list"
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    SEARCH = "search"
    COMPUTATION = "computation"
    CUSTOM = "custom"


@dataclass
class MCPCapability:
    """Represents an MCP server capability."""
    name: str
    type: MCPCapabilityType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    authentication_required: bool = False
    rate_limit: Optional[int] = None
    cache_ttl: Optional[int] = None


@dataclass
class MCPServer:
    """Represents an MCP server configuration."""
    id: str
    name: str
    description: str
    address: str
    version: str = "1.0.0"
    capabilities: List[MCPCapability] = field(default_factory=list)
    status: MCPServerStatus = MCPServerStatus.UNKNOWN
    last_seen: Optional[datetime] = None
    authentication: Optional[Dict[str, str]] = None
    timeout_seconds: float = 30.0
    retry_count: int = 3
    health_check_interval: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPRequest:
    """Represents an MCP request."""
    request_id: str
    server_id: str
    capability_name: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = 30.0
    cache_key: Optional[str] = None
    authentication: Optional[Dict[str, str]] = None


@dataclass
class MCPResponse:
    """Represents an MCP response."""
    request_id: str
    server_id: str
    capability_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPConfig:
    """Configuration for MCP host."""
    discovery_enabled: bool = True
    discovery_timeout: float = 10.0
    default_timeout: float = 30.0
    max_concurrent_requests: int = 10
    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_default_ttl: int = 300
    health_check_enabled: bool = True
    authentication_enabled: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


class MCPHost:
    """
    Enhanced MCP Host for AVA - manages discovery, communication, and caching
    of Model Context Protocol servers for direct data access.
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        """
        Initialize the enhanced MCP host.
        
        Args:
            config: Configuration for MCP host behavior
        """
        self.config = config or MCPConfig()
        self.servers: Dict[str, MCPServer] = {}
        self.response_cache: Dict[str, MCPResponse] = {}
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.request_patterns = self._initialize_request_patterns()
        self._discovery_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced MCP Host initialized")
        
        # Initialize with default local servers
        self._setup_default_servers()
    
    def _initialize_request_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for detecting MCP requests."""
        return {
            # File operations
            r'(?:mcp:)?(?:read|access|open)\s+(?:file|document)\s+["\']?([^"\']+)["\']?': {
                'capability': 'read_file',
                'server_type': 'file_system',
                'param_extraction': lambda m: {'path': m.group(1)}
            },
            r'(?:mcp:)?(?:list|show)\s+(?:directory|folder)\s+["\']?([^"\']+)["\']?': {
                'capability': 'list_directory',
                'server_type': 'file_system',
                'param_extraction': lambda m: {'path': m.group(1)}
            },
            
            # Database operations
            r'(?:mcp:)?(?:query|search)\s+(?:database|db)\s+["\']?([^"\']+)["\']?': {
                'capability': 'query_database',
                'server_type': 'database',
                'param_extraction': lambda m: {'query': m.group(1)}
            },
            
            # Search operations
            r'(?:mcp:)?search\s+(?:for\s+)?["\']?([^"\']+)["\']?': {
                'capability': 'search',
                'server_type': 'search',
                'param_extraction': lambda m: {'query': m.group(1)}
            }
        }
    
    def _setup_default_servers(self):
        """Setup default local MCP servers."""
        # Local file system server
        file_capabilities = [
            MCPCapability(
                name='read_file',
                type=MCPCapabilityType.FILE_READ,
                description='Read contents of a local file',
                required_params=['path'],
                cache_ttl=300
            ),
            MCPCapability(
                name='list_directory',
                type=MCPCapabilityType.DIRECTORY_LIST,
                description='List contents of a local directory',
                required_params=['path'],
                cache_ttl=60
            )
        ]
        
        file_server = MCPServer(
            id='local_file_system',
            name='Local File System',
            description='Access to local files and directories',
            address='file://localhost',
            capabilities=file_capabilities,
            status=MCPServerStatus.AVAILABLE,
            last_seen=datetime.now()
        )
        
        self.servers[file_server.id] = file_server
        logger.info(f"Registered default server: {file_server.name}")
    
    async def discover_servers(self) -> List[MCPServer]:
        """
        Discover available MCP servers through multiple methods.
        
        Returns:
            List of discovered servers
        """
        discovered_servers = []
        
        try:
            # Method 1: Configuration file discovery
            config_servers = await self._discover_from_config()
            discovered_servers.extend(config_servers)
            
            # Update server registry
            for server in discovered_servers:
                if server.id not in self.servers:
                    self.servers[server.id] = server
                    logger.info(f"Discovered new MCP server: {server.name} ({server.id})")
                else:
                    # Update existing server info
                    self.servers[server.id].last_seen = datetime.now()
                    self.servers[server.id].status = MCPServerStatus.AVAILABLE
        
        except Exception as e:
            logger.error(f"Error during server discovery: {e}")
        
        return discovered_servers
    
    async def _discover_from_config(self) -> List[MCPServer]:
        """Discover servers from configuration files."""
        servers = []
        
        try:
            config_paths = [
                'config/mcp_servers.json',
                'mcp_servers.json',
                '.mcp/servers.json'
            ]
            
            for config_path in config_paths:
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    for server_config in config_data.get('servers', []):
                        server = self._create_server_from_config(server_config)
                        if server:
                            servers.append(server)
                    
                    logger.info(f"Loaded {len(servers)} servers from {config_path}")
                    break
                    
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.warning(f"Error reading config {config_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in config discovery: {e}")
        
        return servers
    
    def _create_server_from_config(self, config: Dict[str, Any]) -> Optional[MCPServer]:
        """Create MCPServer object from configuration dictionary."""
        try:
            capabilities = []
            for cap_config in config.get('capabilities', []):
                capability = MCPCapability(
                    name=cap_config['name'],
                    type=MCPCapabilityType(cap_config.get('type', 'custom')),
                    description=cap_config.get('description', ''),
                    parameters=cap_config.get('parameters', {}),
                    required_params=cap_config.get('required_params', []),
                    optional_params=cap_config.get('optional_params', []),
                    authentication_required=cap_config.get('authentication_required', False),
                    rate_limit=cap_config.get('rate_limit'),
                    cache_ttl=cap_config.get('cache_ttl')
                )
                capabilities.append(capability)
            
            server = MCPServer(
                id=config['id'],
                name=config['name'],
                description=config['description'],
                address=config['address'],
                version=config.get('version', '1.0.0'),
                capabilities=capabilities,
                status=MCPServerStatus.AVAILABLE,
                last_seen=datetime.now(),
                authentication=config.get('authentication'),
                timeout_seconds=config.get('timeout_seconds', self.config.default_timeout),
                retry_count=config.get('retry_count', self.config.retry_attempts),
                metadata=config.get('metadata', {})
            )
            
            return server
            
        except Exception as e:
            logger.error(f"Error creating server from config: {e}")
            return None
    
    def is_mcp_request(self, text_or_plan: Union[str, Dict[str, Any]]) -> tuple[bool, Optional[MCPRequest]]:
        """
        Enhanced detection of MCP requests from text or plan.
        
        Args:
            text_or_plan: Text string or structured plan to analyze
            
        Returns:
            Tuple of (is_mcp_request, parsed_request_or_None)
        """
        text_to_analyze = ""
        
        if isinstance(text_or_plan, dict):
            # Extract text from structured plan
            text_to_analyze = text_or_plan.get('action_description', '')
            text_to_analyze += ' ' + text_or_plan.get('content', '')
            text_to_analyze += ' ' + str(text_or_plan.get('parameters', {}))
        else:
            text_to_analyze = str(text_or_plan)
        
        text_to_analyze = text_to_analyze.lower().strip()
        
        # Check against patterns
        for pattern, pattern_info in self.request_patterns.items():
            match = re.search(pattern, text_to_analyze, re.IGNORECASE)
            if match:
                try:
                    # Extract parameters using pattern's extraction function
                    params = pattern_info['param_extraction'](match)
                    
                    # Find appropriate server
                    server_id = self._find_server_for_capability(
                        pattern_info['capability'],
                        pattern_info['server_type']
                    )
                    
                    if server_id:
                        request = MCPRequest(
                            request_id=str(uuid.uuid4()),
                            server_id=server_id,
                            capability_name=pattern_info['capability'],
                            parameters=params,
                            cache_key=self._generate_cache_key(
                                server_id, pattern_info['capability'], params
                            ) if self.config.cache_enabled else None
                        )
                        
                        logger.info(f"Detected MCP request: {pattern_info['capability']} on {server_id}")
                        return True, request
                        
                except Exception as e:
                    logger.error(f"Error parsing MCP request: {e}")
        
        return False, None
    
    def _find_server_for_capability(self, capability_name: str, server_type: str) -> Optional[str]:
        """Find the best server for a specific capability."""
        for server_id, server in self.servers.items():
            if server.status != MCPServerStatus.AVAILABLE:
                continue
            
            for capability in server.capabilities:
                if capability.name == capability_name:
                    return server_id
                    
            # Fallback: check if server type matches
            if server_type in server.name.lower() or server_type in server.description.lower():
                return server_id
        
        return None
    
    def _generate_cache_key(self, server_id: str, capability: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        key_data = f"{server_id}:{capability}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def execute_request(self, request: MCPRequest) -> MCPResponse:
        """
        Execute an MCP request with caching, retries, and error handling.
        
        Args:
            request: The MCP request to execute
            
        Returns:
            MCP response with results or error information
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.config.cache_enabled and request.cache_key:
                cached_response = self._get_cached_response(request.cache_key)
                if cached_response:
                    cached_response.cached = True
                    logger.debug(f"Returning cached response for {request.capability_name}")
                    return cached_response
            
            # Validate server exists and is available
            if request.server_id not in self.servers:
                return MCPResponse(
                    request_id=request.request_id,
                    server_id=request.server_id,
                    capability_name=request.capability_name,
                    success=False,
                    error=f"Server {request.server_id} not found",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            server = self.servers[request.server_id]
            if server.status != MCPServerStatus.AVAILABLE:
                return MCPResponse(
                    request_id=request.request_id,
                    server_id=request.server_id,
                    capability_name=request.capability_name,
                    success=False,
                    error=f"Server {request.server_id} is not available (status: {server.status.value})",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Execute request
            response = await self._execute_single_request(request, server)
            
            # Cache successful response
            if (self.config.cache_enabled and request.cache_key and 
                response.success and not response.cached):
                self._cache_response(request.cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing MCP request: {e}")
            return MCPResponse(
                request_id=request.request_id,
                server_id=request.server_id,
                capability_name=request.capability_name,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _execute_single_request(self, request: MCPRequest, server: MCPServer) -> MCPResponse:
        """Execute a single request to an MCP server."""
        start_time = time.time()
        
        # For file:// protocol, handle locally
        if server.address.startswith('file://'):
            return await self._handle_file_request(request, server)
        
        # Unsupported protocol fallback
        return MCPResponse(
            request_id=request.request_id,
            server_id=request.server_id,
            capability_name=request.capability_name,
            success=False,
            error=f"Unsupported protocol in address: {server.address}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _handle_file_request(self, request: MCPRequest, server: MCPServer) -> MCPResponse:
        """Handle file system requests locally."""
        start_time = time.time()
        
        try:
            if request.capability_name == 'read_file':
                return await self._read_file_local(request, start_time)
            elif request.capability_name == 'list_directory':
                return await self._list_directory_local(request, start_time)
            else:
                return MCPResponse(
                    request_id=request.request_id,
                    server_id=request.server_id,
                    capability_name=request.capability_name,
                    success=False,
                    error=f"Unsupported file capability: {request.capability_name}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                server_id=request.server_id,
                capability_name=request.capability_name,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _read_file_local(self, request: MCPRequest, start_time: float) -> MCPResponse:
        """Read file from local filesystem."""
        import mimetypes
        
        file_path = request.parameters.get('path')
        if not file_path:
            raise ValueError("Missing 'path' parameter for read_file")
        
        # Security check - prevent path traversal
        if '..' in file_path or os.path.isabs(file_path):
            raise ValueError("Invalid file path for security reasons")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try binary mode for non-text files
            with open(file_path, 'rb') as f:
                content = f.read()
                content = f"Binary file ({len(content)} bytes)"
        
        # Get file metadata
        stat = os.stat(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return MCPResponse(
            request_id=request.request_id,
            server_id=request.server_id,
            capability_name=request.capability_name,
            success=True,
            data={
                'content': content,
                'metadata': {
                    'path': file_path,
                    'size': stat.st_size,
                    'mime_type': mime_type,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'encoding': 'utf-8'
                }
            },
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _list_directory_local(self, request: MCPRequest, start_time: float) -> MCPResponse:
        """List directory contents from local filesystem."""
        dir_path = request.parameters.get('path', '.')
        
        # Security check
        if '..' in dir_path:
            raise ValueError("Invalid directory path for security reasons")
        
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if not os.path.isdir(dir_path):
            raise ValueError(f"Path is not a directory: {dir_path}")
        
        # List directory contents
        items = []
        for item_name in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item_name)
            stat = os.stat(item_path)
            
            items.append({
                'name': item_name,
                'path': item_path,
                'type': 'directory' if os.path.isdir(item_path) else 'file',
                'size': stat.st_size if os.path.isfile(item_path) else None,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return MCPResponse(
            request_id=request.request_id,
            server_id=request.server_id,
            capability_name=request.capability_name,
            success=True,
            data={
                'directory': dir_path,
                'items': items,
                'count': len(items)
            },
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _get_cached_response(self, cache_key: str) -> Optional[MCPResponse]:
        """Get response from cache if not expired."""
        if cache_key in self.response_cache:
            response = self.response_cache[cache_key]
            # Check if cache entry is still valid
            cache_age = (datetime.now() - response.timestamp).total_seconds()
            if cache_age < self.config.cache_default_ttl:
                return response
            else:
                # Remove expired entry
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: MCPResponse):
        """Cache a response."""
        # Simple LRU-like eviction if cache is full
        if len(self.response_cache) >= self.config.cache_max_size:
            # Remove oldest entry
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k].timestamp)
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = response
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get statistics about registered servers."""
        stats = {
            'total_servers': len(self.servers),
            'available_servers': sum(1 for s in self.servers.values() 
                                   if s.status == MCPServerStatus.AVAILABLE),
            'total_capabilities': sum(len(s.capabilities) for s in self.servers.values()),
            'cache_size': len(self.response_cache),
            'active_requests': len(self.active_requests)
        }
        
        # Server status breakdown
        status_counts = {}
        for server in self.servers.values():
            status_counts[server.status.value] = status_counts.get(server.status.value, 0) + 1
        stats['status_breakdown'] = status_counts
        
        return stats
    
    def list_available_servers(self) -> Dict[str, MCPServer]:
        """Get all available servers."""
        return {k: v for k, v in self.servers.items() 
                if v.status == MCPServerStatus.AVAILABLE}
    
    def get_server_capabilities(self, server_id: str) -> List[MCPCapability]:
        """Get capabilities for a specific server."""
        if server_id in self.servers:
            return self.servers[server_id].capabilities
        return []


# Example usage and testing
async def test_mcp_host():
    """Test the enhanced MCP host functionality."""
    print("=== Enhanced MCP Host Test ===")
    
    mcp_host = MCPHost()
    
    # Test server discovery
    print("\nDiscovering servers...")
    servers = await mcp_host.discover_servers()
    print(f"Total registered servers: {len(mcp_host.servers)}")
    
    # Test request detection
    test_phrases = [
        "Please read file README.md",
        "mcp:read_file config.yaml",
        "List directory contents of ./docs",
        "Can you access the local file notes.txt?",
        "Search for documentation about MCP"
    ]
    
    for phrase in test_phrases:
        print(f"\nTesting phrase: '{phrase}'")
        is_mcp, request = mcp_host.is_mcp_request(phrase)
        print(f"  Is MCP request: {is_mcp}")
        
        if is_mcp and request:
            print(f"  Server: {request.server_id}")
            print(f"  Capability: {request.capability_name}")
            print(f"  Parameters: {request.parameters}")
            
            # Execute request if server available
            response = await mcp_host.execute_request(request)
            print(f"  Success: {response.success}")
            if response.success:
                data_preview = str(response.data)[:100] + "..." if len(str(response.data)) > 100 else str(response.data)
                print(f"  Data: {data_preview}")
            else:
                print(f"  Error: {response.error}")
    
    # Print server stats
    stats = mcp_host.get_server_stats()
    print(f"\nMCP Host Stats: {stats}")
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_mcp_host()) 