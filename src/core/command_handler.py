#!/usr/bin/env python3
"""
AVA Core Command Handler Module

This module provides comprehensive command processing, validation, and routing for AVA
(Afsah's Virtual Assistant). Handles CLI commands, API endpoints, system commands, and
agentic task coordination. Optimized for NVIDIA RTX A2000 4GB VRAM constraints.

Author: Assistant
Date: 2024
"""

import re
import json
import uuid
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import shlex
import inspect
from functools import wraps
import argparse
import logging

# Import AVA core components
from .config import get_config_manager, AVAConfig
from .logger import get_logger, LogCategory, log_performance, AVALogger
from .assistant import AssistantRequest, AssistantResponse, ResponseType


class CommandType(Enum):
    """Types of commands that can be processed."""
    SYSTEM = "system"          # System management commands
    AGENT = "agent"            # Agentic task commands
    TOOL = "tool"              # Tool execution commands
    CONFIG = "config"          # Configuration commands
    STATUS = "status"          # Status and monitoring commands
    CONVERSATION = "conversation"  # Conversation management
    HELP = "help"              # Help and documentation
    CUSTOM = "custom"          # Custom user-defined commands


class CommandPriority(Enum):
    """Command execution priorities."""
    CRITICAL = 0   # System critical commands
    HIGH = 1       # User-facing operations
    NORMAL = 2     # Regular commands
    LOW = 3        # Background operations


class CommandSource(Enum):
    """Source of command execution."""
    CLI = "cli"
    API = "api"
    INTERNAL = "internal"
    SCHEDULED = "scheduled"
    MCP = "mcp"


@dataclass
class CommandContext:
    """Context information for command execution."""
    source: CommandSource
    session_id: str
    user_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    working_directory: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommandDefinition:
    """Definition of a command including its parameters and validation."""
    name: str
    description: str
    command_type: CommandType
    handler: Callable
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    priority: CommandPriority = CommandPriority.NORMAL
    async_execution: bool = False
    timeout_seconds: Optional[float] = None
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandRequest:
    """Request structure for command execution."""
    command: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_input: Optional[str] = None
    context: Optional[CommandContext] = None
    options: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    command: str = ""
    context: Optional[CommandContext] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class CommandValidator:
    """Validates command inputs and permissions."""
    
    def __init__(self, config: AVAConfig):
        """Initialize command validator."""
        self.config = config
        self.logger = get_logger("ava.command.validator")
    
    def validate_command(self, request: CommandRequest, definition: CommandDefinition) -> Tuple[bool, Optional[str]]:
        """Validate a command request against its definition."""
        try:
            # Check required permissions
            if definition.required_permissions and request.context:
                user_permissions = request.context.permissions
                for required_perm in definition.required_permissions:
                    if required_perm not in user_permissions:
                        return False, f"Missing required permission: {required_perm}"
            
            # Validate required parameters
            for param_name, param_def in definition.parameters.items():
                if param_def.get('required', False) and param_name not in request.arguments:
                    return False, f"Missing required parameter: {param_name}"
            
            # Validate parameter types and values
            for param_name, value in request.arguments.items():
                if param_name in definition.parameters:
                    param_def = definition.parameters[param_name]
                    
                    # Type validation
                    expected_type = param_def.get('type', str)
                    if not isinstance(value, expected_type):
                        try:
                            # Attempt type conversion
                            if expected_type == int:
                                request.arguments[param_name] = int(value)
                            elif expected_type == float:
                                request.arguments[param_name] = float(value)
                            elif expected_type == bool:
                                request.arguments[param_name] = str(value).lower() in ('true', '1', 'yes', 'on')
                        except (ValueError, TypeError):
                            return False, f"Invalid type for parameter {param_name}: expected {expected_type.__name__}"
                    
                    # Range validation
                    if 'min' in param_def and value < param_def['min']:
                        return False, f"Parameter {param_name} below minimum value: {param_def['min']}"
                    if 'max' in param_def and value > param_def['max']:
                        return False, f"Parameter {param_name} above maximum value: {param_def['max']}"
                    
                    # Choice validation
                    if 'choices' in param_def and value not in param_def['choices']:
                        return False, f"Invalid choice for {param_name}: {value}. Valid choices: {param_def['choices']}"
            
            # Security validation
            if self.config.security.enable_input_validation:
                if not self._validate_security(request):
                    return False, "Command failed security validation"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error validating command: {e}", LogCategory.SECURITY)
            return False, f"Validation error: {e}"
    
    def _validate_security(self, request: CommandRequest) -> bool:
        """Perform security validation on command request."""
        try:
            # Check input length
            if request.raw_input and len(request.raw_input) > self.config.security.max_input_length:
                return False
            
            # Check for blocked patterns
            if request.raw_input:
                for pattern in self.config.security.blocked_patterns:
                    if re.search(pattern, request.raw_input, re.IGNORECASE):
                        self.logger.warning(f"Blocked pattern detected: {pattern}", LogCategory.SECURITY)
                        return False
            
            # Rate limiting check (simplified)
            if self.config.security.enable_rate_limiting:
                # In a full implementation, this would check rate limits per user/session
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation error: {e}", LogCategory.SECURITY)
            return False


class CommandParser:
    """Parses raw command input into structured command requests."""
    
    def __init__(self):
        """Initialize command parser."""
        self.logger = get_logger("ava.command.parser")
    
    def parse_command(self, raw_input: str, context: Optional[CommandContext] = None) -> Optional[CommandRequest]:
        """Parse raw command input into a CommandRequest."""
        try:
            if not raw_input or not raw_input.strip():
                return None
            
            # Handle different command formats
            if raw_input.startswith('/'):
                # Slash commands (e.g., /help, /status)
                return self._parse_slash_command(raw_input, context)
            elif raw_input.startswith('!'):
                # System commands (e.g., !config, !restart)
                return self._parse_system_command(raw_input, context)
            elif self._looks_like_cli_command(raw_input):
                # CLI-style commands (e.g., ava query "what is the weather")
                return self._parse_cli_command(raw_input, context)
            else:
                # Natural language input - treat as agent conversation
                return self._parse_natural_language(raw_input, context)
            
        except Exception as e:
            self.logger.error(f"Error parsing command: {e}", LogCategory.SYSTEM)
            return None
    
    def _parse_slash_command(self, raw_input: str, context: Optional[CommandContext]) -> CommandRequest:
        """Parse slash-style commands."""
        parts = raw_input[1:].split()  # Remove leading slash
        command = parts[0] if parts else ""
        
        arguments = {}
        if len(parts) > 1:
            # Simple key=value parsing for slash commands
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    arguments[key] = value
                else:
                    # Positional arguments
                    arguments[f'arg_{len(arguments)}'] = part
        
        return CommandRequest(
            command=command,
            arguments=arguments,
            raw_input=raw_input,
            context=context
        )
    
    def _parse_system_command(self, raw_input: str, context: Optional[CommandContext]) -> CommandRequest:
        """Parse system commands starting with !."""
        parts = raw_input[1:].split()  # Remove leading !
        command = f"system.{parts[0]}" if parts else "system.unknown"
        
        arguments = {}
        if len(parts) > 1:
            arguments['args'] = parts[1:]
        
        return CommandRequest(
            command=command,
            arguments=arguments,
            raw_input=raw_input,
            context=context
        )
    
    def _parse_cli_command(self, raw_input: str, context: Optional[CommandContext]) -> CommandRequest:
        """Parse CLI-style commands."""
        try:
            # Use shlex to properly handle quoted arguments
            parts = shlex.split(raw_input)
            if not parts:
                return CommandRequest(command="unknown", raw_input=raw_input, context=context)
            
            # First part is the command, rest are arguments
            command = parts[0]
            
            # Simple argument parsing (could be enhanced with argparse)
            arguments = {}
            i = 1
            while i < len(parts):
                arg = parts[i]
                if arg.startswith('--'):
                    # Long option
                    key = arg[2:]
                    if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                        arguments[key] = parts[i + 1]
                        i += 2
                    else:
                        arguments[key] = True
                        i += 1
                elif arg.startswith('-'):
                    # Short option
                    key = arg[1:]
                    if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                        arguments[key] = parts[i + 1]
                        i += 2
                    else:
                        arguments[key] = True
                        i += 1
                else:
                    # Positional argument
                    arguments[f'arg_{len([k for k in arguments.keys() if k.startswith("arg_")])}'] = arg
                    i += 1
            
            return CommandRequest(
                command=command,
                arguments=arguments,
                raw_input=raw_input,
                context=context
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing CLI command: {e}", LogCategory.SYSTEM)
            return CommandRequest(command="parse_error", raw_input=raw_input, context=context)
    
    def _parse_natural_language(self, raw_input: str, context: Optional[CommandContext]) -> CommandRequest:
        """Parse natural language input as an agent conversation."""
        return CommandRequest(
            command="agent.conversation",
            arguments={'input': raw_input},
            raw_input=raw_input,
            context=context
        )
    
    def _looks_like_cli_command(self, raw_input: str) -> bool:
        """Heuristic to determine if input looks like a CLI command."""
        # Simple heuristics - could be enhanced
        cli_indicators = [
            raw_input.startswith('ava '),
            '--' in raw_input,
            raw_input.count(' ') < 10,  # Commands typically have fewer words
            any(raw_input.startswith(cmd) for cmd in ['query', 'execute', 'run', 'get', 'set', 'list'])
        ]
        return any(cli_indicators)


class CommandHandler:
    """Main command handler for AVA system."""
    
    def __init__(self, config: Optional[AVAConfig] = None):
        """Initialize command handler."""
        self.config = config or get_config_manager().get_config()
        self.logger = get_logger("ava.command.handler", {
            'log_level': self.config.logging.level.value
        })
        
        # Core components
        self.validator = CommandValidator(self.config)
        self.parser = CommandParser()
        
        # Command registry
        self._commands: Dict[str, CommandDefinition] = {}
        self._command_aliases: Dict[str, str] = {}
        
        # Execution tracking
        self._execution_stats = {
            'total_commands': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'execution_times': []
        }
        
        # Background processing
        self._command_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = threading.Event()
        
        # Initialize built-in commands
        self._register_builtin_commands()
        
        self.logger.info("Command handler initialized", LogCategory.SYSTEM)
    
    def _register_builtin_commands(self):
        """Register built-in system commands."""
        try:
            # Help command
            self.register_command(CommandDefinition(
                name="help",
                description="Show help information for commands",
                command_type=CommandType.HELP,
                handler=self._handle_help,
                parameters={
                    'command': {'type': str, 'required': False, 'description': 'Specific command to get help for'}
                },
                aliases=['h', '?'],
                examples=['help', 'help status', 'help config.set']
            ))
            
            # Status commands
            self.register_command(CommandDefinition(
                name="status",
                description="Show system status",
                command_type=CommandType.STATUS,
                handler=self._handle_status,
                aliases=['stat', 'info']
            ))
            
            # Configuration commands
            self.register_command(CommandDefinition(
                name="config.get",
                description="Get configuration value",
                command_type=CommandType.CONFIG,
                handler=self._handle_config_get,
                parameters={
                    'key': {'type': str, 'required': True, 'description': 'Configuration key to retrieve'}
                }
            ))
            
            self.register_command(CommandDefinition(
                name="config.set",
                description="Set configuration value",
                command_type=CommandType.CONFIG,
                handler=self._handle_config_set,
                parameters={
                    'key': {'type': str, 'required': True, 'description': 'Configuration key to set'},
                    'value': {'type': str, 'required': True, 'description': 'Configuration value to set'}
                },
                required_permissions=['config.write']
            ))
            
            # System commands
            self.register_command(CommandDefinition(
                name="system.restart",
                description="Restart AVA system",
                command_type=CommandType.SYSTEM,
                handler=self._handle_system_restart,
                required_permissions=['system.admin'],
                priority=CommandPriority.CRITICAL
            ))
            
            # Agent conversation command
            self.register_command(CommandDefinition(
                name="agent.conversation",
                description="Process natural language input with AVA agent",
                command_type=CommandType.AGENT,
                handler=self._handle_agent_conversation,
                parameters={
                    'input': {'type': str, 'required': True, 'description': 'Natural language input'}
                },
                async_execution=True
            ))
            
            # Session management
            self.register_command(CommandDefinition(
                name="session.list",
                description="List active sessions",
                command_type=CommandType.CONVERSATION,
                handler=self._handle_session_list
            ))
            
            self.logger.info("Built-in commands registered", LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Error registering built-in commands: {e}", LogCategory.SYSTEM)
    
    def register_command(self, definition: CommandDefinition) -> bool:
        """Register a new command."""
        try:
            # Validate command definition
            if not definition.name or not definition.handler:
                self.logger.error("Invalid command definition: missing name or handler", LogCategory.SYSTEM)
                return False
            
            # Check for conflicts
            if definition.name in self._commands:
                self.logger.warning(f"Overriding existing command: {definition.name}", LogCategory.SYSTEM)
            
            # Register main command
            self._commands[definition.name] = definition
            
            # Register aliases
            for alias in definition.aliases:
                if alias in self._command_aliases:
                    self.logger.warning(f"Overriding existing alias: {alias}", LogCategory.SYSTEM)
                self._command_aliases[alias] = definition.name
            
            self.logger.info(f"Registered command: {definition.name}", LogCategory.SYSTEM)
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering command {definition.name}: {e}", LogCategory.SYSTEM)
            return False
    
    def unregister_command(self, command_name: str) -> bool:
        """Unregister a command."""
        try:
            if command_name not in self._commands:
                return False
            
            definition = self._commands[command_name]
            
            # Remove aliases
            for alias in definition.aliases:
                if alias in self._command_aliases:
                    del self._command_aliases[alias]
            
            # Remove main command
            del self._commands[command_name]
            
            self.logger.info(f"Unregistered command: {command_name}", LogCategory.SYSTEM)
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering command {command_name}: {e}", LogCategory.SYSTEM)
            return False
    
    async def execute_command(self, raw_input: str, context: Optional[CommandContext] = None) -> CommandResult:
        """Execute a command from raw input."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Parse the command
            request = self.parser.parse_command(raw_input, context)
            if not request:
                return CommandResult(
                    success=False,
                    error="Failed to parse command",
                    command=raw_input,
                    context=context
                )
            
            # Execute the parsed command
            result = await self.execute_command_request(request)
            
            # Update execution time
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.logger.error(f"Error executing command: {e}", LogCategory.SYSTEM)
            return CommandResult(
                success=False,
                error=f"Command execution error: {e}",
                command=raw_input,
                context=context,
                execution_time_ms=execution_time
            )
    
    async def execute_command_request(self, request: CommandRequest) -> CommandResult:
        """Execute a structured command request."""
        try:
            self.logger.info(f"Executing command: {request.command}", LogCategory.SYSTEM,
                           request_id=request.context.request_id if request.context else None)
            
            # Update stats
            self._execution_stats['total_commands'] += 1
            
            # Resolve command name (handle aliases)
            command_name = self._resolve_command_name(request.command)
            if command_name not in self._commands:
                return CommandResult(
                    success=False,
                    error=f"Unknown command: {request.command}",
                    command=request.command,
                    context=request.context
                )
            
            definition = self._commands[command_name]
            
            # Validate command
            is_valid, validation_error = self.validator.validate_command(request, definition)
            if not is_valid:
                return CommandResult(
                    success=False,
                    error=f"Command validation failed: {validation_error}",
                    command=request.command,
                    context=request.context
                )
            
            # Execute command
            if definition.async_execution:
                result = await self._execute_async_command(request, definition)
            else:
                result = await self._execute_sync_command(request, definition)
            
            # Update stats
            if result.success:
                self._execution_stats['successful_commands'] += 1
            else:
                self._execution_stats['failed_commands'] += 1
            
            self.logger.info(f"Command executed: {request.command} (success: {result.success})",
                           LogCategory.SYSTEM, duration_ms=result.execution_time_ms)
            
            return result
            
        except Exception as e:
            self._execution_stats['failed_commands'] += 1
            self.logger.error(f"Error executing command request: {e}", LogCategory.SYSTEM)
            return CommandResult(
                success=False,
                error=f"Command execution error: {e}",
                command=request.command,
                context=request.context
            )
    
    def _resolve_command_name(self, command: str) -> str:
        """Resolve command name from aliases."""
        return self._command_aliases.get(command, command)
    
    async def _execute_async_command(self, request: CommandRequest, definition: CommandDefinition) -> CommandResult:
        """Execute an asynchronous command."""
        try:
            # Apply timeout if specified
            if definition.timeout_seconds:
                result = await asyncio.wait_for(
                    definition.handler(request),
                    timeout=definition.timeout_seconds
                )
            else:
                result = await definition.handler(request)
            
            return CommandResult(
                success=True,
                output=result,
                command=request.command,
                context=request.context
            )
            
        except asyncio.TimeoutError:
            return CommandResult(
                success=False,
                error=f"Command timed out after {definition.timeout_seconds} seconds",
                command=request.command,
                context=request.context
            )
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                command=request.command,
                context=request.context
            )
    
    async def _execute_sync_command(self, request: CommandRequest, definition: CommandDefinition) -> CommandResult:
        """Execute a synchronous command."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, definition.handler, request)
            
            return CommandResult(
                success=True,
                output=result,
                command=request.command,
                context=request.context
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                command=request.command,
                context=request.context
            )
    
    # Built-in command handlers
    def _handle_help(self, request: CommandRequest) -> str:
        """Handle help command."""
        command_name = request.arguments.get('command')
        
        if command_name:
            # Help for specific command
            resolved_name = self._resolve_command_name(command_name)
            if resolved_name in self._commands:
                definition = self._commands[resolved_name]
                help_text = f"Command: {definition.name}\n"
                help_text += f"Description: {definition.description}\n"
                help_text += f"Type: {definition.command_type.value}\n"
                
                if definition.parameters:
                    help_text += "\nParameters:\n"
                    for param_name, param_def in definition.parameters.items():
                        required = " (required)" if param_def.get('required') else ""
                        help_text += f"  {param_name}: {param_def.get('description', 'No description')}{required}\n"
                
                if definition.aliases:
                    help_text += f"\nAliases: {', '.join(definition.aliases)}\n"
                
                if definition.examples:
                    help_text += "\nExamples:\n"
                    for example in definition.examples:
                        help_text += f"  {example}\n"
                
                return help_text
            else:
                return f"Command not found: {command_name}"
        else:
            # General help
            help_text = "Available commands:\n\n"
            
            # Group commands by type
            commands_by_type = {}
            for cmd_name, definition in self._commands.items():
                cmd_type = definition.command_type.value
                if cmd_type not in commands_by_type:
                    commands_by_type[cmd_type] = []
                commands_by_type[cmd_type].append((cmd_name, definition.description))
            
            for cmd_type, commands in commands_by_type.items():
                help_text += f"{cmd_type.upper()} Commands:\n"
                for cmd_name, description in commands:
                    help_text += f"  {cmd_name}: {description}\n"
                help_text += "\n"
            
            help_text += "Use 'help <command>' for detailed information about a specific command."
            return help_text
    
    def _handle_status(self, request: CommandRequest) -> Dict[str, Any]:
        """Handle status command."""
        try:
            # Get system status from assistant if available
            from .assistant import get_assistant
            
            # This would get the actual assistant status in a real implementation
            status = {
                'timestamp': datetime.now().isoformat(),
                'commands': {
                    'total_registered': len(self._commands),
                    'total_executed': self._execution_stats['total_commands'],
                    'successful': self._execution_stats['successful_commands'],
                    'failed': self._execution_stats['failed_commands'],
                    'success_rate': self._execution_stats['successful_commands'] / max(1, self._execution_stats['total_commands'])
                },
                'system': 'AVA Command Handler Active'
            }
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def _handle_config_get(self, request: CommandRequest) -> Any:
        """Handle config get command."""
        key = request.arguments.get('key')
        try:
            # Navigate nested configuration
            config_dict = asdict(self.config)
            keys = key.split('.')
            value = config_dict
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return f"Configuration key not found: {key}"
            
            return f"{key} = {value}"
            
        except Exception as e:
            return f"Error retrieving configuration: {e}"
    
    def _handle_config_set(self, request: CommandRequest) -> str:
        """Handle config set command."""
        key = request.arguments.get('key')
        value = request.arguments.get('value')
        
        try:
            # This would update the actual configuration
            # For now, just return a placeholder response
            return f"Configuration updated: {key} = {value}"
            
        except Exception as e:
            return f"Error setting configuration: {e}"
    
    def _handle_system_restart(self, request: CommandRequest) -> str:
        """Handle system restart command."""
        # This would trigger an actual system restart
        return "System restart initiated (placeholder)"
    
    async def _handle_agent_conversation(self, request: CommandRequest) -> str:
        """Handle agent conversation command."""
        try:
            user_input = request.arguments.get('input', '')
            
            # Create assistant request
            assistant_request = AssistantRequest(
                request_id=request.context.request_id if request.context else str(uuid.uuid4()),
                session_id=request.context.session_id if request.context else "default",
                user_input=user_input
            )
            
            # This would use the actual assistant to process the request
            # For now, return a placeholder response
            return f"AVA processed: {user_input}"
            
        except Exception as e:
            return f"Error in agent conversation: {e}"
    
    def _handle_session_list(self, request: CommandRequest) -> List[Dict[str, Any]]:
        """Handle session list command."""
        try:
            # This would get actual session information from the assistant
            return [
                {
                    'session_id': 'example_session',
                    'created_at': datetime.now().isoformat(),
                    'message_count': 5,
                    'active': True
                }
            ]
        except Exception as e:
            return [{'error': str(e)}]
    
    def list_commands(self, command_type: Optional[CommandType] = None) -> List[Dict[str, Any]]:
        """List all registered commands."""
        commands = []
        for name, definition in self._commands.items():
            if command_type is None or definition.command_type == command_type:
                commands.append({
                    'name': name,
                    'description': definition.description,
                    'type': definition.command_type.value,
                    'aliases': definition.aliases,
                    'priority': definition.priority.value,
                    'async': definition.async_execution
                })
        
        return sorted(commands, key=lambda x: x['name'])
    
    def get_command_info(self, command_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific command."""
        resolved_name = self._resolve_command_name(command_name)
        if resolved_name not in self._commands:
            return None
        
        definition = self._commands[resolved_name]
        return {
            'name': definition.name,
            'description': definition.description,
            'type': definition.command_type.value,
            'aliases': definition.aliases,
            'parameters': definition.parameters,
            'required_permissions': definition.required_permissions,
            'priority': definition.priority.value,
            'async_execution': definition.async_execution,
            'timeout_seconds': definition.timeout_seconds,
            'examples': definition.examples
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get command execution statistics."""
        avg_execution_time = (
            sum(self._execution_stats['execution_times']) / 
            len(self._execution_stats['execution_times'])
            if self._execution_stats['execution_times'] else 0
        )
        
        return {
            'total_commands': self._execution_stats['total_commands'],
            'successful_commands': self._execution_stats['successful_commands'],
            'failed_commands': self._execution_stats['failed_commands'],
            'success_rate': self._execution_stats['successful_commands'] / max(1, self._execution_stats['total_commands']),
            'avg_execution_time_ms': avg_execution_time,
            'registered_commands': len(self._commands)
        }
    
    async def shutdown(self):
        """Shutdown the command handler."""
        try:
            self.logger.info("Shutting down command handler", LogCategory.SYSTEM)
            self._shutdown_event.set()
            
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Command handler shutdown completed", LogCategory.SYSTEM)
            
        except Exception as e:
            self.logger.error(f"Error during command handler shutdown: {e}", LogCategory.SYSTEM)


# Global command handler instance
_global_handler: Optional[CommandHandler] = None


def get_command_handler(config: Optional[AVAConfig] = None) -> CommandHandler:
    """Get global command handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = CommandHandler(config)
    return _global_handler


async def execute_command(raw_input: str, context: Optional[CommandContext] = None) -> CommandResult:
    """Execute a command using the global handler."""
    return await get_command_handler().execute_command(raw_input, context)


# Decorator for easy command registration
def command(name: str, description: str, command_type: CommandType = CommandType.CUSTOM, 
           aliases: Optional[List[str]] = None, **kwargs):
    """Decorator for registering command functions."""
    def decorator(func):
        # Extract parameter information from function signature
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name != 'request':  # Skip the request parameter
                param_info = {'type': str}  # Default type
                
                # Extract type from annotation
                if param.annotation != inspect.Parameter.empty:
                    param_info['type'] = param.annotation
                
                # Check if parameter is required
                if param.default == inspect.Parameter.empty:
                    param_info['required'] = True
                else:
                    param_info['required'] = False
                    param_info['default'] = param.default
                
                parameters[param_name] = param_info
        
        # Create command definition
        definition = CommandDefinition(
            name=name,
            description=description,
            command_type=command_type,
            handler=func,
            parameters=parameters,
            aliases=aliases or [],
            **kwargs
        )
        
        # Register with global handler
        get_command_handler().register_command(definition)
        
        return func
    
    return decorator


# Testing functions
async def test_command_handler():
    """Test command handler functionality."""
    print("Testing AVA Command Handler...")
    
    handler = CommandHandler()
    
    # Test help command
    result = await handler.execute_command("help")
    print(f"Help command result: {result.success}")
    if result.output:
        print(f"Help output:\n{result.output[:200]}...")
    
    # Test status command
    result = await handler.execute_command("status")
    print(f"Status command result: {result.success}")
    print(f"Status output: {result.output}")
    
    # Test natural language processing
    result = await handler.execute_command("Hello AVA, how are you today?")
    print(f"Natural language result: {result.success}")
    print(f"Agent response: {result.output}")
    
    # Test command registration
    @command("test.hello", "Test hello command", CommandType.CUSTOM)
    def hello_command(request: CommandRequest) -> str:
        name = request.arguments.get('name', 'World')
        return f"Hello, {name}!"
    
    result = await handler.execute_command("test.hello name=AVA")
    print(f"Custom command result: {result.success}")
    print(f"Custom command output: {result.output}")
    
    # Get command statistics
    stats = handler.get_execution_stats()
    print(f"Execution stats: {stats}")
    
    await handler.shutdown()
    print("Command handler test completed!")


if __name__ == "__main__":
    asyncio.run(test_command_handler())
