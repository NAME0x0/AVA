#!/usr/bin/env python3
"""
AVA CLI Interface Module

This module provides a comprehensive command-line interface for AVA (Afsah's Virtual Assistant)
with features like interactive shell, command history, auto-completion, streaming responses,
and full integration with AVA's core systems. Optimized for the 4GB VRAM constraint.

Author: Assistant
Date: 2024
"""

import os
import sys
import asyncio
import argparse
import readline
import atexit
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import time

try:
    import rich
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import AVA core components
from ..core.config import get_config_manager, AVAConfig
from ..core.logger import get_logger, LogCategory
from ..core.assistant import get_assistant, AssistantRequest, AssistantResponse, ResponseType
from ..core.command_handler import get_command_handler, CommandContext, CommandSource


class CLIMode(Enum):
    """CLI operation modes."""
    INTERACTIVE = "interactive"
    COMMAND = "command"
    SCRIPT = "script"
    STREAMING = "streaming"


class OutputFormat(Enum):
    """Output format options."""
    PLAIN = "plain"
    JSON = "json"
    MARKDOWN = "markdown"
    TABLE = "table"
    RICH = "rich"


@dataclass
class CLISession:
    """CLI session state."""
    session_id: str
    start_time: datetime
    command_count: int = 0
    last_command: Optional[str] = None
    working_directory: str = os.getcwd()
    history: List[str] = None
    mode: CLIMode = CLIMode.INTERACTIVE
    output_format: OutputFormat = OutputFormat.RICH if RICH_AVAILABLE else OutputFormat.PLAIN
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class AVACLIInterface:
    """Main CLI interface for AVA."""
    
    def __init__(self, config: Optional[AVAConfig] = None):
        """Initialize CLI interface."""
        self.config = config or get_config_manager().get_config()
        self.logger = get_logger("ava.cli", {
            'log_level': self.config.logging.level.value,
            'enable_console_logging': False  # Disable console logging to avoid conflicts
        })
        
        # Initialize components
        self.command_handler = get_command_handler(self.config)
        self.assistant = None  # Will be initialized asynchronously
        
        # CLI state
        self.session = CLISession(
            session_id=f"cli_{int(time.time())}",
            start_time=datetime.now()
        )
        
        # Rich console for enhanced output
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        # Command history
        self.history_file = Path.home() / '.ava' / 'cli_history'
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-completion
        self.completion_commands = []
        
        # Shutdown flag
        self._shutdown = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
        self.logger.info("CLI interface initialized", LogCategory.INTERFACE)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.print_message("\n🛑 Shutting down AVA CLI...", style="warning")
        self._shutdown = True
    
    def _cleanup(self):
        """Cleanup on exit."""
        try:
            self._save_history()
            if hasattr(self, 'assistant') and self.assistant:
                # Shutdown will be handled by the main loop
                pass
        except Exception as e:
            self.logger.error(f"Error during CLI cleanup: {e}", LogCategory.INTERFACE)
    
    def _setup_readline(self):
        """Setup readline for command history and completion."""
        try:
            # Load command history
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            
            # Set history length
            readline.set_history_length(1000)
            
            # Setup completion
            readline.set_completer(self._completer)
            readline.parse_and_bind("tab: complete")
            
            # Enable vi or emacs mode based on preference
            if self.config.interface.cli_prompt.endswith("vi> "):
                readline.parse_and_bind("set editing-mode vi")
            else:
                readline.parse_and_bind("set editing-mode emacs")
            
            self.logger.debug("Readline setup completed", LogCategory.INTERFACE)
            
        except Exception as e:
            self.logger.error(f"Error setting up readline: {e}", LogCategory.INTERFACE)
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Auto-completion function for readline."""
        try:
            if state == 0:
                # First call - generate completions
                self.completion_matches = []
                
                # Get available commands
                commands = self.command_handler.list_commands()
                command_names = [cmd['name'] for cmd in commands]
                
                # Add aliases
                for cmd in commands:
                    command_names.extend(cmd.get('aliases', []))
                
                # Add common patterns
                common_commands = [
                    'help', 'status', 'exit', 'quit', 'clear',
                    '/help', '/status', '/exit',
                    '!restart', '!config'
                ]
                
                all_commands = list(set(command_names + common_commands))
                
                # Filter based on current text
                self.completion_matches = [
                    cmd for cmd in all_commands 
                    if cmd.startswith(text)
                ]
            
            # Return next match
            if state < len(self.completion_matches):
                return self.completion_matches[state]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error in auto-completion: {e}", LogCategory.INTERFACE)
            return None
    
    def _save_history(self):
        """Save command history to file."""
        try:
            readline.write_history_file(str(self.history_file))
            self.logger.debug("Command history saved", LogCategory.INTERFACE)
        except Exception as e:
            self.logger.error(f"Error saving history: {e}", LogCategory.INTERFACE)
    
    def print_message(self, message: str, style: str = "info", **kwargs):
        """Print a message with appropriate formatting."""
        try:
            if self.console and RICH_AVAILABLE:
                if style == "error":
                    self.console.print(f"❌ {message}", style="bold red")
                elif style == "warning":
                    self.console.print(f"⚠️  {message}", style="bold yellow")
                elif style == "success":
                    self.console.print(f"✅ {message}", style="bold green")
                elif style == "info":
                    self.console.print(f"ℹ️  {message}", style="bold blue")
                elif style == "system":
                    self.console.print(f"🔧 {message}", style="bold cyan")
                else:
                    self.console.print(message, **kwargs)
            else:
                # Fallback to plain text
                prefix_map = {
                    "error": "[ERROR] ",
                    "warning": "[WARNING] ",
                    "success": "[SUCCESS] ",
                    "info": "[INFO] ",
                    "system": "[SYSTEM] "
                }
                prefix = prefix_map.get(style, "")
                print(f"{prefix}{message}")
                
        except Exception as e:
            # Ultimate fallback
            print(f"[PRINT_ERROR] {message}")
    
    def print_banner(self):
        """Print welcome banner."""
        if self.console and RICH_AVAILABLE:
            banner_text = """
    ╔═══════════════════════════════════════════╗
    ║              AVA Assistant                ║
    ║        Afsah's Virtual Assistant          ║
    ║                                           ║
    ║    Local Agentic AI - RTX A2000 4GB      ║
    ║           Optimized & Ready               ║
    ╚═══════════════════════════════════════════╝
            """
            
            panel = Panel(
                banner_text,
                title="[bold blue]Welcome to AVA[/bold blue]",
                border_style="blue",
                padding=(1, 2)
            )
            self.console.print(panel)
            
            # Show quick help
            help_text = """
💡 **Quick Commands:**
• Type your questions naturally: "What's the weather like?"
• Use `/help` for command reference
• Use `/status` to check system status
• Use `/exit` or `Ctrl+C` to quit
• Press `Tab` for auto-completion
            """
            self.console.print(Markdown(help_text))
            
        else:
            print("\n" + "="*50)
            print("    AVA Assistant - Afsah's Virtual Assistant")
            print("    Local Agentic AI - RTX A2000 4GB Optimized")
            print("="*50)
            print("\nType 'help' for available commands or just start chatting!")
            print("Use 'exit' or Ctrl+C to quit.\n")
    
    def format_output(self, content: Any, output_format: Optional[OutputFormat] = None) -> str:
        """Format output based on the specified format."""
        try:
            fmt = output_format or self.session.output_format
            
            if fmt == OutputFormat.JSON:
                return json.dumps(content, indent=2, default=str)
            elif fmt == OutputFormat.MARKDOWN and isinstance(content, str):
                if self.console and RICH_AVAILABLE:
                    self.console.print(Markdown(content))
                    return ""
                else:
                    return content
            elif fmt == OutputFormat.TABLE and isinstance(content, (list, dict)):
                if self.console and RICH_AVAILABLE:
                    table = Table()
                    if isinstance(content, dict):
                        table.add_column("Key", style="cyan")
                        table.add_column("Value", style="magenta")
                        for key, value in content.items():
                            table.add_row(str(key), str(value))
                    elif isinstance(content, list) and content:
                        if isinstance(content[0], dict):
                            # List of dictionaries
                            for key in content[0].keys():
                                table.add_column(str(key), style="cyan")
                            for item in content:
                                table.add_row(*[str(item.get(key, "")) for key in content[0].keys()])
                        else:
                            # Simple list
                            table.add_column("Items", style="cyan")
                            for item in content:
                                table.add_row(str(item))
                    
                    self.console.print(table)
                    return ""
                else:
                    return str(content)
            else:
                # Plain format or fallback
                return str(content)
                
        except Exception as e:
            self.logger.error(f"Error formatting output: {e}", LogCategory.INTERFACE)
            return str(content)
    
    async def stream_response(self, response_stream):
        """Display streaming response with live updates."""
        try:
            if self.console and RICH_AVAILABLE:
                with Live("🤔 AVA is thinking...", console=self.console, auto_refresh=True) as live:
                    accumulated_text = ""
                    async for token in response_stream:
                        accumulated_text += token
                        live.update(Text(accumulated_text))
                        await asyncio.sleep(0.01)  # Small delay for smooth updates
            else:
                # Fallback for plain text
                print("AVA: ", end="", flush=True)
                async for token in response_stream:
                    print(token, end="", flush=True)
                print()  # New line at the end
                
        except Exception as e:
            self.logger.error(f"Error streaming response: {e}", LogCategory.INTERFACE)
            self.print_message(f"Error displaying streaming response: {e}", "error")
    
    async def execute_command(self, user_input: str) -> bool:
        """Execute a command and return whether to continue."""
        try:
            # Handle special CLI commands
            if user_input.lower() in ['exit', 'quit', '/exit', '/quit']:
                return False
            elif user_input.lower() in ['clear', '/clear']:
                os.system('clear' if os.name == 'posix' else 'cls')
                return True
            elif user_input.lower() in ['history', '/history']:
                self._show_history()
                return True
            
            # Create command context
            context = CommandContext(
                source=CommandSource.CLI,
                session_id=self.session.session_id,
                working_directory=self.session.working_directory,
                permissions=['user.basic']  # Basic user permissions
            )
            
            # Show processing indicator
            if self.console and RICH_AVAILABLE:
                with self.console.status("[bold green]Processing command...") as status:
                    # Execute command through command handler
                    result = await self.command_handler.execute_command(user_input, context)
            else:
                print("Processing...", end="", flush=True)
                result = await self.command_handler.execute_command(user_input, context)
                print("\r" + " "*15 + "\r", end="")  # Clear processing message
            
            # Update session
            self.session.command_count += 1
            self.session.last_command = user_input
            self.session.history.append(user_input)
            
            # Display result
            if result.success:
                if result.output is not None:
                    formatted_output = self.format_output(result.output)
                    if formatted_output:  # Only print if there's content
                        print(formatted_output)
                
                # Log performance if enabled
                if result.execution_time_ms > 0:
                    self.logger.debug(f"Command executed in {result.execution_time_ms:.2f}ms", 
                                    LogCategory.INTERFACE)
            else:
                self.print_message(f"Command failed: {result.error}", "error")
            
            return True
            
        except KeyboardInterrupt:
            self.print_message("\nCommand interrupted by user", "warning")
            return True
        except Exception as e:
            self.logger.error(f"Error executing command: {e}", LogCategory.INTERFACE)
            self.print_message(f"Error executing command: {e}", "error")
            return True
    
    def _show_history(self):
        """Display command history."""
        try:
            if not self.session.history:
                self.print_message("No command history available.", "info")
                return
            
            if self.console and RICH_AVAILABLE:
                table = Table(title="Command History")
                table.add_column("#", style="cyan", width=4)
                table.add_column("Command", style="white")
                table.add_column("Time", style="dim")
                
                for i, cmd in enumerate(self.session.history[-20:], 1):  # Show last 20 commands
                    # Simple timestamp (in real implementation, you'd store actual timestamps)
                    time_str = f"{i} commands ago"
                    table.add_row(str(i), cmd, time_str)
                
                self.console.print(table)
            else:
                print("\nCommand History:")
                for i, cmd in enumerate(self.session.history[-10:], 1):
                    print(f"{i:2d}. {cmd}")
                print()
                
        except Exception as e:
            self.logger.error(f"Error showing history: {e}", LogCategory.INTERFACE)
            self.print_message(f"Error displaying history: {e}", "error")
    
    async def interactive_mode(self):
        """Run interactive CLI mode."""
        try:
            # Initialize assistant
            self.assistant = await get_assistant()
            
            # Setup readline
            self._setup_readline()
            
            # Show banner
            self.print_banner()
            
            # Main interactive loop
            while not self._shutdown:
                try:
                    # Get user input
                    prompt = self.config.interface.cli_prompt
                    if self.console and RICH_AVAILABLE:
                        user_input = Prompt.ask(f"[bold green]{prompt}[/bold green]").strip()
                    else:
                        user_input = input(prompt).strip()
                    
                    if not user_input:
                        continue
                    
                    # Execute command
                    should_continue = await self.execute_command(user_input)
                    if not should_continue:
                        break
                        
                except KeyboardInterrupt:
                    if self.console and RICH_AVAILABLE:
                        if Confirm.ask("\n🤔 Do you want to exit AVA?", default=False):
                            break
                    else:
                        print("\nUse 'exit' to quit or Ctrl+C again to force quit.")
                        try:
                            # Give user a chance to type 'exit'
                            continue
                        except KeyboardInterrupt:
                            break
                except EOFError:
                    # Ctrl+D pressed
                    break
                except Exception as e:
                    self.logger.error(f"Error in interactive loop: {e}", LogCategory.INTERFACE)
                    self.print_message(f"An error occurred: {e}", "error")
            
            # Shutdown
            self.print_message("Goodbye! 👋", "success")
            
        except Exception as e:
            self.logger.error(f"Error in interactive mode: {e}", LogCategory.INTERFACE)
            self.print_message(f"Fatal error in interactive mode: {e}", "error")
        finally:
            await self._shutdown_components()
    
    async def command_mode(self, command: str):
        """Execute a single command and exit."""
        try:
            # Initialize assistant
            self.assistant = await get_assistant()
            
            # Execute the command
            await self.execute_command(command)
            
        except Exception as e:
            self.logger.error(f"Error in command mode: {e}", LogCategory.INTERFACE)
            self.print_message(f"Error executing command: {e}", "error")
            return 1
        finally:
            await self._shutdown_components()
        
        return 0
    
    async def _shutdown_components(self):
        """Shutdown AVA components gracefully."""
        try:
            if self.assistant:
                await self.assistant.shutdown()
            
            if self.command_handler:
                await self.command_handler.shutdown()
            
            self.logger.info("CLI components shut down successfully", LogCategory.INTERFACE)
            
        except Exception as e:
            self.logger.error(f"Error shutting down components: {e}", LogCategory.INTERFACE)


class CLIArgumentParser:
    """Command-line argument parser for AVA CLI."""
    
    def __init__(self):
        """Initialize argument parser."""
        self.parser = argparse.ArgumentParser(
            prog='ava',
            description='AVA - Afsah\'s Virtual Assistant CLI',
            epilog='For more information, visit: https://github.com/NAME0x0/AVA'
        )
        
        self._setup_arguments()
    
    def _setup_arguments(self):
        """Setup command-line arguments."""
        # Main command argument
        self.parser.add_argument(
            'command',
            nargs='?',
            help='Command to execute (if not provided, starts interactive mode)'
        )
        
        # Configuration options
        self.parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file'
        )
        
        # Output format
        self.parser.add_argument(
            '--format', '-f',
            choices=['plain', 'json', 'markdown', 'table'],
            default='rich' if RICH_AVAILABLE else 'plain',
            help='Output format'
        )
        
        # Verbosity
        self.parser.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help='Increase verbosity (use multiple times for more verbose)'
        )
        
        # Quiet mode
        self.parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-essential output'
        )
        
        # Version
        self.parser.add_argument(
            '--version',
            action='version',
            version='AVA CLI v1.0.0'
        )
        
        # Debug mode
        self.parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode'
        )
        
        # No color output
        self.parser.add_argument(
            '--no-color',
            action='store_true',
            help='Disable colored output'
        )
        
        # Streaming mode
        self.parser.add_argument(
            '--stream',
            action='store_true',
            help='Enable streaming responses'
        )
    
    def parse_args(self, args=None):
        """Parse command-line arguments."""
        return self.parser.parse_args(args)


async def main():
    """Main entry point for AVA CLI."""
    try:
        # Parse arguments
        parser = CLIArgumentParser()
        args = parser.parse_args()
        
        # Setup configuration
        config = None
        if args.config:
            config_manager = get_config_manager(args.config)
            config = config_manager.get_config()
        
        # Create CLI interface
        cli = AVACLIInterface(config)
        
        # Override output format if specified
        if hasattr(args, 'format'):
            try:
                cli.session.output_format = OutputFormat(args.format)
            except ValueError:
                cli.print_message(f"Invalid output format: {args.format}", "error")
                return 1
        
        # Handle no-color option
        if args.no_color and cli.console:
            cli.console._color_system = None
        
        # Run appropriate mode
        if args.command:
            # Command mode - execute single command
            return await cli.command_mode(args.command)
        else:
            # Interactive mode
            await cli.interactive_mode()
            return 0
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


# Entry point for direct execution
if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


# Convenience function for embedding in other modules
async def run_cli(config: Optional[AVAConfig] = None, command: Optional[str] = None):
    """Run CLI programmatically."""
    cli = AVACLIInterface(config)
    
    if command:
        return await cli.command_mode(command)
    else:
        await cli.interactive_mode()
        return 0


# Testing function
async def test_cli():
    """Test CLI functionality."""
    print("Testing AVA CLI Interface...")
    
    # Test argument parsing
    parser = CLIArgumentParser()
    test_args = parser.parse_args(['--help'])
    
    # Test CLI creation
    cli = AVACLIInterface()
    
    # Test command execution
    result = await cli.execute_command("help")
    print(f"Help command test: {'✓' if result else '✗'}")
    
    # Test formatting
    test_data = {"test": "value", "number": 42}
    formatted = cli.format_output(test_data, OutputFormat.JSON)
    print(f"JSON formatting test: {'✓' if 'test' in formatted else '✗'}")
    
    await cli._shutdown_components()
    print("CLI test completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_cli())
    else:
        asyncio.run(main())
