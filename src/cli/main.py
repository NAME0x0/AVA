#!/usr/bin/env python3
"""
AVA CLI Interface
Enhanced command-line interface for AVA local agentic AI

Provides commands for:
- Interactive querying and chat
- Status monitoring  
- Tool execution
- Configuration management
- Model management
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from typing_extensions import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import AVA core components
try:
    from ava_core.agent import Agent, AgentConfig
    from ava_core.dialogue_manager import DialogueManager
    from ava_core.function_calling import FunctionCaller
    AVA_CORE_AVAILABLE = True
except ImportError:
    AVA_CORE_AVAILABLE = False

# Initialize console and app
console = Console()
app = typer.Typer(
    name="ava",
    help="AVA - Afsah's Virtual Assistant: Local agentic AI for RTX A2000 4GB",
    add_completion=False,
    rich_markup_mode="rich",
)

# Global agent instance
_agent: Optional[Agent] = None


def get_agent() -> Agent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        if not AVA_CORE_AVAILABLE:
            console.print("[red]Error: AVA core components not available. Please check installation.[/red]")
            raise typer.Exit(1)
        
        config = AgentConfig()
        _agent = Agent(config=config)
    
    return _agent


def display_banner():
    """Display AVA banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                        AVA Assistant                          ║
    ║              Local Agentic AI for RTX A2000 4GB               ║
    ║                                                               ║
    ║  🚀 Optimized • 🛠️  Agentic • 🔒 Local • ⚡ Fast            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


@app.command()
def query(
    prompt: Annotated[str, typer.Argument(help="The natural language query for AVA")],
    model: Annotated[str, typer.Option("--model", "-m", help="Specify the Ollama model to use")] = "ava-agent:latest",
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="Stream the response token by token")] = True,
    reasoning: Annotated[bool, typer.Option("--reasoning/--no-reasoning", help="Enable reasoning mechanisms")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed processing information")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output response in JSON format")] = False,
):
    """Send a query to AVA and display the response."""
    
    if verbose:
        display_banner()
        console.print(f"[dim]Model: {model} | Stream: {stream} | Reasoning: {reasoning}[/dim]\n")
    
    try:
        agent = get_agent()
        
        # Update agent configuration
        agent.update_config(
            model_name=model,
            stream_response=stream,
            enable_reasoning=reasoning
        )
        
        if verbose:
            console.print(f"[blue]Processing query:[/blue] {prompt}\n")
        
        # Process the query with loading indication
        if stream and not json_output:
            with Live(console=console, refresh_per_second=10) as live:
                live.update(Panel("[yellow]🤔 AVA is thinking...[/yellow]", title="Processing"))
                response = agent.process_input(prompt)
                live.update("")
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                response = agent.process_input(prompt)
                progress.remove_task(task)
        
        # Display response
        if json_output:
            output = {
                "query": prompt,
                "response": response,
                "model": model,
                "stream": stream,
                "reasoning": reasoning
            }
            console.print(json.dumps(output, indent=2))
        else:
            console.print(Panel(
                Markdown(response),
                title="[bold green]AVA Response[/bold green]",
                border_style="green"
            ))
            
            if verbose:
                console.print(f"\n[dim]Response generated using model: {model}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error processing query: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def chat(
    model: Annotated[str, typer.Option("--model", "-m", help="Specify the Ollama model to use")] = "ava-agent:latest",
    reasoning: Annotated[bool, typer.Option("--reasoning/--no-reasoning", help="Enable reasoning mechanisms")] = True,
    history_length: Annotated[int, typer.Option("--history", "-h", help="Maximum conversation history length")] = 20,
):
    """Start an interactive chat session with AVA."""
    
    display_banner()
    
    try:
        agent = get_agent()
        
        # Update agent configuration
        agent.update_config(
            model_name=model,
            enable_reasoning=reasoning,
            max_history_length=history_length
        )
        
        console.print(f"[green]Starting interactive chat with AVA[/green]")
        console.print(f"[dim]Model: {model} | Reasoning: {reasoning} | History: {history_length}[/dim]")
        console.print(f"[dim]Type 'exit', 'quit', or 'bye' to end the session[/dim]\n")
        
        session_count = 0
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask(
                    "[bold blue]You[/bold blue]",
                    console=console
                ).strip()
                
                if user_input.lower() in ["exit", "quit", "bye", "q"]:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                if not user_input:
                    continue
                
                # Special commands
                if user_input.startswith("/"):
                    _handle_chat_command(user_input, agent)
                    continue
                
                session_count += 1
                
                # Process input with loading indication
                with Live(console=console, refresh_per_second=10) as live:
                    live.update(Panel("[yellow]🤔 AVA is thinking...[/yellow]", title="Processing"))
                    response = agent.process_input(user_input)
                    live.update("")
                
                # Display response
                console.print(Panel(
                    Markdown(response),
                    title=f"[bold green]AVA[/bold green] [dim]#{session_count}[/dim]",
                    border_style="green"
                ))
                console.print("")  # Add spacing
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Chat interrupted. Goodbye! 👋[/yellow]")
                break
            except EOFError:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
                
    except Exception as e:
        console.print(f"[red]Error starting chat session: {e}[/red]")
        raise typer.Exit(1)


def _handle_chat_command(command: str, agent: Agent):
    """Handle special chat commands."""
    cmd = command[1:].lower().strip()
    
    if cmd == "help":
        help_text = """
Available chat commands:
• /help - Show this help message
• /status - Show agent status
• /clear - Clear conversation history
• /config - Show current configuration
• /tools - List available tools
        """
        console.print(Panel(help_text.strip(), title="Chat Commands", border_style="blue"))
    
    elif cmd == "status":
        status = agent.get_status()
        _display_status_table(status)
    
    elif cmd == "clear":
        agent.clear_history()
        console.print("[green]Conversation history cleared.[/green]")
    
    elif cmd == "config":
        config = agent.config.__dict__
        table = Table(title="Agent Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in config.items():
            table.add_row(key, str(value))
        
        console.print(table)
    
    elif cmd == "tools":
        tools = agent.function_caller.available_tools.keys()
        console.print(f"[green]Available tools:[/green] {', '.join(tools)}")
    
    else:
        console.print(f"[red]Unknown command: {command}[/red]. Type /help for available commands.")


@app.command()
def status():
    """Check the status of AVA and its connection to services like Ollama."""
    
    console.print("[bold]Checking AVA status...[/bold]\n")
    
    # Check core availability
    status_data = {
        "AVA Core Available": AVA_CORE_AVAILABLE,
        "Ollama Available": OLLAMA_AVAILABLE,
    }
    
    if AVA_CORE_AVAILABLE:
        try:
            agent = get_agent()
            agent_status = agent.get_status()
            status_data.update(agent_status)
        except Exception as e:
            status_data["Agent Error"] = str(e)
    
    _display_status_table(status_data)


def _display_status_table(status_data: Dict[str, Any]):
    """Display status information in a formatted table."""
    table = Table(title="AVA System Status")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")
    
    for key, value in status_data.items():
        if isinstance(value, bool):
            status_text = "[green]✓ OK[/green]" if value else "[red]✗ Error[/red]"
            details = ""
        elif isinstance(value, (list, dict)):
            status_text = "[green]✓ OK[/green]"
            details = str(len(value)) + " items" if isinstance(value, list) else json.dumps(value, indent=2)[:100]
        else:
            status_text = "[green]✓ OK[/green]" if value else "[red]✗ Error[/red]"
            details = str(value)[:100]
        
        table.add_row(key, status_text, details)
    
    console.print(table)


@app.command()
def tool_exec(
    tool_name: Annotated[str, typer.Argument(help="The name of the tool to execute")],
    args: Annotated[List[str], typer.Argument(help="Arguments for the tool in key=value format")] = [],
    json_output: Annotated[bool, typer.Option("--json", help="Output result in JSON format")] = False,
):
    """Execute a specified tool with given arguments (for testing/development)."""
    
    # Parse arguments
    parsed_args = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to parse as JSON, fallback to string
            try:
                parsed_args[key] = json.loads(value)
            except json.JSONDecodeError:
                parsed_args[key] = value
        else:
            console.print(f"[yellow]Warning: Argument '{arg}' is not in key=value format and will be ignored.[/yellow]")
    
    try:
        agent = get_agent()
        
        console.print(f"[blue]Executing tool:[/blue] {tool_name}")
        console.print(f"[dim]Arguments: {parsed_args}[/dim]\n")
        
        # Execute the tool
        result = agent.function_caller.execute_function(tool_name, parsed_args)
        
        # Display result
        if json_output:
            output = {
                "tool": tool_name,
                "arguments": parsed_args,
                "result": result,
                "success": not str(result).startswith("Error")
            }
            console.print(json.dumps(output, indent=2))
        else:
            console.print(Panel(
                str(result),
                title=f"[bold green]Tool Result: {tool_name}[/bold green]",
                border_style="green"
            ))
        
    except Exception as e:
        console.print(f"[red]Error executing tool: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def models():
    """List available models in Ollama."""
    
    if not OLLAMA_AVAILABLE:
        console.print("[red]Ollama is not available. Please install it: pip install ollama[/red]")
        raise typer.Exit(1)
    
    try:
        client = ollama.Client()
        models_response = client.list()
        
        if not models_response.get("models"):
            console.print("[yellow]No models found in Ollama.[/yellow]")
            return
        
        table = Table(title="Available Ollama Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Size", style="white")
        table.add_column("Modified", style="dim")
        
        for model in models_response["models"]:
            name = model.get("name", "Unknown")
            size = model.get("size", 0)
            modified = model.get("modified_at", "Unknown")
            
            # Format size
            if size > 0:
                size_gb = size / (1024**3)
                size_str = f"{size_gb:.1f} GB"
            else:
                size_str = "Unknown"
            
            table.add_row(name, size_str, modified[:19] if isinstance(modified, str) else str(modified))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
    model: Annotated[Optional[str], typer.Option("--model", help="Set default model")] = None,
    temperature: Annotated[Optional[float], typer.Option("--temperature", help="Set temperature (0.0-2.0)")] = None,
    max_tokens: Annotated[Optional[int], typer.Option("--max-tokens", help="Set max tokens")] = None,
):
    """Manage AVA configuration."""
    
    if show or (model is None and temperature is None and max_tokens is None):
        try:
            agent = get_agent()
            config_data = agent.config.__dict__
            
            table = Table(title="AVA Configuration")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Type", style="dim")
            
            for key, value in config_data.items():
                table.add_row(key, str(value), type(value).__name__)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error reading configuration: {e}[/red]")
            raise typer.Exit(1)
    
    else:
        # Update configuration
        try:
            agent = get_agent()
            updates = {}
            
            if model is not None:
                updates["model_name"] = model
            if temperature is not None:
                if not 0.0 <= temperature <= 2.0:
                    console.print("[red]Temperature must be between 0.0 and 2.0[/red]")
                    raise typer.Exit(1)
                updates["temperature"] = temperature
            if max_tokens is not None:
                if max_tokens < 1:
                    console.print("[red]Max tokens must be greater than 0[/red]")
                    raise typer.Exit(1)
                updates["max_tokens"] = max_tokens
            
            agent.update_config(**updates)
            
            console.print("[green]Configuration updated successfully:[/green]")
            for key, value in updates.items():
                console.print(f"  {key}: {value}")
            
        except Exception as e:
            console.print(f"[red]Error updating configuration: {e}[/red]")
            raise typer.Exit(1)


@app.callback()
def main():
    """AVA - Afsah's Virtual Assistant: Local agentic AI optimized for RTX A2000 4GB VRAM."""
    pass


if __name__ == "__main__":
    app() 