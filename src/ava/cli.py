#!/usr/bin/env python3
"""
AVA Command Line Interface
==========================

Unified CLI for all AVA operations.

Usage:
    ava serve      # Start HTTP API server
    ava tui        # Start Terminal UI
    ava chat       # Quick interactive chat
    ava status     # Check system status
    ava doctor     # Diagnose issues
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = typer.Typer(
    name="ava",
    help="AVA - Adaptive Virtual Agent with dual-brain architecture",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8085, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable hot reload"),
    simulation: bool = typer.Option(False, "--simulation", "-s", help="Enable simulation mode"),
):
    """Start the AVA HTTP API server."""
    console.print(Panel.fit(
        "[bold cyan]AVA[/bold cyan] API Server",
        subtitle=f"http://{host}:{port}"
    ))

    # Import and run server
    try:
        from aiohttp import web

        # Set environment variables
        import os
        if simulation:
            os.environ["AVA_SIMULATION_MODE"] = "true"

        # Import server module
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        import server

        # Run the server
        asyncio.run(server.main(host=host, port=port))

    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing dependency: {e}")
        console.print("Run: [cyan]pip install aiohttp[/cyan]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


@app.command()
def tui():
    """Start the Terminal User Interface."""
    console.print(Panel.fit(
        "[bold cyan]AVA[/bold cyan] Terminal UI",
        subtitle="Press Ctrl+Q to quit"
    ))

    try:
        # Import and run TUI
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from tui.app import AVATUI

        app_instance = AVATUI()
        app_instance.run()

    except ImportError as e:
        console.print(f"[red]Error:[/red] TUI not available: {e}")
        console.print("Run: [cyan]pip install textual[/cyan]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]TUI closed.[/yellow]")


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send (or omit for interactive mode)"),
    deep: bool = typer.Option(False, "--deep", "-d", help="Force deep thinking (Cortex)"),
):
    """Quick chat with AVA."""
    async def _chat():
        try:
            from ava import AVA

            ava = AVA()
            await ava.start()

            if message:
                # Single message mode
                console.print(f"[dim]You:[/dim] {message}")
                if deep:
                    response = await ava.think(message)
                else:
                    response = await ava.chat(message)
                console.print(f"[cyan]AVA:[/cyan] {response.text}")
            else:
                # Interactive mode
                console.print("[dim]Interactive mode. Type 'exit' to quit.[/dim]\n")
                while True:
                    try:
                        user_input = console.input("[bold]You:[/bold] ")
                        if user_input.lower() in ("exit", "quit", "q"):
                            break
                        if not user_input.strip():
                            continue

                        if deep:
                            response = await ava.think(user_input)
                        else:
                            response = await ava.chat(user_input)
                        console.print(f"[cyan]AVA:[/cyan] {response.text}\n")

                    except KeyboardInterrupt:
                        break

            await ava.stop()

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(_chat())


@app.command()
def status():
    """Check AVA system status."""
    import httpx
    import os

    host = os.getenv("AVA_HOST", "127.0.0.1")
    port = os.getenv("AVA_PORT", "8085")
    url = f"http://{host}:{port}"

    table = Table(title="AVA System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Check API server
    try:
        response = httpx.get(f"{url}/health", timeout=5.0)
        if response.status_code == 200:
            table.add_row("API Server", "[green]Online[/green]", url)
        else:
            table.add_row("API Server", "[yellow]Degraded[/yellow]", f"Status: {response.status_code}")
    except httpx.ConnectError:
        table.add_row("API Server", "[red]Offline[/red]", f"Not running at {url}")
    except Exception as e:
        table.add_row("API Server", "[red]Error[/red]", str(e))

    # Check Ollama
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = httpx.get(f"{ollama_host}/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models[:3]]
            table.add_row("Ollama", "[green]Online[/green]", f"Models: {', '.join(model_names) or 'none'}")
        else:
            table.add_row("Ollama", "[yellow]Degraded[/yellow]", f"Status: {response.status_code}")
    except httpx.ConnectError:
        table.add_row("Ollama", "[red]Offline[/red]", f"Start with: ollama serve")
    except Exception as e:
        table.add_row("Ollama", "[red]Error[/red]", str(e))

    # Check GPU
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                name, mem_used, mem_total, temp = parts[:4]
                table.add_row(
                    "GPU",
                    "[green]Available[/green]",
                    f"{name} ({mem_used}/{mem_total}MB, {temp}C)"
                )
            else:
                table.add_row("GPU", "[green]Available[/green]", result.stdout.strip())
        else:
            table.add_row("GPU", "[yellow]Unknown[/yellow]", "nvidia-smi failed")
    except FileNotFoundError:
        table.add_row("GPU", "[yellow]N/A[/yellow]", "nvidia-smi not found")
    except Exception as e:
        table.add_row("GPU", "[red]Error[/red]", str(e))

    console.print(table)


@app.command()
def doctor():
    """Diagnose common issues with AVA setup."""
    console.print(Panel.fit("[bold cyan]AVA Doctor[/bold cyan] - Checking your setup..."))

    checks = []

    # Check Python version
    import platform
    py_version = platform.python_version()
    if tuple(map(int, py_version.split(".")[:2])) >= (3, 10):
        checks.append(("Python Version", True, f"Python {py_version}"))
    else:
        checks.append(("Python Version", False, f"Python {py_version} (need 3.10+)"))

    # Check required packages
    required_packages = [
        ("aiohttp", "API server"),
        ("httpx", "HTTP client"),
        ("typer", "CLI"),
        ("rich", "Terminal formatting"),
        ("textual", "TUI"),
        ("pydantic", "Data validation"),
    ]

    for package, purpose in required_packages:
        try:
            __import__(package)
            checks.append((f"Package: {package}", True, purpose))
        except ImportError:
            checks.append((f"Package: {package}", False, f"pip install {package}"))

    # Check config files
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "cortex_medulla.yaml"
    if config_path.exists():
        checks.append(("Config File", True, str(config_path.name)))
    else:
        checks.append(("Config File", False, "config/cortex_medulla.yaml not found"))

    # Check .env file
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        checks.append(("Environment", True, ".env file found"))
    else:
        checks.append(("Environment", None, "No .env file (using defaults)"))

    # Print results
    table = Table(title="Diagnostic Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    all_passed = True
    for check, passed, details in checks:
        if passed is True:
            status = "[green]PASS[/green]"
        elif passed is False:
            status = "[red]FAIL[/red]"
            all_passed = False
        else:
            status = "[yellow]WARN[/yellow]"
        table.add_row(check, status, details)

    console.print(table)

    if all_passed:
        console.print("\n[green]All checks passed![/green] AVA should work correctly.")
    else:
        console.print("\n[yellow]Some issues found.[/yellow] Fix the failed checks above.")
        console.print("\nQuick fix: [cyan]pip install -r requirements.txt[/cyan]")


@app.command()
def version():
    """Show AVA version information."""
    try:
        from ava import __version__
        version_str = __version__
    except ImportError:
        version_str = "unknown"

    console.print(Panel.fit(
        f"[bold cyan]AVA[/bold cyan] v{version_str}\n"
        "[dim]Adaptive Virtual Agent with dual-brain architecture[/dim]",
        title="Version Info"
    ))


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
