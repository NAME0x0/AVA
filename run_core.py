#!/usr/bin/env python3
"""
AVA v3 Core Runner - Cortex-Medulla Architecture
=================================================

This script launches the AVA v3 system with the Cortex-Medulla architecture
designed for autonomous, always-on operation on constrained hardware.

Target Hardware: NVIDIA RTX A2000 (4GB VRAM)

Usage:
    python run_core.py                    # Interactive mode
    python run_core.py --config custom.yaml  # Custom config
    python run_core.py --simulation       # Simulation mode (no GPU)
    python run_core.py --stats            # Show system stats
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

console = Console()

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     █████╗ ██╗   ██╗ █████╗     ██╗   ██╗██████╗                  ║
║    ██╔══██╗██║   ██║██╔══██╗    ██║   ██║╚════██╗                 ║
║    ███████║██║   ██║███████║    ██║   ██║ █████╔╝                 ║
║    ██╔══██║╚██╗ ██╔╝██╔══██║    ╚██╗ ██╔╝ ╚═══██╗                 ║
║    ██║  ██║ ╚████╔╝ ██║  ██║     ╚████╔╝ ██████╔╝                 ║
║    ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝      ╚═══╝  ╚═════╝                  ║
║                                                                   ║
║             CORTEX-MEDULLA ARCHITECTURE                           ║
║      Autonomous Agent for Constrained Hardware                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the system."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    import yaml
    
    default_path = Path(__file__).parent / "config" / "cortex_medulla.yaml"
    path = Path(config_path) if config_path else default_path
    
    if not path.exists():
        console.print(f"[yellow]Config not found at {path}, using defaults[/yellow]")
        return {}
    
    with open(path) as f:
        return yaml.safe_load(f)


def display_stats(ava_system) -> None:
    """Display system statistics in a nice table."""
    stats = ava_system.get_stats()
    
    # System stats
    sys_table = Table(title="System Statistics", show_header=True)
    sys_table.add_column("Metric", style="cyan")
    sys_table.add_column("Value", style="green")
    
    sys_stats = stats.get("system", {})
    sys_table.add_row("State", sys_stats.get("state", "unknown"))
    sys_table.add_row("Uptime", f"{sys_stats.get('uptime_seconds', 0):.1f}s")
    sys_table.add_row("Total Interactions", str(sys_stats.get("total_interactions", 0)))
    
    console.print(sys_table)
    
    # Medulla stats
    med_table = Table(title="Medulla (Reflexive Core)", show_header=True)
    med_table.add_column("Metric", style="cyan")
    med_table.add_column("Value", style="green")
    
    med_stats = stats.get("medulla", {})
    med_table.add_row("State", med_stats.get("state", "unknown"))
    med_table.add_row("Interactions", str(med_stats.get("interaction_count", 0)))
    med_table.add_row("Cortex Invocations", str(med_stats.get("cortex_invocations", 0)))
    med_table.add_row("Avg Surprise", f"{med_stats.get('avg_surprise', 0):.3f}")
    
    console.print(med_table)
    
    # Agency stats
    agency_table = Table(title="Agency (Active Inference)", show_header=True)
    agency_table.add_column("Metric", style="cyan")
    agency_table.add_column("Value", style="green")
    
    agency_stats = stats.get("agency", {})
    agency_table.add_row("Belief Entropy", f"{agency_stats.get('belief_entropy', 0):.3f}")
    agency_table.add_row("Total Actions", str(agency_stats.get("total_actions", 0)))
    agency_table.add_row("Avg Free Energy", f"{agency_stats.get('avg_expected_free_energy', 0):.3f}")
    
    console.print(agency_table)


async def interactive_mode(ava_system, simulation: bool = False) -> None:
    """Run AVA in interactive console mode."""
    console.print(BANNER, style="bold blue")
    
    console.print(Panel(
        "[bold green]AVA v3 Cortex-Medulla System[/bold green]\n"
        f"[dim]Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n"
        f"[dim]Mode: {'Simulation' if simulation else 'Production'}[/dim]\n\n"
        "Commands:\n"
        "  [cyan]/stats[/cyan]  - Show system statistics\n"
        "  [cyan]/think[/cyan]  - Force Cortex reasoning\n"
        "  [cyan]/clear[/cyan]  - Clear conversation\n"
        "  [cyan]/quit[/cyan]   - Exit\n",
        title="Welcome",
        border_style="blue",
    ))
    
    # Set up output callback
    def output_callback(text: str):
        if text.startswith("[Thinking]"):
            console.print(f"[dim italic]{text}[/dim italic]")
        elif text.startswith("[Proactive]"):
            console.print(f"[yellow]{text}[/yellow]")
        else:
            console.print(f"[bold green]AVA:[/bold green] {text}")
    
    ava_system.set_output_callback(output_callback)
    
    # Main interaction loop
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().strip()
                
                if cmd == "quit" or cmd == "exit":
                    console.print("[yellow]Shutting down...[/yellow]")
                    break
                    
                elif cmd == "stats":
                    display_stats(ava_system)
                    continue
                    
                elif cmd == "think":
                    console.print("[dim]Forcing Cortex reasoning...[/dim]")
                    next_input = console.input("[bold blue]Query for Cortex:[/bold blue] ")
                    response = await ava_system.process_input(next_input, force_cortex=True)
                    output_callback(response)
                    continue
                    
                elif cmd == "clear":
                    ava_system.conversation_history.clear()
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue
                    
                else:
                    console.print(f"[red]Unknown command: {cmd}[/red]")
                    continue
            
            # Process normal input
            with console.status("[bold green]Processing...", spinner="dots"):
                response = await ava_system.process_input(user_input)
            
            output_callback(response)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Use /quit to exit.[/yellow]")
        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logging.exception("Error in interactive loop")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AVA v3 Core System - Cortex-Medulla Architecture"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Run in simulation mode (no real models)",
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Just show stats and exit",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Run in autonomous mode (continuous loop)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Import core system
    try:
        from src.core import AVACoreSystem, CoreConfig
    except ImportError as e:
        console.print(f"[red]Failed to import core system: {e}[/red]")
        console.print("[yellow]Make sure you're running from the project root.[/yellow]")
        sys.exit(1)
    
    # Create configuration
    core_config = CoreConfig()
    
    # Apply config file settings
    if "system" in config_dict:
        sys_cfg = config_dict["system"]
        core_config.data_dir = sys_cfg.get("data_dir", core_config.data_dir)
        core_config.log_level = sys_cfg.get("log_level", core_config.log_level)
    
    # Create system
    ava = AVACoreSystem(config=core_config)
    
    try:
        # Initialize
        console.print("[bold]Initializing AVA Core System...[/bold]")
        await ava.initialize()
        
        if args.stats:
            display_stats(ava)
            return
        
        if args.autonomous:
            console.print("[bold]Starting autonomous mode...[/bold]")
            console.print("[dim]Press Ctrl+C to stop[/dim]")
            await ava.run_forever()
        else:
            # Interactive mode
            await interactive_mode(ava, simulation=args.simulation)
            
    finally:
        await ava.shutdown()
        console.print("[green]AVA Core System shut down successfully.[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logging.exception("Fatal error")
        sys.exit(1)
