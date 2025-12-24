"""
AVA CLI - Command Line Interface (v3)
=====================================

A rich terminal interface for interacting with AVA v3 (Cortex-Medulla Architecture).

Features:
- Interactive chat mode with the v3 engine
- Status display showing cognitive state
- System monitoring and tool access
- Force modes: /think (Cortex), /search (Search-First)

Note: This CLI uses the unified AVA class from src.ava which implements
the Cortex-Medulla architecture. Legacy features like developmental stages
and emotions have been removed in favor of cognitive states and VFE.
"""

import asyncio

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False


# Initialize console
console = Console() if RICH_AVAILABLE else None

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     █████╗ ██╗   ██╗ █████╗      ██████╗██╗     ██╗               ║
║    ██╔══██╗██║   ██║██╔══██╗    ██╔════╝██║     ██║               ║
║    ███████║██║   ██║███████║    ██║     ██║     ██║               ║
║    ██╔══██║╚██╗ ██╔╝██╔══██║    ██║     ██║     ██║               ║
║    ██║  ██║ ╚████╔╝ ██║  ██║    ╚██████╗███████╗██║               ║
║    ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝     ╚═════╝╚══════╝╚═╝               ║
║                                                                   ║
║              COMMAND LINE INTERFACE v3                            ║
║         Cortex-Medulla • Search-First • Always-On                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def print_output(text: str, style: str = ""):
    """Print output with or without rich formatting."""
    if console:
        console.print(text, style=style)
    else:
        print(text)


def print_error(text: str):
    """Print error message."""
    if console:
        console.print(f"[red]Error:[/red] {text}")
    else:
        print(f"Error: {text}")


def print_panel(content: str, title: str = "", style: str = "blue"):
    """Print a panel with content."""
    if console:
        console.print(Panel(content, title=title, border_style=style))
    else:
        print(f"\n=== {title} ===")
        print(content)
        print("=" * (len(title) + 8))


class AVACli:
    """
    Command-line interface for AVA v3.

    Provides an interactive chat interface with commands for
    status, tools, and system monitoring.

    Note: Uses the v3 Cortex-Medulla architecture via src.ava.AVA
    """

    def __init__(self, data_dir: str = "data", model: str = "llama3.2"):
        """
        Initialize the CLI.

        Args:
            data_dir: Data directory for AVA
            model: Ollama model name (for v3, this is configured in cortex_medulla.yaml)
        """
        self.data_dir = data_dir
        self.model = model
        self.ava = None  # v3 AVA instance
        self.last_response = None  # Store last response for reference

        # Command handlers
        self.commands = {
            "/help": self.cmd_help,
            "/status": self.cmd_status,
            "/memory": self.cmd_memory,
            "/quit": self.cmd_quit,
            "/exit": self.cmd_quit,
            "/tools": self.cmd_tools,
            "/think": self.cmd_think,
            "/search": self.cmd_search,
        }

    def initialize_ava(self) -> bool:
        """Initialize the AVA v3 system."""
        try:
            from .ava import AVA

            self.ava = AVA()
            return True
        except Exception as e:
            print_error(f"Failed to initialize AVA: {e}")
            return False

    async def start_ava(self) -> bool:
        """Start the AVA engine (async initialization)."""
        if self.ava:
            try:
                success = await self.ava.start()
                if success:
                    return True
                else:
                    print_error("AVA engine failed to start")
                    return False
            except Exception as e:
                print_error(f"Failed to start AVA engine: {e}")
                return False
        return False

    def print_welcome(self):
        """Print welcome message with ASCII banner."""
        # Print the main banner
        if console:
            console.print(BANNER, style="cyan")
        else:
            print(BANNER)

        # Print command help
        help_text = """
  Commands:
    /help    - Show all commands
    /status  - Show AVA's system status
    /think   - Force deep thinking (Cortex)
    /search  - Search-first mode
    /quit    - Exit
"""
        if console:
            console.print(help_text, style="dim")
        else:
            print(help_text)

    def print_prompt(self) -> str:
        """Print the input prompt."""
        return "🤖 You: "

    def format_response(self, result) -> str:
        """Format AVA's response for display."""
        if not result:
            return ""

        # Extract text from response object
        text = getattr(result, "text", str(result))
        cognitive_state = getattr(result, "cognitive_state", "FLOW")
        used_cortex = getattr(result, "used_cortex", False)

        # Add indicator if Cortex was used
        indicator = "🧠" if used_cortex else "⚡"

        return f"{indicator} AVA [{cognitive_state}]: {text}"

    async def process_input(self, user_input: str) -> bool:
        """
        Process user input.

        Args:
            user_input: What the user typed

        Returns:
            False if should quit, True otherwise
        """
        user_input = user_input.strip()

        if not user_input:
            return True

        # Check for commands
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            args = user_input[len(cmd) :].strip()

            if cmd in self.commands:
                return await self.commands[cmd](args)
            else:
                print_error(f"Unknown command: {cmd}. Type /help for available commands.")
                return True

        # Regular interaction with AVA v3
        if not self.ava:
            print_error("AVA not initialized. Please restart.")
            return True

        # Show thinking indicator
        if console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]AVA is processing...[/cyan]"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("thinking", total=None)
                result = await self.ava.chat(user_input)
        else:
            print("AVA is processing...")
            result = await self.ava.chat(user_input)

        # Store response for reference
        self.last_response = result

        # Print response
        formatted = self.format_response(result)
        print_output("")
        if console:
            console.print(formatted, style="cyan")
        else:
            print(formatted)
        print_output("")

        # Show tools used if any
        tools_used = getattr(result, "tools_used", [])
        if tools_used:
            print_output(f"[Tools: {', '.join(tools_used)}]", style="dim")

        return True

    async def cmd_help(self, args: str) -> bool:
        """Show help message."""
        help_text = """
Available Commands:

  Chat Commands:
    /help           Show this help message
    /quit, /exit    Exit AVA

  Processing Modes:
    /think <query>  Force deep thinking (Cortex mode)
    /search <query> Force search-first mode

  Status Commands:
    /status         Show AVA's system status
    /memory         Show memory statistics
    /tools          Show available tools
"""
        print_panel(help_text.strip(), "Help", "green")
        return True

    async def cmd_status(self, args: str) -> bool:
        """Show AVA v3 system status."""
        if not self.ava or not self.ava._started:
            print_error("AVA not initialized")
            return True

        # Get status from engine if available
        if console:
            table = Table(title="AVA v3 Status", show_header=True)
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")

            table.add_row("Engine", "Running" if self.ava._started else "Stopped")
            table.add_row("Architecture", "Cortex-Medulla v3")
            table.add_row("Mode", "Search-First Enabled")

            # Show last response info if available
            if self.last_response:
                table.add_row(
                    "Last Cognitive State", getattr(self.last_response, "cognitive_state", "N/A")
                )
                table.add_row(
                    "Last Used Cortex",
                    "Yes" if getattr(self.last_response, "used_cortex", False) else "No",
                )

            console.print(table)
        else:
            print("\n=== AVA v3 Status ===")
            print(f"Engine: {'Running' if self.ava._started else 'Stopped'}")
            print("Architecture: Cortex-Medulla v3")

        return True

    async def cmd_memory(self, args: str) -> bool:
        """Show memory statistics."""
        if not self.ava or not self.ava._memory:
            print_error("AVA not initialized or memory not available")
            return True

        # Get memory stats from v3 memory system
        memory = self.ava._memory

        print_panel(
            f"Conversation turns: {len(memory.messages) if hasattr(memory, 'messages') else 'N/A'}\n"
            f"Architecture: Titans Neural Memory (when loaded)",
            "Memory Statistics",
            "yellow",
        )

        return True

    async def cmd_tools(self, args: str) -> bool:
        """Show available tools."""
        if not self.ava or not self.ava._tools:
            print_error("AVA not initialized or tools not available")
            return True

        tools = self.ava._tools
        available = list(tools.tools.keys()) if hasattr(tools, "tools") else []

        if console:
            table = Table(title="Available Tools")
            table.add_column("Tool", style="cyan")
            table.add_column("Status", style="green")

            for tool in available:
                table.add_row(tool, "[green]✓ Available[/green]")

            if not available:
                table.add_row("(No tools loaded)", "[yellow]Configure tools in tools.yaml[/yellow]")

            console.print(table)
        else:
            print("\n=== Tools ===")
            for tool in available:
                print(f"  {tool}: Available")

        return True

    async def cmd_think(self, args: str) -> bool:
        """Force deep thinking (Cortex mode)."""
        if not args:
            print_error("Usage: /think <query>")
            return True

        if not self.ava:
            print_error("AVA not initialized")
            return True

        if console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Cortex thinking deeply...[/cyan]"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("thinking", total=None)
                result = await self.ava.think(args)
        else:
            print("Cortex thinking...")
            result = await self.ava.think(args)

        self.last_response = result
        formatted = self.format_response(result)
        print_output("")
        if console:
            console.print(formatted, style="cyan")
        else:
            print(formatted)
        print_output("")

        return True

    async def cmd_search(self, args: str) -> bool:
        """Force search-first mode."""
        if not args:
            print_error("Usage: /search <query>")
            return True

        if not self.ava:
            print_error("AVA not initialized")
            return True

        # Search-first is the default in v3, but we make it explicit
        if console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Searching for information...[/cyan]"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("searching", total=None)
                result = await self.ava.chat(args, tools=["web_search"])
        else:
            print("Searching...")
            result = await self.ava.chat(args, tools=["web_search"])

        self.last_response = result
        formatted = self.format_response(result)
        print_output("")
        if console:
            console.print(formatted, style="cyan")
        else:
            print(formatted)
        print_output("")

        return True

    async def cmd_quit(self, args: str) -> bool:
        """Exit the CLI."""
        if self.ava:
            await self.ava.stop()
            print_output("AVA stopped. Goodbye!", style="green")
        return False

    async def run(self):
        """Run the interactive CLI loop."""
        # Initialize AVA
        if not self.initialize_ava():
            return

        # Start AVA engine
        print_output("Starting AVA v3 engine...", style="yellow")
        if not await self.start_ava():
            return

        # Print welcome
        self.print_welcome()

        # Main loop
        running = True
        while running:
            try:
                # Get input
                if console:
                    user_input = Prompt.ask(self.print_prompt())
                else:
                    user_input = input(self.print_prompt())

                # Process
                running = await self.process_input(user_input)

            except KeyboardInterrupt:
                print_output("\n\nInterrupted. Shutting down...", style="yellow")
                if self.ava:
                    await self.ava.stop()
                break
            except EOFError:
                break
            except Exception as e:
                print_error(f"Error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AVA v3 - Cortex-Medulla AI Assistant")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--model", default="llama3.2", help="Model name (configured in cortex_medulla.yaml)"
    )

    args = parser.parse_args()

    cli = AVACli(data_dir=args.data_dir, model=args.model)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
