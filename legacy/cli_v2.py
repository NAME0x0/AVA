"""
AVA CLI - Command Line Interface

A rich terminal interface for interacting with AVA v3 (Cortex-Medulla Architecture).

Features:
- Interactive chat mode with the v3 engine
- Status display showing cognitive state
- System monitoring and tool access
- Administrative commands

Note: This CLI uses the unified AVA class from src.ava which implements
the Cortex-Medulla architecture. Legacy features like developmental stages
and emotions have been removed in favor of cognitive states and VFE.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.style import Style
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import typer
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False


# Initialize console
console = Console() if RICH_AVAILABLE else None


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
    
    def initialize_agent(self):
        """Initialize the AVA v3 system."""
        try:
            from .ava import AVA
            
            self.ava = AVA()
            return True
        except Exception as e:
            print_error(f"Failed to initialize AVA: {e}")
            return False
    
    async def start_ava(self):
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
        """Print welcome message."""
        welcome = """
╔══════════════════════════════════════════════════════════════╗
║                    Welcome to AVA v3                          ║
║          Cortex-Medulla Architecture                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Commands:                                                    ║
║    /help    - Show all commands                               ║
║    /status  - Show AVA's system status                        ║
║    /think   - Force deep thinking (Cortex)                    ║
║    /search  - Search-first mode                               ║
║    /quit    - Exit                                            ║
╚══════════════════════════════════════════════════════════════╝
"""
        if console:
            console.print(Panel(welcome.strip(), title="AVA CLI", border_style="cyan"))
        else:
            print(welcome)
    
    def print_prompt(self):
        """Print the input prompt."""
        return "🤖 You: "
    
    def format_response(self, result) -> str:
        """Format AVA's response for display."""
        if not result:
            return ""
        
        # Extract text from response object
        text = getattr(result, 'text', str(result))
        cognitive_state = getattr(result, 'cognitive_state', 'FLOW')
        used_cortex = getattr(result, 'used_cortex', False)
        
        # Add indicator if Cortex was used
        indicator = "🧠" if used_cortex else "⚡"
        
        return f"{indicator} AVA [{cognitive_state}]: {text}"
    
    def print_emotion_bar(self, emotions: dict):
        """Print a visual representation of emotions."""
        if not console:
            return
        
        emotion_colors = {
            "hope": "green",
            "fear": "red",
            "joy": "yellow",
            "surprise": "magenta",
            "ambition": "blue",
        }
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        
        for emotion, value in emotions.items():
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            color = emotion_colors.get(emotion, "white")
            table.add_row(
                f"{emotion:10}",
                f"[{color}]{bar}[/{color}]",
                f"{value:.2f}"
            )
        
        console.print(table)
    
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
            args = user_input[len(cmd):].strip()
            
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
        tools_used = getattr(result, 'tools_used', [])
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
        """Show AVA's status."""
        if not self.agent:
            print_error("Agent not initialized")
            return True
        
        status = self.agent.get_status()
        
        if console:
            table = Table(title="AVA Status", show_header=True)
            table.add_column("Category", style="cyan")
            table.add_column("Detail", style="white")
            table.add_column("Value", style="green")
            
            # Developmental
            table.add_row("Developmental", "Stage", status["developmental"]["stage"])
            table.add_row("", "Age (days)", f"{status['developmental']['age_days']:.1f}")
            table.add_row("", "Interactions", str(status["developmental"]["interaction_count"]))
            table.add_row("", "Milestones", str(len(status["developmental"]["achieved_milestones"])))
            
            # Emotional
            table.add_row("Emotional", "Dominant", status["emotional"]["dominant_emotion"] or "None")
            for emotion, value in status["emotional"]["current_state"].items():
                table.add_row("", emotion.capitalize(), f"{value:.2f}")
            
            # Learning
            table.add_row("Learning", "Samples", str(status["learning"]["samples_collected"]))
            table.add_row("", "Ready for training", str(status["learning"]["available_for_training"]))
            table.add_row("", "Training runs", str(status["learning"]["training_runs"]))
            
            # Session
            table.add_row("Session", "Started", status["session"]["started"][:19])
            table.add_row("", "Interactions", str(status["session"]["interactions"]))
            
            console.print(table)
        else:
            print("\n=== AVA Status ===")
            print(f"Stage: {status['developmental']['stage']}")
            print(f"Age: {status['developmental']['age_days']:.1f} days")
            print(f"Interactions: {status['developmental']['interaction_count']}")
            print(f"Samples: {status['learning']['samples_collected']}")
        
        return True
    
    async def cmd_emotions(self, args: str) -> bool:
        """Show emotional state."""
        if not self.agent:
            print_error("Agent not initialized")
            return True
        
        emotions = self.agent.emotions.get_emotion_dict()
        
        print_output("\nCurrent Emotional State:", style="bold")
        self.print_emotion_bar(emotions)
        print_output("")
        
        return True
    
    async def cmd_memory(self, args: str) -> bool:
        """Show memory statistics."""
        if not self.agent:
            print_error("Agent not initialized")
            return True
        
        stats = self.agent.memory.get_stats()
        
        print_panel(
            f"Episodic memories: {stats.get('episodic_count', 0)}\n"
            f"Semantic memories: {stats.get('semantic_count', 0)}\n"
            f"Total memories: {stats.get('total', 0)}",
            "Memory Statistics",
            "yellow"
        )
        
        return True
    
    async def cmd_stage(self, args: str) -> bool:
        """Show developmental stage info."""
        if not self.agent:
            print_error("Agent not initialized")
            return True
        
        from .developmental import STAGE_PROPERTIES
        
        stage = self.agent.development.current_stage
        props = STAGE_PROPERTIES[stage]
        
        info = f"""
Stage: {stage.name}
Clarity: {props.clarity * 100:.0f}%
Vocabulary Range: {props.vocabulary_range * 100:.0f}%
Tool Level: {props.tool_level}
Learning Rate Modifier: {props.learning_rate_modifier:.1f}x
Max Response Tokens: {props.max_response_tokens}
Thinking Budget: {props.thinking_budget}
"""
        print_panel(info.strip(), f"Developmental Stage: {stage.name}", "blue")
        
        return True
    
    async def cmd_tools(self, args: str) -> bool:
        """Show available tools."""
        if not self.agent:
            print_error("Agent not initialized")
            return True
        
        available = self.agent.tool_progression.get_available_tools(
            stage=self.agent.development.current_stage,
            milestones=self.agent.development.get_achieved_milestones(),
        )
        
        all_tools = list(self.agent.tools.tools.keys())
        
        if console:
            table = Table(title="Tool Availability")
            table.add_column("Tool", style="cyan")
            table.add_column("Status", style="white")
            
            for tool in all_tools:
                status = "[green]✓ Available[/green]" if tool in available else "[red]✗ Locked[/red]"
                table.add_row(tool, status)
            
            console.print(table)
        else:
            print("\n=== Tools ===")
            for tool in all_tools:
                status = "Available" if tool in available else "Locked"
                print(f"  {tool}: {status}")
        
        return True
    
    async def cmd_good_feedback(self, args: str) -> bool:
        """Provide positive feedback."""
        if not self.last_sample_id:
            print_error("No previous response to rate")
            return True
        
        self.agent.provide_feedback(
            sample_id=self.last_sample_id,
            positive=True,
            feedback_text=args if args else None,
        )
        
        print_output("✓ Positive feedback recorded. AVA is happy!", style="green")
        return True
    
    async def cmd_bad_feedback(self, args: str) -> bool:
        """Provide negative feedback."""
        if not self.last_sample_id:
            print_error("No previous response to rate")
            return True
        
        self.agent.provide_feedback(
            sample_id=self.last_sample_id,
            positive=False,
            feedback_text=args if args else None,
        )
        
        print_output("✓ Negative feedback recorded. AVA will learn from this.", style="yellow")
        return True
    
    async def cmd_correct(self, args: str) -> bool:
        """Provide a correction."""
        if not self.last_sample_id:
            print_error("No previous response to correct")
            return True
        
        if not args:
            print_error("Please provide the correction text: /correct <your correction>")
            return True
        
        self.agent.provide_feedback(
            sample_id=self.last_sample_id,
            positive=False,
            correction=args,
        )
        
        print_output("✓ Correction recorded. AVA will learn the right response.", style="yellow")
        return True
    
    async def cmd_save(self, args: str) -> bool:
        """Save state."""
        if self.agent:
            self.agent.save_state()
            print_output("✓ State saved", style="green")
        return True
    
    async def cmd_reset(self, args: str) -> bool:
        """Reset AVA to infant stage."""
        if args.lower() != "confirm":
            print_panel(
                "This will reset AVA to infant stage, losing all progress and memories.\n"
                "Type '/reset confirm' to proceed.",
                "Warning",
                "red"
            )
            return True
        
        if self.agent:
            self.agent.reset_to_infant(confirm=True)
            print_output("AVA has been reset to infant stage.", style="yellow")
        
        return True
    
    async def cmd_quit(self, args: str) -> bool:
        """Exit the CLI."""
        if self.agent:
            self.agent.save_state()
            print_output("State saved. Goodbye!", style="green")
        return False
    
    async def run(self):
        """Run the interactive CLI loop."""
        # Initialize agent
        if not self.initialize_agent():
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
                print_output("\n\nInterrupted. Saving state...", style="yellow")
                if self.agent:
                    self.agent.save_state()
                break
            except EOFError:
                break
            except Exception as e:
                print_error(f"Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AVA - Developmental AI Assistant")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    
    args = parser.parse_args()
    
    cli = AVACli(data_dir=args.data_dir, model=args.model)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
