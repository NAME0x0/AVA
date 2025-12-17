#!/usr/bin/env python3
"""
AVA Neural Node - Entry Point
=============================

This is the main entry point for running AVA as a cognitive system.
It initializes the Bicameral Architecture:

1. CONSCIOUS SYSTEM (Online):
   - Executive: High-level orchestration
   - ConsciousStream: Real-time processing with Titans Neural Memory
   - Tool Registry: Available capabilities

2. SUBCONSCIOUS SYSTEM (Offline):
   - Dreamer: Background consolidation
   - FastSlowWeightManager: Multi-timescale learning
   - QLoRA Trainer: Fine-tuning capabilities

Usage:
    python run_node.py                 # Interactive mode
    python run_node.py --daemon        # Background service mode
    python run_node.py --api           # API server mode

Commands (in interactive mode):
    /dream          - Force a consolidation cycle
    /stats          - Show system statistics
    /mode <mode>    - Set operating mode (normal/learning/performance)
    /save           - Save current state
    /quit           - Exit gracefully

Requirements:
    - Ollama running locally (ollama serve)
    - Model pulled (ollama pull llama3.2)
    - Embedding model (ollama pull nomic-embed-text)
"""

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/ava.log"),
    ],
)
logger = logging.getLogger("AVA")


# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     █████╗ ██╗   ██╗ █████╗     ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗     ║
║    ██╔══██╗██║   ██║██╔══██╗    ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗    ║
║    ███████║██║   ██║███████║    ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║    ║
║    ██╔══██║╚██╗ ██╔╝██╔══██║    ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║    ║
║    ██║  ██║ ╚████╔╝ ██║  ██║    ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝    ║
║    ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝    ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝     ║
║                                                                   ║
║    Bicameral Cognitive Architecture                               ║
║    Neural Memory Online | Nested Learning Active                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""


class AVANode:
    """
    Main AVA Node - orchestrates the entire cognitive system.
    """
    
    def __init__(
        self,
        model: str = "llama3.2:latest",
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        enable_dreaming: bool = True,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
        self.enable_dreaming = enable_dreaming
        
        # Components (initialized in start())
        self.executive = None
        self.weight_manager = None
        
        # State
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Initialize and start all systems."""
        logger.info("Starting AVA Neural Node...")
        
        # Create data directories
        Path("data/learning/checkpoints").mkdir(parents=True, exist_ok=True)
        Path("data/memory").mkdir(parents=True, exist_ok=True)
        Path("models/fine_tuned_adapters").mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize Weight Manager (Nested Learning)
            from src.learning.nested import create_nested_learning_system
            
            try:
                context_manager, self.weight_manager = create_nested_learning_system(
                    data_dir="data/learning"
                )
                logger.info("Nested Learning system initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Nested Learning: {e}")
                self.weight_manager = None
            
            # Initialize Executive (Cortex)
            from src.cortex.executive import Executive, ExecutiveConfig
            
            config = ExecutiveConfig(
                model_name=self.model,
                embedding_model=self.embedding_model,
                ollama_host=self.ollama_host,
                enable_background_dreaming=self.enable_dreaming,
            )
            
            self.executive = Executive(
                config=config,
                weight_manager=self.weight_manager,
            )
            
            # Initialize Executive (loads all subsystems)
            await self.executive.initialize()
            
            self.is_running = True
            logger.info("AVA Neural Node is online")
            
        except Exception as e:
            logger.error(f"Failed to start AVA Node: {e}")
            raise
    
    async def process(self, user_input: str) -> str:
        """Process user input and return response."""
        if not self.is_running:
            return "System not initialized. Call start() first."
        
        return await self.executive.process(user_input)
    
    async def shutdown(self):
        """Gracefully shutdown all systems."""
        logger.info("Shutting down AVA Neural Node...")
        
        self.is_running = False
        self._shutdown_event.set()
        
        if self.executive:
            await self.executive.shutdown()
        
        logger.info("AVA Neural Node shutdown complete")
    
    def get_stats(self):
        """Get system statistics."""
        if self.executive:
            return self.executive.get_statistics()
        return {}


async def interactive_mode(node: AVANode):
    """Run AVA in interactive mode."""
    print(BANNER)
    print("\n[AVA] System is quasi-sentient. Ready.")
    print("[AVA] Type /help for commands, /quit to exit.\n")
    
    while node.is_running:
        try:
            # Get input
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(">> ")
            )
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                await handle_command(node, user_input)
                continue
            
            # Process input
            start_time = time.time()
            response = await node.process(user_input)
            elapsed = time.time() - start_time
            
            print(f"\nAVA: {response}")
            print(f"     [{elapsed:.2f}s]\n")
            
        except KeyboardInterrupt:
            print("\n[AVA] Received interrupt signal...")
            break
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            print(f"\n[AVA] Error: {e}\n")
    
    await node.shutdown()


async def handle_command(node: AVANode, command: str):
    """Handle system commands."""
    cmd_parts = command.lower().split()
    cmd = cmd_parts[0]
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    
    if cmd == "/help":
        print("""
Available Commands:
  /dream [type]    - Force a consolidation cycle (fast/slow/full)
  /stats           - Show system statistics
  /mode <mode>     - Set operating mode (normal/learning/performance)
  /save            - Save current state
  /models          - List available Ollama models
  /history         - Show dream history
  /quit            - Exit gracefully
  /help            - Show this help
""")
    
    elif cmd == "/dream":
        cycle_type = args[0] if args else "fast"
        print(f"[AVA] Forcing {cycle_type} consolidation cycle...")
        result = await node.executive.force_dream(cycle_type)
        if result:
            print(f"[AVA] Dream complete: {result.samples_processed} samples processed")
        else:
            print("[AVA] Dream cycle returned no results")
    
    elif cmd == "/stats":
        stats = node.get_stats()
        print("\n=== AVA System Statistics ===")
        
        if "executive" in stats:
            exec_stats = stats["executive"]
            print(f"  Total Interactions: {exec_stats.get('total_interactions', 0)}")
            print(f"  Avg Response Time: {exec_stats.get('avg_response_time', 0):.2f}s")
            print(f"  Errors: {exec_stats.get('errors', 0)}")
        
        if "conscious_stream" in stats:
            stream_stats = stats["conscious_stream"]
            print(f"\n  Neural Memory Updates: {stream_stats.get('memory_updates', 0)}")
            print(f"  Avg Surprise: {stream_stats.get('avg_surprise', 0):.4f}")
            print(f"  Replay Buffer: {stream_stats.get('replay_buffer_size', 0)} samples")
        
        if "dreamer" in stats:
            dream_stats = stats["dreamer"]
            print(f"\n  Dream Cycles: {dream_stats.get('total_dream_cycles', 0)}")
            print(f"  Fast Updates: {dream_stats.get('successful_fast_updates', 0)}")
            print(f"  Slow Updates: {dream_stats.get('successful_slow_updates', 0)}")
        
        print("")
    
    elif cmd == "/mode":
        if not args:
            print(f"[AVA] Current mode: {node.executive.mode.value}")
        else:
            from src.cortex.executive import ExecutiveMode
            try:
                mode = ExecutiveMode(args[0])
                node.executive.set_mode(mode)
                print(f"[AVA] Mode set to: {mode.value}")
            except ValueError:
                print(f"[AVA] Invalid mode. Use: normal, learning, or performance")
    
    elif cmd == "/save":
        print("[AVA] Saving state...")
        if node.executive._conscious_stream:
            node.executive._conscious_stream.save_state("data/stream_state.json")
        print("[AVA] State saved")
    
    elif cmd == "/models":
        if node.executive._llm_interface:
            models = await node.executive._llm_interface.list_models()
            print(f"[AVA] Available models: {', '.join(models)}")
        else:
            print("[AVA] LLM interface not available")
    
    elif cmd == "/history":
        if node.executive._dreamer:
            history = node.executive._dreamer.get_dream_history(5)
            print("\n=== Recent Dream History ===")
            for entry in history:
                print(f"  {entry['cycle_type']}: {entry['samples_processed']} samples "
                      f"({'success' if entry['success'] else 'failed'})")
            print("")
        else:
            print("[AVA] Dreamer not available")
    
    elif cmd == "/quit":
        print("[AVA] Initiating shutdown...")
        node.is_running = False
    
    else:
        print(f"[AVA] Unknown command: {cmd}. Type /help for available commands.")


def setup_signal_handlers(node: AVANode, loop: asyncio.AbstractEventLoop):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        loop.create_task(node.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AVA Neural Node - Bicameral Cognitive Architecture"
    )
    parser.add_argument(
        "--model", 
        default="llama3.2:latest",
        help="Ollama model for generation"
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Ollama model for embeddings"
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama API host"
    )
    parser.add_argument(
        "--no-dreaming",
        action="store_true",
        help="Disable background dreaming"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as background daemon"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run as API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="API server port"
    )
    
    args = parser.parse_args()
    
    # Create node
    node = AVANode(
        model=args.model,
        embedding_model=args.embedding_model,
        ollama_host=args.host,
        enable_dreaming=not args.no_dreaming,
    )
    
    # Run
    if args.daemon:
        print("[AVA] Daemon mode not yet implemented")
        sys.exit(1)
    elif args.api:
        print("[AVA] API mode not yet implemented")
        sys.exit(1)
    else:
        # Interactive mode
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(node.start())
            loop.run_until_complete(interactive_mode(node))
        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(node.shutdown())
            loop.close()


if __name__ == "__main__":
    main()
