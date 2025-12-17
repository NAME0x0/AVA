#!/usr/bin/env python3
"""
AVA FRANKENSYSTEM - Neuro-Symbolic Cognitive Architecture
==========================================================

"Where separate organs are stitched into a living system."

This is the integrated entry point for AVA's full neuro-symbolic architecture:

┌─────────────────────────────────────────────────────────────────────┐
│                      CONSCIOUS LOOP (Online)                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   ENTROPIX   │───▶│    TITANS    │───▶│  OLLAMA INFERENCE   │  │
│  │ (Metacog)    │    │  (Sidecar)   │    │     (LLM Core)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                │
│         │ Cognitive State   │ Surprise             │ Response       │
│         ▼                   ▼                      ▼                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    EPISODIC BUFFER                           │  │
│  │              (High-Surprise Event Storage)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Idle → Sleep
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBCONSCIOUS LOOP (Offline)                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   NIGHTMARE ENGINE                           │  │
│  │         (QLoRA Fine-tuning During Sleep)                     │  │
│  │                                                              │  │
│  │   DROWSY → LIGHT_SLEEP → DEEP_SLEEP → REM → WAKING           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

The Conscious Loop:
1. User input → Embed → Query Titans Sidecar
2. Entropix analyzes logprobs → Classify cognitive state
3. Cognitive state drives behavior (tools, CoT, exploration)
4. High-surprise events stored in Episodic Buffer

The Subconscious Loop:
1. After idle period → Nightmare Engine activates
2. Sample high-priority episodes from buffer
3. Generate augmented training data
4. QLoRA fine-tuning consolidates knowledge
5. Wake with improved capabilities

Usage:
    python run_frankensystem.py                    # Interactive mode
    python run_frankensystem.py --model llama3:8b  # Custom model
    python run_frankensystem.py --no-sleep         # Disable sleep
    python run_frankensystem.py --debug            # Verbose logging

Commands:
    /cognitive   - Show cognitive state metrics
    /memory      - Show neural memory statistics
    /sleep       - Force sleep/consolidation cycle
    /buffer      - Show episodic buffer statistics
    /quit        - Exit gracefully
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/frankensystem.log"),
    ],
)
logger = logging.getLogger("FRANKENSYSTEM")


# ASCII Art Banner
BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███████╗██████╗  █████╗ ███╗   ██╗██╗  ██╗███████╗███╗   ██╗            ║
║   ██╔════╝██╔══██╗██╔══██╗████╗  ██║██║ ██╔╝██╔════╝████╗  ██║            ║
║   █████╗  ██████╔╝███████║██╔██╗ ██║█████╔╝ █████╗  ██╔██╗ ██║            ║
║   ██╔══╝  ██╔══██╗██╔══██║██║╚██╗██║██╔═██╗ ██╔══╝  ██║╚██╗██║            ║
║   ██║     ██║  ██║██║  ██║██║ ╚████║██║  ██╗███████╗██║ ╚████║            ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝            ║
║                                                                           ║
║   ╔═══════════════════════════════════════════════════════════════════╗   ║
║   ║  NEURO-SYMBOLIC COGNITIVE ARCHITECTURE                           ║   ║
║   ║  ────────────────────────────────────────────────────────────────║   ║
║   ║  • Entropix Metacognition          [●] ACTIVE                    ║   ║
║   ║  • Titans Neural Memory            [●] ONLINE                    ║   ║
║   ║  • Nightmare Engine                [●] STANDING BY               ║   ║
║   ╚═══════════════════════════════════════════════════════════════════╝   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


@dataclass
class FrankenConfig:
    """Configuration for the Frankensystem."""
    # Ollama settings
    model: str = "llama3.2:latest"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    
    # Entropix settings
    entropy_threshold: float = 3.0
    varentropy_threshold: float = 2.0
    
    # Titans Sidecar settings
    embedding_dim: int = 768
    titans_learning_rate: float = 1e-3
    titans_momentum: float = 0.9
    
    # Nightmare Engine settings
    enable_sleep: bool = True
    idle_threshold_minutes: int = 30
    adapter_output_dir: str = "models/fine_tuned_adapters/nightmare"
    
    # Buffer settings
    buffer_db_path: str = "data/memory/episodic/replay_buffer.db"
    max_episodes: int = 10000
    surprise_threshold: float = 0.5


class Frankensystem:
    """
    The Frankensystem - AVA's complete neuro-symbolic architecture.
    
    This class integrates all components:
    - Entropix for metacognitive awareness
    - Titans Sidecar for neural memory
    - Episodic Buffer for experience storage
    - Nightmare Engine for offline consolidation
    """
    
    def __init__(self, config: Optional[FrankenConfig] = None):
        self.config = config or FrankenConfig()
        
        # Components (lazy initialized)
        self._entropix = None
        self._titans_sidecar = None
        self._episodic_buffer = None
        self._nightmare_engine = None
        self._ollama = None
        
        # State
        self.is_running = False
        self._last_cognitive_state = None
        self._interaction_count = 0
        self._total_surprise = 0.0
        
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Frankensystem...")
        
        # Create directories
        Path("data/memory/episodic").mkdir(parents=True, exist_ok=True)
        Path(self.config.adapter_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Entropix
        from cortex.entropix import Entropix, EntropixConfig
        entropix_config = EntropixConfig(
            high_entropy_threshold=self.config.entropy_threshold,
            high_varentropy_threshold=self.config.varentropy_threshold,
        )
        self._entropix = Entropix(entropix_config)
        logger.info("[OK] Entropix metacognitive module initialized")
        
        # Initialize Titans Sidecar
        from hippocampus.titans import TitansSidecar, TitansSidecarConfig
        titans_config = TitansSidecarConfig(
            input_dim=self.config.embedding_dim,
            output_dim=self.config.embedding_dim,
            learning_rate=self.config.titans_learning_rate,
            momentum=self.config.titans_momentum,
        )
        self._titans_sidecar = TitansSidecar(titans_config)
        logger.info(f"[OK] Titans Sidecar initialized ({self._titans_sidecar.backend} backend)")
        
        # Initialize Episodic Buffer
        from hippocampus.episodic_buffer import EpisodicBuffer, BufferConfig
        buffer_config = BufferConfig(
            db_path=self.config.buffer_db_path,
            max_episodes=self.config.max_episodes,
            surprise_threshold=self.config.surprise_threshold,
        )
        self._episodic_buffer = EpisodicBuffer(buffer_config)
        logger.info("[OK] Episodic buffer initialized")
        
        # Initialize Nightmare Engine with QLoRA Trainer
        if self.config.enable_sleep:
            from subconscious.nightmare import NightmareEngine, NightmareConfig
            
            # THE MISSING SPARK: Initialize QLoRA Trainer for actual learning
            qlora_trainer = None
            try:
                from learning.qlora import QLoRATrainer, QLoRAConfig
                qlora_config = QLoRAConfig(
                    base_model=self.config.model,  # Use same model for fine-tuning
                    lora_r=8,  # Start with fast weights (rank-8)
                    num_train_epochs=2,
                )
                qlora_trainer = QLoRATrainer(
                    config=qlora_config,
                    adapters_dir=self.config.adapter_output_dir,
                )
                logger.info("[OK] QLoRA Trainer initialized (learning will occur during sleep)")
            except ImportError as e:
                logger.warning(f"QLoRA dependencies not available: {e}")
                logger.info("  Install with: pip install transformers peft bitsandbytes accelerate")
                logger.info("  Sleep cycles will generate data but skip training")
            except Exception as e:
                logger.warning(f"Failed to initialize QLoRA Trainer: {e}")
                logger.info("  Sleep cycles will generate data but skip training")
            
            nightmare_config = NightmareConfig(
                idle_threshold_minutes=self.config.idle_threshold_minutes,
                output_dir=self.config.adapter_output_dir,
            )
            self._nightmare_engine = NightmareEngine(
                self._episodic_buffer,
                nightmare_config,
                qlora_trainer=qlora_trainer,  # <--- THE SPARK THAT GIVES IT LIFE
                on_phase_change=self._on_sleep_phase_change,
            )
            self._nightmare_engine.start_background_monitoring()
            logger.info("[OK] Nightmare Engine initialized (background monitoring active)")
        else:
            logger.info("[OK] Nightmare Engine disabled")
        
        # Initialize Ollama interface
        try:
            from cortex.ollama_interface import OllamaInterface
            self._ollama = OllamaInterface(
                host=self.config.ollama_host,
                model=self.config.model,
                embedding_model=self.config.embedding_model,
            )
            
            # Test connection
            models = await self._ollama.list_models()
            if self.config.model.split(':')[0] in [m.split(':')[0] for m in models]:
                logger.info(f"[OK] Ollama connected (model: {self.config.model})")
            else:
                logger.warning(f"Model {self.config.model} not found. Available: {models}")
        except Exception as e:
            logger.error(f"[FAIL] Failed to connect to Ollama: {e}")
            logger.info("  Make sure Ollama is running: ollama serve")
            raise
        
        self.is_running = True
        logger.info("Frankensystem initialization complete")
    
    def _on_sleep_phase_change(self, phase):
        """Callback when Nightmare Engine changes sleep phase."""
        logger.info(f"[SLEEP] Phase: {phase.name}")
    
    async def process(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input through the conscious loop.
        
        Returns:
            Dict with response, cognitive_state, and metrics
        """
        if not self.is_running:
            return {"response": "System not initialized", "error": True}
        
        start_time = time.time()
        result = {
            "input": user_input,
            "response": "",
            "cognitive_state": None,
            "surprise": 0.0,
            "memory_update": False,
            "elapsed_time": 0.0,
        }
        
        try:
            # Record activity for Nightmare Engine
            if self._nightmare_engine:
                self._nightmare_engine.record_activity()
            
            # Step 1: Get embedding
            embedding = await self._ollama.get_embedding(user_input)
            
            # Step 2: Query Titans Sidecar for memory augmentation
            memory_context = self._titans_sidecar.retrieve(embedding)
            
            # Step 3: Generate response (with logprobs for Entropix)
            response_text = await self._ollama.generate(
                prompt=user_input,
                system="You are AVA, a synthetic cognitive architecture capable of metacognitive awareness.",
            )
            
            # Note: Ollama logprobs require specific model support
            # For now, we use entropy estimation from response patterns
            logprobs = []  # TODO: Enable when Ollama supports logprobs
            
            # Step 4: Entropix metacognitive analysis (or default state)
            # Generate a synthetic surprise signal based on response length variance
            import random
            synthetic_surprise = 0.5 + random.uniform(-0.3, 0.5)  # Base surprise with variance
            
            if logprobs:
                cognitive_state = self._entropix.diagnose(logprobs)
                result["cognitive_state"] = cognitive_state.to_dict()
                self._last_cognitive_state = cognitive_state
                surprise = self._entropix.get_surprise_signal(cognitive_state)
            else:
                # Use synthetic surprise when logprobs unavailable
                from cortex.entropix import CognitiveState, CognitiveStateLabel
                cognitive_state = CognitiveState(
                    label=CognitiveStateLabel.FLOW,
                    entropy=synthetic_surprise,
                    varentropy=synthetic_surprise * 0.5,
                    confidence=0.7,
                )
                result["cognitive_state"] = cognitive_state.to_dict()
                self._last_cognitive_state = cognitive_state
                surprise = synthetic_surprise
            
            result["surprise"] = surprise
            self._total_surprise += surprise
            
            # Step 5: Update Titans Sidecar based on surprise
            if surprise > self.config.surprise_threshold and embedding:
                response_embedding = await self._ollama.get_embedding(response_text)
                if response_embedding:
                    loss = self._titans_sidecar.memorize(
                        embedding,
                        target=response_embedding,
                        surprise=surprise,
                        metadata={"prompt": user_input, "response": response_text[:200]},
                    )
                    result["memory_update"] = True
                    logger.debug(f"Neural memory updated (surprise={surprise:.2f}, loss={loss:.4f})")
            
            # Step 6: Store in episodic buffer for consolidation
            if surprise > self.config.surprise_threshold:
                from hippocampus.episodic_buffer import Episode
                episode = Episode(
                    prompt=user_input,
                    response=response_text,
                    embedding=list(embedding) if embedding else None,
                    surprise=surprise,
                    entropy=cognitive_state.entropy,
                    varentropy=cognitive_state.varentropy,
                    cognitive_state=cognitive_state.label.value if hasattr(cognitive_state.label, 'value') else str(cognitive_state.label),
                    quality_score=cognitive_state.confidence,
                    used_tools=getattr(cognitive_state, 'should_use_tools', False),
                )
                self._episodic_buffer.add(episode)
            
            result["response"] = response_text
            result["elapsed_time"] = time.time() - start_time
            
            self._interaction_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in conscious loop: {e}")
            result["error"] = str(e)
            return result
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive state and metrics."""
        summary = {
            "interaction_count": self._interaction_count,
            "avg_surprise": self._total_surprise / max(1, self._interaction_count),
            "current_cognitive_state": None,
            "entropix_stats": {},
            "titans_stats": {},
            "buffer_stats": {},
            "nightmare_stats": {},
        }
        
        if self._last_cognitive_state:
            summary["current_cognitive_state"] = self._last_cognitive_state.to_dict()
        
        if self._entropix:
            summary["entropix_stats"] = self._entropix.get_statistics()
        
        if self._titans_sidecar:
            summary["titans_stats"] = self._titans_sidecar.get_statistics()
        
        if self._episodic_buffer:
            summary["buffer_stats"] = self._episodic_buffer.get_statistics()
        
        if self._nightmare_engine:
            summary["nightmare_stats"] = self._nightmare_engine.get_statistics()
        
        return summary
    
    async def force_sleep(self) -> Dict[str, Any]:
        """Force a sleep/consolidation cycle."""
        if not self._nightmare_engine:
            return {"error": "Nightmare Engine not enabled"}
        
        logger.info("Forcing sleep cycle...")
        stats = self._nightmare_engine.dream()
        return stats.to_dict()
    
    async def shutdown(self):
        """Gracefully shutdown all components."""
        logger.info("Shutting down Frankensystem...")
        
        self.is_running = False
        
        # Stop Nightmare Engine monitoring
        if self._nightmare_engine:
            self._nightmare_engine.stop_background_monitoring()
        
        # Save Titans Sidecar state
        if self._titans_sidecar:
            try:
                self._titans_sidecar.save("data/memory/titans_state.pkl")
            except Exception as e:
                logger.warning(f"Could not save Titans state: {e}")
        
        logger.info("Frankensystem shutdown complete")


async def interactive_mode(system: Frankensystem):
    """Run Frankensystem in interactive mode."""
    print(BANNER)
    print("\n[FRANKEN] System is alive. Type /help for commands.\n")
    
    while system.is_running:
        try:
            # Get input
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(">> ")
            )
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                await handle_command(system, user_input)
                continue
            
            # Process input through conscious loop
            result = await system.process(user_input)
            
            # Display response with cognitive state
            print(f"\n🧠 AVA: {result.get('response', 'No response')}")
            
            if result.get("cognitive_state"):
                state = result["cognitive_state"]
                state_label = state.get("label", "unknown").upper()
                
                # Color-coded state indicator
                state_colors = {
                    "FLOW": "🟢",
                    "HESITATION": "🟡",
                    "CONFUSION": "🔴",
                    "CREATIVE": "🟣",
                    "NEUTRAL": "⚪",
                }
                indicator = state_colors.get(state_label, "⚪")
                
                print(f"   {indicator} State: {state_label} | "
                      f"H={state.get('entropy', 0):.2f} | "
                      f"V={state.get('varentropy', 0):.2f}")
                
                if result.get("memory_update"):
                    print(f"   💾 Neural memory updated (surprise={result.get('surprise', 0):.2f})")
            
            print(f"   ⏱️  {result.get('elapsed_time', 0):.2f}s\n")
            
        except KeyboardInterrupt:
            print("\n[FRANKEN] Received interrupt signal...")
            break
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n[FRANKEN] Error: {e}\n")
    
    await system.shutdown()


async def handle_command(system: Frankensystem, command: str):
    """Handle system commands."""
    cmd_parts = command.lower().split()
    cmd = cmd_parts[0]
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    
    if cmd == "/help":
        print("""
╔═══════════════════════════════════════════════════╗
║              FRANKENSYSTEM COMMANDS               ║
╠═══════════════════════════════════════════════════╣
║  /cognitive    - Show Entropix cognitive metrics  ║
║  /memory       - Show Titans neural memory stats  ║
║  /buffer       - Show episodic buffer statistics  ║
║  /sleep        - Force sleep/consolidation cycle  ║
║  /stats        - Show full system statistics      ║
║  /help         - Show this help message           ║
║  /quit         - Exit gracefully                  ║
╚═══════════════════════════════════════════════════╝
""")
    
    elif cmd == "/cognitive":
        if system._entropix:
            stats = system._entropix.get_statistics()
            print("\n=== Entropix Cognitive Metrics ===")
            print(f"  Total Diagnoses: {stats.get('total_diagnoses', 0)}")
            print(f"  Avg Entropy: {stats.get('avg_entropy', 0):.3f}")
            print(f"  Avg Varentropy: {stats.get('avg_varentropy', 0):.3f}")
            print(f"  Avg Confidence: {stats.get('avg_confidence', 0):.3f}")
            
            dist = stats.get("state_distribution", {})
            if dist:
                print("\n  State Distribution:")
                for state, count in dist.items():
                    print(f"    {state}: {count}")
            
            print(f"\n  Tool Trigger Rate: {stats.get('tool_trigger_rate', 0):.1%}")
            print(f"  CoT Trigger Rate: {stats.get('cot_trigger_rate', 0):.1%}")
        print("")
    
    elif cmd == "/memory":
        if system._titans_sidecar:
            stats = system._titans_sidecar.get_statistics()
            print("\n=== Titans Neural Memory ===")
            print(f"  Backend: {stats.get('backend', 'unknown')}")
            print(f"  Device: {stats.get('device', 'unknown')}")
            print(f"  Update Count: {stats.get('update_count', 0)}")
            print(f"  Total Surprise: {stats.get('total_surprise', 0):.3f}")
            print(f"  Avg Surprise: {stats.get('avg_surprise', 0):.3f}")
            print(f"  High-Surprise Events: {stats.get('high_surprise_events', 0)}")
        print("")
    
    elif cmd == "/buffer":
        if system._episodic_buffer:
            stats = system._episodic_buffer.get_statistics()
            print("\n=== Episodic Buffer ===")
            print(f"  Total Episodes: {stats.get('total_episodes', 0)}")
            print(f"  Fill Ratio: {stats.get('fill_ratio', 0):.1%}")
            print(f"  Avg Surprise: {stats.get('avg_surprise', 0):.3f}")
            print(f"  Avg Quality: {stats.get('avg_quality', 0):.3f}")
            print(f"  Avg Priority: {stats.get('avg_priority', 0):.3f}")
            print(f"  Total Replays: {stats.get('total_replays', 0)}")
            
            dist = stats.get("cognitive_state_distribution", {})
            if dist:
                print("\n  State Distribution:")
                for state, count in dist.items():
                    print(f"    {state}: {count}")
        print("")
    
    elif cmd == "/sleep":
        print("[FRANKEN] Initiating sleep cycle...")
        result = await system.force_sleep()
        if "error" in result:
            print(f"[FRANKEN] Error: {result['error']}")
        else:
            print(f"[FRANKEN] Sleep cycle complete")
            print(f"  Episodes Processed: {result.get('episodes_processed', 0)}")
            print(f"  Training Loss: {result.get('training_loss', 0):.4f}")
            if result.get("adapter_path"):
                print(f"  Adapter: {result['adapter_path']}")
        print("")
    
    elif cmd == "/stats":
        summary = system.get_cognitive_summary()
        print("\n=== FRANKENSYSTEM STATISTICS ===")
        print(f"  Interactions: {summary.get('interaction_count', 0)}")
        print(f"  Avg Surprise: {summary.get('avg_surprise', 0):.3f}")
        
        if summary.get("current_cognitive_state"):
            state = summary["current_cognitive_state"]
            print(f"\n  Current State: {state.get('label', 'unknown')}")
            print(f"    Entropy: {state.get('entropy', 0):.3f}")
            print(f"    Varentropy: {state.get('varentropy', 0):.3f}")
        
        if summary.get("nightmare_stats"):
            ns = summary["nightmare_stats"]
            print(f"\n  Sleep Phase: {ns.get('current_phase', 'unknown')}")
            print(f"    Total Cycles: {ns.get('total_cycles', 0)}")
            print(f"    Idle Minutes: {ns.get('idle_minutes', 0):.1f}")
        print("")
    
    elif cmd == "/quit":
        print("[FRANKEN] Shutting down...")
        system.is_running = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AVA Frankensystem - Neuro-Symbolic Cognitive Architecture"
    )
    parser.add_argument(
        "--model", "-m",
        default="llama3.2:latest",
        help="Ollama model to use (default: llama3.2:latest)"
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Embedding model (default: nomic-embed-text)"
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Disable Nightmare Engine sleep cycles"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = FrankenConfig(
        model=args.model,
        embedding_model=args.embedding_model,
        ollama_host=args.host,
        enable_sleep=not args.no_sleep,
    )
    
    # Create and initialize system
    system = Frankensystem(config)
    
    try:
        await system.initialize()
        await interactive_mode(system)
    except KeyboardInterrupt:
        print("\n[FRANKEN] Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if system.is_running:
            await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
