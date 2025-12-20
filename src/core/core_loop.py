"""
AVA CORE SYSTEM - The Unified Cortex-Medulla Architecture
==========================================================

This module integrates all components of the Cortex-Medulla architecture
into a unified system that implements continuous, autonomous operation.

System Flow:
============

1. SENSING (Medulla):
   - Continuous ingestion of sensory inputs (text, audio, logs)
   - Update SSM hidden state with O(1) memory
   - Calculate surprise signal

2. PERCEIVING (Medulla + Titans):
   - Update Titans neural memory based on surprise
   - Retrieve relevant memories

3. DECIDING (Agency):
   - Active Inference calculates Expected Free Energy
   - Select optimal policy (Reflex, DeepThought, Tool, Wait, etc.)

4. ACTING:
   - If REFLEX: Medulla generates quick response
   - If DEEP_THOUGHT: Bridge projects state → Cortex generates
   - If TOOL: Execute tool and feed results back

5. LEARNING (Titans):
   - Compress experience into neural memory weights
   - No context window growth

Hardware Budget (RTX A2000 4GB):
================================
- System Overhead:    ~300 MB
- Medulla (Mamba):    ~800 MB
- Titans Memory:      ~200 MB
- Bridge Adapter:     ~50 MB
- Cortex Buffer:      ~1,600 MB (paged)
- Headroom:           ~1,050 MB
- TOTAL:              ~3,000 MB / 4,096 MB available
"""

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

# Import architecture components
from .medulla import Medulla, MedullaConfig, SurpriseSignal
from .cortex_engine import Cortex, CortexConfig, GenerationResult
from .bridge import Bridge, BridgeConfig
from .agency import (
    ActiveInferenceController, AgencyConfig, 
    PolicyType, Observation, BeliefState
)

# Import existing systems for integration
try:
    from ..hippocampus.titans import TitansSidecarNumpy as TitansSidecar
except ImportError:
    TitansSidecar = None

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system operating state."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class CoreConfig:
    """
    Master configuration for the AVA Core System.
    
    This aggregates all component configs and provides system-wide settings.
    """
    
    # Component Configurations
    medulla_config: MedullaConfig = field(default_factory=MedullaConfig)
    cortex_config: CortexConfig = field(default_factory=CortexConfig)
    bridge_config: BridgeConfig = field(default_factory=BridgeConfig)
    agency_config: AgencyConfig = field(default_factory=AgencyConfig)
    
    # System Settings
    data_dir: str = "data"
    log_level: str = "INFO"
    
    # Main Loop Settings
    main_loop_interval: float = 0.1       # 100ms tick rate
    idle_loop_interval: float = 1.0       # 1s when idle
    
    # Sensory Input Configuration
    enable_audio_input: bool = False       # Requires faster-whisper
    enable_log_monitoring: bool = True     # Monitor system logs
    
    # Output Configuration
    enable_voice_output: bool = False      # Requires TTS
    
    # Safety Settings
    max_cortex_time: float = 300.0         # 5 min max per Cortex call
    emergency_shutdown_phrase: str = "ava shutdown"
    
    # Persistence
    autosave_interval: int = 100           # Save state every N interactions


@dataclass 
class InteractionRecord:
    """Record of a single interaction for logging and learning."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Input
    user_input: str = ""
    input_type: str = "text"  # text, audio, system
    
    # Processing
    surprise_value: float = 0.0
    selected_policy: str = ""
    used_cortex: bool = False
    
    # Output
    response: str = ""
    response_time_ms: float = 0.0
    
    # State
    belief_entropy: float = 0.0
    medulla_state_norm: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "input_type": self.input_type,
            "surprise_value": self.surprise_value,
            "selected_policy": self.selected_policy,
            "used_cortex": self.used_cortex,
            "response": self.response,
            "response_time_ms": self.response_time_ms,
            "belief_entropy": self.belief_entropy,
        }


class AVACoreSystem:
    """
    The AVA Core System - Unified Cortex-Medulla Architecture.
    
    This is the main entry point for the v3 architecture. It orchestrates:
    
    1. Medulla: Always-on reflexive processing
    2. Cortex: Deep reasoning via AirLLM
    3. Bridge: Neural state projection for context transfer
    4. Agency: Active Inference for autonomous behavior
    5. Titans: Test-time learning for infinite memory
    
    The system runs continuously, responding to user input and proactively
    managing its own operation through Free Energy minimization.
    """
    
    def __init__(
        self,
        config: Optional[CoreConfig] = None,
    ):
        """
        Initialize the AVA Core System.
        
        Args:
            config: Master configuration
        """
        self.config = config or CoreConfig()
        
        # System state
        self.state = SystemState.INITIALIZING
        self.is_running = False
        
        # Initialize components (lazy loading)
        self._medulla: Optional[Medulla] = None
        self._cortex: Optional[Cortex] = None
        self._bridge: Optional[Bridge] = None
        self._agency: Optional[ActiveInferenceController] = None
        self._titans: Optional[Any] = None
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 50
        
        # Interaction tracking
        self.interaction_records: List[InteractionRecord] = []
        self.total_interactions = 0
        self.session_start = datetime.now()
        
        # Callbacks for external integration
        self._output_callback: Optional[Callable[[str], None]] = None
        self._input_callback: Optional[Callable[[], Optional[str]]] = None
        
        logger.info("AVA Core System created")
    
    async def initialize(self) -> None:
        """
        Initialize all system components.
        
        This performs startup checks and loads models:
        1. Initialize Titans memory
        2. Initialize Medulla (loads to VRAM)
        3. Initialize Cortex (loads to System RAM)
        4. Initialize Bridge
        5. Initialize Agency controller
        6. Register action callbacks
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING AVA CORE SYSTEM v3")
        logger.info("Cortex-Medulla Architecture")
        logger.info("=" * 60)
        
        try:
            # 1. Initialize Titans Memory
            logger.info("[1/5] Initializing Titans Neural Memory...")
            await self._init_titans()
            
            # 2. Initialize Medulla (VRAM-resident)
            logger.info("[2/5] Initializing Medulla (Reflexive Core)...")
            await self._init_medulla()
            
            # 3. Initialize Cortex (System RAM)
            logger.info("[3/5] Initializing Cortex (Reflective Core)...")
            await self._init_cortex()
            
            # 4. Initialize Bridge
            logger.info("[4/5] Initializing Bridge (State Projection)...")
            await self._init_bridge()
            
            # 5. Initialize Agency
            logger.info("[5/5] Initializing Agency (Active Inference)...")
            await self._init_agency()
            
            # Register action callbacks
            self._register_action_callbacks()
            
            self.state = SystemState.RUNNING
            logger.info("=" * 60)
            logger.info("AVA CORE SYSTEM INITIALIZED SUCCESSFULLY")
            logger.info(f"Session ID: {self.session_start.strftime('%Y%m%d_%H%M%S')}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state = SystemState.ERROR
            raise
    
    async def _init_titans(self) -> None:
        """Initialize Titans neural memory."""
        if TitansSidecar is not None:
            self._titans = TitansSidecar()
            logger.info("Titans memory initialized")
        else:
            logger.warning("Titans not available - using no persistent memory")
            self._titans = None
    
    async def _init_medulla(self) -> None:
        """Initialize Medulla with Titans memory."""
        self._medulla = Medulla(
            config=self.config.medulla_config,
            titans_memory=self._titans,
        )
        await self._medulla.initialize()
    
    async def _init_cortex(self) -> None:
        """Initialize Cortex (loads to System RAM)."""
        self._cortex = Cortex(config=self.config.cortex_config)
        # Don't initialize yet - lazy load on first use
        logger.info("Cortex configured (will load on first use)")
    
    async def _init_bridge(self) -> None:
        """Initialize Bridge for state projection."""
        self._bridge = Bridge(config=self.config.bridge_config)
    
    async def _init_agency(self) -> None:
        """Initialize Active Inference controller."""
        self._agency = ActiveInferenceController(config=self.config.agency_config)
    
    def _register_action_callbacks(self) -> None:
        """Register action callbacks with the Agency controller."""
        
        async def handle_reflex(obs: Observation, beliefs: BeliefState) -> Dict:
            """Handle reflexive response."""
            if obs.text:
                surprise, response = await self._medulla.perceive(input_text=obs.text)
                return {"response": response, "surprise": surprise.value}
            return {"response": None}
        
        async def handle_deep_thought(obs: Observation, beliefs: BeliefState) -> Dict:
            """Handle deep reasoning via Cortex."""
            if obs.text:
                response = await self._invoke_cortex(obs.text)
                return {"response": response}
            return {"response": None}
        
        async def handle_wait(obs: Observation, beliefs: BeliefState) -> Dict:
            """Handle waiting (no action)."""
            return {"action": "wait"}
        
        async def handle_ask_clarification(obs: Observation, beliefs: BeliefState) -> Dict:
            """Handle asking for clarification."""
            clarification = "Could you please provide more details about what you need?"
            if self._output_callback:
                self._output_callback(clarification)
            return {"response": clarification}
        
        # Register callbacks
        self._agency.register_action_callback(PolicyType.REFLEX_REPLY, handle_reflex)
        self._agency.register_action_callback(PolicyType.DEEP_THOUGHT, handle_deep_thought)
        self._agency.register_action_callback(PolicyType.WAIT, handle_wait)
        self._agency.register_action_callback(PolicyType.ASK_CLARIFICATION, handle_ask_clarification)
    
    def set_output_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for outputting responses."""
        self._output_callback = callback
    
    def set_input_callback(self, callback: Callable[[], Optional[str]]) -> None:
        """Set callback for getting user input."""
        self._input_callback = callback
    
    async def process_input(
        self,
        user_input: str,
        force_cortex: bool = False,
    ) -> str:
        """
        Process user input and generate a response.
        
        This is the main interaction method that:
        1. Passes input through Medulla for surprise calculation
        2. Uses Agency to select optimal policy
        3. Executes the selected policy
        4. Returns the response
        
        Args:
            user_input: User's input text
            force_cortex: Force Cortex processing regardless of surprise
            
        Returns:
            Generated response string
        """
        start_time = time.time()
        record = InteractionRecord(user_input=user_input)
        
        try:
            # 1. Process through Medulla
            surprise, medulla_response = await self._medulla.perceive(input_text=user_input)
            record.surprise_value = surprise.value
            
            # 2. Create observation for Agency
            observation = Observation(
                text=user_input,
                surprise_signal=surprise.value,
                query_complexity=self._estimate_complexity(user_input),
            )
            
            # 3. Let Agency decide (or force Cortex)
            if force_cortex or surprise.requires_cortex:
                # Skip agency decision - go directly to Cortex
                record.selected_policy = PolicyType.DEEP_THOUGHT.name
                record.used_cortex = True
                response = await self._invoke_cortex(user_input)
            else:
                # Let Agency decide
                policy, result = await self._agency.process_observation(observation)
                record.selected_policy = policy.name
                
                if policy == PolicyType.DEEP_THOUGHT:
                    record.used_cortex = True
                    response = await self._invoke_cortex(user_input)
                elif "response" in result.get("action_result", {}):
                    response = result["action_result"]["response"]
                else:
                    response = medulla_response
            
            # 4. Record interaction
            record.response = response or ""
            record.response_time_ms = (time.time() - start_time) * 1000
            record.belief_entropy = self._agency.beliefs.entropy
            
            # Update conversation history
            self._update_history(user_input, response)
            
            # Store record
            self.interaction_records.append(record)
            self.total_interactions += 1
            
            # Autosave
            if self.total_interactions % self.config.autosave_interval == 0:
                await self._autosave()
            
            return response or "I'm not sure how to respond to that."
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"I encountered an error: {str(e)}"
    
    async def _invoke_cortex(self, query: str) -> str:
        """
        Invoke the Cortex for deep reasoning.
        
        This method:
        1. Gets Medulla's current state
        2. Uses Bridge to project state to Cortex embeddings
        3. Prepares the full prompt
        4. Calls Cortex for generation
        
        Args:
            query: User's query
            
        Returns:
            Cortex-generated response
        """
        logger.info("Invoking Cortex for deep reasoning...")
        
        # Output waiting message
        if self._output_callback:
            waiting_msg = await self._medulla._generate_waiting_response()
            self._output_callback(f"[Thinking] {waiting_msg}")
        
        # 1. Get Medulla state
        medulla_state = self._medulla.get_state_vector()
        
        # 2. Prepare Cortex input via Bridge
        cortex_input = await self._bridge.prepare_cortex_input(
            medulla_state=medulla_state,
            current_query=query,
            conversation_history=self.conversation_history,
        )
        
        # 3. Generate with Cortex
        result = await self._cortex.generate(
            prompt=cortex_input["text_prompt"],
            projected_state=cortex_input["soft_prompts"],
        )
        
        if result.error:
            logger.error(f"Cortex generation error: {result.error}")
            return f"I had trouble thinking about that: {result.error}"
        
        logger.info(
            f"Cortex complete: {result.output_tokens} tokens in "
            f"{result.total_time_seconds:.1f}s"
        )
        
        return result.text
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate query complexity for Agency decision-making."""
        complexity = 0.0
        
        # Length factor
        words = len(text.split())
        complexity += min(words / 50, 0.3)
        
        # Question words
        question_words = ["how", "why", "explain", "analyze", "compare", "evaluate"]
        if any(w in text.lower() for w in question_words):
            complexity += 0.3
        
        # Technical indicators
        technical_words = ["code", "implement", "algorithm", "function", "debug"]
        if any(w in text.lower() for w in technical_words):
            complexity += 0.3
        
        # Multiple questions
        complexity += min(text.count("?") * 0.1, 0.2)
        
        return min(complexity, 1.0)
    
    def _update_history(self, user_input: str, response: str) -> None:
        """Update conversation history."""
        self.conversation_history.append({"role": "user", "content": user_input})
        if response:
            self.conversation_history.append({"role": "assistant", "content": response})
        
        # Trim history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    async def run_forever(self) -> None:
        """
        Run the system in continuous autonomous mode.
        
        This implements the "always-on" behavior where the system:
        1. Monitors for user input
        2. Processes observations through Agency
        3. Takes proactive actions when uncertainty is high
        4. Never exits until explicitly stopped
        """
        logger.info("Starting AVA Core System continuous operation...")
        self.is_running = True
        
        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        last_input_time = time.time()
        
        while self.is_running:
            try:
                # Check for user input
                user_input = None
                if self._input_callback:
                    user_input = self._input_callback()
                
                if user_input:
                    # Check for emergency shutdown
                    if self.config.emergency_shutdown_phrase in user_input.lower():
                        logger.info("Emergency shutdown phrase detected")
                        break
                    
                    # Process input
                    response = await self.process_input(user_input)
                    
                    # Output response
                    if self._output_callback:
                        self._output_callback(response)
                    
                    last_input_time = time.time()
                    
                    # Short sleep for responsive interaction
                    await asyncio.sleep(self.config.main_loop_interval)
                else:
                    # No input - run Agency inference
                    time_since_input = time.time() - last_input_time
                    
                    # Create observation from current state
                    observation = Observation(
                        silence_duration=time_since_input,
                        surprise_signal=0.0,
                    )
                    
                    # Let Agency decide
                    policy, result = await self._agency.process_observation(observation)
                    
                    # Handle proactive actions
                    if policy == PolicyType.ASK_CLARIFICATION and time_since_input > 300:
                        proactive_msg = "Is there anything I can help you with?"
                        if self._output_callback:
                            self._output_callback(f"[Proactive] {proactive_msg}")
                        last_input_time = time.time()  # Reset timer
                    
                    # Longer sleep when idle
                    await asyncio.sleep(self.config.idle_loop_interval)
                    
            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1.0)
        
        await self.shutdown()
    
    async def _autosave(self) -> None:
        """Autosave system state."""
        logger.debug("Autosaving state...")
        try:
            self._medulla._save_state()
            self._bridge.save_state()
            self._agency.save_state()
        except Exception as e:
            logger.error(f"Autosave failed: {e}")
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the system.
        
        Saves all state and releases resources.
        """
        logger.info("Initiating AVA Core System shutdown...")
        self.state = SystemState.SHUTTING_DOWN
        self.is_running = False
        
        try:
            # Save state
            logger.info("Saving system state...")
            await self._autosave()
            
            # Shutdown components
            if self._medulla:
                await self._medulla.shutdown()
            
            if self._cortex:
                await self._cortex.shutdown()
            
            logger.info("AVA Core System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        self.state = SystemState.INITIALIZING
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        uptime = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "system": {
                "state": self.state.value,
                "uptime_seconds": uptime,
                "total_interactions": self.total_interactions,
                "session_start": self.session_start.isoformat(),
            },
            "medulla": self._medulla.get_stats() if self._medulla else {},
            "cortex": self._cortex.get_stats() if self._cortex else {},
            "bridge": self._bridge.get_stats() if self._bridge else {},
            "agency": self._agency.get_stats() if self._agency else {},
        }
    
    def get_interaction_log(self, n: int = 10) -> List[Dict]:
        """Get the last N interaction records."""
        return [r.to_dict() for r in self.interaction_records[-n:]]


# Convenience function for quick start
async def create_and_run_ava(config: Optional[CoreConfig] = None) -> AVACoreSystem:
    """
    Create and initialize an AVA Core System instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized AVA Core System
    """
    ava = AVACoreSystem(config)
    await ava.initialize()
    return ava
