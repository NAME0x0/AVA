"""
THE EXECUTIVE - High-Level Decision Making & Orchestration
==========================================================

The Executive is the "frontal lobe" of AVA - it coordinates:
- The Conscious Stream (real-time processing)
- The Dreamer (background consolidation)  
- The Emotional Engine (state modulation)
- The Developmental Tracker (growth progression)

This is the main entry point for interaction with AVA.
It handles:
1. Input preprocessing and context injection
2. Routing to appropriate processing pipelines
3. Post-processing and response formatting
4. State management across components

Architecture:
                         ┌───────────────┐
                         │   Executive   │
                         │  (Conductor)  │
                         └───────┬───────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
   ┌───────────┐          ┌───────────┐          ┌───────────┐
   │ Conscious │          │  Dreamer  │          │ Emotional │
   │  Stream   │◄────────►│   (BG)    │◄────────►│  Engine   │
   │ (Online)  │          │ (Offline) │          │  (State)  │
   └───────────┘          └───────────┘          └───────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    Output     │
                         │ (Articulation)│
                         └───────────────┘
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExecutiveMode(Enum):
    """Operating modes for the Executive."""
    NORMAL = "normal"              # Standard operation
    LEARNING = "learning"          # Focused on learning (more exploration)
    PERFORMANCE = "performance"    # Focused on quality (less exploration)
    DEBUG = "debug"                # Verbose logging, no learning


@dataclass
class ExecutiveConfig:
    """Configuration for the Executive."""
    
    # Model settings
    model_name: str = "llama3.2:latest"
    embedding_model: str = "nomic-embed-text"
    
    # Ollama connection
    ollama_host: str = "http://localhost:11434"
    
    # Mode settings
    default_mode: ExecutiveMode = ExecutiveMode.NORMAL
    
    # Context limits
    max_context_length: int = 4096
    
    # Dreaming settings
    enable_background_dreaming: bool = True
    dream_interval_seconds: float = 300.0
    
    # Emotional modulation
    enable_emotional_modulation: bool = True
    
    # Data directories
    data_dir: str = "data"
    models_dir: str = "models"
    
    # Logging
    log_interactions: bool = True
    interaction_log_path: str = "data/interactions.jsonl"


@dataclass  
class InteractionContext:
    """Context for a single interaction."""
    
    # User input
    user_input: str = ""
    
    # Contextual information
    conversation_id: Optional[str] = None
    turn_number: int = 0
    topic: str = ""
    
    # State snapshots
    emotional_state: Optional[Dict[str, float]] = None
    developmental_stage: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    mode: ExecutiveMode = ExecutiveMode.NORMAL
    
    # Additional context
    custom_context: Dict[str, Any] = field(default_factory=dict)


class Executive:
    """
    The Executive - Central coordinator of the AVA system.
    
    This is the main interface for interacting with AVA.
    It orchestrates all subsystems to produce coherent,
    adaptive, and emotionally-aware responses.
    """
    
    def __init__(
        self,
        config: Optional[ExecutiveConfig] = None,
        weight_manager: Optional[Any] = None,
    ):
        """
        Initialize the Executive.
        
        Args:
            config: Executive configuration
            weight_manager: FastSlowWeightManager for learning
        """
        self.config = config or ExecutiveConfig()
        self.weight_manager = weight_manager
        
        # Current state
        self.mode = self.config.default_mode
        self.is_initialized = False
        self.is_running = False
        
        # Subsystems (initialized lazily)
        self._llm_interface: Optional[Any] = None
        self._conscious_stream: Optional[Any] = None
        self._dreamer: Optional[Any] = None
        self._emotional_engine: Optional[Any] = None
        self._tool_registry: Optional[Any] = None
        
        # Conversation state
        self.current_conversation_id: Optional[str] = None
        self.turn_count: int = 0
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "total_tokens_generated": 0,
            "avg_response_time": 0.0,
            "errors": 0,
        }
        
        logger.info(f"Executive created with config: {self.config}")
    
    async def initialize(self):
        """
        Initialize all subsystems.
        
        This should be called before processing any interactions.
        """
        if self.is_initialized:
            return
        
        logger.info("Initializing Executive subsystems...")
        
        try:
            # 1. Initialize LLM Interface (Ollama)
            await self._initialize_llm()
            
            # 2. Initialize Tool Registry
            await self._initialize_tools()
            
            # 3. Initialize Conscious Stream
            await self._initialize_conscious_stream()
            
            # 4. Initialize Dreamer
            await self._initialize_dreamer()
            
            # 5. Initialize Emotional Engine (optional)
            if self.config.enable_emotional_modulation:
                await self._initialize_emotional_engine()
            
            # 6. Start background processes
            if self.config.enable_background_dreaming and self._dreamer:
                self._dreamer.start_background_dreaming()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("Executive initialization complete")
            
        except Exception as e:
            logger.error(f"Executive initialization failed: {e}")
            raise
    
    async def _initialize_llm(self):
        """Initialize the LLM interface (Ollama)."""
        try:
            from .ollama_interface import OllamaInterface
            
            self._llm_interface = OllamaInterface(
                host=self.config.ollama_host,
                model=self.config.model_name,
                embedding_model=self.config.embedding_model,
            )
            
            # Test connection
            await self._llm_interface.health_check()
            logger.info(f"LLM interface connected to {self.config.ollama_host}")
            
        except ImportError:
            logger.warning("OllamaInterface not found, using mock LLM")
            self._llm_interface = MockLLMInterface()
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}, using mock")
            self._llm_interface = MockLLMInterface()
    
    async def _initialize_tools(self):
        """Initialize the tool registry."""
        try:
            from ..tools.registry import ToolRegistry
            
            self._tool_registry = ToolRegistry()
            logger.info(f"Tool registry initialized with {len(self._tool_registry.list_tools())} tools")
            
        except ImportError:
            logger.warning("Tool registry not available")
            self._tool_registry = None
    
    async def _initialize_conscious_stream(self):
        """Initialize the Conscious Stream."""
        from .stream import ConsciousStream, StreamConfig
        
        stream_config = StreamConfig(
            model_dim=4096,  # Adjust based on your model
            device="cuda" if self._check_cuda() else "cpu",
        )
        
        self._conscious_stream = ConsciousStream(
            config=stream_config,
            llm_interface=self._llm_interface,
            tool_registry=self._tool_registry,
            weight_manager=self.weight_manager,
        )
        
        logger.info("Conscious Stream initialized")
    
    async def _initialize_dreamer(self):
        """Initialize the Dreamer."""
        from .dreaming import Dreamer, DreamerConfig
        
        dreamer_config = DreamerConfig(
            dream_interval=self.config.dream_interval_seconds,
        )
        
        self._dreamer = Dreamer(
            config=dreamer_config,
            weight_manager=self.weight_manager,
            conscious_stream=self._conscious_stream,
        )
        
        logger.info("Dreamer initialized")
    
    async def _initialize_emotional_engine(self):
        """Initialize the Emotional Engine."""
        try:
            from ..emotional.engine import EmotionalEngine
            
            self._emotional_engine = EmotionalEngine()
            logger.info("Emotional Engine initialized")
            
        except ImportError:
            logger.warning("Emotional Engine not available")
            self._emotional_engine = None
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def process(
        self,
        user_input: str,
        context: Optional[InteractionContext] = None,
    ) -> str:
        """
        Process a user input and generate a response.
        
        This is the main entry point for interacting with AVA.
        
        Args:
            user_input: The user's input text
            context: Optional interaction context
            
        Returns:
            AVA's response
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Create context if not provided
        if context is None:
            context = InteractionContext(
                user_input=user_input,
                turn_number=self.turn_count,
                mode=self.mode,
            )
        
        try:
            # ============================================
            # PRE-PROCESSING
            # ============================================
            
            # Get emotional state if available
            if self._emotional_engine:
                context.emotional_state = self._emotional_engine.get_current_state()
            
            # Build context dictionary for stream
            stream_context = self._build_stream_context(context)
            
            # ============================================
            # MAIN PROCESSING (Conscious Stream)
            # ============================================
            
            response = await self._conscious_stream.process(
                user_input=user_input,
                context=stream_context,
            )
            
            # ============================================
            # POST-PROCESSING
            # ============================================
            
            # Update emotional state based on interaction
            if self._emotional_engine:
                self._emotional_engine.process_interaction(
                    user_input=user_input,
                    response=response,
                )
            
            # Update conversation state
            self.turn_count += 1
            
            # Update statistics
            elapsed = time.time() - start_time
            self._update_statistics(elapsed)
            
            # Log interaction if enabled
            if self.config.log_interactions:
                self._log_interaction(context, response, elapsed)
            
            return response
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.stats["errors"] += 1
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _build_stream_context(
        self,
        context: InteractionContext,
    ) -> Dict[str, Any]:
        """Build context dictionary for the Conscious Stream."""
        stream_context = {
            "conversation_id": context.conversation_id,
            "turn_number": context.turn_number,
            "topic": context.topic,
            "mode": context.mode.value,
        }
        
        if context.emotional_state:
            stream_context["emotional_state"] = context.emotional_state
        
        if context.developmental_stage:
            stream_context["developmental_stage"] = context.developmental_stage
        
        stream_context.update(context.custom_context)
        
        return stream_context
    
    def _update_statistics(self, elapsed: float):
        """Update executive statistics."""
        n = self.stats["total_interactions"]
        old_avg = self.stats["avg_response_time"]
        
        self.stats["total_interactions"] += 1
        self.stats["avg_response_time"] = (old_avg * n + elapsed) / (n + 1)
    
    def _log_interaction(
        self,
        context: InteractionContext,
        response: str,
        elapsed: float,
    ):
        """Log an interaction to disk."""
        import json
        
        log_entry = {
            "timestamp": context.timestamp.isoformat(),
            "user_input": context.user_input,
            "response": response,
            "turn_number": context.turn_number,
            "mode": context.mode.value,
            "elapsed_seconds": elapsed,
        }
        
        log_path = Path(self.config.interaction_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    async def force_dream(self, cycle_type: str = "fast"):
        """Force an immediate dream cycle."""
        if self._dreamer:
            return self._dreamer.force_dream(cycle_type)
        return None
    
    def set_mode(self, mode: ExecutiveMode):
        """Set the operating mode."""
        self.mode = mode
        logger.info(f"Executive mode set to: {mode.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all subsystems."""
        stats = {
            "executive": self.stats,
            "mode": self.mode.value,
            "is_running": self.is_running,
            "turn_count": self.turn_count,
        }
        
        if self._conscious_stream:
            stats["conscious_stream"] = self._conscious_stream.get_statistics()
        
        if self._dreamer:
            stats["dreamer"] = self._dreamer.get_statistics()
        
        if self._emotional_engine:
            stats["emotional"] = self._emotional_engine.get_current_state()
        
        return stats
    
    async def shutdown(self):
        """Gracefully shutdown all subsystems."""
        logger.info("Shutting down Executive...")
        
        self.is_running = False
        
        # Stop dreamer
        if self._dreamer:
            self._dreamer.stop_background_dreaming()
        
        # Save state
        if self._conscious_stream:
            self._conscious_stream.save_state("data/stream_state.json")
        
        logger.info("Executive shutdown complete")
    
    def new_conversation(self) -> str:
        """Start a new conversation."""
        import uuid
        
        self.current_conversation_id = str(uuid.uuid4())
        self.turn_count = 0
        
        logger.info(f"New conversation started: {self.current_conversation_id}")
        return self.current_conversation_id


class MockLLMInterface:
    """Mock LLM interface for testing without Ollama."""
    
    async def generate(self, prompt: str) -> str:
        """Generate a mock response."""
        return f"[Mock Response] Received prompt of {len(prompt)} characters."
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate a mock embedding."""
        import numpy as np
        return np.random.randn(4096).tolist()
    
    async def health_check(self) -> bool:
        """Mock health check."""
        return True
