"""
AVA CORE V2 - Unified Cortex-Medulla System
============================================

This is the working implementation of the Cortex-Medulla architecture.
It integrates:

1. MedullaV2: Fast reflexive responses via small Ollama model
2. CortexV2: Deep reasoning via larger Ollama model
3. Simple routing based on surprise and complexity
4. Proper conversation context management
5. Cognitive state tracking

The system follows the blueprint:
- Medulla handles 95% of interactions (fast, reflexive)
- Cortex activates for complex queries (slow, thoughtful)
- Routing is based on surprise (embedding distance) and complexity

Hardware Note:
- This uses Ollama as the backend
- Can work with just CPU Ollama
- With GPU: phi3:mini (~2GB) + llama3:8b (~4GB, paged)
- Fits within RTX A2000 4GB via model swapping
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from .medulla_v2 import MedullaV2, MedullaConfig, SurpriseSignal, CognitiveState, CognitiveLabel
from .cortex_v2 import CortexV2, CortexConfig, GenerationResult

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Available action policies."""
    REFLEX_REPLY = auto()       # Quick Medulla response
    DEEP_THOUGHT = auto()       # Invoke Cortex
    ACKNOWLEDGE = auto()        # Quick acknowledgment
    WAIT = auto()               # Continue monitoring


@dataclass
class CoreConfig:
    """Master configuration for AVA Core V2."""
    
    # Component configs
    medulla_config: MedullaConfig = field(default_factory=MedullaConfig)
    cortex_config: CortexConfig = field(default_factory=CortexConfig)
    
    # Routing thresholds
    cortex_surprise_threshold: float = 0.5   # Surprise level to invoke Cortex
    cortex_complexity_threshold: float = 0.4  # Complexity level to invoke Cortex
    
    # Force cortex keywords
    cortex_keywords: List[str] = field(default_factory=lambda: [
        "analyze", "explain in detail", "think carefully", "step by step",
        "compare", "why", "how does", "what if", "debug", "optimize"
    ])


@dataclass
class ProcessResult:
    """Result of processing user input."""
    response: str = ""
    used_cortex: bool = False
    policy: PolicyType = PolicyType.REFLEX_REPLY
    surprise: float = 0.0
    complexity: float = 0.0
    cognitive_state: Optional[CognitiveState] = None
    response_time_ms: float = 0.0
    error: Optional[str] = None


class AVACoreV2:
    """
    AVA Core V2 - The Working Cortex-Medulla System.
    
    This orchestrates:
    1. Input processing via Medulla (embedding, surprise)
    2. Policy selection (Medulla vs Cortex)
    3. Response generation
    4. State tracking
    
    The system is designed to:
    - Be always responsive (Medulla)
    - Provide deep reasoning when needed (Cortex)
    - Track cognitive state accurately
    - Manage conversation context
    """
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config or CoreConfig()
        
        # Initialize components
        self._medulla = MedullaV2(self.config.medulla_config)
        self._cortex = CortexV2(self.config.cortex_config)
        
        # State tracking
        self.is_initialized = False
        self._force_cortex = False
        
        # Statistics
        self.interaction_count = 0
        self.cortex_count = 0
        self.total_response_time = 0.0
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"AVA Core V2 initialized - Session: {self.session_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize the system components.
        
        Returns:
            True if initialization successful
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING AVA CORE V2")
        logger.info("Cortex-Medulla Architecture")
        logger.info("=" * 60)
        
        # Check Ollama health
        if not await self._medulla.health_check():
            logger.error("Ollama is not available. Please start Ollama first.")
            logger.error("Run: ollama serve")
            return False
        
        logger.info("✓ Ollama connection established")
        
        # Initialize Cortex
        cortex_ready = await self._cortex.initialize()
        if cortex_ready:
            logger.info(f"✓ Cortex ready with model: {self._cortex._active_model}")
        else:
            logger.warning("✗ Cortex not available - Medulla only mode")
        
        logger.info("=" * 60)
        logger.info("AVA CORE V2 READY")
        logger.info("=" * 60)
        
        self.is_initialized = True
        return True
    
    def _should_use_cortex(
        self,
        input_text: str,
        surprise: SurpriseSignal,
    ) -> Tuple[bool, str]:
        """
        Determine if Cortex should handle this query.
        
        Args:
            input_text: User input
            surprise: Surprise signal from Medulla
            
        Returns:
            Tuple of (should_use_cortex, reason)
        """
        # Forced Cortex mode
        if self._force_cortex:
            self._force_cortex = False
            return True, "forced"
        
        # Check keywords
        input_lower = input_text.lower()
        for keyword in self.config.cortex_keywords:
            if keyword in input_lower:
                return True, f"keyword: {keyword}"
        
        # Check surprise threshold
        if surprise.value >= self.config.cortex_surprise_threshold:
            return True, f"high surprise: {surprise.value:.2f}"
        
        # Check complexity threshold
        if surprise.query_complexity >= self.config.cortex_complexity_threshold:
            return True, f"high complexity: {surprise.query_complexity:.2f}"
        
        # Check if surprise signal explicitly requires Cortex
        if surprise.requires_cortex:
            return True, "surprise signal"
        
        return False, "within medulla capacity"
    
    async def process_input(
        self,
        user_input: str,
        force_cortex: bool = False,
    ) -> ProcessResult:
        """
        Process user input through the Cortex-Medulla system.
        
        This is the main entry point. It:
        1. Perceives input via Medulla (embedding, surprise)
        2. Decides policy (Medulla vs Cortex)
        3. Generates response
        4. Tracks cognitive state
        
        Args:
            user_input: The user's input text
            force_cortex: Force Cortex processing
            
        Returns:
            ProcessResult with response and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        result = ProcessResult()
        start_time = time.time()
        
        if force_cortex:
            self._force_cortex = True
        
        try:
            # 1. PERCEIVE: Medulla processes input
            surprise, _ = await self._medulla.perceive(user_input)
            result.surprise = surprise.value
            result.complexity = surprise.query_complexity
            
            # 2. DECIDE: Should we use Cortex?
            use_cortex, reason = self._should_use_cortex(user_input, surprise)
            
            if use_cortex:
                logger.info(f"Routing to Cortex: {reason}")
                result.used_cortex = True
                result.policy = PolicyType.DEEP_THOUGHT
                self.cortex_count += 1
                
                # 3a. GENERATE via Cortex
                # Build context from Medulla's conversation history
                context = self._build_context()
                
                cortex_result = await self._cortex.generate(
                    query=user_input,
                    context=context,
                    conversation_history=self._medulla.conversation_history,
                )
                
                if cortex_result.error:
                    logger.error(f"Cortex failed: {cortex_result.error}")
                    # Fallback to Medulla
                    response, cognitive_state = await self._medulla.respond(user_input)
                    result.response = response
                    result.cognitive_state = cognitive_state
                    result.used_cortex = False
                else:
                    result.response = cortex_result.text
                    # Cortex response = high confidence creative state
                    result.cognitive_state = CognitiveState(
                        label=CognitiveLabel.CREATIVE,
                        entropy=0.5,
                        varentropy=0.6,
                        confidence=0.8,
                        surprise=surprise.value,
                        should_think=False,  # Already thought
                    )
                    # Update Medulla's history with the Cortex response
                    self._medulla.conversation_history.append(
                        {"role": "user", "content": user_input}
                    )
                    self._medulla.conversation_history.append(
                        {"role": "assistant", "content": result.response}
                    )
            else:
                # 3b. GENERATE via Medulla
                result.policy = PolicyType.REFLEX_REPLY
                response, cognitive_state = await self._medulla.respond(user_input)
                result.response = response
                result.cognitive_state = cognitive_state
            
            # 4. TRACK
            self.interaction_count += 1
            result.response_time_ms = (time.time() - start_time) * 1000
            self.total_response_time += result.response_time_ms
            
            logger.info(
                f"Response generated via {'Cortex' if result.used_cortex else 'Medulla'} "
                f"in {result.response_time_ms:.0f}ms"
            )
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            result.error = str(e)
            result.response = f"I encountered an error: {str(e)}"
        
        return result
    
    def _build_context(self) -> str:
        """Build context string from recent conversation."""
        if not self._medulla.conversation_history:
            return ""
        
        context_lines = []
        for msg in self._medulla.conversation_history[-6:]:
            role = msg["role"].capitalize()
            content = msg["content"][:200]  # Truncate long messages
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)
    
    def force_cortex_next(self):
        """Force the next response to use Cortex."""
        self._force_cortex = True
        logger.info("Next response will use Cortex")
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state."""
        state = self._medulla.cognitive_state
        return {
            "label": state.label.value,
            "entropy": state.entropy,
            "varentropy": state.varentropy,
            "confidence": state.confidence,
            "surprise": state.surprise,
            "should_think": state.should_think,
        }
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get full system state."""
        return {
            "session_id": self.session_id,
            "is_initialized": self.is_initialized,
            "interaction_count": self.interaction_count,
            "cortex_count": self.cortex_count,
            "cortex_ratio": self.cortex_count / max(1, self.interaction_count),
            "avg_response_time_ms": self.total_response_time / max(1, self.interaction_count),
            "medulla_stats": self._medulla.get_stats(),
            "cortex_stats": self._cortex.get_stats(),
            "cognitive_state": self.get_cognitive_state(),
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory/context stats."""
        return {
            "conversation_length": len(self._medulla.conversation_history),
            "embedding_history_length": len(self._medulla._embedding_history),
            "avg_surprise": np.mean(self._medulla.surprise_history) if self._medulla.surprise_history else 0.0,
            "surprise_trend": self._medulla.surprise_history[-10:] if self._medulla.surprise_history else [],
        }
    
    def clear_context(self):
        """Clear conversation context."""
        self._medulla.clear_history()
        logger.info("Conversation context cleared")
    
    async def close(self):
        """Close all connections."""
        await self._medulla.close()
        await self._cortex.close()
        logger.info("AVA Core V2 shutdown complete")


# Singleton instance for easy access
_instance: Optional[AVACoreV2] = None


async def get_core() -> AVACoreV2:
    """Get or create the AVA Core V2 singleton."""
    global _instance
    if _instance is None:
        _instance = AVACoreV2()
        await _instance.initialize()
    return _instance


async def shutdown_core():
    """Shutdown the AVA Core V2 singleton."""
    global _instance
    if _instance is not None:
        await _instance.close()
        _instance = None
