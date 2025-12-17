"""
THE DREAMER - Subconscious Consolidation
========================================

The background process that consolidates learning from the Conscious Stream.
This is the "sleeping" mind that:

1. FAST WEIGHT UPDATE (Session Adaptation):
   - Triggered after N interactions or on high-quality sessions
   - Trains a lightweight LoRA adapter (rank=8) quickly
   - Provides immediate behavioral adaptation

2. SLOW WEIGHT UPDATE (Long-term Consolidation):
   - Triggered after significant data accumulation
   - Trains a heavier adapter (rank=64) more thoroughly
   - Updates consolidated reference weights for drift penalty

3. KNOWLEDGE DISTILLATION:
   - Extracts Chain-of-Thought reasoning from interactions
   - Augments training data with tool-use patterns
   - Enables self-improvement loop

The Dreamer runs in a background thread, periodically processing
the replay buffer from the Conscious Stream.

Reference Papers:
- "Nested Learning: A New Paradigm for Continual Learning" (Google, 2025)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (arXiv:2305.14314)
- "Distilling Step-by-Step!" (ACL Findings, 2023)
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DreamState(Enum):
    """Current state of the Dreamer."""
    AWAKE = "awake"                # Not dreaming
    LIGHT_SLEEP = "light_sleep"    # Processing replay buffer
    REM = "rem"                    # Fast weight training
    DEEP_SLEEP = "deep_sleep"      # Slow weight consolidation
    LUCID = "lucid"                # Self-distillation


@dataclass
class DreamerConfig:
    """Configuration for the Dreamer."""
    
    # Timing (in seconds)
    dream_interval: float = 300.0  # Dream every 5 minutes
    min_interactions_for_fast: int = 5
    min_interactions_for_slow: int = 100
    
    # Fast weight settings (session adaptation)
    fast_adapter_rank: int = 8
    fast_training_epochs: int = 2
    fast_learning_rate: float = 1e-4
    
    # Slow weight settings (consolidation)
    slow_adapter_rank: int = 64
    slow_training_epochs: int = 5
    slow_learning_rate: float = 5e-5
    
    # Quality filtering
    min_quality_for_training: float = 0.6
    prefer_surprising_samples: bool = True
    
    # Distillation settings
    enable_cot_distillation: bool = True
    enable_tool_distillation: bool = True
    
    # Storage
    checkpoints_dir: str = "data/learning/checkpoints"
    adapters_dir: str = "models/fine_tuned_adapters"
    
    # Resource limits
    max_samples_per_batch: int = 100
    max_memory_gb: float = 6.0  # For consumer GPUs


@dataclass
class DreamCycleResult:
    """Result of a dream cycle."""
    
    cycle_type: str  # "fast", "slow", "distillation"
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    samples_processed: int = 0
    training_loss: Optional[float] = None
    adapter_path: Optional[str] = None
    
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_type": self.cycle_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "samples_processed": self.samples_processed,
            "training_loss": self.training_loss,
            "adapter_path": self.adapter_path,
            "success": self.success,
            "error_message": self.error_message,
        }


class Dreamer:
    """
    The Dreamer - Background consolidation and optimization.
    
    This is the "subconscious" that processes experiences during
    idle periods, consolidating learning into permanent adapters.
    
    Architecture:
        Replay Buffer → Quality Filter → Training Data
                                              ↓
                            ┌─────────────────┴─────────────────┐
                            ↓                                   ↓
                    Fast Weight Update              Slow Weight Update
                    (Rank-8 LoRA, 2 epochs)        (Rank-64 LoRA, 5 epochs)
                            ↓                                   ↓
                    Session Adapter                 Long-term Adapter
                            ↓                                   ↓
                            └─────────────────┬─────────────────┘
                                              ↓
                                    Consolidated Weights
                                    (Drift Penalty Reference)
    """
    
    def __init__(
        self,
        config: Optional[DreamerConfig] = None,
        weight_manager: Optional[Any] = None,
        conscious_stream: Optional[Any] = None,
    ):
        """
        Initialize the Dreamer.
        
        Args:
            config: Dreamer configuration
            weight_manager: FastSlowWeightManager for recording
            conscious_stream: ConsciousStream to get replay buffer from
        """
        self.config = config or DreamerConfig()
        self.weight_manager = weight_manager
        self.conscious_stream = conscious_stream
        
        # State
        self.state = DreamState.AWAKE
        self.is_running = False
        self._dream_thread: Optional[threading.Thread] = None
        
        # Directories
        self.checkpoints_dir = Path(self.config.checkpoints_dir)
        self.adapters_dir = Path(self.config.adapters_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Training components
        self.qlora_trainer: Optional[Any] = None
        self._initialize_trainer()
        
        # History
        self.dream_history: List[DreamCycleResult] = []
        self.total_samples_processed = 0
        self.last_fast_update: Optional[datetime] = None
        self.last_slow_update: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            "total_dream_cycles": 0,
            "successful_fast_updates": 0,
            "successful_slow_updates": 0,
            "total_training_time": 0.0,
        }
        
        logger.info(f"Dreamer initialized with config: {self.config}")
    
    def _initialize_trainer(self):
        """Initialize QLoRA trainer if available."""
        try:
            from ..learning.qlora import QLoRATrainer
            self.qlora_trainer = QLoRATrainer(
                checkpoints_dir=str(self.checkpoints_dir),
                adapters_dir=str(self.adapters_dir),
            )
            logger.info("QLoRA trainer initialized")
        except ImportError:
            logger.warning("QLoRA trainer not available - dreaming disabled")
            self.qlora_trainer = None
    
    def start_background_dreaming(self):
        """Start the background dreaming thread."""
        if self.is_running:
            logger.warning("Dreamer is already running")
            return
        
        self.is_running = True
        self._dream_thread = threading.Thread(
            target=self._dream_loop,
            daemon=True,
            name="AVA-Dreamer",
        )
        self._dream_thread.start()
        logger.info("Background dreaming started")
    
    def stop_background_dreaming(self):
        """Stop the background dreaming thread."""
        self.is_running = False
        if self._dream_thread:
            self._dream_thread.join(timeout=5.0)
        logger.info("Background dreaming stopped")
    
    def _dream_loop(self):
        """The main dreaming loop - runs in background thread."""
        while self.is_running:
            try:
                time.sleep(self.config.dream_interval)
                
                if not self.is_running:
                    break
                
                # Execute a dream cycle
                self.dream()
                
            except Exception as e:
                logger.error(f"Error in dream loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def dream(self) -> List[DreamCycleResult]:
        """
        Execute a dream cycle - the main consolidation routine.
        
        Returns:
            List of results from this dream cycle
        """
        logger.info("Entering dream state...")
        results = []
        
        try:
            self.state = DreamState.LIGHT_SLEEP
            
            # Get replay buffer from conscious stream
            replay_buffer = self._get_replay_buffer()
            
            if not replay_buffer:
                logger.info("No samples to dream about")
                self.state = DreamState.AWAKE
                return results
            
            # Filter by quality
            training_samples = self._filter_samples(replay_buffer)
            
            if not training_samples:
                logger.info("No samples passed quality filter")
                self.state = DreamState.AWAKE
                return results
            
            # ============================================
            # FAST WEIGHT UPDATE (REM Sleep)
            # ============================================
            if len(training_samples) >= self.config.min_interactions_for_fast:
                self.state = DreamState.REM
                fast_result = self._fast_weight_update(training_samples)
                results.append(fast_result)
                
                if fast_result.success:
                    self.last_fast_update = datetime.now()
                    self.stats["successful_fast_updates"] += 1
            
            # ============================================
            # SLOW WEIGHT UPDATE (Deep Sleep)
            # ============================================
            if self._should_consolidate():
                self.state = DreamState.DEEP_SLEEP
                slow_result = self._slow_weight_update()
                results.append(slow_result)
                
                if slow_result.success:
                    self.last_slow_update = datetime.now()
                    self.stats["successful_slow_updates"] += 1
            
            # ============================================
            # SELF-DISTILLATION (Lucid Dreaming)
            # ============================================
            if self.config.enable_cot_distillation:
                self.state = DreamState.LUCID
                distill_result = self._self_distillation(training_samples)
                if distill_result:
                    results.append(distill_result)
            
            self.stats["total_dream_cycles"] += 1
            
        except Exception as e:
            logger.error(f"Dream cycle failed: {e}")
            results.append(DreamCycleResult(
                cycle_type="error",
                success=False,
                error_message=str(e),
            ))
        finally:
            self.state = DreamState.AWAKE
        
        logger.info(f"Dream cycle complete: {len(results)} operations")
        return results
    
    def _get_replay_buffer(self) -> List[Dict[str, Any]]:
        """Get replay buffer from conscious stream."""
        if self.conscious_stream:
            buffer = self.conscious_stream.get_replay_buffer()
            return [r.to_dict() if hasattr(r, 'to_dict') else r for r in buffer]
        
        # Fallback to weight manager
        if self.weight_manager:
            return self.weight_manager.get_pending_samples()
        
        return []
    
    def _filter_samples(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter samples by quality for training.
        
        Applies quality threshold and optionally prioritizes
        surprising samples.
        """
        filtered = []
        
        for sample in samples:
            quality = sample.get("quality_score", 0.0)
            surprise = sample.get("surprise_value", 0.0)
            
            # Apply surprise boost
            if self.config.prefer_surprising_samples and surprise > 0.5:
                quality *= 1.5
            
            if quality >= self.config.min_quality_for_training:
                filtered.append(sample)
        
        # Sort by quality (descending) and take top N
        filtered.sort(
            key=lambda x: x.get("quality_score", 0) * (
                1 + x.get("surprise_value", 0) * 0.5
            ),
            reverse=True,
        )
        
        return filtered[:self.config.max_samples_per_batch]
    
    def _fast_weight_update(
        self,
        samples: List[Dict[str, Any]],
    ) -> DreamCycleResult:
        """
        Perform fast weight update (session adaptation).
        
        Trains a lightweight LoRA adapter quickly on recent interactions.
        """
        result = DreamCycleResult(cycle_type="fast")
        start_time = time.time()
        
        try:
            logger.info(f"Fast weight update: {len(samples)} samples")
            
            if not self.qlora_trainer:
                result.error_message = "QLoRA trainer not available"
                return result
            
            # Prepare training data
            training_data = self._prepare_training_data(samples)
            
            if not training_data:
                result.error_message = "No training data prepared"
                return result
            
            # Train fast adapter
            adapter_path = self.qlora_trainer.train_adapter(
                data=training_data,
                adapter_name=f"fast_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rank=self.config.fast_adapter_rank,
                epochs=self.config.fast_training_epochs,
                learning_rate=self.config.fast_learning_rate,
            )
            
            result.samples_processed = len(samples)
            result.adapter_path = adapter_path
            result.success = True
            result.completed_at = datetime.now()
            
            # Clear processed samples from replay buffer
            if self.conscious_stream:
                self.conscious_stream.clear_replay_buffer()
            
            self.total_samples_processed += len(samples)
            
        except Exception as e:
            logger.error(f"Fast weight update failed: {e}")
            result.error_message = str(e)
        finally:
            elapsed = time.time() - start_time
            self.stats["total_training_time"] += elapsed
        
        self.dream_history.append(result)
        return result
    
    def _slow_weight_update(self) -> DreamCycleResult:
        """
        Perform slow weight update (long-term consolidation).
        
        Trains a heavier adapter on accumulated data and updates
        the consolidated reference weights for drift penalty.
        """
        result = DreamCycleResult(cycle_type="slow")
        start_time = time.time()
        
        try:
            logger.info("Slow weight consolidation starting...")
            
            if not self.qlora_trainer:
                result.error_message = "QLoRA trainer not available"
                return result
            
            # Get accumulated data from weight manager
            if not self.weight_manager:
                result.error_message = "Weight manager not available"
                return result
            
            accumulated_samples = self.weight_manager.get_slow_update_batch()
            
            if not accumulated_samples:
                result.error_message = "No accumulated samples for consolidation"
                return result
            
            # Prepare training data
            training_data = self._prepare_training_data(accumulated_samples)
            
            # Train slow adapter
            adapter_path = self.qlora_trainer.train_adapter(
                data=training_data,
                adapter_name=f"long_term_memory_{datetime.now().strftime('%Y%m%d')}",
                rank=self.config.slow_adapter_rank,
                epochs=self.config.slow_training_epochs,
                learning_rate=self.config.slow_learning_rate,
            )
            
            result.samples_processed = len(accumulated_samples)
            result.adapter_path = adapter_path
            result.success = True
            result.completed_at = datetime.now()
            
            # Update consolidated reference weights
            if self.qlora_trainer and result.success:
                try:
                    current_weights = self.qlora_trainer.get_model_weights()
                    self.weight_manager.update_consolidated_state(current_weights)
                except Exception as e:
                    logger.warning(f"Failed to update consolidated state: {e}")
            
        except Exception as e:
            logger.error(f"Slow weight update failed: {e}")
            result.error_message = str(e)
        finally:
            elapsed = time.time() - start_time
            self.stats["total_training_time"] += elapsed
        
        self.dream_history.append(result)
        return result
    
    def _self_distillation(
        self,
        samples: List[Dict[str, Any]],
    ) -> Optional[DreamCycleResult]:
        """
        Perform self-distillation on high-quality samples.
        
        This extracts Chain-of-Thought reasoning and tool-use patterns,
        then creates enhanced training samples for future learning.
        """
        result = DreamCycleResult(cycle_type="distillation")
        
        try:
            # Extract samples with reasoning steps
            cot_samples = [
                s for s in samples
                if s.get("reasoning_steps") and len(s.get("reasoning_steps", [])) > 0
            ]
            
            if not cot_samples:
                return None
            
            logger.info(f"Self-distillation: {len(cot_samples)} samples with CoT")
            
            # Create enhanced training data
            enhanced_samples = []
            
            for sample in cot_samples:
                # Create structured output with reasoning
                reasoning = "\n".join(sample.get("reasoning_steps", []))
                output = sample.get("output_text", "")
                
                enhanced_output = f"<think>\n{reasoning}\n</think>\n\n{output}"
                
                enhanced_samples.append({
                    "instruction": sample.get("input_text", ""),
                    "output": enhanced_output,
                    "quality_score": sample.get("quality_score", 0.5) * 1.2,  # Boost for CoT
                })
            
            # Save enhanced samples for future training
            distill_path = self.checkpoints_dir / "distillation_samples.json"
            
            existing_samples = []
            if distill_path.exists():
                with open(distill_path, "r") as f:
                    existing_samples = json.load(f)
            
            existing_samples.extend(enhanced_samples)
            
            # Keep bounded
            existing_samples = existing_samples[-1000:]
            
            with open(distill_path, "w") as f:
                json.dump(existing_samples, f, indent=2)
            
            result.samples_processed = len(enhanced_samples)
            result.success = True
            result.completed_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Self-distillation failed: {e}")
            result.error_message = str(e)
        
        return result
    
    def _should_consolidate(self) -> bool:
        """
        Determine if slow weight consolidation should occur.
        
        Triggers on:
        - Accumulated sample count threshold
        - Time since last consolidation
        - Manual trigger
        """
        if not self.weight_manager:
            return False
        
        stats = self.weight_manager.get_statistics()
        
        # Check sample count
        if stats.get("interaction_count", 0) >= self.config.min_interactions_for_slow:
            if stats.get("interaction_count", 0) % self.config.min_interactions_for_slow == 0:
                return True
        
        # Check time since last consolidation
        if self.last_slow_update:
            hours_since = (datetime.now() - self.last_slow_update).total_seconds() / 3600
            if hours_since >= 24:  # Daily consolidation
                return True
        
        return False
    
    def _prepare_training_data(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Prepare samples for QLoRA training.
        
        Converts interaction records to instruction/output format.
        """
        training_data = []
        
        for sample in samples:
            input_text = sample.get("input_text", "")
            output_text = sample.get("output_text", "")
            
            if not input_text or not output_text:
                continue
            
            # Include reasoning if available
            reasoning = sample.get("reasoning_steps", [])
            if reasoning:
                structured_output = f"<think>\n{chr(10).join(reasoning)}\n</think>\n\n{output_text}"
            else:
                structured_output = output_text
            
            # Include tool calls if available
            tool_calls = sample.get("tool_calls", [])
            tool_results = sample.get("tool_results", [])
            
            if tool_calls and tool_results:
                tool_section = ""
                for call, result in zip(tool_calls, tool_results):
                    tool_section += f"[{call.get('name', '')}:{call.get('args', '')}] → {result.get('output', '')}\n"
                structured_output = tool_section + "\n" + structured_output
            
            training_data.append({
                "instruction": input_text,
                "output": structured_output.strip(),
            })
        
        return training_data
    
    def force_dream(self, cycle_type: str = "fast") -> DreamCycleResult:
        """
        Force an immediate dream cycle.
        
        Args:
            cycle_type: "fast", "slow", or "full"
        """
        logger.info(f"Forcing dream cycle: {cycle_type}")
        
        results = []
        
        if cycle_type in ["fast", "full"]:
            replay_buffer = self._get_replay_buffer()
            samples = self._filter_samples(replay_buffer)
            if samples:
                self.state = DreamState.REM
                results.append(self._fast_weight_update(samples))
        
        if cycle_type in ["slow", "full"]:
            self.state = DreamState.DEEP_SLEEP
            results.append(self._slow_weight_update())
        
        self.state = DreamState.AWAKE
        
        # Return most recent result
        return results[-1] if results else DreamCycleResult(
            cycle_type=cycle_type,
            success=False,
            error_message="No operations performed",
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dreamer statistics."""
        return {
            **self.stats,
            "current_state": self.state.value,
            "is_running": self.is_running,
            "total_samples_processed": self.total_samples_processed,
            "last_fast_update": self.last_fast_update.isoformat() if self.last_fast_update else None,
            "last_slow_update": self.last_slow_update.isoformat() if self.last_slow_update else None,
            "dream_history_count": len(self.dream_history),
        }
    
    def get_dream_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent dream history."""
        return [r.to_dict() for r in self.dream_history[-limit:]]
