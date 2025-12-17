"""
THE NIGHTMARE ENGINE: Subconscious Consolidation System
=======================================================

"Where memories become knowledge."

The Nightmare Engine is AVA's offline learning system that runs during
"sleep" periods (when AVA is idle). It consolidates high-value experiences
from the episodic buffer into long-term knowledge via QLoRA fine-tuning.

Architecture:
1. Sleep Phase: Triggered after periods of inactivity
2. Episode Sampling: Select high-surprise, high-quality episodes
3. Data Augmentation: Generate CoT reasoning and tool usage
4. QLoRA Training: Fine-tune on augmented data
5. Wake Phase: Merge adapters back into active model

The name "Nightmare" comes from the idea that learning happens through
intensive replay of significant (sometimes challenging) experiences.

Reference:
- Sleep Replay in Biological Learning
- QLoRA: Efficient Finetuning of Quantized LLMs (2023)
- Experience Replay (Mnih et al., 2015)
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SleepPhase(Enum):
    """Phases of the sleep/consolidation cycle."""
    AWAKE = auto()           # Normal operation
    DROWSY = auto()          # Preparing for sleep
    LIGHT_SLEEP = auto()     # Fast weight updates
    DEEP_SLEEP = auto()      # Slow weight consolidation
    REM = auto()             # Intensive replay (QLoRA training)
    WAKING = auto()          # Transitioning back to awake


@dataclass
class NightmareConfig:
    """
    Configuration for the Nightmare Engine.
    
    Attributes:
        idle_threshold_minutes: Minutes of inactivity before sleep
        min_episodes_for_training: Minimum episodes needed to train
        batch_size: Training batch size
        fast_epochs: Epochs for fast weight update
        slow_epochs: Epochs for slow weight consolidation
        fast_rank: LoRA rank for fast updates
        slow_rank: LoRA rank for slow consolidation
        learning_rate: Base learning rate
        output_dir: Directory for adapter outputs
        min_surprise: Minimum surprise for episode selection
        min_quality: Minimum quality for episode selection
        max_training_time_minutes: Maximum training time per cycle
    """
    idle_threshold_minutes: int = 30
    min_episodes_for_training: int = 10
    batch_size: int = 4
    fast_epochs: int = 2
    slow_epochs: int = 5
    fast_rank: int = 8
    slow_rank: int = 64
    learning_rate: float = 1e-4
    output_dir: str = "models/fine_tuned_adapters/nightmare"
    min_surprise: float = 1.0
    min_quality: float = 0.6
    max_training_time_minutes: int = 60
    
    # Sleep cycle timing
    light_sleep_duration_minutes: int = 5
    deep_sleep_duration_minutes: int = 15
    rem_duration_minutes: int = 30
    
    # Memory consolidation
    replay_batch_size: int = 32
    priority_alpha: float = 0.6


@dataclass
class SleepCycleStats:
    """Statistics from a sleep cycle."""
    cycle_id: str
    start_time: str
    end_time: Optional[str] = None
    phase: SleepPhase = SleepPhase.AWAKE
    episodes_processed: int = 0
    training_loss: float = 0.0
    fast_updates: int = 0
    slow_updates: int = 0
    adapter_path: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "phase": self.phase.name,
            "episodes_processed": self.episodes_processed,
            "training_loss": self.training_loss,
            "fast_updates": self.fast_updates,
            "slow_updates": self.slow_updates,
            "adapter_path": self.adapter_path,
            "error": self.error,
        }


class NightmareEngine:
    """
    The Nightmare Engine - Offline consolidation via QLoRA.
    
    This system runs in the background during idle periods,
    consolidating high-value experiences into long-term knowledge.
    
    Usage:
        from src.hippocampus import EpisodicBuffer
        from src.subconscious import NightmareEngine
        
        buffer = EpisodicBuffer()
        nightmare = NightmareEngine(buffer)
        
        # Start background sleep monitoring
        nightmare.start_background_monitoring()
        
        # Or trigger manual sleep
        stats = nightmare.dream()
        
        # Check status
        print(nightmare.current_phase)
    """
    
    def __init__(
        self,
        episodic_buffer,  # EpisodicBuffer instance
        config: Optional[NightmareConfig] = None,
        qlora_trainer=None,  # QLoRATrainer instance
        on_phase_change: Optional[Callable[[SleepPhase], None]] = None,
    ):
        """
        Initialize the Nightmare Engine.
        
        Args:
            episodic_buffer: EpisodicBuffer for episode sampling
            config: Nightmare configuration
            qlora_trainer: QLoRATrainer for fine-tuning (lazy-loaded if None)
            on_phase_change: Callback when sleep phase changes
        """
        self.buffer = episodic_buffer
        self.config = config or NightmareConfig()
        self.qlora_trainer = qlora_trainer
        self.on_phase_change = on_phase_change
        
        # State
        self._current_phase = SleepPhase.AWAKE
        self._last_activity = datetime.now()
        self._sleep_cycle_stats: List[SleepCycleStats] = []
        self._current_stats: Optional[SleepCycleStats] = None
        
        # Background monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._is_sleeping = threading.Event()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info("NightmareEngine initialized")
    
    @property
    def current_phase(self) -> SleepPhase:
        """Get current sleep phase."""
        return self._current_phase
    
    def _set_phase(self, phase: SleepPhase):
        """Set sleep phase with callback."""
        if phase != self._current_phase:
            old_phase = self._current_phase
            self._current_phase = phase
            
            if self._current_stats:
                self._current_stats.phase = phase
            
            logger.info(f"Sleep phase: {old_phase.name} → {phase.name}")
            
            if self.on_phase_change:
                try:
                    self.on_phase_change(phase)
                except Exception as e:
                    logger.error(f"Error in phase change callback: {e}")
    
    def record_activity(self):
        """Record activity (resets idle timer)."""
        self._last_activity = datetime.now()
    
    def get_idle_time(self) -> timedelta:
        """Get time since last activity."""
        return datetime.now() - self._last_activity
    
    def should_sleep(self) -> bool:
        """Check if conditions are met for sleep."""
        idle_minutes = self.get_idle_time().total_seconds() / 60
        
        if idle_minutes < self.config.idle_threshold_minutes:
            return False
        
        # Check if we have enough episodes
        stats = self.buffer.get_statistics()
        if stats["total_episodes"] < self.config.min_episodes_for_training:
            return False
        
        return True
    
    def start_background_monitoring(self):
        """Start background thread to monitor for sleep conditions."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Background monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="NightmareEngine-Monitor",
        )
        self._monitor_thread.start()
        logger.info("Background sleep monitoring started")
    
    def stop_background_monitoring(self):
        """Stop background monitoring thread."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Background sleep monitoring stopped")
    
    def _monitoring_loop(self):
        """Background loop that monitors for sleep conditions."""
        while not self._stop_monitoring.is_set():
            try:
                if self.should_sleep() and not self._is_sleeping.is_set():
                    logger.info("Sleep conditions met, initiating dream cycle")
                    self._is_sleeping.set()
                    
                    try:
                        self.dream()
                    finally:
                        self._is_sleeping.clear()
                
                # Check every minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def dream(self) -> SleepCycleStats:
        """
        Execute a full sleep/consolidation cycle.
        
        This is the main entry point for offline learning.
        It runs through all sleep phases sequentially.
        
        Returns:
            SleepCycleStats with cycle results
        """
        import uuid
        
        # Initialize cycle stats
        self._current_stats = SleepCycleStats(
            cycle_id=str(uuid.uuid4())[:8],
            start_time=datetime.now().isoformat(),
            phase=SleepPhase.DROWSY,
        )
        
        try:
            # Phase 1: DROWSY - Prepare for sleep
            self._set_phase(SleepPhase.DROWSY)
            episodes = self._prepare_for_sleep()
            
            if not episodes:
                logger.info("No suitable episodes for consolidation")
                self._current_stats.error = "No suitable episodes"
                self._set_phase(SleepPhase.AWAKE)
                return self._finalize_stats()
            
            # Phase 2: LIGHT_SLEEP - Fast weight updates
            self._set_phase(SleepPhase.LIGHT_SLEEP)
            self._light_sleep_phase(episodes)
            
            # Phase 3: DEEP_SLEEP - Slow weight consolidation
            self._set_phase(SleepPhase.DEEP_SLEEP)
            self._deep_sleep_phase(episodes)
            
            # Phase 4: REM - Intensive replay (QLoRA)
            self._set_phase(SleepPhase.REM)
            self._rem_phase(episodes)
            
            # Phase 5: WAKING - Finalize and return
            self._set_phase(SleepPhase.WAKING)
            self._waking_phase()
            
        except Exception as e:
            logger.error(f"Error during dream cycle: {e}")
            self._current_stats.error = str(e)
            
        finally:
            self._set_phase(SleepPhase.AWAKE)
        
        return self._finalize_stats()
    
    def _prepare_for_sleep(self) -> List[Any]:
        """
        Prepare for sleep: sample high-priority episodes.
        
        Returns:
            List of episodes for consolidation
        """
        logger.info("Preparing for sleep - sampling episodes")
        
        episodes = self.buffer.sample(
            batch_size=self.config.replay_batch_size * 4,  # Sample more, filter later
            min_surprise=self.config.min_surprise,
            prioritized=True,
        )
        
        # Filter by quality
        episodes = [
            ep for ep in episodes 
            if ep.quality_score >= self.config.min_quality
        ]
        
        logger.info(f"Selected {len(episodes)} episodes for consolidation")
        self._current_stats.episodes_processed = len(episodes)
        
        return episodes
    
    def _light_sleep_phase(self, episodes: List[Any]):
        """
        Light sleep: fast weight updates.
        
        Rapid adaptation to recent experiences using low-rank updates.
        """
        logger.info(f"Light sleep: fast weight updates ({len(episodes)} episodes)")
        
        # In a real implementation, this would update the Titans sidecar
        # with recent high-surprise events
        for ep in episodes[:self.config.batch_size]:
            # Simulate fast weight update
            self._current_stats.fast_updates += 1
        
        # Brief pause to simulate processing
        time.sleep(min(
            self.config.light_sleep_duration_minutes * 60,
            30  # Cap at 30 seconds for responsiveness
        ))
    
    def _deep_sleep_phase(self, episodes: List[Any]):
        """
        Deep sleep: slow weight consolidation.
        
        Consolidate important patterns into long-term memory.
        """
        logger.info(f"Deep sleep: slow weight consolidation")
        
        # In a real implementation, this would run longer-term
        # consolidation in the nested learning system
        self._current_stats.slow_updates = len(episodes)
        
        # Simulate processing time
        time.sleep(min(
            self.config.deep_sleep_duration_minutes * 60,
            60  # Cap at 60 seconds
        ))
    
    def _rem_phase(self, episodes: List[Any]):
        """
        REM sleep: intensive replay with QLoRA training.
        
        This is where the actual fine-tuning happens.
        """
        logger.info(f"REM sleep: QLoRA training on {len(episodes)} episodes")
        
        if len(episodes) < self.config.min_episodes_for_training:
            logger.warning("Not enough episodes for QLoRA training")
            return
        
        # Prepare training data
        training_data = self._prepare_training_data(episodes)
        
        if not training_data:
            logger.warning("No training data generated")
            return
        
        # Save training data
        training_path = os.path.join(
            self.config.output_dir,
            f"training_data_{self._current_stats.cycle_id}.jsonl"
        )
        self._save_training_data(training_data, training_path)
        
        # Run QLoRA training if trainer is available
        if self.qlora_trainer:
            try:
                adapter_path = os.path.join(
                    self.config.output_dir,
                    f"adapter_{self._current_stats.cycle_id}"
                )
                
                loss = self.qlora_trainer.train(
                    training_path,
                    output_dir=adapter_path,
                    num_epochs=self.config.slow_epochs,
                )
                
                self._current_stats.training_loss = loss
                self._current_stats.adapter_path = adapter_path
                
                logger.info(f"QLoRA training complete. Loss: {loss:.4f}")
                
            except Exception as e:
                logger.error(f"QLoRA training failed: {e}")
                self._current_stats.error = f"Training failed: {e}"
        else:
            logger.info("No QLoRA trainer available - skipping actual training")
            logger.info(f"Training data saved to: {training_path}")
        
        # Mark episodes as replayed
        episode_ids = [ep.episode_id for ep in episodes]
        self.buffer.mark_replayed(episode_ids)
    
    def _prepare_training_data(
        self,
        episodes: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Prepare training data from episodes.
        
        Converts episodes to prompt/response format with optional
        chain-of-thought augmentation.
        """
        training_data = []
        
        for ep in episodes:
            # Basic format
            item = {
                "instruction": ep.prompt,
                "output": ep.response,
            }
            
            # Add tool calls if present
            if ep.used_tools and ep.tool_calls:
                item["tools"] = ep.tool_calls
            
            # Add metadata for filtering
            item["_metadata"] = {
                "surprise": ep.surprise,
                "quality": ep.quality_score,
                "cognitive_state": ep.cognitive_state,
            }
            
            training_data.append(item)
        
        logger.info(f"Prepared {len(training_data)} training samples")
        return training_data
    
    def _save_training_data(
        self,
        data: List[Dict[str, Any]],
        path: str,
    ):
        """Save training data to JSONL file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            for item in data:
                # Remove metadata for actual training
                clean_item = {k: v for k, v in item.items() if not k.startswith('_')}
                f.write(json.dumps(clean_item) + '\n')
        
        logger.info(f"Training data saved to {path}")
    
    def _waking_phase(self):
        """
        Waking phase: finalize consolidation.
        
        Optionally merge adapters back into base model.
        """
        logger.info("Waking phase: finalizing consolidation")
        
        # In a full implementation, this would:
        # 1. Merge QLoRA adapter if training was successful
        # 2. Update the active model weights
        # 3. Clear temporary training files
        
        if self._current_stats.adapter_path:
            logger.info(f"Adapter available at: {self._current_stats.adapter_path}")
    
    def _finalize_stats(self) -> SleepCycleStats:
        """Finalize and return sleep cycle statistics."""
        if self._current_stats:
            self._current_stats.end_time = datetime.now().isoformat()
            self._sleep_cycle_stats.append(self._current_stats)
            
            # Keep last 100 cycles
            if len(self._sleep_cycle_stats) > 100:
                self._sleep_cycle_stats.pop(0)
        
        stats = self._current_stats
        self._current_stats = None
        
        return stats
    
    def get_sleep_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sleep cycle history."""
        return [s.to_dict() for s in self._sleep_cycle_stats[-limit:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall nightmare engine statistics."""
        total_cycles = len(self._sleep_cycle_stats)
        
        if not self._sleep_cycle_stats:
            return {
                "total_cycles": 0,
                "current_phase": self._current_phase.name,
                "idle_minutes": self.get_idle_time().total_seconds() / 60,
            }
        
        successful = [s for s in self._sleep_cycle_stats if not s.error]
        avg_loss = sum(s.training_loss for s in successful) / max(1, len(successful))
        total_episodes = sum(s.episodes_processed for s in self._sleep_cycle_stats)
        
        return {
            "total_cycles": total_cycles,
            "successful_cycles": len(successful),
            "failed_cycles": total_cycles - len(successful),
            "avg_training_loss": avg_loss,
            "total_episodes_processed": total_episodes,
            "current_phase": self._current_phase.name,
            "idle_minutes": self.get_idle_time().total_seconds() / 60,
            "should_sleep": self.should_sleep(),
        }


# Factory function
def create_nightmare_engine(
    episodic_buffer,
    idle_threshold_minutes: int = 30,
    output_dir: str = "models/fine_tuned_adapters/nightmare",
) -> NightmareEngine:
    """
    Factory function to create a NightmareEngine.
    
    Args:
        episodic_buffer: EpisodicBuffer for episode sampling
        idle_threshold_minutes: Minutes before sleep triggers
        output_dir: Directory for adapter outputs
        
    Returns:
        Configured NightmareEngine
    """
    config = NightmareConfig(
        idle_threshold_minutes=idle_threshold_minutes,
        output_dir=output_dir,
    )
    
    return NightmareEngine(episodic_buffer, config)
