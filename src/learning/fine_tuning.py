"""
Fine-Tuning Scheduler for AVA

Manages periodic QLoRA fine-tuning cycles based on:
- Sample count thresholds
- Time since last fine-tune
- Developmental stage transitions
- Emotional state (high ambition + joy triggers training)

Also implements distillation pipelines for:
- Chain-of-Thought (CoT) reasoning transfer
- Toolformer-style tool-use distillation
- Self-distillation loops

Reference Papers:
- "QLoRA: Efficient Finetuning of Quantized LLMs" (arXiv:2305.14314)
- "Distilling Step-by-Step!" (ACL Findings, 2023)
- "Toolformer: Language Models Teach Themselves to Use Tools" (2023)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FineTuningTrigger(Enum):
    """Reasons for triggering fine-tuning."""
    SAMPLE_THRESHOLD = "sample_threshold"      # Enough samples accumulated
    TIME_THRESHOLD = "time_threshold"          # Enough time passed
    STAGE_TRANSITION = "stage_transition"      # Developmental stage changed
    EMOTIONAL_BOOST = "emotional_boost"        # High ambition + joy
    PERFORMANCE_DROP = "performance_drop"      # Detected degradation
    MANUAL = "manual"                          # User requested


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    
    # QLoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    
    # 4-bit quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio,
            "max_seq_length": self.max_seq_length,
            "use_4bit": self.use_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
        }


@dataclass
class FineTuningResult:
    """Result of a fine-tuning cycle."""
    
    id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    trigger: FineTuningTrigger = FineTuningTrigger.MANUAL
    samples_used: int = 0
    
    # Training metrics
    train_loss: float = 0.0
    eval_loss: float = 0.0
    epochs_completed: int = 0
    
    # Output
    adapter_path: str = ""
    success: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "trigger": self.trigger.value,
            "samples_used": self.samples_used,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "epochs_completed": self.epochs_completed,
            "adapter_path": self.adapter_path,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FineTuningResult":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            trigger=FineTuningTrigger(data.get("trigger", "manual")),
            samples_used=data.get("samples_used", 0),
            train_loss=data.get("train_loss", 0.0),
            eval_loss=data.get("eval_loss", 0.0),
            epochs_completed=data.get("epochs_completed", 0),
            adapter_path=data.get("adapter_path", ""),
            success=data.get("success", False),
            error_message=data.get("error_message", ""),
        )


class FineTuningScheduler:
    """
    Schedules and manages fine-tuning cycles.
    
    Decides when to trigger fine-tuning based on various conditions
    and orchestrates the training process.
    """
    
    def __init__(
        self,
        checkpoints_dir: str = "data/learning/checkpoints",
        adapters_dir: str = "models/fine_tuned_adapters",
        sample_threshold: int = 100,
        time_threshold_days: int = 7,
    ):
        """
        Initialize the fine-tuning scheduler.
        
        Args:
            checkpoints_dir: Directory for training checkpoints
            adapters_dir: Directory for trained adapters
            sample_threshold: Minimum samples to trigger training
            time_threshold_days: Days between automatic training
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.adapters_dir = Path(adapters_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_threshold = sample_threshold
        self.time_threshold = timedelta(days=time_threshold_days)
        
        self.config = FineTuningConfig()
        
        # Training history
        self.history: List[FineTuningResult] = []
        self.last_training: Optional[datetime] = None
        
        # Current state
        self.is_training = False
        self.current_run: Optional[FineTuningResult] = None
        
        self._load_history()
    
    def should_trigger(
        self,
        sample_count: int,
        developmental_stage: str,
        previous_stage: Optional[str] = None,
        emotional_state: Optional[Dict[str, float]] = None,
        performance_metric: Optional[float] = None,
    ) -> Optional[FineTuningTrigger]:
        """
        Check if fine-tuning should be triggered.
        
        Args:
            sample_count: Number of available training samples
            developmental_stage: Current stage
            previous_stage: Previous stage (if transition occurred)
            emotional_state: Current emotional state
            performance_metric: Recent performance (0-1, lower is worse)
            
        Returns:
            Trigger reason if should train, None otherwise
        """
        if self.is_training:
            return None
        
        # Check sample threshold
        if sample_count >= self.sample_threshold:
            return FineTuningTrigger.SAMPLE_THRESHOLD
        
        # Check time threshold
        if self.last_training:
            time_since = datetime.now() - self.last_training
            if time_since >= self.time_threshold and sample_count >= 50:
                return FineTuningTrigger.TIME_THRESHOLD
        elif sample_count >= 50:
            # Never trained before, but have enough samples
            return FineTuningTrigger.TIME_THRESHOLD
        
        # Check stage transition
        if previous_stage and previous_stage != developmental_stage:
            if sample_count >= 30:  # Lower threshold for stage transitions
                return FineTuningTrigger.STAGE_TRANSITION
        
        # Check emotional boost (high ambition + joy)
        if emotional_state:
            ambition = emotional_state.get("ambition", 0)
            joy = emotional_state.get("joy", 0)
            if ambition > 0.7 and joy > 0.6 and sample_count >= 30:
                return FineTuningTrigger.EMOTIONAL_BOOST
        
        # Check performance degradation
        if performance_metric is not None and performance_metric < 0.5:
            if sample_count >= 20:
                return FineTuningTrigger.PERFORMANCE_DROP
        
        return None
    
    def prepare_training_data(
        self,
        samples: List[Any],  # List[LearningSample]
    ) -> List[Dict[str, str]]:
        """
        Prepare training data from learning samples.
        
        Args:
            samples: Learning samples to use
            
        Returns:
            List of prompt/completion pairs
        """
        training_data = []
        
        for sample in samples:
            try:
                # Convert to training format
                data_point = sample.to_training_format()
                training_data.append(data_point)
            except Exception as e:
                logger.warning(f"Failed to convert sample {sample.id}: {e}")
        
        return training_data
    
    def start_training(
        self,
        training_data: List[Dict[str, str]],
        trigger: FineTuningTrigger,
        base_model: str = "ollama/llama3.2",
    ) -> FineTuningResult:
        """
        Start a fine-tuning run.
        
        Note: This is a placeholder implementation. In production,
        this would integrate with actual training infrastructure.
        
        Args:
            training_data: Prepared training data
            trigger: What triggered this training
            base_model: Base model to fine-tune
            
        Returns:
            FineTuningResult with status
        """
        import uuid
        
        self.is_training = True
        
        result = FineTuningResult(
            id=str(uuid.uuid4())[:8],
            trigger=trigger,
            samples_used=len(training_data),
        )
        self.current_run = result
        
        logger.info(f"Starting fine-tuning run {result.id} with {len(training_data)} samples")
        
        try:
            # Save training data
            training_file = self.checkpoints_dir / f"training_data_{result.id}.json"
            with open(training_file, "w") as f:
                json.dump(training_data, f, indent=2)
            
            # In a real implementation, this would:
            # 1. Load the base model with 4-bit quantization
            # 2. Setup LoRA adapters
            # 3. Train for specified epochs
            # 4. Save the trained adapter
            
            # For now, we'll create a placeholder result
            adapter_path = self.adapters_dir / f"adapter_{result.id}"
            adapter_path.mkdir(exist_ok=True)
            
            # Save adapter config (placeholder)
            adapter_config = {
                "base_model": base_model,
                "training_run": result.id,
                "samples_used": len(training_data),
                "config": self.config.to_dict(),
                "created_at": datetime.now().isoformat(),
            }
            
            with open(adapter_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f, indent=2)
            
            # Update result
            result.completed_at = datetime.now()
            result.adapter_path = str(adapter_path)
            result.success = True
            result.epochs_completed = self.config.num_epochs
            
            # Simulated losses (in real implementation, these come from training)
            result.train_loss = 0.5
            result.eval_loss = 0.6
            
            logger.info(f"Fine-tuning run {result.id} completed successfully")
            
        except Exception as e:
            result.completed_at = datetime.now()
            result.success = False
            result.error_message = str(e)
            logger.error(f"Fine-tuning run {result.id} failed: {e}")
        
        finally:
            self.is_training = False
            self.current_run = None
            self.last_training = datetime.now()
            
            # Add to history
            self.history.append(result)
            self._save_history()
        
        return result
    
    def get_latest_adapter(self) -> Optional[str]:
        """Get path to the latest trained adapter."""
        successful_runs = [r for r in self.history if r.success]
        
        if not successful_runs:
            return None
        
        latest = max(successful_runs, key=lambda r: r.completed_at or r.started_at)
        
        if latest.adapter_path and Path(latest.adapter_path).exists():
            return latest.adapter_path
        
        return None
    
    def get_training_history(self, limit: int = 10) -> List[FineTuningResult]:
        """Get recent training history."""
        return sorted(
            self.history,
            key=lambda r: r.started_at,
            reverse=True,
        )[:limit]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "is_training": self.is_training,
            "current_run": self.current_run.to_dict() if self.current_run else None,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "total_training_runs": len(self.history),
            "successful_runs": sum(1 for r in self.history if r.success),
            "latest_adapter": self.get_latest_adapter(),
            "config": self.config.to_dict(),
            "thresholds": {
                "sample_threshold": self.sample_threshold,
                "time_threshold_days": self.time_threshold.days,
            },
        }
    
    def update_config(self, **kwargs):
        """Update fine-tuning configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _load_history(self):
        """Load training history from disk."""
        history_file = self.checkpoints_dir / "training_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    self.history = [FineTuningResult.from_dict(r) for r in data.get("runs", [])]
                    
                    if data.get("last_training"):
                        self.last_training = datetime.fromisoformat(data["last_training"])
                        
            except Exception as e:
                logger.warning(f"Failed to load training history: {e}")
    
    def _save_history(self):
        """Save training history to disk."""
        history_file = self.checkpoints_dir / "training_history.json"
        
        try:
            data = {
                "runs": [r.to_dict() for r in self.history],
                "last_training": self.last_training.isoformat() if self.last_training else None,
            }
            
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")


# ==============================================================================
# DISTILLATION PIPELINE (Distilling Step-by-Step + Toolformer Integration)
# ==============================================================================

@dataclass
class DistillationSample:
    """
    A single distillation sample containing input, rationale, and tool calls.
    
    Reference: "Distilling Step-by-Step!" (ACL Findings 2023) emphasizes
    extracting intermediate rationales, not just final predictions.
    """
    input_text: str
    output_text: str
    rationale: Optional[str] = None  # Chain-of-thought reasoning
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0  # Perplexity-based quality metric
    teacher_model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_training_format(self) -> Dict[str, str]:
        """
        Convert to training format suitable for QLoRA fine-tuning.
        
        If rationale exists, include it in the expected output format.
        If tool calls exist, include them in Toolformer-style format.
        """
        # Build structured output with CoT if available
        structured_output = ""
        
        if self.rationale:
            structured_output += f"<think>\n{self.rationale}\n</think>\n"
        
        # Include tool calls in Toolformer-style format
        for i, (call, result) in enumerate(zip(self.tool_calls, self.tool_results)):
            tool_name = call.get("name", "Unknown")
            tool_args = call.get("args", "")
            tool_output = result.get("output", "")
            structured_output += f"[{tool_name}:{tool_args}] → {tool_output}\n"
        
        structured_output += f"\n{self.output_text}"
        
        return {
            "instruction": self.input_text,
            "output": structured_output.strip(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "rationale": self.rationale,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "quality_score": self.quality_score,
            "teacher_model": self.teacher_model,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistillationSample":
        return cls(**data)


@dataclass
class DistillationConfig:
    """Configuration for distillation pipeline."""
    
    # Teacher model settings
    teacher_model: str = "llama3.2:latest"  # Default Ollama model
    teacher_temperature: float = 0.7
    
    # Quality filtering
    min_quality_score: float = 0.6  # Minimum perplexity-based score
    max_samples_per_iteration: int = 100
    
    # Toolformer distillation
    enable_tool_distillation: bool = True
    tool_augmentation_threshold: float = 0.05  # Perplexity improvement threshold
    
    # CoT distillation
    enable_cot_distillation: bool = True
    cot_prompt_template: str = "Think step by step before answering."
    
    # Self-distillation (student teaches itself on filtered samples)
    enable_self_distillation: bool = True
    self_distill_iterations: int = 3
    
    # Storage
    samples_dir: str = "data/learning/distillation_samples"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_model": self.teacher_model,
            "teacher_temperature": self.teacher_temperature,
            "min_quality_score": self.min_quality_score,
            "max_samples_per_iteration": self.max_samples_per_iteration,
            "enable_tool_distillation": self.enable_tool_distillation,
            "tool_augmentation_threshold": self.tool_augmentation_threshold,
            "enable_cot_distillation": self.enable_cot_distillation,
            "cot_prompt_template": self.cot_prompt_template,
            "enable_self_distillation": self.enable_self_distillation,
            "self_distill_iterations": self.self_distill_iterations,
            "samples_dir": self.samples_dir,
        }


class DistillationPipeline:
    """
    Distillation pipeline for transferring knowledge from teacher to student.
    
    Implements three distillation strategies:
    1. Chain-of-Thought Distillation: Extract reasoning chains from teacher
    2. Tool-Use Distillation: Learn when/how to call tools (Toolformer-style)
    3. Self-Distillation: Student improves on its own high-quality outputs
    
    Reference Papers:
    - "Distilling Step-by-Step!" (Hsieh et al., 2023): Uses rationale extraction
      to achieve SOTA with 770x less data than full fine-tuning
    - "Toolformer" (Schick et al., 2023): Self-supervised tool use learning
    """
    
    def __init__(
        self,
        config: Optional[DistillationConfig] = None,
        ollama_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
    ):
        self.config = config or DistillationConfig()
        self.ollama_client = ollama_client
        self.tool_registry = tool_registry
        
        # Storage
        self.samples_dir = Path(self.config.samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Current batch of samples
        self.pending_samples: List[DistillationSample] = []
        self.iteration_count: int = 0
        
        # Statistics
        self.stats = {
            "total_samples_collected": 0,
            "samples_filtered_quality": 0,
            "tool_augmentations_successful": 0,
            "cot_extractions": 0,
            "self_distill_iterations": 0,
        }
        
        logger.info(f"DistillationPipeline initialized with config: {self.config.to_dict()}")
    
    async def collect_cot_sample(
        self,
        input_text: str,
        context: Optional[str] = None,
    ) -> Optional[DistillationSample]:
        """
        Collect a Chain-of-Thought sample from the teacher model.
        
        Uses explicit CoT prompting to extract intermediate reasoning steps.
        
        Reference: "Distilling Step-by-Step!" shows that rationale extraction
        significantly improves student model performance on reasoning tasks.
        """
        if not self.config.enable_cot_distillation:
            return None
        
        if not self.ollama_client:
            logger.warning("No Ollama client available for CoT collection")
            return None
        
        try:
            # Create CoT prompt
            cot_prompt = self._create_cot_prompt(input_text, context)
            
            # Query teacher model
            response = await self._query_teacher(cot_prompt)
            
            if not response:
                return None
            
            # Parse CoT response
            rationale, final_answer = self._parse_cot_response(response)
            
            # Score quality
            quality_score = await self._score_sample_quality(input_text, response)
            
            if quality_score < self.config.min_quality_score:
                self.stats["samples_filtered_quality"] += 1
                return None
            
            sample = DistillationSample(
                input_text=input_text,
                output_text=final_answer,
                rationale=rationale,
                quality_score=quality_score,
                teacher_model=self.config.teacher_model,
            )
            
            self.stats["cot_extractions"] += 1
            self.pending_samples.append(sample)
            
            return sample
            
        except Exception as e:
            logger.error(f"Failed to collect CoT sample: {e}")
            return None
    
    async def collect_tool_augmented_sample(
        self,
        input_text: str,
        base_output: str,
        available_tools: Optional[List[str]] = None,
    ) -> Optional[DistillationSample]:
        """
        Collect a tool-augmented sample in Toolformer style.
        
        Process:
        1. Given base output, identify positions where tool calls could help
        2. Insert candidate tool calls
        3. Execute tools and compare perplexity
        4. Keep augmentation if it improves perplexity beyond threshold
        
        Reference: Toolformer Section 3 describes the self-supervised
        augmentation process that enables tool learning without manual annotation.
        """
        if not self.config.enable_tool_distillation:
            return None
        
        if not self.tool_registry:
            logger.warning("No tool registry available for tool distillation")
            return None
        
        try:
            # Get available tools
            tools = available_tools or self.tool_registry.get_available_tools()
            
            # Find tool insertion points
            tool_proposals = await self._propose_tool_insertions(input_text, base_output, tools)
            
            if not tool_proposals:
                return None
            
            # Execute proposed tools
            tool_calls = []
            tool_results = []
            
            for proposal in tool_proposals:
                try:
                    result = await self._execute_tool(proposal)
                    if result:
                        tool_calls.append(proposal)
                        tool_results.append(result)
                except Exception as e:
                    logger.debug(f"Tool execution failed: {e}")
                    continue
            
            if not tool_calls:
                return None
            
            # Score augmentation quality
            augmented_output = self._create_augmented_output(base_output, tool_calls, tool_results)
            
            base_score = await self._score_sample_quality(input_text, base_output)
            augmented_score = await self._score_sample_quality(input_text, augmented_output)
            
            # Only keep if augmentation improves beyond threshold
            improvement = augmented_score - base_score
            if improvement < self.config.tool_augmentation_threshold:
                return None
            
            sample = DistillationSample(
                input_text=input_text,
                output_text=augmented_output,
                tool_calls=tool_calls,
                tool_results=tool_results,
                quality_score=augmented_score,
                teacher_model=self.config.teacher_model,
            )
            
            self.stats["tool_augmentations_successful"] += 1
            self.pending_samples.append(sample)
            
            return sample
            
        except Exception as e:
            logger.error(f"Failed to collect tool-augmented sample: {e}")
            return None
    
    async def self_distillation_iteration(
        self,
        student_model: str = "llama3.2:latest",
    ) -> List[DistillationSample]:
        """
        Perform one iteration of self-distillation.
        
        Process:
        1. Take existing high-quality samples
        2. Have student generate on same inputs
        3. Score student outputs
        4. Add student outputs that exceed quality threshold
        
        This creates a virtuous cycle where the student improves
        by learning from its own best outputs (filtered by quality).
        
        Reference: Similar to "Self-Improving Language Models" concept
        where filtering on verifiable dimensions enables improvement.
        """
        if not self.config.enable_self_distillation:
            return []
        
        if self.iteration_count >= self.config.self_distill_iterations:
            logger.info("Max self-distillation iterations reached")
            return []
        
        try:
            # Load existing samples
            existing_samples = self._load_samples()
            
            if not existing_samples:
                logger.info("No existing samples for self-distillation")
                return []
            
            new_samples = []
            
            for sample in existing_samples[:self.config.max_samples_per_iteration]:
                # Generate student response
                student_response = await self._query_model(
                    sample.input_text,
                    model=student_model,
                )
                
                if not student_response:
                    continue
                
                # Score student output
                student_score = await self._score_sample_quality(
                    sample.input_text,
                    student_response,
                )
                
                # Only add if quality exceeds threshold
                if student_score >= self.config.min_quality_score:
                    new_sample = DistillationSample(
                        input_text=sample.input_text,
                        output_text=student_response,
                        rationale=sample.rationale,  # Preserve original rationale
                        quality_score=student_score,
                        teacher_model=student_model,  # Student is now teacher
                    )
                    new_samples.append(new_sample)
            
            # Add to pending samples
            self.pending_samples.extend(new_samples)
            self.iteration_count += 1
            self.stats["self_distill_iterations"] += 1
            
            logger.info(f"Self-distillation iteration {self.iteration_count}: {len(new_samples)} new samples")
            
            return new_samples
            
        except Exception as e:
            logger.error(f"Self-distillation iteration failed: {e}")
            return []
    
    def prepare_training_data(self) -> Tuple[str, int]:
        """
        Prepare distillation samples for QLoRA fine-tuning.
        
        Returns:
            Tuple of (path to training data JSON, number of samples)
        """
        # Deduplicate and filter samples
        filtered_samples = self._filter_samples(self.pending_samples)
        
        if not filtered_samples:
            logger.warning("No samples to prepare for training")
            return "", 0
        
        # Convert to training format
        training_data = [sample.to_training_format() for sample in filtered_samples]
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.samples_dir / f"distillation_train_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)
        
        # Clear pending samples
        self.stats["total_samples_collected"] += len(filtered_samples)
        self.pending_samples = []
        
        logger.info(f"Prepared {len(training_data)} samples for training: {output_file}")
        
        return str(output_file), len(training_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distillation statistics."""
        return {
            **self.stats,
            "pending_samples": len(self.pending_samples),
            "iteration_count": self.iteration_count,
            "config": self.config.to_dict(),
        }
    
    # ==================== Private Helper Methods ====================
    
    def _create_cot_prompt(self, input_text: str, context: Optional[str] = None) -> str:
        """Create a Chain-of-Thought prompt."""
        prompt = f"{self.config.cot_prompt_template}\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += f"Question: {input_text}\n\n"
        prompt += "Let me think through this step by step:\n"
        
        return prompt
    
    def _parse_cot_response(self, response: str) -> Tuple[str, str]:
        """
        Parse CoT response into rationale and final answer.
        
        Looks for explicit markers or infers from structure.
        """
        # Check for explicit think tags
        if "<think>" in response and "</think>" in response:
            start = response.index("<think>") + len("<think>")
            end = response.index("</think>")
            rationale = response[start:end].strip()
            final_answer = response[end + len("</think>"):].strip()
            return rationale, final_answer
        
        # Check for "Therefore" or "So" markers
        markers = ["Therefore,", "So,", "Thus,", "In conclusion,", "The answer is"]
        
        for marker in markers:
            if marker in response:
                parts = response.split(marker, 1)
                rationale = parts[0].strip()
                final_answer = marker + parts[1].strip() if len(parts) > 1 else ""
                return rationale, final_answer
        
        # No clear separation - treat entire response as answer
        return "", response
    
    async def _query_teacher(self, prompt: str) -> Optional[str]:
        """Query the teacher model."""
        return await self._query_model(
            prompt,
            model=self.config.teacher_model,
            temperature=self.config.teacher_temperature,
        )
    
    async def _query_model(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Query an Ollama model."""
        if not self.ollama_client:
            return None
        
        try:
            # This assumes ollama_client has a generate method
            # Actual implementation depends on AVA's Ollama wrapper
            response = await self.ollama_client.generate(
                model=model or self.config.teacher_model,
                prompt=prompt,
                temperature=temperature,
            )
            return response.get("response", "") if isinstance(response, dict) else str(response)
        except Exception as e:
            logger.error(f"Model query failed: {e}")
            return None
    
    async def _score_sample_quality(self, input_text: str, output: str) -> float:
        """
        Score sample quality using perplexity-based metric.
        
        Lower perplexity = higher quality score.
        """
        # Simple heuristic scoring (actual implementation would use model perplexity)
        score = 0.5
        
        # Length bonus (prefer detailed responses)
        if len(output) > 100:
            score += 0.1
        if len(output) > 300:
            score += 0.1
        
        # Structure bonus (has reasoning markers)
        reasoning_markers = ["because", "since", "therefore", "step", "first", "then"]
        for marker in reasoning_markers:
            if marker.lower() in output.lower():
                score += 0.05
        
        # Relevance penalty (very short or generic)
        if len(output) < 20:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _propose_tool_insertions(
        self,
        input_text: str,
        output: str,
        tools: List[str],
    ) -> List[Dict[str, Any]]:
        """Propose tool insertions for output."""
        proposals = []
        
        # Simple heuristic tool proposal
        # Actual implementation would use model to identify insertion points
        
        tool_triggers = {
            "Calculator": ["calculate", "compute", "math", "number", "sum", "multiply"],
            "Search": ["search", "find", "lookup", "information about", "who is", "what is"],
            "Time": ["time", "date", "today", "current"],
            "Define": ["define", "meaning", "definition", "what does"],
        }
        
        lower_input = input_text.lower()
        
        for tool_name, triggers in tool_triggers.items():
            if tool_name in tools:
                for trigger in triggers:
                    if trigger in lower_input:
                        proposals.append({
                            "name": tool_name,
                            "args": input_text,
                            "position": 0,  # Beginning of output
                        })
                        break
        
        return proposals[:3]  # Limit proposals
    
    async def _execute_tool(self, proposal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a tool call proposal."""
        if not self.tool_registry:
            return None
        
        try:
            tool_name = proposal.get("name")
            tool_args = proposal.get("args", "")
            
            result = await self.tool_registry.execute(tool_name, tool_args)
            
            return {
                "name": tool_name,
                "output": str(result),
                "success": True,
            }
        except Exception as e:
            logger.debug(f"Tool execution failed: {e}")
            return None
    
    def _create_augmented_output(
        self,
        base_output: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """Create tool-augmented output."""
        augmented = ""
        
        for call, result in zip(tool_calls, tool_results):
            tool_name = call.get("name", "")
            tool_args = call.get("args", "")
            tool_output = result.get("output", "")
            augmented += f"[{tool_name}:{tool_args}] → {tool_output}\n"
        
        augmented += "\n" + base_output
        return augmented.strip()
    
    def _filter_samples(
        self,
        samples: List[DistillationSample],
    ) -> List[DistillationSample]:
        """Filter and deduplicate samples."""
        # Remove low quality
        filtered = [s for s in samples if s.quality_score >= self.config.min_quality_score]
        
        # Deduplicate by input text
        seen_inputs = set()
        unique_samples = []
        
        for sample in filtered:
            input_hash = hash(sample.input_text[:100])  # Use prefix for hash
            if input_hash not in seen_inputs:
                seen_inputs.add(input_hash)
                unique_samples.append(sample)
        
        return unique_samples[:self.config.max_samples_per_iteration]
    
    def _save_samples(self):
        """Save pending samples to disk."""
        if not self.pending_samples:
            return
        
        samples_file = self.samples_dir / "pending_samples.json"
        
        with open(samples_file, "w") as f:
            json.dump(
                [s.to_dict() for s in self.pending_samples],
                f,
                indent=2,
            )
    
    def _load_samples(self) -> List[DistillationSample]:
        """Load samples from disk."""
        samples_file = self.samples_dir / "pending_samples.json"
        
        if not samples_file.exists():
            return []
        
        try:
            with open(samples_file, "r") as f:
                data = json.load(f)
                return [DistillationSample.from_dict(s) for s in data]
        except Exception as e:
            logger.warning(f"Failed to load samples: {e}")
            return []


def create_qlora_training_script(
    config: FineTuningConfig,
    training_data_path: str,
    output_dir: str,
    base_model: str,
) -> str:
    """
    Generate a QLoRA training script.
    
    This creates a Python script that can be run to perform
    actual QLoRA fine-tuning with the given configuration.
    
    Args:
        config: Fine-tuning configuration
        training_data_path: Path to training data JSON
        output_dir: Where to save the adapter
        base_model: Base model name
        
    Returns:
        The training script as a string
    """
    script = f'''"""
QLoRA Fine-Tuning Script for AVA
Auto-generated by FineTuningScheduler
"""

import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Configuration
BASE_MODEL = "{base_model}"
TRAINING_DATA = "{training_data_path}"
OUTPUT_DIR = "{output_dir}"

# QLoRA config
LORA_R = {config.lora_r}
LORA_ALPHA = {config.lora_alpha}
LORA_DROPOUT = {config.lora_dropout}
TARGET_MODULES = {config.target_modules}

# Training config
BATCH_SIZE = {config.batch_size}
GRADIENT_ACCUMULATION = {config.gradient_accumulation_steps}
LEARNING_RATE = {config.learning_rate}
NUM_EPOCHS = {config.num_epochs}
WARMUP_RATIO = {config.warmup_ratio}
MAX_SEQ_LENGTH = {config.max_seq_length}


def main():
    # Load training data
    with open(TRAINING_DATA, "r") as f:
        raw_data = json.load(f)
    
    # Convert to dataset format
    texts = [d["prompt"] + d["completion"] for d in raw_data]
    dataset = Dataset.from_dict({{"text": texts}})
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="{config.bnb_4bit_quant_type}",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Tokenize dataset
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    trainer.train()
    
    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {{OUTPUT_DIR}}")


if __name__ == "__main__":
    main()
'''
    return script
