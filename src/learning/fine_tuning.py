"""
Fine-Tuning Scheduler for AVA

Manages periodic QLoRA fine-tuning cycles based on:
- Sample count thresholds
- Time since last fine-tune
- Developmental stage transitions
- Emotional state (high ambition + joy triggers training)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
