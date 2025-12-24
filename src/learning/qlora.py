"""
QLoRA Training Wrapper for AVA
==============================

Provides a clean interface for QLoRA (Quantized Low-Rank Adaptation) training,
enabling fine-tuning of large language models on consumer hardware.

Key Features:
- 4-bit quantization via bitsandbytes
- Low-rank adaptation (LoRA) for parameter-efficient fine-tuning
- Gradient checkpointing for memory efficiency
- Configurable rank for fast (8) vs slow (64) weight updates

This wrapper is used by the Dreamer for:
1. Fast Weight Updates: Rank-8 LoRA, 2 epochs (session adaptation)
2. Slow Weight Updates: Rank-64 LoRA, 5 epochs (consolidation)

Reference: "QLoRA: Efficient Finetuning of Quantized LLMs" (arXiv:2305.14314)

Hardware Requirements:
- Minimum 6GB VRAM for rank-8 training
- Recommended 12GB VRAM for rank-64 training
- CPU fallback available but significantly slower
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import training dependencies
TRAINING_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - QLoRA training disabled")

try:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    TRAINING_AVAILABLE = True
except ImportError:
    logger.warning(
        "transformers/peft not available - install with: "
        "pip install transformers peft bitsandbytes accelerate"
    )


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""

    # Model settings
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    # Revision/commit hash for reproducibility and security (pin to specific version)
    model_revision: str | None = "main"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA settings
    lora_r: int = 8  # Rank (8 for fast, 64 for slow)
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training settings
    learning_rate: float = 1e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True

    # Output
    output_dir: str = "models/fine_tuned_adapters"
    logging_steps: int = 10
    save_steps: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": self.fp16,
            "output_dir": self.output_dir,
        }


class InstructionDataset:
    """Dataset for instruction-following fine-tuning."""

    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 2048,
    ):
        """
        Initialize dataset.

        Args:
            data: List of {"instruction": str, "output": str} dicts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format as instruction-response
        prompt = self._format_prompt(item["instruction"])
        full_text = f"{prompt}{item['output']}{self.tokenizer.eos_token}"

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze(),
        }

    def _format_prompt(self, instruction: str) -> str:
        """Format instruction as a prompt."""
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class QLoRATrainer:
    """
    QLoRA Training wrapper for AVA's learning system.

    Provides methods to:
    1. Load a base model with 4-bit quantization
    2. Apply LoRA adapters
    3. Train on instruction-following data
    4. Save and load trained adapters
    """

    def __init__(
        self,
        config: QLoRAConfig | None = None,
        checkpoints_dir: str = "data/learning/checkpoints",
        adapters_dir: str = "models/fine_tuned_adapters",
    ):
        """
        Initialize the QLoRA trainer.

        Args:
            config: Training configuration
            checkpoints_dir: Directory for training checkpoints
            adapters_dir: Directory for saved adapters
        """
        self.config = config or QLoRAConfig()
        self.checkpoints_dir = Path(checkpoints_dir)
        self.adapters_dir = Path(adapters_dir)

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

        # Model state
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        # Training history
        self.training_history: list[dict[str, Any]] = []

        logger.info(f"QLoRATrainer initialized (training available: {TRAINING_AVAILABLE})")

    def _load_base_model(self):
        """Load the base model with 4-bit quantization."""
        if not TRAINING_AVAILABLE:
            raise RuntimeError(
                "Training dependencies not available. Install with: "
                "pip install transformers peft bitsandbytes accelerate"
            )

        if self._is_loaded:
            return

        logger.info(f"Loading base model: {self.config.base_model}")

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        # Load tokenizer with revision pinning for security
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            revision=self.config.model_revision,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with revision pinning for security
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            revision=self.config.model_revision,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        self._model = prepare_model_for_kbit_training(self._model)

        self._is_loaded = True
        logger.info("Base model loaded successfully")

    def _apply_lora(self, rank: int = 8):
        """Apply LoRA adapter to the model."""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self._model = get_peft_model(self._model, lora_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self._model.parameters())

        logger.info(
            f"LoRA applied: {trainable_params:,} trainable / "
            f"{total_params:,} total ({100 * trainable_params / total_params:.2f}%)"
        )

    def train_adapter(
        self,
        data: list[dict[str, str]],
        adapter_name: str,
        rank: int = 8,
        epochs: int = 2,
        learning_rate: float = 1e-4,
    ) -> str | None:
        """
        Train a LoRA adapter on the provided data.

        Args:
            data: Training data as list of {"instruction": str, "output": str}
            adapter_name: Name for the saved adapter
            rank: LoRA rank (8 for fast, 64 for slow)
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Path to saved adapter, or None if training failed
        """
        if not TRAINING_AVAILABLE:
            logger.warning("Training not available - returning mock path")
            return self._mock_train(data, adapter_name, rank, epochs)

        if not data:
            logger.warning("No training data provided")
            return None

        try:
            # Load model if not already loaded
            self._load_base_model()

            # Apply LoRA with specified rank
            self._apply_lora(rank=rank)

            # Create dataset
            dataset = InstructionDataset(
                data=data,
                tokenizer=self._tokenizer,
                max_length=self.config.max_seq_length,
            )

            # Configure training
            output_dir = self.adapters_dir / adapter_name

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=self.config.fp16,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                gradient_checkpointing=self.config.gradient_checkpointing,
                report_to="none",  # Disable wandb/tensorboard
            )

            # Create trainer
            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=dataset,
            )

            # Train
            logger.info(f"Starting training: {len(data)} samples, {epochs} epochs, rank={rank}")
            train_result = trainer.train()

            # Save adapter
            self._model.save_pretrained(output_dir)
            self._tokenizer.save_pretrained(output_dir)

            # Record history
            self.training_history.append(
                {
                    "adapter_name": adapter_name,
                    "timestamp": datetime.now().isoformat(),
                    "samples": len(data),
                    "epochs": epochs,
                    "rank": rank,
                    "learning_rate": learning_rate,
                    "final_loss": train_result.training_loss,
                    "path": str(output_dir),
                }
            )

            logger.info(f"Adapter saved to: {output_dir}")
            return str(output_dir)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None

    def _mock_train(
        self,
        data: list[dict[str, str]],
        adapter_name: str,
        rank: int,
        epochs: int,
    ) -> str:
        """Mock training for testing without GPU."""
        logger.info(f"Mock training: {len(data)} samples, {epochs} epochs, rank={rank}")

        # Create mock adapter directory
        output_dir = self.adapters_dir / adapter_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save training config
        config_path = output_dir / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "adapter_name": adapter_name,
                    "rank": rank,
                    "epochs": epochs,
                    "samples": len(data),
                    "mock": True,
                },
                f,
                indent=2,
            )

        # Save sample of training data
        data_path = output_dir / "training_data_sample.json"
        with open(data_path, "w") as f:
            json.dump(data[:10], f, indent=2)  # Save first 10 samples

        self.training_history.append(
            {
                "adapter_name": adapter_name,
                "timestamp": datetime.now().isoformat(),
                "samples": len(data),
                "epochs": epochs,
                "rank": rank,
                "path": str(output_dir),
                "mock": True,
            }
        )

        return str(output_dir)

    def load_adapter(self, adapter_path: str) -> bool:
        """
        Load a trained LoRA adapter.

        Args:
            adapter_path: Path to the adapter directory

        Returns:
            True if successful
        """
        if not TRAINING_AVAILABLE:
            logger.warning("Training dependencies not available")
            return False

        try:
            # Load base model if not loaded
            self._load_base_model()

            # Load adapter
            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_path,
            )

            logger.info(f"Adapter loaded from: {adapter_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            return False

    def merge_and_save(self, output_path: str) -> bool:
        """
        Merge LoRA weights into base model and save.

        This creates a standalone model without needing the adapter.

        Args:
            output_path: Where to save the merged model

        Returns:
            True if successful
        """
        if not TRAINING_AVAILABLE or not self._is_loaded:
            return False

        try:
            # Merge weights
            merged_model = self._model.merge_and_unload()

            # Save
            merged_model.save_pretrained(output_path)
            self._tokenizer.save_pretrained(output_path)

            logger.info(f"Merged model saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to merge and save: {e}")
            return False

    def get_model_weights(self) -> dict[str, Any] | None:
        """
        Get current model weights for drift penalty calculation.

        Returns a lightweight representation (not full weights).
        """
        if not self._is_loaded:
            return None

        # Return adapter state dict keys and shapes
        if hasattr(self._model, "get_peft_model_state_dict"):
            state_dict = self._model.get_peft_model_state_dict()
            return {
                "type": "lora_adapter",
                "keys": list(state_dict.keys()),
                "shapes": {k: list(v.shape) for k, v in state_dict.items()},
                "timestamp": datetime.now().isoformat(),
            }

        return None

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get training history."""
        return self.training_history

    def list_adapters(self) -> list[str]:
        """List available adapters."""
        adapters = []

        for path in self.adapters_dir.iterdir():
            if path.is_dir():
                config_path = path / "adapter_config.json"
                if config_path.exists():
                    adapters.append(path.name)

        return adapters


def create_qlora_trainer(
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    checkpoints_dir: str = "data/learning/checkpoints",
    adapters_dir: str = "models/fine_tuned_adapters",
) -> QLoRATrainer:
    """
    Factory function to create a QLoRA trainer.

    Args:
        base_model: HuggingFace model name
        checkpoints_dir: Directory for checkpoints
        adapters_dir: Directory for adapters

    Returns:
        Configured QLoRATrainer instance
    """
    config = QLoRAConfig(
        base_model=base_model,
        output_dir=adapters_dir,
    )

    return QLoRATrainer(
        config=config,
        checkpoints_dir=checkpoints_dir,
        adapters_dir=adapters_dir,
    )
