#!/usr/bin/env python3
"""
Enhanced Synthetic Data Generation Script for AVA
Production-Ready Implementation for Agentic AI Training Data
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

# Optional imports with fallbacks
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class DataGenerationStrategy(Enum):
    """Different strategies for synthetic data generation."""
    QUESTION_ANSWER = "qa"
    INSTRUCTION_FOLLOWING = "instruction"
    FUNCTION_CALLING = "function_calling"
    REASONING_COT = "chain_of_thought"
    AGENTIC_PLANNING = "agentic_planning"
    TOOL_USAGE = "tool_usage"
    CONVERSATION = "conversation"
    CODE_GENERATION = "code_generation"


class LLMProvider(Enum):
    """Supported LLM providers for data generation."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL_OLLAMA = "ollama"
    MOCK = "mock"  # For testing


class OutputFormat(Enum):
    """Output formats for generated data."""
    JSONL = "jsonl"
    JSON = "json"
    HF_DATASET = "hf_dataset"
    CSV = "csv"


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    strategy: DataGenerationStrategy = DataGenerationStrategy.QUESTION_ANSWER
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-3.5-turbo"
    num_samples: int = 100
    batch_size: int = 10
    max_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    
    # Quality control
    enable_validation: bool = True
    min_answer_length: int = 10
    max_answer_length: int = 500
    diversity_threshold: float = 0.7
    
    # Rate limiting
    requests_per_minute: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Output settings
    output_format: OutputFormat = OutputFormat.JSONL
    include_metadata: bool = True


@dataclass
class SyntheticSample:
    """Represents a single synthetic data sample."""
    instruction: str
    input: str = ""
    output: str = ""
    strategy: str = ""
    difficulty: str = "medium"
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    validation_passed: bool = False


class SyntheticDataGenerator:
    """Enhanced synthetic data generator with multiple strategies and providers."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the synthetic data generator."""
        self.config = config or GenerationConfig()
        self.llm_client = None
        self.generated_samples: List[SyntheticSample] = []
        self.generation_stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "validation_passed": 0,
            "average_quality_score": 0.0
        }
        
        # Initialize LLM client
        self._initialize_llm_client()
        
        # Load generation templates
        self.templates = self._initialize_templates()
        
        logger.info(f"SyntheticDataGenerator initialized with {self.config.strategy.value} strategy")
    
    def _initialize_llm_client(self):
        """Initialize the LLM client based on provider."""
        if self.config.provider == LLMProvider.OPENAI:
            if not HAS_OPENAI:
                raise ImportError("OpenAI library not installed. Run: pip install openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.llm_client = AsyncOpenAI(api_key=api_key)
            
        elif self.config.provider == LLMProvider.ANTHROPIC:
            if not HAS_ANTHROPIC:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
            self.llm_client = anthropic.AsyncAnthropic(api_key=api_key)
            
        elif self.config.provider == LLMProvider.LOCAL_OLLAMA:
            # For local Ollama, we'll use direct HTTP requests
            self.llm_client = None  # Will implement HTTP client
            logger.info("Using local Ollama for data generation")
            
        elif self.config.provider == LLMProvider.MOCK:
            self.llm_client = None  # Mock responses for testing
            logger.info("Using mock provider for testing")
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _initialize_templates(self) -> Dict[DataGenerationStrategy, Dict[str, str]]:
        """Initialize prompt templates for different generation strategies."""
        return {
            DataGenerationStrategy.QUESTION_ANSWER: {
                "system": "You are an expert AI trainer creating high-quality question-answer pairs for training a local AI assistant.",
                "user": """Create a diverse, educational question-answer pair based on this example:

Example: {example}

Requirements:
- Question should be clear and specific
- Answer should be accurate and concise (50-150 words)
- Maintain similar complexity and domain
- Ensure factual accuracy

Generate:
Question: [Your question here]
Answer: [Your answer here]"""
            },
            
            DataGenerationStrategy.INSTRUCTION_FOLLOWING: {
                "system": "You are creating instruction-following training data for an AI assistant. Focus on practical, actionable tasks.",
                "user": """Based on this example instruction-response pair:

Example: {example}

Create a new instruction-following pair:
- Instruction should be clear and actionable
- Response should be helpful and complete
- Vary the task type while maintaining quality

Instruction: [Your instruction here]
Response: [Your response here]"""
            },
            
            DataGenerationStrategy.FUNCTION_CALLING: {
                "system": "You are creating function calling training data. Focus on scenarios where an AI assistant needs to use external tools.",
                "user": """Based on this function calling example:

Example: {example}

Create a new function calling scenario:
- User request that requires a tool/function
- Appropriate function call with parameters
- Expected response using function results

User Request: [Request requiring function use]
Function Call: [JSON function call with parameters]
Expected Response: [Response incorporating function results]"""
            },
            
            DataGenerationStrategy.REASONING_COT: {
                "system": "You are creating chain-of-thought reasoning examples for training an AI to think step-by-step.",
                "user": """Based on this reasoning example:

Example: {example}

Create a new reasoning problem with step-by-step solution:
- Problem should require multi-step thinking
- Solution should show clear reasoning steps
- Conclusion should follow logically

Problem: [Your problem here]
Reasoning: [Step-by-step thinking process]
Answer: [Final answer]"""
            },
            
            DataGenerationStrategy.AGENTIC_PLANNING: {
                "system": "You are creating agentic planning examples where an AI breaks down complex tasks into subtasks.",
                "user": """Based on this planning example:

Example: {example}

Create a new complex task that requires planning:
- Task should require multiple steps
- Plan should be logical and actionable
- Include potential obstacles and solutions

Task: [Complex task description]
Plan: [Step-by-step plan with contingencies]
Expected Outcome: [What success looks like]"""
            },
            
            DataGenerationStrategy.TOOL_USAGE: {
                "system": "You are creating tool usage examples for an AI assistant that can use various external tools.",
                "user": """Based on this tool usage example:

Example: {example}

Create a new tool usage scenario:
- Situation requiring external tool
- Appropriate tool selection
- Proper tool usage and result interpretation

Scenario: [Situation description]
Tool Selection: [Which tool to use and why]
Tool Usage: [How to use the tool]
Result Interpretation: [How to use the results]"""
            }
        }
    
    async def generate_data(
        self, 
        examples: List[Dict[str, Any]], 
        output_path: str,
        num_samples: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Generate synthetic data based on examples.
        
        Args:
            examples: List of example data to guide generation
            output_path: Path to save generated data
            num_samples: Number of samples to generate (overrides config)
            
        Returns:
            Tuple of (success, results_dict)
        """
        start_time = time.time()
        num_samples = num_samples or self.config.num_samples
        
        logger.info(f"Starting generation of {num_samples} samples using {self.config.strategy.value} strategy")
        
        try:
            # Generate samples in batches
            all_samples = []
            
            for batch_start in range(0, num_samples, self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, num_samples)
                batch_size = batch_end - batch_start
                
                logger.info(f"Generating batch {batch_start//self.config.batch_size + 1} ({batch_start+1}-{batch_end})")
                
                batch_samples = await self._generate_batch(examples, batch_size)
                all_samples.extend(batch_samples)
                
                # Rate limiting
                if self.config.requests_per_minute > 0:
                    delay = 60.0 / self.config.requests_per_minute * batch_size
                    await asyncio.sleep(delay)
            
            # Validate samples if enabled
            if self.config.enable_validation:
                logger.info("Validating generated samples...")
                validated_samples = self._validate_samples(all_samples)
            else:
                validated_samples = all_samples
            
            # Save to file
            saved_count = self._save_samples(validated_samples, output_path)
            
            # Update statistics
            self._update_statistics(validated_samples)
            
            # Prepare results
            results = {
                "success": True,
                "total_generated": len(all_samples),
                "validation_passed": len(validated_samples),
                "saved_count": saved_count,
                "generation_time_s": time.time() - start_time,
                "strategy": self.config.strategy.value,
                "provider": self.config.provider.value,
                "stats": self.generation_stats
            }
            
            logger.info(f"Data generation completed successfully!")
            logger.info(f"Generated: {len(all_samples)}, Validated: {len(validated_samples)}, Saved: {saved_count}")
            
            return True, results
            
        except Exception as e:
            logger.error(f"Data generation failed: {str(e)}")
            return False, {"error": str(e)}
    
    async def _generate_batch(self, examples: List[Dict[str, Any]], batch_size: int) -> List[SyntheticSample]:
        """Generate a batch of synthetic samples."""
        batch_samples = []
        
        for i in range(batch_size):
            try:
                # Select random example for variation
                example = random.choice(examples)
                
                # Generate sample
                sample = await self._generate_single_sample(example)
                if sample:
                    batch_samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"Failed to generate sample {i+1} in batch: {e}")
                self.generation_stats["failed"] += 1
        
        return batch_samples
    
    async def _generate_single_sample(self, example: Dict[str, Any]) -> Optional[SyntheticSample]:
        """Generate a single synthetic sample based on an example."""
        try:
            # Get template for current strategy
            template = self.templates.get(self.config.strategy)
            if not template:
                raise ValueError(f"No template found for strategy: {self.config.strategy}")
            
            # Format prompt
            system_prompt = template["system"]
            user_prompt = template["user"].format(example=json.dumps(example, indent=2))
            
            # Generate response using LLM
            response_text = await self._call_llm(system_prompt, user_prompt)
            
            if not response_text:
                return None
            
            # Parse response into structured format
            sample = self._parse_response(response_text, example)
            
            if sample:
                self.generation_stats["successful"] += 1
                self.generation_stats["total_generated"] += 1
            
            return sample
            
        except Exception as e:
            logger.warning(f"Error generating single sample: {e}")
            self.generation_stats["failed"] += 1
            return None
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call the LLM provider to generate response."""
        if self.config.provider == LLMProvider.OPENAI:
            return await self._call_openai(system_prompt, user_prompt)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(system_prompt, user_prompt)
        elif self.config.provider == LLMProvider.MOCK:
            return self._call_mock(system_prompt, user_prompt)
        else:
            raise ValueError(f"Provider {self.config.provider} not implemented")
    
    async def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call OpenAI API."""
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None
    
    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call Anthropic API."""
        try:
            response = await self.llm_client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None
    
    def _call_mock(self, system_prompt: str, user_prompt: str) -> str:
        """Mock LLM call for testing."""
        return f"Mock response for strategy: {self.config.strategy.value}"
    
    def _parse_response(self, response_text: str, original_example: Dict[str, Any]) -> Optional[SyntheticSample]:
        """Parse LLM response into structured sample."""
        try:
            sample = SyntheticSample(
                instruction="",
                input="",
                output="",
                strategy=self.config.strategy.value,
                metadata={
                    "original_example": original_example,
                    "generation_model": self.config.model_name,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            
            # Strategy-specific parsing
            if self.config.strategy == DataGenerationStrategy.QUESTION_ANSWER:
                question_match = re.search(r'Question:\s*(.+?)(?=Answer:)', response_text, re.DOTALL)
                answer_match = re.search(r'Answer:\s*(.+?)$', response_text, re.DOTALL)
                
                if question_match and answer_match:
                    sample.instruction = question_match.group(1).strip()
                    sample.output = answer_match.group(1).strip()
                else:
                    return None
                    
            elif self.config.strategy == DataGenerationStrategy.INSTRUCTION_FOLLOWING:
                instruction_match = re.search(r'Instruction:\s*(.+?)(?=Response:)', response_text, re.DOTALL)
                response_match = re.search(r'Response:\s*(.+?)$', response_text, re.DOTALL)
                
                if instruction_match and response_match:
                    sample.instruction = instruction_match.group(1).strip()
                    sample.output = response_match.group(1).strip()
                else:
                    return None
            
            # Add more parsing for other strategies as needed
            else:
                # Fallback parsing - split on common patterns
                lines = response_text.split('\n')
                if len(lines) >= 2:
                    sample.instruction = lines[0].strip()
                    sample.output = '\n'.join(lines[1:]).strip()
                else:
                    return None
            
            # Basic quality check
            if len(sample.instruction) < 5 or len(sample.output) < self.config.min_answer_length:
                return None
            
            sample.quality_score = self._calculate_quality_score(sample)
            
            return sample
            
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            return None
    
    def _calculate_quality_score(self, sample: SyntheticSample) -> float:
        """Calculate quality score for a sample."""
        score = 0.0
        
        # Length checks
        if self.config.min_answer_length <= len(sample.output) <= self.config.max_answer_length:
            score += 0.3
        
        # Content diversity (simple heuristic)
        unique_words = len(set(sample.output.lower().split()))
        total_words = len(sample.output.split())
        if total_words > 0:
            diversity = unique_words / total_words
            score += diversity * 0.3
        
        # Structure check (has punctuation, proper capitalization)
        if sample.output.endswith('.') or sample.output.endswith('!') or sample.output.endswith('?'):
            score += 0.2
        
        if sample.output[0].isupper():
            score += 0.2
        
        return min(score, 1.0)
    
    def _validate_samples(self, samples: List[SyntheticSample]) -> List[SyntheticSample]:
        """Validate generated samples."""
        validated = []
        
        for sample in samples:
            # Quality threshold check
            if sample.quality_score >= 0.5:
                sample.validation_passed = True
                validated.append(sample)
                self.generation_stats["validation_passed"] += 1
        
        return validated
    
    def _save_samples(self, samples: List[SyntheticSample], output_path: str) -> int:
        """Save samples to file in specified format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.config.output_format == OutputFormat.JSONL:
                return self._save_jsonl(samples, output_path)
            elif self.config.output_format == OutputFormat.JSON:
                return self._save_json(samples, output_path)
            else:
                raise ValueError(f"Output format {self.config.output_format} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to save samples: {e}")
            return 0
    
    def _save_jsonl(self, samples: List[SyntheticSample], output_path: Path) -> int:
        """Save samples in JSONL format."""
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                sample_dict = {
                    "instruction": sample.instruction,
                    "input": sample.input,
                    "output": sample.output
                }
                
                if self.config.include_metadata:
                    sample_dict.update({
                        "strategy": sample.strategy,
                        "quality_score": sample.quality_score,
                        "metadata": sample.metadata
                    })
                
                f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
                count += 1
        
        return count
    
    def _save_json(self, samples: List[SyntheticSample], output_path: Path) -> int:
        """Save samples in JSON format."""
        samples_data = []
        
        for sample in samples:
            sample_dict = {
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output
            }
            
            if self.config.include_metadata:
                sample_dict.update({
                    "strategy": sample.strategy,
                    "quality_score": sample.quality_score,
                    "metadata": sample.metadata
                })
            
            samples_data.append(sample_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
        
        return len(samples_data)
    
    def _update_statistics(self, samples: List[SyntheticSample]):
        """Update generation statistics."""
        if samples:
            avg_quality = sum(s.quality_score for s in samples) / len(samples)
            self.generation_stats["average_quality_score"] = avg_quality


def load_example_data(example_path: str) -> List[Dict[str, Any]]:
    """Load example data from file."""
    try:
        with open(example_path, 'r', encoding='utf-8') as f:
            if example_path.endswith('.jsonl'):
                examples = [json.loads(line) for line in f if line.strip()]
            else:
                examples = json.load(f)
        
        logger.info(f"Loaded {len(examples)} examples from {example_path}")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to load examples from {example_path}: {e}")
        return []


def get_default_examples(strategy: DataGenerationStrategy) -> List[Dict[str, Any]]:
    """Get default examples for different strategies."""
    examples = {
        DataGenerationStrategy.QUESTION_ANSWER: [
            {
                "question": "What is the capital of France?",
                "answer": "Paris is the capital and largest city of France."
            },
            {
                "question": "Explain quantum computing in simple terms.",
                "answer": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot."
            }
        ],
        
        DataGenerationStrategy.INSTRUCTION_FOLLOWING: [
            {
                "instruction": "Write a brief email to schedule a meeting.",
                "response": "Subject: Meeting Request\n\nHi [Name],\n\nI hope this email finds you well. I would like to schedule a meeting to discuss [topic]. Are you available next week? Please let me know your preferred time.\n\nBest regards,\n[Your name]"
            }
        ],
        
        DataGenerationStrategy.FUNCTION_CALLING: [
            {
                "user_request": "What's the weather like in New York?",
                "function_call": '{"name": "get_weather", "parameters": {"location": "New York, NY"}}',
                "response": "Based on the weather data, it's currently 72°F and sunny in New York with light winds from the west."
            }
        ]
    }
    
    return examples.get(strategy, examples[DataGenerationStrategy.QUESTION_ANSWER])


async def test_data_generation():
    """Test the synthetic data generation pipeline."""
    logger.info("=== Testing Synthetic Data Generation ===")
    
    config = GenerationConfig(
        strategy=DataGenerationStrategy.QUESTION_ANSWER,
        provider=LLMProvider.MOCK,
        num_samples=5,
        batch_size=2
    )
    
    generator = SyntheticDataGenerator(config)
    examples = get_default_examples(config.strategy)
    
    success, results = await generator.generate_data(
        examples=examples,
        output_path="./data/test_synthetic.jsonl"
    )
    
    if success:
        logger.info("Test generation successful!")
        logger.info(f"Results: {results}")
    else:
        logger.error(f"Test generation failed: {results.get('error')}")
    
    return success


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Enhanced Synthetic Data Generation for AVA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_synthetic_data.py --strategy qa --num_samples 100 --output ./data/qa_synthetic.jsonl
  python scripts/generate_synthetic_data.py --strategy instruction --provider openai --examples ./examples.json
  python scripts/generate_synthetic_data.py --test  # Run test with mock provider
        """
    )
    
    parser.add_argument("--strategy", type=str, choices=[s.value for s in DataGenerationStrategy],
                       default="qa", help="Data generation strategy")
    parser.add_argument("--provider", type=str, choices=[p.value for p in LLMProvider],
                       default="openai", help="LLM provider")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                       help="Model name for generation")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="./data/synthetic_data.jsonl",
                       help="Output file path")
    parser.add_argument("--examples", type=str,
                       help="Path to example data file (JSON/JSONL)")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Generation temperature")
    parser.add_argument("--enable_validation", action="store_true",
                       help="Enable quality validation")
    parser.add_argument("--test", action="store_true",
                       help="Run test generation with mock provider")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = GenerationConfig(
        strategy=DataGenerationStrategy(args.strategy),
        provider=LLMProvider.MOCK if args.test else LLMProvider(args.provider),
        model_name=args.model_name,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        enable_validation=args.enable_validation
    )
    
    async def run_generation():
        if args.test:
            return await test_data_generation()
        
        # Load examples
        if args.examples:
            examples = load_example_data(args.examples)
            if not examples:
                logger.error("No examples loaded, using defaults")
                examples = get_default_examples(config.strategy)
        else:
            examples = get_default_examples(config.strategy)
        
        # Generate data
        generator = SyntheticDataGenerator(config)
        success, results = await generator.generate_data(examples, args.output)
        
        if success:
            logger.info("Data generation completed successfully!")
            print(f"\nResults: {json.dumps(results, indent=2)}")
        else:
            logger.error(f"Data generation failed: {results.get('error')}")
            return False
        
        return True
    
    # Run async generation
    return asyncio.run(run_generation())


if __name__ == "__main__":
    main() 