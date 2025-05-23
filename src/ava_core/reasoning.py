#!/usr/bin/env python3
"""
Enhanced Reasoning Engine for AVA - Advanced Chain-of-Thought and Multi-Step Reasoning
Optimized for local agentic AI on RTX A2000 4GB VRAM constraints.
Supports multiple reasoning strategies and adaptive reasoning selection.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Different reasoning approaches available."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    PROBLEM_SOLVING = "problem_solving"
    DECOMPOSITION = "decomposition"


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process."""
    step_number: int
    description: str
    content: str
    confidence: float = 1.0
    dependencies: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Complete reasoning result with steps and final answer."""
    question: str
    strategy: ReasoningStrategy
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    reasoning_time_ms: float
    total_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning engine."""
    max_steps: int = 10
    min_confidence_threshold: float = 0.7
    enable_adaptive_strategy: bool = True
    detailed_steps: bool = True
    include_confidence_scores: bool = True
    timeout_seconds: float = 30.0
    
    # Strategy-specific settings
    cot_depth: int = 4
    decomposition_max_levels: int = 3
    comparative_aspects: List[str] = field(default_factory=lambda: [
        "advantages", "disadvantages", "similarities", "differences"
    ])


class ReasoningEngine:
    """
    Advanced reasoning engine supporting multiple reasoning strategies
    and adaptive reasoning selection for complex queries.
    """
    
    def __init__(
        self, 
        llm_querier: Optional[Callable] = None,
        config: Optional[ReasoningConfig] = None
    ):
        """
        Initialize the enhanced reasoning engine.
        
        Args:
            llm_querier: Function to query the LLM (async or sync)
            config: Configuration for reasoning behavior
        """
        self.llm_querier = llm_querier
        self.config = config or ReasoningConfig()
        self._reasoning_templates = self._initialize_templates()
        self._strategy_patterns = self._initialize_strategy_patterns()
        
        logger.info("Enhanced Reasoning Engine initialized with strategy support")
    
    def _initialize_templates(self) -> Dict[ReasoningStrategy, str]:
        """Initialize reasoning prompt templates for different strategies."""
        return {
            ReasoningStrategy.CHAIN_OF_THOUGHT: """
Let's approach this step by step using chain-of-thought reasoning:

Question: {question}

I need to think through this carefully:
1. What is being asked?
2. What information do I have?
3. What logical steps lead to the answer?
4. What is my conclusion?

Let me work through this:
""",
            
            ReasoningStrategy.STEP_BY_STEP: """
I'll solve this using a systematic step-by-step approach:

Question: {question}

Step-by-step analysis:
""",
            
            ReasoningStrategy.ANALYTICAL: """
Let me analyze this question systematically:

Question: {question}

Analysis framework:
- Context and background
- Key components
- Relationships and dependencies
- Implications and conclusions
""",
            
            ReasoningStrategy.COMPARATIVE: """
I'll compare the relevant aspects to answer this question:

Question: {question}

Comparative analysis:
""",
            
            ReasoningStrategy.CAUSAL: """
Let me trace the causal relationships to understand this:

Question: {question}

Causal analysis:
- Root causes
- Contributing factors
- Direct effects
- Indirect consequences
""",
            
            ReasoningStrategy.PROBLEM_SOLVING: """
I'll approach this as a problem-solving exercise:

Question: {question}

Problem-solving framework:
1. Problem definition
2. Available resources/constraints
3. Possible approaches
4. Solution evaluation
5. Recommended solution
""",
            
            ReasoningStrategy.DECOMPOSITION: """
Let me break this complex question into simpler parts:

Question: {question}

Decomposition approach:
""",
        }
    
    def _initialize_strategy_patterns(self) -> Dict[str, ReasoningStrategy]:
        """Initialize patterns to detect appropriate reasoning strategy."""
        return {
            r'\b(why|explain|because|reason|cause)\b': ReasoningStrategy.CAUSAL,
            r'\b(compare|versus|vs|difference|similar)\b': ReasoningStrategy.COMPARATIVE,
            r'\b(analyze|analysis|examine|evaluate)\b': ReasoningStrategy.ANALYTICAL,
            r'\b(solve|problem|issue|challenge|how to)\b': ReasoningStrategy.PROBLEM_SOLVING,
            r'\b(step|process|procedure|method)\b': ReasoningStrategy.STEP_BY_STEP,
            r'\b(complex|complicated|multiple|several)\b': ReasoningStrategy.DECOMPOSITION,
        }
    
    def detect_reasoning_strategy(self, question: str) -> ReasoningStrategy:
        """
        Automatically detect the most appropriate reasoning strategy.
        
        Args:
            question: The question to analyze
            
        Returns:
            Most appropriate reasoning strategy
        """
        question_lower = question.lower()
        
        if not self.config.enable_adaptive_strategy:
            return ReasoningStrategy.CHAIN_OF_THOUGHT
        
        # Check for strategy patterns
        strategy_scores = {}
        for pattern, strategy in self._strategy_patterns.items():
            if re.search(pattern, question_lower):
                strategy_scores[strategy] = strategy_scores.get(strategy, 0) + 1
        
        # Additional heuristics
        word_count = len(question.split())
        if word_count > 20:
            strategy_scores[ReasoningStrategy.DECOMPOSITION] = \
                strategy_scores.get(ReasoningStrategy.DECOMPOSITION, 0) + 1
        
        if '?' in question and question.count('?') > 1:
            strategy_scores[ReasoningStrategy.DECOMPOSITION] = \
                strategy_scores.get(ReasoningStrategy.DECOMPOSITION, 0) + 1
        
        # Return strategy with highest score, default to CoT
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Selected reasoning strategy: {best_strategy.value}")
            return best_strategy
        
        logger.debug("Using default Chain-of-Thought strategy")
        return ReasoningStrategy.CHAIN_OF_THOUGHT
    
    def needs_reasoning_enhancement(
        self, 
        user_query: str, 
        llm_direct_response: str
    ) -> bool:
        """
        Enhanced logic to determine if a query needs reasoning enhancement.
        
        Args:
            user_query: Original user question
            llm_direct_response: Initial LLM response
            
        Returns:
            True if reasoning enhancement would be beneficial
        """
        query_lower = user_query.lower()
        response_length = len(llm_direct_response.split())
        
        # Complex question indicators
        complexity_indicators = [
            'why', 'explain', 'how', 'compare', 'analyze', 'evaluate',
            'pros and cons', 'advantages and disadvantages', 'step by step'
        ]
        
        has_complexity = any(indicator in query_lower for indicator in complexity_indicators)
        
        # Response quality indicators
        is_short_response = response_length < 15
        has_uncertainty = any(phrase in llm_direct_response.lower() for phrase in [
            'not sure', 'maybe', 'possibly', 'might be', 'could be'
        ])
        
        # Multi-part questions
        is_multipart = '?' in user_query and user_query.count('?') > 1
        
        # Technical complexity
        has_technical_terms = any(term in query_lower for term in [
            'algorithm', 'model', 'optimization', 'quantization', 'fine-tuning',
            'neural', 'architecture', 'performance', 'efficiency'
        ])
        
        reasoning_needed = (
            has_complexity or 
            (is_short_response and has_technical_terms) or
            has_uncertainty or
            is_multipart
        )
        
        if reasoning_needed:
            logger.info(f"Reasoning enhancement recommended for query: {user_query[:50]}...")
        
        return reasoning_needed
    
    async def apply_reasoning_async(
        self, 
        question: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[str] = None
    ) -> ReasoningResult:
        """
        Apply reasoning asynchronously with the specified or auto-detected strategy.
        
        Args:
            question: Question requiring reasoning
            strategy: Specific strategy to use (auto-detect if None)
            context: Additional context for reasoning
            
        Returns:
            Complete reasoning result
        """
        start_time = time.time()
        
        try:
            # Auto-detect strategy if not provided
            if strategy is None:
                strategy = self.detect_reasoning_strategy(question)
            
            logger.info(f"Applying {strategy.value} reasoning for: {question[:50]}...")
            
            # Build reasoning prompt
            reasoning_prompt = self._build_reasoning_prompt(question, strategy, context)
            
            # Execute reasoning with timeout
            reasoning_response = await asyncio.wait_for(
                self._execute_reasoning(reasoning_prompt),
                timeout=self.config.timeout_seconds
            )
            
            # Parse reasoning steps
            steps = self._parse_reasoning_steps(reasoning_response, strategy)
            
            # Extract final answer
            final_answer = self._extract_final_answer(reasoning_response, steps)
            
            # Calculate confidence
            confidence = self._calculate_confidence(steps, reasoning_response)
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ReasoningResult(
                question=question,
                strategy=strategy,
                steps=steps,
                final_answer=final_answer,
                confidence=confidence,
                reasoning_time_ms=execution_time,
                metadata={
                    'context_provided': context is not None,
                    'steps_count': len(steps),
                    'strategy_auto_detected': strategy is None
                }
            )
            
            logger.info(f"Reasoning completed in {execution_time:.2f}ms with confidence {confidence:.2f}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Reasoning timeout after {self.config.timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            raise
    
    def apply_reasoning(
        self, 
        question: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[str] = None
    ) -> ReasoningResult:
        """
        Synchronous wrapper for reasoning application.
        
        Args:
            question: Question requiring reasoning
            strategy: Specific strategy to use
            context: Additional context
            
        Returns:
            Complete reasoning result
        """
        return asyncio.run(self.apply_reasoning_async(question, strategy, context))
    
    def _build_reasoning_prompt(
        self, 
        question: str, 
        strategy: ReasoningStrategy, 
        context: Optional[str]
    ) -> str:
        """Build the complete reasoning prompt."""
        base_template = self._reasoning_templates[strategy]
        prompt = base_template.format(question=question)
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        if self.config.include_confidence_scores:
            prompt += "\n\nPlease include confidence levels (0-1) for key reasoning steps."
        
        return prompt
    
    async def _execute_reasoning(self, prompt: str) -> str:
        """Execute the reasoning prompt through the LLM."""
        if self.llm_querier:
            if asyncio.iscoroutinefunction(self.llm_querier):
                return await self.llm_querier(prompt)
            else:
                return self.llm_querier(prompt)
        else:
            # Fallback simulation for testing
            return f"[Simulated reasoning response for prompt: {prompt[:100]}...]"
    
    def _parse_reasoning_steps(
        self, 
        response: str, 
        strategy: ReasoningStrategy
    ) -> List[ReasoningStep]:
        """Parse individual reasoning steps from the response."""
        steps = []
        
        # Look for numbered steps or bullet points
        step_patterns = [
            r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]*)*)',  # Numbered steps
            r'[-*]\s*([^\n]+(?:\n(?![-*])[^\n]*)*)',      # Bullet points
            r'Step\s*(\d+)[:\s]*([^\n]+(?:\n(?!Step)[^\n]*)*)'  # "Step N:" format
        ]
        
        step_number = 1
        for pattern in step_patterns:
            matches = re.finditer(pattern, response, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    if match.group(1).isdigit():
                        step_num = int(match.group(1))
                        content = match.group(2).strip()
                    else:
                        step_num = step_number
                        content = match.group(1).strip()
                else:
                    step_num = step_number
                    content = match.group(0).strip()
                
                if content and len(content) > 10:  # Filter out too short steps
                    confidence = self._extract_step_confidence(content)
                    steps.append(ReasoningStep(
                        step_number=step_num,
                        description=f"{strategy.value.title()} Step {step_num}",
                        content=content,
                        confidence=confidence
                    ))
                    step_number += 1
        
        # If no structured steps found, create single step
        if not steps and response.strip():
            steps.append(ReasoningStep(
                step_number=1,
                description=f"{strategy.value.title()} Analysis",
                content=response.strip(),
                confidence=0.8
            ))
        
        return steps[:self.config.max_steps]
    
    def _extract_step_confidence(self, content: str) -> float:
        """Extract confidence score from step content."""
        if not self.config.include_confidence_scores:
            return 1.0
        
        # Look for confidence patterns
        confidence_patterns = [
            r'confidence[:\s]*(\d*\.?\d+)',
            r'\((\d*\.?\d+)\)',
            r'(\d*\.?\d+)%'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content.lower())
            if match:
                try:
                    value = float(match.group(1))
                    return min(1.0, value if value <= 1.0 else value / 100.0)
                except ValueError:
                    continue
        
        return 0.8  # Default confidence
    
    def _extract_final_answer(self, response: str, steps: List[ReasoningStep]) -> str:
        """Extract the final answer from the reasoning response."""
        # Look for answer indicators
        answer_patterns = [
            r'(?:final\s*)?answer[:\s]*([^\n]+(?:\n(?!(?:question|step|\d+\.)).+)*)',
            r'conclusion[:\s]*([^\n]+(?:\n(?!(?:question|step|\d+\.)).+)*)',
            r'therefore[,\s]*([^\n]+(?:\n(?!(?:question|step|\d+\.)).+)*)',
            r'in summary[,\s]*([^\n]+(?:\n(?!(?:question|step|\d+\.)).+)*)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                if len(answer) > 10:  # Ensure substantial answer
                    return answer
        
        # Fallback: use last step if no explicit answer found
        if steps:
            return steps[-1].content
        
        # Last resort: return last paragraph
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        return paragraphs[-1] if paragraphs else response.strip()
    
    def _calculate_confidence(self, steps: List[ReasoningStep], response: str) -> float:
        """Calculate overall confidence for the reasoning result."""
        if not steps:
            return 0.5
        
        # Average step confidences
        step_confidence = sum(step.confidence for step in steps) / len(steps)
        
        # Adjust based on response characteristics
        response_length = len(response.split())
        length_factor = min(1.0, response_length / 100)  # Longer responses may be more thorough
        
        # Check for uncertainty indicators
        uncertainty_indicators = ['maybe', 'possibly', 'might', 'could be', 'not sure']
        uncertainty_count = sum(1 for indicator in uncertainty_indicators 
                              if indicator in response.lower())
        uncertainty_penalty = min(0.3, uncertainty_count * 0.1)
        
        final_confidence = (step_confidence * 0.7 + length_factor * 0.3) - uncertainty_penalty
        return max(0.1, min(1.0, final_confidence))
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning engine usage."""
        return {
            'config': {
                'max_steps': self.config.max_steps,
                'adaptive_strategy': self.config.enable_adaptive_strategy,
                'timeout_seconds': self.config.timeout_seconds
            },
            'supported_strategies': [strategy.value for strategy in ReasoningStrategy],
            'template_count': len(self._reasoning_templates)
        }


# Example usage and testing
async def test_reasoning_engine():
    """Test the enhanced reasoning engine with various question types."""
    print("=== Enhanced Reasoning Engine Test ===")
    
    engine = ReasoningEngine()
    
    test_questions = [
        "Why is 4-bit quantization effective for LLMs on resource-constrained hardware?",
        "Compare QLoRA and full fine-tuning approaches for model adaptation.",
        "How can I optimize my model to run on 4GB VRAM?",
        "What are the step-by-step requirements for setting up AVA?",
        "Analyze the trade-offs between model size and performance in local AI deployment."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}: {question[:50]}... ---")
        
        # Test strategy detection
        strategy = engine.detect_reasoning_strategy(question)
        print(f"Detected strategy: {strategy.value}")
        
        # Test reasoning enhancement detection
        mock_response = "This is effective."  # Short response to trigger enhancement
        needs_enhancement = engine.needs_reasoning_enhancement(question, mock_response)
        print(f"Needs reasoning enhancement: {needs_enhancement}")
        
        # Apply reasoning (simulated)
        try:
            result = await engine.apply_reasoning_async(question, strategy)
            print(f"Steps generated: {len(result.steps)}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Reasoning time: {result.reasoning_time_ms:.2f}ms")
        except Exception as e:
            print(f"Error in reasoning: {e}")
    
    # Print engine stats
    print(f"\nEngine stats: {engine.get_reasoning_stats()}")
    print("=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_reasoning_engine()) 