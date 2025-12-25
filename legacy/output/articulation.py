"""
Articulation Model for AVA

Simulates speech development - at early stages, output is garbled
and unclear, gradually becoming more coherent as AVA matures.

Also provides Chain-of-Thought (CoT) reasoning enforcement for
improved reasoning quality.
Reference: "Distilling Step-by-Step!" (ACL, 2023)
"""

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ArticulationModel:
    """
    Simulates speech/articulation development.

    At low clarity levels:
    - Letters are substituted (r -> w, th -> f)
    - Words are dropped or simplified
    - Sentences are shortened

    At high clarity levels:
    - Full articulation with occasional minor errors
    """

    def __init__(self):
        """Initialize the articulation model."""
        # Letter substitutions common in child speech development
        self.substitutions: List[Tuple[str, str, float]] = [
            # (from, to, max_clarity_for_substitution)
            ("r", "w", 0.35),      # "rabbit" -> "wabbit"
            ("l", "w", 0.30),      # "love" -> "wove"
            ("th", "f", 0.40),     # "think" -> "fink"
            ("th", "d", 0.35),     # "the" -> "de"
            ("s", "th", 0.30),     # "see" -> "thee" (lisp)
            ("ch", "t", 0.25),     # "chair" -> "tair"
            ("sh", "s", 0.30),     # "ship" -> "sip"
            ("j", "d", 0.25),      # "jump" -> "dump"
            ("v", "b", 0.25),      # "very" -> "bery"
            ("z", "d", 0.20),      # "zoo" -> "doo"
        ]

        # Word simplifications
        self.simplifications: Dict[str, str] = {
            "because": "cause",
            "going to": "gonna",
            "want to": "wanna",
            "give me": "gimme",
            "let me": "lemme",
            "i am": "i'm",
            "you are": "you're",
            "cannot": "can't",
            "do not": "don't",
            "it is": "it's",
        }

        # Filler words/sounds for very young stages
        self.fillers = ["um", "uh", "like", "well", "so"]

        # Emotion interjections
        self.emotion_sounds = {
            "joy": ["yay", "hehe", "wow"],
            "fear": ["oh no", "eep", "uh oh"],
            "surprise": ["oh", "whoa", "ooh"],
            "ambition": ["yes", "yeah", "okay"],
        }

    def apply(
        self,
        text: str,
        clarity: float,
        emotional_state: dict | None = None,
    ) -> str:
        """
        Apply articulation model to text.

        Args:
            text: The text to process
            clarity: Articulation clarity (0.0 to 1.0)
            emotional_state: Optional emotional context

        Returns:
            Articulated text
        """
        if clarity >= 0.95:
            return text  # Near-perfect articulation

        # Apply transformations based on clarity
        result = text

        # Very low clarity: heavy transformation
        if clarity < 0.3:
            result = self._apply_heavy_garbling(result, clarity)
        elif clarity < 0.5:
            result = self._apply_moderate_garbling(result, clarity)
        elif clarity < 0.8:
            result = self._apply_light_garbling(result, clarity)

        # Add emotional interjections if appropriate
        if emotional_state and clarity < 0.7:
            result = self._add_emotional_sounds(result, emotional_state, clarity)

        return result

    def _apply_heavy_garbling(self, text: str, clarity: float) -> str:
        """Apply heavy garbling for infant stage."""
        words = text.split()

        # Drop some words randomly
        drop_rate = 0.3 - (clarity * 0.5)  # More drops at lower clarity
        words = [w for w in words if random.random() > drop_rate]

        # Simplify remaining words
        words = [self._simplify_word(w, clarity) for w in words]

        # Apply all substitutions
        result = " ".join(words)
        for from_str, to_str, max_clarity in self.substitutions:
            if clarity < max_clarity:
                result = self._random_substitute(result, from_str, to_str, 0.7)

        # Shorten sentences
        sentences = result.split(".")
        sentences = [self._shorten_sentence(s, clarity) for s in sentences]
        result = ". ".join(s for s in sentences if s.strip())

        # Add fillers
        result = self._add_fillers(result, clarity)

        return result

    def _apply_moderate_garbling(self, text: str, clarity: float) -> str:
        """Apply moderate garbling for toddler stage."""
        words = text.split()

        # Occasional word drops
        drop_rate = 0.15 - (clarity * 0.2)
        words = [w for w in words if random.random() > drop_rate]

        # Some simplifications
        words = [self._simplify_word(w, clarity) for w in words]

        result = " ".join(words)

        # Apply some substitutions
        for from_str, to_str, max_clarity in self.substitutions:
            if clarity < max_clarity:
                result = self._random_substitute(result, from_str, to_str, 0.4)

        # Occasional fillers
        if random.random() < 0.3:
            result = self._add_fillers(result, clarity)

        return result

    def _apply_light_garbling(self, text: str, clarity: float) -> str:
        """Apply light garbling for child stage."""
        # Only occasional substitutions
        result = text

        for from_str, to_str, max_clarity in self.substitutions[:3]:  # Only common ones
            if clarity < max_clarity:
                result = self._random_substitute(result, from_str, to_str, 0.2)

        return result

    def _simplify_word(self, word: str, clarity: float) -> str:
        """Simplify a word based on clarity."""
        word_lower = word.lower()

        # Check for simplifications
        for full, simple in self.simplifications.items():
            if word_lower == full:
                if random.random() > clarity:
                    return simple

        # Shorten long words at very low clarity
        if clarity < 0.3 and len(word) > 8:
            # Drop middle of word
            if random.random() > clarity * 2:
                return word[:3] + word[-2:]

        return word

    def _random_substitute(
        self,
        text: str,
        from_str: str,
        to_str: str,
        probability: float
    ) -> str:
        """Randomly substitute characters with given probability."""
        result = []
        i = 0
        while i < len(text):
            # Check for multi-character substitution
            if text[i:i+len(from_str)].lower() == from_str:
                if random.random() < probability:
                    # Preserve case
                    if text[i].isupper():
                        result.append(to_str.capitalize())
                    else:
                        result.append(to_str)
                    i += len(from_str)
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1

        return "".join(result)

    def _shorten_sentence(self, sentence: str, clarity: float) -> str:
        """Shorten a sentence based on clarity."""
        words = sentence.split()

        if len(words) <= 3:
            return sentence

        # Calculate target length
        max_words = int(3 + (clarity * 20))  # 3-23 words

        if len(words) > max_words:
            words = words[:max_words]

        return " ".join(words)

    def _add_fillers(self, text: str, clarity: float) -> str:
        """Add filler words/sounds."""
        if clarity > 0.6:
            return text

        sentences = text.split(".")
        result = []

        for sentence in sentences:
            if sentence.strip() and random.random() < (0.5 - clarity):
                filler = random.choice(self.fillers)
                sentence = f"{filler}, {sentence.strip()}"
            result.append(sentence)

        return ".".join(result)

    def _add_emotional_sounds(
        self,
        text: str,
        emotional_state: dict,
        clarity: float
    ) -> str:
        """Add emotional interjections based on emotional state."""
        # Find dominant emotion
        dominant = max(emotional_state.items(), key=lambda x: x[1])
        emotion_name, intensity = dominant

        if intensity < 0.6:
            return text  # Not intense enough

        if emotion_name in self.emotion_sounds:
            sounds = self.emotion_sounds[emotion_name]
            if random.random() < (intensity - 0.5):
                sound = random.choice(sounds)
                if random.random() < 0.5:
                    text = f"{sound}! {text}"
                else:
                    text = f"{text} {sound}!"

        return text


# Vocabulary constraints by developmental stage
VOCABULARY_LEVELS = {
    # Very basic words for infant stage
    0.1: [
        "yes", "no", "hi", "bye", "good", "bad", "want", "need", "like",
        "help", "please", "thank", "sorry", "happy", "sad", "big", "small",
        "up", "down", "in", "out", "on", "off", "go", "stop", "come",
        "look", "see", "hear", "eat", "drink", "sleep", "play", "work",
    ],
    # Expanded toddler vocabulary
    0.3: [
        "think", "know", "understand", "remember", "forget", "learn", "try",
        "hope", "wish", "feel", "believe", "wonder", "question", "answer",
        "maybe", "probably", "because", "but", "and", "or", "if", "then",
        "before", "after", "now", "later", "today", "tomorrow", "always",
    ],
    # Child vocabulary
    0.5: [
        "explain", "describe", "compare", "different", "similar", "example",
        "reason", "cause", "effect", "result", "problem", "solution",
        "important", "interesting", "difficult", "easy", "possible",
        "impossible", "certain", "uncertain", "correct", "incorrect",
    ],
    # Teen vocabulary
    0.7: [
        "analyze", "evaluate", "synthesize", "conclude", "hypothesize",
        "evidence", "argument", "perspective", "context", "significant",
        "relevant", "appropriate", "effective", "efficient", "complex",
        "nuanced", "subtle", "explicit", "implicit", "fundamental",
    ],
}


def get_allowed_vocabulary(vocabulary_range: float) -> set:
    """Get the set of allowed words for a vocabulary range."""
    allowed = set()

    for level, words in VOCABULARY_LEVELS.items():
        if level <= vocabulary_range:
            allowed.update(words)

    return allowed


def constrain_vocabulary(text: str, vocabulary_range: float) -> str:
    """
    Constrain text to age-appropriate vocabulary.

    Replaces complex words with simpler alternatives when possible.
    """
    if vocabulary_range >= 0.9:
        return text  # Full vocabulary access

    # Simple word replacements for lower vocabulary levels
    replacements = {
        "analyze": "look at",
        "synthesize": "put together",
        "evaluate": "check",
        "hypothesize": "guess",
        "significant": "important",
        "fundamental": "basic",
        "nuanced": "detailed",
        "appropriate": "right",
        "efficient": "good",
        "complex": "hard",
        "conclude": "think",
        "perspective": "view",
        "relevant": "useful",
        "demonstrate": "show",
        "implement": "do",
        "utilize": "use",
        "facilitate": "help",
        "comprehensive": "complete",
        "subsequently": "then",
        "approximately": "about",
        "consequently": "so",
        "furthermore": "also",
        "nevertheless": "but",
        "specifically": "exactly",
        "essentially": "basically",
    }

    # Only apply replacements based on vocabulary range
    words = text.split()
    result = []

    for word in words:
        word_lower = word.lower().strip(".,!?")
        replaced = False

        if vocabulary_range < 0.7:
            for complex_word, simple in replacements.items():
                if word_lower == complex_word:
                    # Preserve punctuation
                    punct = ""
                    if word[-1] in ".,!?":
                        punct = word[-1]
                    result.append(simple + punct)
                    replaced = True
                    break

        if not replaced:
            result.append(word)

    return " ".join(result)


# =============================================================================
# CHAIN-OF-THOUGHT (CoT) REASONING ENFORCEMENT
# =============================================================================
# Reference: "Distilling Step-by-Step!" (ACL Findings, 2023)
#
# Key concepts:
# - Force step-by-step reasoning in model output
# - Extract rationales for training data
# - Improve reasoning quality of small models
# =============================================================================


class ReasoningStyle(Enum):
    """Different styles of reasoning prompts."""
    STEP_BY_STEP = "step_by_step"
    BREAKDOWN = "breakdown"
    ANALYSIS = "analysis"
    REFLECTION = "reflection"
    PLANNING = "planning"


@dataclass
class CoTPromptTemplate:
    """A template for Chain-of-Thought prompting."""
    name: str
    prefix: str  # Added before the query
    suffix: str  # Added after the query
    reasoning_style: ReasoningStyle
    min_stage: int = 0  # Minimum developmental stage
    
    def apply(self, query: str) -> str:
        """Apply template to a query."""
        return f"{self.prefix}\n{query}\n{self.suffix}"


# CoT templates adapted for different developmental stages
COT_TEMPLATES = {
    ReasoningStyle.STEP_BY_STEP: [
        CoTPromptTemplate(
            name="basic_steps",
            prefix="",
            suffix="Let's think about this step by step:",
            reasoning_style=ReasoningStyle.STEP_BY_STEP,
            min_stage=1,  # TODDLER
        ),
        CoTPromptTemplate(
            name="numbered_steps",
            prefix="I'll work through this carefully.",
            suffix="Let's reason through this step by step:\nStep 1:",
            reasoning_style=ReasoningStyle.STEP_BY_STEP,
            min_stage=2,  # CHILD
        ),
        CoTPromptTemplate(
            name="detailed_reasoning",
            prefix="Let me analyze this methodically.",
            suffix="My reasoning process:\n1. First, I'll identify the key elements.\n2. Then, I'll consider the relationships.\n3. Finally, I'll synthesize my understanding.",
            reasoning_style=ReasoningStyle.STEP_BY_STEP,
            min_stage=4,  # YOUNG_ADULT
        ),
    ],
    ReasoningStyle.BREAKDOWN: [
        CoTPromptTemplate(
            name="simple_breakdown",
            prefix="",
            suffix="Let me break this down:",
            reasoning_style=ReasoningStyle.BREAKDOWN,
            min_stage=2,  # CHILD
        ),
        CoTPromptTemplate(
            name="component_analysis",
            prefix="I'll decompose this into parts.",
            suffix="Components to consider:\n- Main concept:\n- Supporting details:\n- Connections:",
            reasoning_style=ReasoningStyle.BREAKDOWN,
            min_stage=3,  # ADOLESCENT
        ),
    ],
    ReasoningStyle.ANALYSIS: [
        CoTPromptTemplate(
            name="pros_cons",
            prefix="",
            suffix="Analyzing this:\nPositives:\nNegatives:\nConclusion:",
            reasoning_style=ReasoningStyle.ANALYSIS,
            min_stage=3,  # ADOLESCENT
        ),
        CoTPromptTemplate(
            name="deep_analysis",
            prefix="I'll provide a thorough analysis.",
            suffix="Analysis framework:\n1. Context and background\n2. Key factors\n3. Implications\n4. Synthesis",
            reasoning_style=ReasoningStyle.ANALYSIS,
            min_stage=4,  # YOUNG_ADULT
        ),
    ],
    ReasoningStyle.REFLECTION: [
        CoTPromptTemplate(
            name="self_check",
            prefix="",
            suffix="Let me think about this... What do I know? What don't I know?",
            reasoning_style=ReasoningStyle.REFLECTION,
            min_stage=3,  # ADOLESCENT
        ),
        CoTPromptTemplate(
            name="metacognitive",
            prefix="I'll reflect on my understanding.",
            suffix="Metacognitive check:\n- What I'm certain about:\n- What I'm uncertain about:\n- How confident am I overall:",
            reasoning_style=ReasoningStyle.REFLECTION,
            min_stage=5,  # MATURE
        ),
    ],
    ReasoningStyle.PLANNING: [
        CoTPromptTemplate(
            name="simple_plan",
            prefix="",
            suffix="My plan:\n1. Understand the question\n2. Think about what I know\n3. Give my answer",
            reasoning_style=ReasoningStyle.PLANNING,
            min_stage=2,  # CHILD
        ),
        CoTPromptTemplate(
            name="strategic_plan",
            prefix="I'll approach this strategically.",
            suffix="Strategic approach:\n- Goal identification\n- Available resources/knowledge\n- Step-by-step execution\n- Verification",
            reasoning_style=ReasoningStyle.PLANNING,
            min_stage=4,  # YOUNG_ADULT
        ),
    ],
}


@dataclass
class RationaleExtraction:
    """Extracted rationale from a response."""
    steps: List[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.5
    reasoning_style: Optional[ReasoningStyle] = None
    quality_score: float = 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "steps": self.steps,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "reasoning_style": self.reasoning_style.value if self.reasoning_style else None,
            "quality_score": self.quality_score,
        }


class ChainOfThoughtEnforcer:
    """
    Enforces Chain-of-Thought reasoning in model output.
    
    Provides:
    1. CoT prompt templates adapted to developmental stage
    2. Rationale extraction from responses
    3. Quality scoring of reasoning
    """
    
    def __init__(self):
        """Initialize the CoT enforcer."""
        self.templates = COT_TEMPLATES
        
        # Patterns to detect reasoning markers
        self.step_patterns = [
            re.compile(r'step\s*\d+[:\.]', re.IGNORECASE),
            re.compile(r'\d+\.\s+\w+', re.IGNORECASE),
            re.compile(r'first[,\s]', re.IGNORECASE),
            re.compile(r'then[,\s]', re.IGNORECASE),
            re.compile(r'next[,\s]', re.IGNORECASE),
            re.compile(r'finally[,\s]', re.IGNORECASE),
            re.compile(r'because[,\s]', re.IGNORECASE),
            re.compile(r'therefore[,\s]', re.IGNORECASE),
            re.compile(r'thus[,\s]', re.IGNORECASE),
        ]
        
        self.conclusion_patterns = [
            re.compile(r'(in conclusion|to conclude|therefore|thus|so)[,\s](.+?)(?:\.|$)', re.IGNORECASE),
            re.compile(r'(the answer is|my answer is)[:\s](.+?)(?:\.|$)', re.IGNORECASE),
            re.compile(r'(finally|ultimately)[,\s](.+?)(?:\.|$)', re.IGNORECASE),
        ]
    
    def get_appropriate_template(
        self,
        query: str,
        stage: int,
        style: Optional[ReasoningStyle] = None,
    ) -> Optional[CoTPromptTemplate]:
        """
        Get an appropriate CoT template based on query and stage.
        
        Args:
            query: The user's query
            stage: Current developmental stage
            style: Preferred reasoning style (auto-detected if None)
            
        Returns:
            Appropriate template or None if stage too low
        """
        # Auto-detect style if not specified
        if style is None:
            style = self._detect_reasoning_style(query)
        
        # Get templates for this style
        templates = self.templates.get(style, [])
        
        # Filter by stage
        valid_templates = [t for t in templates if t.min_stage <= stage]
        
        if not valid_templates:
            return None
        
        # Return the most advanced valid template
        return max(valid_templates, key=lambda t: t.min_stage)
    
    def _detect_reasoning_style(self, query: str) -> ReasoningStyle:
        """Detect the most appropriate reasoning style for a query."""
        query_lower = query.lower()
        
        # Keywords for different styles
        if any(w in query_lower for w in ["steps", "how to", "process", "procedure"]):
            return ReasoningStyle.STEP_BY_STEP
        
        if any(w in query_lower for w in ["compare", "analyze", "evaluate", "pros", "cons"]):
            return ReasoningStyle.ANALYSIS
        
        if any(w in query_lower for w in ["break down", "components", "parts", "elements"]):
            return ReasoningStyle.BREAKDOWN
        
        if any(w in query_lower for w in ["think about", "reflect", "consider"]):
            return ReasoningStyle.REFLECTION
        
        if any(w in query_lower for w in ["plan", "strategy", "approach", "how should"]):
            return ReasoningStyle.PLANNING
        
        # Default to step-by-step
        return ReasoningStyle.STEP_BY_STEP
    
    def create_cot_prompt(
        self,
        query: str,
        context: str = "",
        stage: int = 3,
        style: Optional[ReasoningStyle] = None,
    ) -> str:
        """
        Create a Chain-of-Thought enhanced prompt.
        
        Args:
            query: User's query
            context: Additional context
            stage: Developmental stage
            style: Reasoning style
            
        Returns:
            Enhanced prompt with CoT guidance
        """
        template = self.get_appropriate_template(query, stage, style)
        
        if template is None:
            # Stage too low for CoT
            return query
        
        # Build the enhanced prompt
        parts = []
        
        if context:
            parts.append(f"Context: {context}")
        
        parts.append(f"Question: {query}")
        parts.append("")
        parts.append(template.suffix)
        
        if template.prefix:
            parts.insert(0, template.prefix)
        
        return "\n".join(parts)
    
    def extract_rationale(self, response: str) -> RationaleExtraction:
        """
        Extract the reasoning rationale from a response.
        
        Args:
            response: Model's response
            
        Returns:
            RationaleExtraction with steps and analysis
        """
        extraction = RationaleExtraction()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Identify step-like sentences
        for sentence in sentences:
            is_step = any(p.search(sentence) for p in self.step_patterns)
            if is_step:
                extraction.steps.append(sentence)
        
        # Extract conclusion
        for pattern in self.conclusion_patterns:
            match = pattern.search(response)
            if match:
                extraction.conclusion = match.group(2).strip()
                break
        
        # If no explicit conclusion, use last sentence
        if not extraction.conclusion and sentences:
            extraction.conclusion = sentences[-1]
        
        # Detect reasoning style
        extraction.reasoning_style = self._detect_style_from_response(response)
        
        # Calculate quality score
        extraction.quality_score = self._score_rationale_quality(extraction, response)
        
        # Set confidence based on quality
        extraction.confidence = min(extraction.quality_score + 0.2, 1.0)
        
        return extraction
    
    def _detect_style_from_response(self, response: str) -> Optional[ReasoningStyle]:
        """Detect what reasoning style was used in a response."""
        response_lower = response.lower()
        
        if any(marker in response_lower for marker in ["step 1", "step 2", "first,", "then,", "finally,"]):
            return ReasoningStyle.STEP_BY_STEP
        
        if any(marker in response_lower for marker in ["analyzing", "pros:", "cons:", "positive", "negative"]):
            return ReasoningStyle.ANALYSIS
        
        if any(marker in response_lower for marker in ["breaking down", "components", "parts:"]):
            return ReasoningStyle.BREAKDOWN
        
        if any(marker in response_lower for marker in ["reflecting", "thinking about", "uncertain"]):
            return ReasoningStyle.REFLECTION
        
        if any(marker in response_lower for marker in ["plan:", "strategy", "approach:"]):
            return ReasoningStyle.PLANNING
        
        return None
    
    def _score_rationale_quality(
        self,
        extraction: RationaleExtraction,
        response: str,
    ) -> float:
        """Score the quality of extracted rationale."""
        score = 0.3  # Base score
        
        # Bonus for having steps
        if extraction.steps:
            score += min(len(extraction.steps) * 0.1, 0.3)
        
        # Bonus for having a conclusion
        if extraction.conclusion:
            score += 0.15
        
        # Bonus for detected reasoning style
        if extraction.reasoning_style:
            score += 0.1
        
        # Bonus for response length (indicates thorough reasoning)
        word_count = len(response.split())
        if word_count > 50:
            score += 0.1
        if word_count > 100:
            score += 0.05
        
        # Check for reasoning keywords
        reasoning_keywords = ["because", "therefore", "thus", "since", "reason", "conclude"]
        keyword_count = sum(1 for kw in reasoning_keywords if kw in response.lower())
        score += min(keyword_count * 0.05, 0.15)
        
        return min(score, 1.0)
    
    def enhance_response_with_cot(
        self,
        response: str,
        stage: int,
        force_structure: bool = False,
    ) -> str:
        """
        Enhance a response to include clearer reasoning structure.
        
        Args:
            response: Original response
            stage: Developmental stage
            force_structure: Force adding structure even if present
            
        Returns:
            Enhanced response
        """
        # Check if already has structure
        extraction = self.extract_rationale(response)
        
        if extraction.steps and not force_structure:
            # Already has reasoning structure
            return response
        
        if stage < 2:
            # Too young for enhanced structure
            return response
        
        # Add basic structure for responses without it
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return response
        
        # Restructure into reasoning format
        enhanced_parts = []
        
        if stage >= 3:
            enhanced_parts.append("Here's my reasoning:")
        
        for i, sentence in enumerate(sentences[:-1], 1):
            if stage >= 4:
                enhanced_parts.append(f"  {i}. {sentence}.")
            else:
                enhanced_parts.append(f"  - {sentence}.")
        
        # Add conclusion
        if stage >= 3:
            enhanced_parts.append(f"\nSo: {sentences[-1]}.")
        else:
            enhanced_parts.append(sentences[-1] + ".")
        
        return "\n".join(enhanced_parts)
    
    def extract_reasoning_steps(self, response: str) -> List[str]:
        """
        Extract intermediate reasoning steps from a response.
        
        Looks for <think>...</think> sections first, then falls back
        to extracting step-like sentences.
        
        Args:
            response: Model's response with potential reasoning
            
        Returns:
            List of extracted reasoning steps
        """
        # First try to extract <think>...</think> blocks
        think_matches = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_matches:
            steps = []
            for match in think_matches:
                # Split the think block into individual steps
                lines = [line.strip() for line in match.strip().split('\n') if line.strip()]
                steps.extend(lines)
            return steps
        
        # Fall back to using extract_rationale
        extraction = self.extract_rationale(response)
        return extraction.steps
    
    def format_structured_reasoning(
        self,
        question: str,
        reasoning: str,
        answer: str,
    ) -> str:
        """
        Format question, reasoning, and answer into a structured output.
        
        Args:
            question: The original question
            reasoning: The reasoning/rationale
            answer: The final answer
            
        Returns:
            Structured string with question, reasoning, and answer
        """
        parts = []
        
        if question:
            parts.append(f"<question>{question}</question>")
        
        if reasoning:
            parts.append(f"<reasoning>{reasoning}</reasoning>")
        
        if answer:
            parts.append(f"<answer>{answer}</answer>")
        
        return "\n".join(parts)


@dataclass
class CoTDistillationSample:
    """A sample for CoT distillation training."""
    query: str
    response_with_rationale: str
    extracted_steps: List[str]
    conclusion: str
    quality_score: float
    reasoning_style: Optional[str] = None
    
    def to_training_format(self) -> Dict[str, str]:
        """Convert to format suitable for training."""
        return {
            "instruction": self.query,
            "output": self.response_with_rationale,
            "rationale_steps": self.extracted_steps,
        }


class CoTDistillationCollector:
    """
    Collects high-quality CoT samples for distillation training.
    
    These samples help transfer reasoning capabilities from
    larger models or high-quality interactions to the small model.
    """
    
    def __init__(
        self,
        min_quality: float = 0.6,
        max_samples: int = 1000,
    ):
        """
        Initialize collector.
        
        Args:
            min_quality: Minimum quality score to collect
            max_samples: Maximum samples to store
        """
        self.min_quality = min_quality
        self.max_samples = max_samples
        self.samples: List[CoTDistillationSample] = []
        self.enforcer = ChainOfThoughtEnforcer()
    
    def collect(
        self,
        query: str,
        response: str,
    ) -> bool:
        """
        Attempt to collect a sample.
        
        Args:
            query: Original query
            response: Model response
            
        Returns:
            True if collected, False if rejected
        """
        extraction = self.enforcer.extract_rationale(response)
        
        if extraction.quality_score < self.min_quality:
            return False
        
        sample = CoTDistillationSample(
            query=query,
            response_with_rationale=response,
            extracted_steps=extraction.steps,
            conclusion=extraction.conclusion,
            quality_score=extraction.quality_score,
            reasoning_style=extraction.reasoning_style.value if extraction.reasoning_style else None,
        )
        
        self.samples.append(sample)
        
        # Trim to max size
        if len(self.samples) > self.max_samples:
            self.samples.sort(key=lambda x: x.quality_score, reverse=True)
            self.samples = self.samples[:self.max_samples]
        
        return True
    
    def get_training_batch(self, batch_size: int = 32) -> List[Dict]:
        """Get a batch of samples for training."""
        import random
        
        if not self.samples:
            return []
        
        batch_size = min(batch_size, len(self.samples))
        batch = random.sample(self.samples, batch_size)
        
        return [s.to_training_format() for s in batch]
    
    def get_statistics(self) -> Dict:
        """Get collector statistics."""
        if not self.samples:
            return {"count": 0}
        
        qualities = [s.quality_score for s in self.samples]
        styles = [s.reasoning_style for s in self.samples if s.reasoning_style]
        
        return {
            "count": len(self.samples),
            "avg_quality": sum(qualities) / len(qualities),
            "style_distribution": {
                style: styles.count(style) for style in set(styles)
            },
        }


def create_cot_system(
    min_quality: float = 0.6,
    max_samples: int = 1000,
) -> Tuple[ChainOfThoughtEnforcer, CoTDistillationCollector]:
    """
    Factory function to create CoT components.
    
    Args:
        min_quality: Minimum quality for collection
        max_samples: Maximum samples to store
        
    Returns:
        Tuple of (ChainOfThoughtEnforcer, CoTDistillationCollector)
    """
    enforcer = ChainOfThoughtEnforcer()
    collector = CoTDistillationCollector(
        min_quality=min_quality,
        max_samples=max_samples,
    )
    
    return enforcer, collector
