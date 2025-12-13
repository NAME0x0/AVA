"""
Developmental Filter for AVA

Applies stage-appropriate filtering to responses, ensuring that
AVA's output matches its developmental level in terms of:
- Vocabulary complexity
- Sentence structure
- Response length
- Articulation clarity
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .articulation import ArticulationModel, constrain_vocabulary


@dataclass
class FilteredResponse:
    """Result of applying developmental filtering."""
    
    original_text: str
    filtered_text: str
    clarity_applied: float
    vocabulary_range: float
    max_sentence_length: int
    transformations_applied: List[str] = field(default_factory=list)
    
    @property
    def was_modified(self) -> bool:
        """Check if the text was modified during filtering."""
        return self.original_text != self.filtered_text
    
    @property
    def reduction_ratio(self) -> float:
        """Calculate how much the response was reduced."""
        if len(self.original_text) == 0:
            return 0.0
        return 1.0 - (len(self.filtered_text) / len(self.original_text))


class DevelopmentalFilter:
    """
    Filters responses based on developmental stage.
    
    This is the main interface for applying all developmental
    constraints to AVA's output.
    """
    
    def __init__(self):
        """Initialize the developmental filter."""
        self.articulation = ArticulationModel()
        
        # Stage-based constraints
        self.stage_constraints = {
            "INFANT": {
                "clarity": 0.2,
                "vocabulary_range": 0.1,
                "max_sentence_length": 5,
                "max_sentences": 2,
                "allow_questions": False,
                "allow_complex_punctuation": False,
            },
            "TODDLER": {
                "clarity": 0.4,
                "vocabulary_range": 0.3,
                "max_sentence_length": 8,
                "max_sentences": 4,
                "allow_questions": True,
                "allow_complex_punctuation": False,
            },
            "CHILD": {
                "clarity": 0.7,
                "vocabulary_range": 0.5,
                "max_sentence_length": 15,
                "max_sentences": 6,
                "allow_questions": True,
                "allow_complex_punctuation": True,
            },
            "ADOLESCENT": {
                "clarity": 0.85,
                "vocabulary_range": 0.7,
                "max_sentence_length": 25,
                "max_sentences": 10,
                "allow_questions": True,
                "allow_complex_punctuation": True,
            },
            "YOUNG_ADULT": {
                "clarity": 0.95,
                "vocabulary_range": 0.9,
                "max_sentence_length": 40,
                "max_sentences": 15,
                "allow_questions": True,
                "allow_complex_punctuation": True,
            },
            "MATURE": {
                "clarity": 1.0,
                "vocabulary_range": 1.0,
                "max_sentence_length": 100,
                "max_sentences": 50,
                "allow_questions": True,
                "allow_complex_punctuation": True,
            },
        }
    
    def filter(
        self,
        text: str,
        stage: str,
        emotional_state: Optional[Dict[str, float]] = None,
        override_clarity: Optional[float] = None,
    ) -> FilteredResponse:
        """
        Apply developmental filtering to text.
        
        Args:
            text: The text to filter
            stage: Developmental stage name
            emotional_state: Optional emotional context for articulation
            override_clarity: Optional clarity override (for testing)
            
        Returns:
            FilteredResponse with original and filtered text
        """
        constraints = self.stage_constraints.get(stage, self.stage_constraints["INFANT"])
        transformations = []
        
        clarity = override_clarity if override_clarity is not None else constraints["clarity"]
        vocabulary_range = constraints["vocabulary_range"]
        max_sentence_length = constraints["max_sentence_length"]
        
        result = text
        
        # Step 1: Limit response length (number of sentences)
        result = self._limit_sentences(result, constraints["max_sentences"])
        if result != text:
            transformations.append("sentence_limit")
        
        # Step 2: Limit sentence length
        prev_result = result
        result = self._limit_sentence_length(result, max_sentence_length)
        if result != prev_result:
            transformations.append("sentence_truncation")
        
        # Step 3: Handle questions based on stage
        prev_result = result
        if not constraints["allow_questions"]:
            result = self._convert_questions(result)
            if result != prev_result:
                transformations.append("question_conversion")
        
        # Step 4: Handle complex punctuation
        prev_result = result
        if not constraints["allow_complex_punctuation"]:
            result = self._simplify_punctuation(result)
            if result != prev_result:
                transformations.append("punctuation_simplification")
        
        # Step 5: Constrain vocabulary
        prev_result = result
        result = constrain_vocabulary(result, vocabulary_range)
        if result != prev_result:
            transformations.append("vocabulary_constraint")
        
        # Step 6: Apply articulation model (garbling, substitutions, etc.)
        prev_result = result
        result = self.articulation.apply(result, clarity, emotional_state)
        if result != prev_result:
            transformations.append("articulation")
        
        return FilteredResponse(
            original_text=text,
            filtered_text=result,
            clarity_applied=clarity,
            vocabulary_range=vocabulary_range,
            max_sentence_length=max_sentence_length,
            transformations_applied=transformations,
        )
    
    def _limit_sentences(self, text: str, max_sentences: int) -> str:
        """Limit the number of sentences in the text."""
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Keep only max_sentences
        limited = sentences[:max_sentences]
        result = " ".join(limited)
        
        # Ensure it ends with punctuation
        if result and result[-1] not in ".!?":
            result += "."
        
        return result
    
    def _limit_sentence_length(self, text: str, max_words: int) -> str:
        """Limit the length of each sentence."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > max_words:
                # Truncate and add ellipsis or period
                truncated = " ".join(words[:max_words])
                if truncated[-1] not in ".!?":
                    truncated += "..."
                result_sentences.append(truncated)
            else:
                result_sentences.append(sentence)
        
        return " ".join(result_sentences)
    
    def _convert_questions(self, text: str) -> str:
        """Convert questions to statements for early stages."""
        # Simple conversion: replace ? with .
        # More sophisticated would rephrase, but that's complex
        return text.replace("?", ".")
    
    def _simplify_punctuation(self, text: str) -> str:
        """Remove complex punctuation for early stages."""
        # Remove semicolons, colons, parentheses, quotes
        text = re.sub(r'[;:()]', '', text)
        text = re.sub(r'["""]', '', text)
        text = re.sub(r"[''']", "'", text)
        # Replace em-dash with comma or nothing
        text = re.sub(r'[—–-]{2,}', ', ', text)
        return text
    
    def get_constraints_for_stage(self, stage: str) -> Dict[str, Any]:
        """Get the constraints for a specific stage."""
        return self.stage_constraints.get(stage, self.stage_constraints["INFANT"]).copy()
    
    def get_stage_description(self, stage: str) -> str:
        """Get a human-readable description of the stage's capabilities."""
        constraints = self.get_constraints_for_stage(stage)
        
        clarity_pct = int(constraints["clarity"] * 100)
        vocab_pct = int(constraints["vocabulary_range"] * 100)
        
        descriptions = {
            "INFANT": f"Infant stage: {clarity_pct}% clarity, very limited vocabulary ({vocab_pct}%), short responses",
            "TODDLER": f"Toddler stage: {clarity_pct}% clarity, basic vocabulary ({vocab_pct}%), can ask simple questions",
            "CHILD": f"Child stage: {clarity_pct}% clarity, growing vocabulary ({vocab_pct}%), longer responses",
            "ADOLESCENT": f"Adolescent stage: {clarity_pct}% clarity, advanced vocabulary ({vocab_pct}%), complex expression",
            "YOUNG_ADULT": f"Young adult stage: {clarity_pct}% clarity, near-full vocabulary ({vocab_pct}%)",
            "MATURE": f"Mature stage: Full clarity and vocabulary, no constraints",
        }
        
        return descriptions.get(stage, f"Unknown stage: {stage}")


class ResponseLengthManager:
    """
    Manages response length based on developmental stage and context.
    
    Early stages have shorter responses not just due to vocabulary,
    but also due to limited "attention span" and processing capability.
    """
    
    # Token limits by stage (approximate)
    STAGE_TOKEN_LIMITS = {
        "INFANT": 20,
        "TODDLER": 50,
        "CHILD": 150,
        "ADOLESCENT": 300,
        "YOUNG_ADULT": 500,
        "MATURE": 1000,
    }
    
    def get_max_tokens(self, stage: str, context_complexity: float = 0.5) -> int:
        """
        Get maximum response tokens for a stage.
        
        Args:
            stage: Developmental stage
            context_complexity: How complex the question/task is (0-1)
            
        Returns:
            Maximum tokens to generate
        """
        base_limit = self.STAGE_TOKEN_LIMITS.get(stage, 50)
        
        # Adjust for complexity - more complex questions get longer responses
        # but still bounded by stage
        multiplier = 0.5 + (context_complexity * 0.5)  # 0.5 to 1.0
        
        return int(base_limit * multiplier)
    
    def should_truncate(self, text: str, stage: str) -> bool:
        """
        Check if a response should be truncated.
        
        Uses rough word count as proxy for tokens.
        """
        max_tokens = self.STAGE_TOKEN_LIMITS.get(stage, 50)
        word_count = len(text.split())
        
        # Rough approximation: 1 token ≈ 0.75 words
        estimated_tokens = word_count / 0.75
        
        return estimated_tokens > max_tokens
    
    def truncate(self, text: str, stage: str) -> str:
        """Truncate text to stage-appropriate length."""
        max_tokens = self.STAGE_TOKEN_LIMITS.get(stage, 50)
        max_words = int(max_tokens * 0.75)
        
        words = text.split()
        if len(words) <= max_words:
            return text
        
        truncated = " ".join(words[:max_words])
        
        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        last_exclaim = truncated.rfind("!")
        last_question = truncated.rfind("?")
        
        last_punct = max(last_period, last_exclaim, last_question)
        
        if last_punct > len(truncated) * 0.5:  # At least halfway through
            truncated = truncated[:last_punct + 1]
        else:
            truncated += "..."
        
        return truncated


class EmotionalToneFilter:
    """
    Adjusts the emotional tone of responses based on stage and emotions.
    
    Early stages have more raw emotional expression, while mature
    stages can modulate and control emotional expression.
    """
    
    def __init__(self):
        # Emotion words that might be intensified or modulated
        self.emotion_intensifiers = {
            "happy": ["happy", "very happy", "so happy", "really happy"],
            "sad": ["sad", "very sad", "so sad", "really sad"],
            "angry": ["upset", "angry", "very angry", "mad"],
            "scared": ["scared", "very scared", "afraid", "really scared"],
            "excited": ["excited", "very excited", "so excited", "really excited"],
        }
    
    def apply_emotional_tone(
        self,
        text: str,
        stage: str,
        dominant_emotion: str,
        emotion_intensity: float,
    ) -> str:
        """
        Apply emotional tone adjustments to text.
        
        Args:
            text: The text to adjust
            stage: Developmental stage
            dominant_emotion: The dominant emotion
            emotion_intensity: How intense the emotion is (0-1)
            
        Returns:
            Emotionally adjusted text
        """
        # Early stages show more raw emotion
        stage_modulation = {
            "INFANT": 1.5,      # Amplified emotion
            "TODDLER": 1.3,
            "CHILD": 1.1,
            "ADOLESCENT": 1.0,  # Normal expression
            "YOUNG_ADULT": 0.9,
            "MATURE": 0.8,      # More controlled
        }
        
        modulation = stage_modulation.get(stage, 1.0)
        effective_intensity = min(1.0, emotion_intensity * modulation)
        
        # Add emotional markers based on intensity
        if effective_intensity > 0.8:
            # Very intense - add exclamation marks
            text = self._intensify_punctuation(text)
        
        return text
    
    def _intensify_punctuation(self, text: str) -> str:
        """Add emotional intensity through punctuation."""
        # Replace some periods with exclamation marks
        sentences = text.split(". ")
        result = []
        
        for i, sentence in enumerate(sentences):
            if i == 0 and sentence and sentence[-1] not in "!?":
                sentence = sentence.rstrip(".") + "!"
            result.append(sentence)
        
        return ". ".join(result)
