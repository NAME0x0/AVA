"""
Articulation Model for AVA

Simulates speech development - at early stages, output is garbled
and unclear, gradually becoming more coherent as AVA matures.
"""

import random
import re
from typing import Dict, List, Tuple


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
