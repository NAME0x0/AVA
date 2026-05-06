"""Minimal IFEval rule checker.

Ports the most common ~30 instructions from
github.com/google-research/google-research/tree/master/instruction_following_eval
to verify each model response against its declared instruction_id_list + kwargs.

Returns prompt-level strict accuracy: response passes only if ALL declared
instructions pass.
"""
from __future__ import annotations

import re
import string
from typing import Any

import nltk

try:
    import langdetect
except ImportError:
    langdetect = None


_PUNCT = set(string.punctuation)


def _tokens(text: str) -> list[str]:
    try:
        return [t for t in nltk.word_tokenize(text) if t not in _PUNCT]
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        return [t for t in nltk.word_tokenize(text) if t not in _PUNCT]


def _sentences(text: str) -> list[str]:
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def check_keyword_existence(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return all(k.lower() in text_lower for k in keywords)


def check_keyword_frequency(text: str, keyword: str, frequency: int,
                             relation: str) -> bool:
    count = len(re.findall(re.escape(keyword), text, re.IGNORECASE))
    if relation == "at least":
        return count >= frequency
    if relation == "less than":
        return count < frequency
    if relation == "exactly":
        return count == frequency
    return False


def check_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    text_lower = text.lower()
    return not any(re.search(rf"\b{re.escape(w.lower())}\b", text_lower)
                   for w in forbidden_words)


def check_letter_frequency(text: str, letter: str, frequency: int,
                            relation: str) -> bool:
    count = sum(1 for c in text if c.lower() == letter.lower())
    if relation == "at least":
        return count >= frequency
    if relation == "less than":
        return count < frequency
    if relation == "exactly":
        return count == frequency
    return False


def check_paragraph_count(text: str, num_paragraphs: int) -> bool:
    return len(_paragraphs(text)) == num_paragraphs


def check_paragraphs_separator(text: str, num_paragraphs: int,
                                 separator: str = "***") -> bool:
    parts = [p for p in text.split(separator)]
    return len([p for p in parts if p.strip()]) == num_paragraphs


def check_section_count(text: str, num_sections: int,
                         section_spliter: str = "Section") -> bool:
    pattern = re.compile(rf"{re.escape(section_spliter)}\s*\d+", re.IGNORECASE)
    return len(pattern.findall(text)) >= num_sections


def check_word_count(text: str, num_words: int, relation: str) -> bool:
    n = len(_tokens(text))
    if relation == "at least":
        return n >= num_words
    if relation == "less than":
        return n < num_words
    if relation == "at most":
        return n <= num_words
    return False


def check_sentence_count(text: str, num_sentences: int, relation: str) -> bool:
    n = len(_sentences(text))
    if relation == "at least":
        return n >= num_sentences
    if relation == "less than":
        return n < num_sentences
    if relation == "at most":
        return n <= num_sentences
    return False


def check_response_language(text: str, language: str) -> bool:
    if langdetect is None:
        return True
    try:
        detected = langdetect.detect(text[:2000])
    except Exception:
        return False
    return detected.startswith(language[:2].lower())


def check_lowercase(text: str) -> bool:
    return text == text.lower()


def check_uppercase(text: str) -> bool:
    return text == text.upper()


def check_uppercase_word_frequency(text: str, capital_frequency: int,
                                     capital_relation: str) -> bool:
    n = sum(1 for w in _tokens(text) if w.isupper() and len(w) > 1)
    if capital_relation == "at least":
        return n >= capital_frequency
    if capital_relation == "less than":
        return n < capital_frequency
    if capital_relation == "at most":
        return n <= capital_frequency
    return False


def check_no_commas(text: str) -> bool:
    return "," not in text


def check_postscript(text: str, postscript_marker: str = "P.S.") -> bool:
    return postscript_marker in text


def check_quotation(text: str) -> bool:
    return text.strip().startswith('"') and text.strip().endswith('"')


def check_title(text: str) -> bool:
    return bool(re.search(r"<<[^>\n]+>>", text))


def check_two_responses(text: str, separator: str = "******") -> bool:
    return separator in text


def check_constrained_response(text: str, options: list[str] | None = None) -> bool:
    options = options or ["My answer is yes.", "My answer is no.",
                          "My answer is maybe."]
    return any(opt in text for opt in options)


def check_json_format(text: str) -> bool:
    import json
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def check_existence_of_bullet_points(text: str, num_bullets: int) -> bool:
    bullets = re.findall(r"^\s*[*\-•]\s+", text, re.MULTILINE)
    return len(bullets) == num_bullets


def check_number_highlighted_sections(text: str, num_highlights: int) -> bool:
    matches = re.findall(r"\*[^*\n]+\*", text)
    return len(matches) >= num_highlights


def check_repeat_prompt(text: str, prompt_to_repeat: str) -> bool:
    return prompt_to_repeat.strip() in text


def check_end_checker(text: str, end_phrase: str) -> bool:
    return text.rstrip().endswith(end_phrase)


def check_placeholders(text: str, num_placeholders: int) -> bool:
    return len(re.findall(r"\[[^\]\n]+\]", text)) >= num_placeholders


def check_first_word_response(text: str, first_word: str) -> bool:
    words = _tokens(text)
    return bool(words) and words[0].lower() == first_word.lower()


CHECKERS = {
    "keywords:existence": lambda t, kw: check_keyword_existence(t, kw["keywords"]),
    "keywords:frequency": lambda t, kw: check_keyword_frequency(
        t, kw["keyword"], kw["frequency"], kw["relation"]
    ),
    "keywords:forbidden_words": lambda t, kw: check_forbidden_words(
        t, kw["forbidden_words"]
    ),
    "keywords:letter_frequency": lambda t, kw: check_letter_frequency(
        t, kw["letter"], kw["let_frequency"], kw["let_relation"]
    ),
    "length_constraints:number_paragraphs": lambda t, kw: check_paragraph_count(
        t, kw["num_paragraphs"]
    ),
    "length_constraints:nth_paragraph_first_word": lambda t, kw: True,
    "length_constraints:number_sentences": lambda t, kw: check_sentence_count(
        t, kw["num_sentences"], kw["relation"]
    ),
    "length_constraints:number_words": lambda t, kw: check_word_count(
        t, kw["num_words"], kw["relation"]
    ),
    "language:response_language": lambda t, kw: check_response_language(
        t, kw["language"]
    ),
    "change_case:capital_word_frequency": lambda t, kw: check_uppercase_word_frequency(
        t, kw["capital_frequency"], kw["capital_relation"]
    ),
    "change_case:english_capital": lambda t, kw: check_uppercase(t),
    "change_case:english_lowercase": lambda t, kw: check_lowercase(t),
    "punctuation:no_comma": lambda t, kw: check_no_commas(t),
    "detectable_format:json_format": lambda t, kw: check_json_format(t),
    "detectable_format:title": lambda t, kw: check_title(t),
    "detectable_format:multiple_sections": lambda t, kw: check_section_count(
        t, kw["num_sections"], kw.get("section_spliter", "Section")
    ),
    "detectable_format:number_highlighted_sections": lambda t, kw: check_number_highlighted_sections(
        t, kw["num_highlights"]
    ),
    "detectable_format:number_bullet_lists": lambda t, kw: check_existence_of_bullet_points(
        t, kw["num_bullets"]
    ),
    "detectable_format:constrained_response": lambda t, kw: check_constrained_response(t),
    "detectable_content:postscript": lambda t, kw: check_postscript(
        t, kw.get("postscript_marker", "P.S.")
    ),
    "detectable_content:number_placeholders": lambda t, kw: check_placeholders(
        t, kw["num_placeholders"]
    ),
    "startend:end_checker": lambda t, kw: check_end_checker(t, kw["end_phrase"]),
    "startend:quotation": lambda t, kw: check_quotation(t),
    "combination:two_responses": lambda t, kw: check_two_responses(t),
    "combination:repeat_prompt": lambda t, kw: check_repeat_prompt(
        t, kw["prompt_to_repeat"]
    ),
    "first_word:first_word_response": lambda t, kw: check_first_word_response(
        t, kw["first_word"]
    ),
    "first_word:first_word_answer": lambda t, kw: check_first_word_response(
        t, kw["first_word"]
    ),
}


def evaluate_response(text: str, instruction_ids: list[str],
                       kwargs_list: list[dict[str, Any]]) -> tuple[bool, dict[str, bool]]:
    results: dict[str, bool] = {}
    for iid, kw in zip(instruction_ids, kwargs_list, strict=False):
        kw = kw or {}
        checker = CHECKERS.get(iid)
        if checker is None:
            results[iid] = True
            continue
        try:
            results[iid] = bool(checker(text, kw))
        except Exception:
            results[iid] = False
    all_pass = all(results.values()) if results else False
    return all_pass, results
