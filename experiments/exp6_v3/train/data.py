"""v3.0 SFT data pipeline: mixture streaming, masking, FIM, hashline, decontam.

Design constraints this file owns (REVIEW_2026-07.md sections 7, 9, 11):

- **Deterministic + resumable.** The mixture is driven by a seeded RNG and
  per-source counters. A checkpoint stores `cursor()`; `skip_to(cursor)`
  fast-forwards each source independently so resumed data == unseen data.
  (Fixes the naive global fast-forward from the first notebook draft, which
  desynced whenever source schemas changed relative sampling.)
- **Completion-only loss.** Prompt tokens are masked to -100; the model
  learns the response, not the prompt.
- **Two dialects only** (section 11): plain completion (bash/mini-swe-agent
  world needs no schema) and the hashline edit dialect for edit training.
- **FIM** applied opportunistically: only if the tokenizer actually has FIM
  sentinels (runtime check — Qwen3.5's set is verified at C1, not assumed).
- **Decontamination-lite (R21-floor):** 10-gram overlap filter against the
  eval prompts/solutions, built at startup. The full MinHash+AST gate is a
  v3-full requirement; this floor keeps v3.0 SFT publishable.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- text utils

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (donor thinking-mode traces)."""
    return _THINK_RE.sub("", text).strip()


def extract_code(text: str) -> str:
    """First fenced code block if present, else the raw text."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# --------------------------------------------------------------------------- hashline edit dialect

def _line_hash(line: str) -> str:
    """8-hex content hash of a line, whitespace-normalized (OMP hashline idea)."""
    norm = " ".join(line.split())
    return hashlib.blake2b(norm.encode(), digest_size=4).hexdigest()


def render_hashline_edit(old_code: str, new_code: str, path: str = "file.py") -> str | None:
    """Render old->new as hash-anchored hunks — v3's trained edit dialect.

    Each hunk is anchored to the content hash of the nearest unchanged line
    above the change, immune to whitespace drift and line-number shifts:

        <<<EDIT path
        @@ <8hex-anchor-hash> <anchor line text>
        - removed line
        + added line
        >>>

    Returns None when there is no usable diff (identical, or change at cost
    of whole file — fall back to plain rewrite for those).
    """
    import difflib

    old_lines, new_lines = old_code.splitlines(), new_code.splitlines()
    if old_lines == new_lines or not old_lines:
        return None

    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    hunks: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # anchor = last unchanged line before the hunk (or file start sentinel)
        if i1 > 0:
            anchor_line = old_lines[i1 - 1]
            anchor = f"@@ {_line_hash(anchor_line)} {anchor_line.strip()[:60]}"
        else:
            anchor = "@@ 00000000 <file-start>"
        body = [f"- {ln}" for ln in old_lines[i1:i2]] + [f"+ {ln}" for ln in new_lines[j1:j2]]
        hunks.append("\n".join([anchor, *body]))

    if not hunks:
        return None
    # degenerate diff (touches ~whole file) -> not an "edit", skip dialect
    changed = sum(h.count("\n") for h in hunks)
    if changed >= 2 * max(len(old_lines), 1):
        return None
    return f"<<<EDIT {path}\n" + "\n".join(hunks) + "\n>>>"


# --------------------------------------------------------------------------- FIM

FIM_TOKENS = ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")


def tokenizer_has_fim(tokenizer: Any) -> bool:
    vocab = tokenizer.get_vocab()
    return all(t in vocab for t in FIM_TOKENS)


def fim_transform(code: str, rng: random.Random, min_span: int = 3) -> str | None:
    """PSM-format FIM sample from a code string; None if too short."""
    lines = code.splitlines()
    if len(lines) < 3 * min_span:
        return None
    a = rng.randrange(min_span, len(lines) - 2 * min_span)
    b = rng.randrange(a + min_span, len(lines) - min_span)
    prefix, middle, suffix = "\n".join(lines[:a]), "\n".join(lines[a:b]), "\n".join(lines[b:])
    p, m, s = FIM_TOKENS
    return f"{p}{prefix}{s}{suffix}{m}{middle}"


# --------------------------------------------------------------------------- decontamination-lite

def _ngrams(text: str, n: int = 10) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {hash(tuple(words[i : i + n])) for i in range(len(words) - n + 1)}


class DecontamFilter:
    """10-gram collision filter against eval prompts/solutions (R21 floor)."""

    def __init__(self, eval_texts: list[str], n: int = 10, max_collisions: int = 2) -> None:
        self.n = n
        self.max_collisions = max_collisions
        self.eval_grams: set[int] = set()
        for t in eval_texts:
            self.eval_grams |= _ngrams(t, n)
        self.rejected = 0

    def clean(self, sample_text: str) -> bool:
        """True if the sample is clean (allowed into training)."""
        if not self.eval_grams:
            return True
        hits = len(_ngrams(sample_text, self.n) & self.eval_grams)
        if hits > self.max_collisions:
            self.rejected += 1
            return False
        return True


# --------------------------------------------------------------------------- source schema mapping

@dataclass
class Sample:
    prompt: str
    response: str
    kind: str  # "reasoning" | "edit" | "fim" | "plain"


def map_opencodereasoning(ex: dict) -> Sample | None:
    q = ex.get("input") or ex.get("question") or ""
    a = ex.get("output") or ex.get("response") or ""
    if not q or not a:
        return None
    return Sample(prompt=q.strip(), response=a.strip(), kind="reasoning")


def map_commitpackft(ex: dict) -> Sample | None:
    old, new = ex.get("old_contents") or "", ex.get("new_contents") or ""
    msg = ex.get("message") or ex.get("subject") or "apply the requested change"
    if not old or not new:
        return None
    edit = render_hashline_edit(old, new, path=ex.get("new_file") or "file.py")
    if edit is None:  # whole-file rewrite — still useful as plain sample
        prompt = f"Edit the following file. Task: {msg}\n\n```\n{old}\n```"
        return Sample(prompt=prompt, response=f"```\n{new}\n```", kind="plain")
    prompt = (
        f"Edit the following file using hash-anchored hunks. Task: {msg}\n\n```\n{old}\n```"
    )
    return Sample(prompt=prompt, response=edit, kind="edit")


SCHEMA_MAPPERS: dict[str, Callable[[dict], Sample | None]] = {
    "opencodereasoning": map_opencodereasoning,
    "commitpackft": map_commitpackft,
}


# --------------------------------------------------------------------------- mixture stream

@dataclass
class SourceSpec:
    name: str                     # key into SCHEMA_MAPPERS
    weight: float                 # relative sampling weight
    iterator_factory: Callable[[], Iterator[dict]]  # fresh raw-example iterator
    fim_fraction: float = 0.0     # fraction of samples converted to FIM


@dataclass
class MixtureCursor:
    consumed: dict[str, int] = field(default_factory=dict)  # per-source raw count
    draws: int = 0                                          # RNG draw count


class MixtureStream:
    """Seeded weighted mixture with per-source resumable cursors.

    Determinism contract: same seed + same SourceSpec list + same cursor
    => identical continuation, because (a) source choice comes from a
    dedicated RNG advanced exactly once per draw, and (b) each source
    iterator is advanced only when chosen, with counts tracked per source.
    """

    def __init__(
        self,
        sources: list[SourceSpec],
        seed: int,
        decontam: DecontamFilter | None = None,
    ) -> None:
        self.sources = sources
        self.seed = seed
        self.decontam = decontam
        self.rng = random.Random(seed)          # source-choice RNG
        self.fim_rng = random.Random(seed + 1)  # FIM span RNG (advances per FIM sample)
        self.iters = {s.name: s.iterator_factory() for s in sources}
        self.cursor_state = MixtureCursor(consumed={s.name: 0 for s in sources})
        self._weights = [s.weight for s in sources]

    # -- cursor ------------------------------------------------------------
    def cursor(self) -> dict:
        return {
            "consumed": dict(self.cursor_state.consumed),
            "draws": self.cursor_state.draws,
            "seed": self.seed,
        }

    def skip_to(self, cursor: dict) -> None:
        """Fast-forward: replay RNG draws, skip consumed raw examples per source."""
        if cursor.get("seed") != self.seed:
            raise ValueError(
                f"cursor seed {cursor.get('seed')} != stream seed {self.seed}; "
                "refusing to resume with a different data order"
            )
        for _ in range(cursor["draws"]):
            self.rng.choices(self.sources, weights=self._weights)
        for name, count in cursor["consumed"].items():
            it = self.iters[name]
            for _ in range(count):
                next(it, None)
        self.cursor_state = MixtureCursor(consumed=dict(cursor["consumed"]), draws=cursor["draws"])
        # NOTE: fim_rng replay is driven by re-drawing; FIM spans after resume
        # differ from the never-preempted run — acceptable (content, not order).

    # -- iteration ----------------------------------------------------------
    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        for _ in range(10_000):  # bounded skip of exhausted/dirty/None samples
            spec = self.rng.choices(self.sources, weights=self._weights)[0]
            self.cursor_state.draws += 1
            raw = next(self.iters[spec.name], None)
            if raw is None:  # source exhausted -> restart (epoch wrap)
                self.iters[spec.name] = spec.iterator_factory()
                self.cursor_state.consumed[spec.name] = 0
                raw = next(self.iters[spec.name], None)
                if raw is None:
                    continue  # empty source; weight effectively wasted
            self.cursor_state.consumed[spec.name] += 1
            sample = SCHEMA_MAPPERS[spec.name](raw)
            if sample is None:
                continue
            if self.decontam is not None and not self.decontam.clean(
                sample.prompt + "\n" + sample.response
            ):
                continue
            if spec.fim_fraction > 0 and self.fim_rng.random() < spec.fim_fraction:
                fim = fim_transform(extract_code(sample.response), self.fim_rng)
                if fim is not None:
                    return Sample(prompt="", response=fim, kind="fim")
            return sample
        raise RuntimeError("MixtureStream: 10k consecutive rejects — check sources/filters")


# --------------------------------------------------------------------------- tokenization + masking

def encode_completion_masked(
    tokenizer: Any,
    sample: Sample,
    max_len: int,
    use_chat_template: bool = True,
) -> dict | None:
    """input_ids + labels with prompt masked to -100 (completion-only loss).

    FIM samples skip the chat template (raw continuation objective).
    Returns None if the sample doesn't fit or is degenerate.
    """
    if sample.kind == "fim" or not use_chat_template or not sample.prompt:
        text = sample.response
        ids = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt").input_ids[0]
        if len(ids) < 8:
            return None
        return {"input_ids": ids, "labels": ids.clone()}

    if getattr(tokenizer, "chat_template", None) is None:
        # base-model / dry-run fallback: plain concat, same masking contract
        prompt_text = f"USER: {sample.prompt}\nASSISTANT: "
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(
            prompt_text + sample.response, truncation=True, max_length=max_len,
            return_tensors="pt",
        ).input_ids[0]
    else:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample.prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )[0]
        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": sample.prompt},
                {"role": "assistant", "content": sample.response},
            ],
            return_tensors="pt",
        )[0]
    if len(full_ids) > max_len or len(full_ids) - len(prompt_ids) < 4:
        return None
    labels = full_ids.clone()
    labels[: len(prompt_ids)] = -100
    return {"input_ids": full_ids, "labels": labels}
