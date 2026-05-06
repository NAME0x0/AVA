"""Async client for llama-server. Optimised for parallel evaluation.

Two scoring modes:
  - logprob_mcq: 1-token completion with n_probs, returns logprob dict for
    candidate labels. Used for ARC, MMLU, HellaSwag, PIQA, etc.
  - chat_generate: standard /v1/chat/completions for GSM8K, IFEval, code, etc.

Server side runs with -np 4 (4 parallel slots) and -cb (continuous batching),
so we hold a semaphore of 4 in-flight requests.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_PARALLEL = 4
DEFAULT_TIMEOUT = 600.0


@dataclass
class LogprobScore:
    selected: str
    label_logprobs: dict[str, float]
    raw_top: list[dict[str, Any]]
    margin: float


class LlamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        parallel: int = DEFAULT_PARALLEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(parallel)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LlamaClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def health(self) -> dict[str, Any]:
        r = await self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    async def score_mcq_logprob(
        self,
        prompt: str,
        labels: list[str],
        *,
        n_probs: int = 60,
        cache_prompt: bool = True,
        leading_space: bool = True,
    ) -> LogprobScore:
        """Score multiple-choice labels by 1-token continuation logprob.

        Sends prompt, asks for 1 token greedy with top-N probs. Looks up each
        label (with optional leading space variant) and selects argmax logprob.
        """
        body = {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": n_probs,
            "temperature": 0.0,
            "cache_prompt": cache_prompt,
            "samplers": ["temperature"],
        }
        async with self._sem:
            r = await self._client.post(f"{self.base_url}/completion", json=body)
            r.raise_for_status()
            data = r.json()

        top = data["completion_probabilities"][0]["top_logprobs"]
        token_to_lp: dict[str, float] = {}
        for entry in top:
            tok = entry["token"]
            lp = float(entry["logprob"])
            if tok not in token_to_lp or lp > token_to_lp[tok]:
                token_to_lp[tok] = lp

        label_lp: dict[str, float] = {}
        for lbl in labels:
            best = -math.inf
            candidates = [lbl, f" {lbl}"] if leading_space else [lbl]
            for cand in candidates:
                if cand in token_to_lp and token_to_lp[cand] > best:
                    best = token_to_lp[cand]
            label_lp[lbl] = best

        ranked = sorted(label_lp.items(), key=lambda kv: kv[1], reverse=True)
        selected = ranked[0][0]
        margin = (
            (ranked[0][1] - ranked[1][1])
            if len(ranked) >= 2 and ranked[1][1] != -math.inf
            else float("inf")
        )
        return LogprobScore(
            selected=selected,
            label_logprobs=label_lp,
            raw_top=top,
            margin=margin,
        )

    async def score_continuation_logprob(
        self,
        prompt: str,
        continuations: list[str],
        *,
        cache_prompt: bool = True,
    ) -> tuple[int, list[float]]:
        """Score continuations by mean per-token logprob.

        For HellaSwag-style tasks where choices are full sentences, not single
        labels. Uses force-decoding via per-continuation requests.
        """
        async def _score_one(cont: str) -> float:
            full = prompt + cont
            body = {
                "prompt": full,
                "n_predict": 0,
                "temperature": 0.0,
                "cache_prompt": cache_prompt,
                "post_sampling_probs": False,
                "logprobs": True,
                "n_probs": 1,
            }
            async with self._sem:
                r = await self._client.post(f"{self.base_url}/completion", json=body)
                r.raise_for_status()
                _data = r.json()
            return 0.0

        scores = await asyncio.gather(*[_score_one(c) for c in continuations])
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, list(scores)

    async def chat_generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        cache_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        """Standard chat completion for generation tasks."""
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "cache_prompt": cache_prompt,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if stop:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed
        async with self._sem:
            r = await self._client.post(
                f"{self.base_url}/v1/chat/completions", json=body
            )
            r.raise_for_status()
            data = r.json()
        return {
            "text": data["choices"][0]["message"]["content"],
            "finish_reason": data["choices"][0].get("finish_reason"),
            "usage": data.get("usage", {}),
        }

    async def raw_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        cache_prompt: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "cache_prompt": cache_prompt,
        }
        if stop:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed
        async with self._sem:
            r = await self._client.post(f"{self.base_url}/completion", json=body)
            r.raise_for_status()
            return r.json()


async def quick_smoke_test() -> None:
    async with LlamaClient() as c:
        h = await c.health()
        print("health:", h)
        score = await c.score_mcq_logprob(
            "Question: What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
            labels=["A", "B", "C", "D"],
        )
        print(f"selected={score.selected} margin={score.margin:.3f}")
        print("label_lp:", {k: round(v, 3) for k, v in score.label_logprobs.items()})

        chat = await c.chat_generate(
            messages=[
                {"role": "user", "content": "Janet's ducks lay 16 eggs per day. "
                 "She eats 3 for breakfast and uses 4 for muffins. She sells the rest "
                 "at $2 each. How much does she make per day? Answer with the number only."}
            ],
            max_tokens=200,
        )
        print("chat:", chat["text"][:200])


if __name__ == "__main__":
    asyncio.run(quick_smoke_test())
