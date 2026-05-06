"""Translation proxy: legacy-OpenAI logprobs <-> llama-server new format.

lm-eval-harness expects /v1/completions response shape:
  choices[i].logprobs.token_logprobs : list[float]
  choices[i].logprobs.tokens         : list[str]
  choices[i].logprobs.top_logprobs   : list[dict[str, float]]

llama-server returns:
  choices[i].logprobs.content[k]: {token, logprob, top_logprobs:[{token, logprob}]}

Proxy listens on :8766, forwards to llama-server :8765, rewrites response.
Echo + multi-token loglikelihood works because llama-server returns one entry
per generated/echoed token in `content[]`.
"""
from __future__ import annotations

import asyncio
import sys

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

UPSTREAM = "http://127.0.0.1:8765"
PORT = 8766

app = FastAPI()
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    return _client


def _shape_logprobs(logprobs: dict | None) -> dict | None:
    if logprobs is None:
        return None
    if "token_logprobs" in logprobs:
        return logprobs
    content = logprobs.get("content")
    if not content:
        return logprobs
    tokens = []
    token_logprobs = []
    top_logprobs = []
    for entry in content:
        tokens.append(entry.get("token", ""))
        token_logprobs.append(float(entry.get("logprob", 0.0)))
        top = entry.get("top_logprobs", []) or []
        top_map = {t.get("token", ""): float(t.get("logprob", 0.0)) for t in top}
        if entry.get("token", "") not in top_map:
            top_map[entry.get("token", "")] = float(entry.get("logprob", 0.0))
        top_logprobs.append(top_map)
    out = dict(logprobs)
    out["tokens"] = tokens
    out["token_logprobs"] = token_logprobs
    out["top_logprobs"] = top_logprobs
    out["text_offset"] = list(range(len(tokens)))
    return out


def _shape_response(data: dict) -> dict:
    if "choices" not in data:
        return data
    for ch in data["choices"]:
        if "logprobs" in ch:
            ch["logprobs"] = _shape_logprobs(ch["logprobs"])
    return data


@app.post("/v1/completions")
async def completions(req: Request) -> Response:
    body = await req.json()
    body.setdefault("cache_prompt", True)
    client = _get_client()
    r = await client.post(f"{UPSTREAM}/v1/completions", json=body)
    if r.status_code != 200:
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    return JSONResponse(_shape_response(r.json()))


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> Response:
    body = await req.json()
    body.setdefault("cache_prompt", True)
    client = _get_client()
    r = await client.post(f"{UPSTREAM}/v1/chat/completions", json=body)
    if r.status_code != 200:
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    return JSONResponse(_shape_response(r.json()))


@app.get("/v1/models")
async def models() -> Response:
    client = _get_client()
    r = await client.get(f"{UPSTREAM}/v1/models")
    return JSONResponse(r.json())


@app.get("/health")
async def health() -> Response:
    client = _get_client()
    r = await client.get(f"{UPSTREAM}/health")
    return JSONResponse(r.json())


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, req: Request) -> Response:
    client = _get_client()
    body = await req.body()
    method = req.method
    url = f"{UPSTREAM}/{path}"
    if method == "GET":
        r = await client.get(url, params=req.query_params)
    else:
        r = await client.post(url, content=body, headers={"content-type": "application/json"})
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
