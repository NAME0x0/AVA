# HF Research Session: architecture-rag-tokenizer-2026-03-15-v2

## Focus

Recent Hugging Face and arXiv papers that materially affect AVA's tool-use, planning, and multimodal roadmap.

## Recent Papers Selected From The HF Feed

- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (2025-02-07) -> Test-time loops over a shared latent block are the strongest current evidence that extra compute can substitute for extra parameters, but AVA needs a cleaner recurrent core than the current naive looped baseline.
- From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (2025-02-20) -> Graph-style memory plus passage integration is a better target than plain vector lookup for AVA's long-term memory layer.
- Qwen3 Technical Report (2025-05-14) -> Qwen3 is the strongest current open reference for tokenizer quality, multilingual coverage, and explicit thinking-budget control; AVA should steal the tokenizer lesson before the large-scale model lesson.
- In-Context Reinforcement Learning for Tool Use in Large Language Models (2026-03-09) -> Promising later-stage tool RL recipe: warm-start tool behavior with in-context examples during rollouts, then anneal toward zero-shot tool use.
- DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints (2026-01-26) -> AVA should add explicit planning benchmarks before making strong agent claims.
- SkillNet: Create, Evaluate, and Connect AI Skills (2026-02-26) -> Reusable skills belong in the product runtime layer and evaluation stack more than in the base model weights.
- Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders (2026-03-06) -> For AVA's future vision branch, compact multimodal design matters more than copying large VLM recipes.
- InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing (2026-03-10) -> Useful scale reference for a later multimodal family, but too broad and heavy for AVA's current 4 GB mainline.
- ToRL: Scaling Tool-Integrated RL (2025-03-30) -> Pure reward-driven tool use can produce strategic invocation behavior once the RL stack is stable enough.
- ReTool: Reinforcement Learning for Strategic Tool Use in LLMs (2025-04-15) -> Tool RL is strongest when the model learns when not to call tools as well as how to call them.
- AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning (2025-05-30) -> Only invest in large RL systems work if rollout throughput becomes the bottleneck after simpler tool and verifier loops already work.

## What Changes Now

- ICRL is useful, but only after compact supervised tool traces already work; it is not a replacement for AVA's first tool SFT pass.
- Add DeepPlanning to the benchmark contract now so long-horizon planning claims are measured before any agent push.
- Treat SkillNet as a runtime and evaluation reference for reusable skills, not as a base-model scaling recipe.
- Keep Penguin-VL as the leading compact multimodal reference for AVA's future vision branch.

## What Stays Later

- Use ToRL and ReTool as RL baselines after the tool RL environment and verifiers are stable.
- Use InternVL-U as a larger multimodal reference point, not a 4 GB mainline target.
- Use AReaL only if rollout throughput becomes the bottleneck after simpler RL loops already work.

## Relevant Benchmarks

- BFCL (tool_use, foundation)
- DeepPlanning (long_horizon_planning, agentic)
- tau2-bench (policy_constrained_tool_use, scale_only)
- MathVista (visual_math_reasoning, multimodal)
- ScienceQA (multimodal_science_reasoning, multimodal)
- DocVQA (document_reasoning, multimodal)

## Queued Experiments

- exp-013: Latent recurrent-depth AVA-v2 core
- exp-014: Open-tokenizer reboot around Qwen-style segmentation
- exp-015: RAFT-style retrieval-aware distillation
- exp-016: HippoRAG2-style structured memory graph
- exp-009: In-context tool RL after compact tool SFT
- exp-010: Planning benchmarks before agent claims
- exp-011: Compact multimodal branch after text stability
- exp-012: Async RL infra only when rollouts dominate cost

## Next Actions

- Finish a stronger compact tool-supervision packet before opening the ICRL, ToRL, or ReTool branch.
- Keep DeepPlanning, BFCL, and tau2-bench visible together so tool use and planning are measured separately.
- Do not start a multimodal training branch until the text tokenizer, tool data, and compliance behavior are stable.
- When multimodal work starts, begin with a compact encoder path and MathVista or ScienceQA style targets rather than a broad unified model.
