# Research Roadmap

This roadmap maps current arXiv work to concrete AVA experiments.

## Paper Set

- `Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity` (`arXiv:2101.03961`, submitted January 11, 2021; revised June 16, 2022)
  AVA takeaway: sparse MoE raises parameter count at near-constant compute, but routing complexity and instability remain real costs.
- `ST-MoE: Designing Stable and Transferable Sparse Expert Models` (`arXiv:2202.08906`, submitted February 17, 2022; revised April 29, 2022)
  AVA takeaway: stable sparse experts are possible, but the useful regimes are still far above AVA's hardware envelope.
- `LIMO: Less is More for Reasoning` (`arXiv:2502.03387`, submitted February 5, 2025; revised July 29, 2025)
  AVA takeaway: use a small, carefully designed reasoning set before scaling data volume.
- `DeepSeek-R1` (`arXiv:2501.12948`, submitted January 22, 2025; revised January 4, 2026)
  AVA takeaway: use verifiable RL after a stable base model exists.
- `s1: Simple test-time scaling` (`arXiv:2501.19393`, submitted January 31, 2025; revised March 1, 2025)
  AVA takeaway: apply extra thinking budget only on the hardest reasoning tasks, beginning with math.
- `ToolACE-R` (`arXiv:2504.01400`, submitted April 2, 2025; revised January 10, 2026)
  AVA takeaway: refine tool data iteratively against the current model.
- `Toolformer` (`arXiv:2302.04761`, submitted February 9, 2023)
  AVA takeaway: tool use can be taught with very few demonstrations if traces are structured well.
- `DeepSeekMath` (`arXiv:2402.03300`, submitted February 5, 2024; revised April 27, 2024)
  AVA takeaway: domain-adaptive pretraining on math-heavy tokens matters.
- `Titans` (`arXiv:2501.00663`, submitted December 31, 2024)
  AVA takeaway: keep long-term memory external and gated by surprise.
- `Phi-4 Technical Report` (`arXiv:2412.08905`, submitted December 12, 2024)
  AVA takeaway: data quality and synthetic curriculum can dominate raw size.
- `Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach` (`arXiv:2502.05171`, submitted February 7, 2025)
  AVA takeaway: if AVA wants more reasoning without more parameters, recurrent latent loops are the strongest current architecture lead.
- `From RAG to Memory: Non-Parametric Continual Learning for Large Language Models` (`arXiv:2502.14802`, submitted February 20, 2025)
  AVA takeaway: structured memory is a stronger target than flat vector lookup for the next retrieval branch.
- `RAFT: Adapting Language Model to Domain Specific RAG` (`arXiv:2403.10131`, submitted March 15, 2024)
  AVA takeaway: distillation should teach evidence selection under distractors, not just answer memorization.
- `Qwen3 Technical Report` (`arXiv:2505.09388`, submitted May 14, 2025)
  AVA takeaway: AVA should steal the open-tokenizer and thinking-budget lessons before copying any large-model scale.
- `DAPO` (`arXiv:2503.14476`, submitted March 18, 2025; revised May 20, 2025)
  AVA takeaway: RL infrastructure quality matters if we add verifiable post-training.
- `In-Context Reinforcement Learning for Tool Use in Large Language Models` (`arXiv:2603.08068`, submitted March 9, 2026)
  AVA takeaway: promising tool-RL recipe, but only after compact supervised tool traces already work.
- `ToRL: Scaling Tool-Integrated RL` (`arXiv:2503.23383`, submitted March 30, 2025)
  AVA takeaway: tool-integrated RL can improve strategic tool choice once the RL stack is stable.
- `ReTool: Reinforcement Learning for Strategic Tool Use in LLMs` (`arXiv:2504.11536`, submitted April 15, 2025; revised April 17, 2025)
  AVA takeaway: tool-use quality includes good abstention, not only successful calls.
- `DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints` (`arXiv:2601.18137`, submitted January 26, 2026)
  AVA takeaway: planning should be benchmarked explicitly before AVA makes strong agentic claims.
- `SkillNet: Create, Evaluate, and Connect AI Skills` (`arXiv:2603.04448`, submitted February 26, 2026)
  AVA takeaway: reusable skills belong in the runtime and evaluation stack more than in the base weights.
- `Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders` (`arXiv:2603.06569`, submitted March 6, 2026)
  AVA takeaway: compact multimodal design should guide AVA's future vision branch.
- `InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing` (`arXiv:2603.09877`, submitted March 10, 2026)
  AVA takeaway: strong multimodal reference point, but too broad for AVA's current 4 GB mainline.
- `AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning` (`arXiv:2505.24298`, submitted May 30, 2025; revised March 2, 2026)
  AVA takeaway: async RL infra only matters if rollout throughput becomes the bottleneck.
- `Mixtral of Experts` (`arXiv:2401.04088`, submitted January 8, 2024)
  AVA takeaway: sparse MoE can beat larger dense models, but storage cost still dominates at 4 GB.
- `DeepSeekMoE` (`arXiv:2401.06066`, submitted January 11, 2024)
  AVA takeaway: better expert specialization helps efficiency, but not enough to erase 4 GB residency limits.
- `JetMoE` (`arXiv:2404.07413`, submitted April 11, 2024)
  AVA takeaway: accessible MoE is real, but still not a comfortable 4 GB standalone target.
- `BitNet b1.58 2B4T Technical Report` (`arXiv:2504.12285`, submitted April 16, 2025; revised April 25, 2025)
  AVA takeaway: compression is a deployment and inference branch, not the only path to quality.
- `DeepSeek-V3 Technical Report` (`arXiv:2412.19437`, submitted December 27, 2024; revised February 18, 2025)
  AVA takeaway: frontier-competitive MoE exists, but only at scales that are incompatible with AVA's local-first target.
- `QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models` (`arXiv:2310.16795`, submitted October 25, 2023)
  AVA takeaway: even aggressive MoE compression is still a server-scale story, not a 4 GB laptop story.
- `SqueezeLLM: Dense-and-Sparse Quantization` (`arXiv:2306.07629`, submitted June 13, 2023; revised June 5, 2024)
  AVA takeaway: single-batch inference is memory-bandwidth bound, so expert offload can erase sparse-compute gains.

## Phase Order

1. Dense base model
2. High-quality language, math, science, and code curriculum
3. Compact tool-use traces
4. Planning and skill evaluation scaffolding
5. Verifiable RL on math, science, coding, and tool tasks
6. Recurrent-depth AVA-v2 architecture branch
7. Selective test-time scaling on the hardest queries
8. External memory for product behavior
9. Compact multimodal branch after the text line is stable
10. Aggressive deployment compression
11. Tiny MoE branch only after a strong dense teacher exists

## Mainline Decision

The mainline AVA path stays compact and text-first, but it is no longer just “dense GPT plus patches.”

Current AVA remains a hybrid product stack: compact checkpoint plus transparent external retrieval and memory. AVA-v2 should open three deeper branches in parallel: a stronger open tokenizer, a cleaner recurrent-depth student core, and a more structured retrieval/memory layer.

Sparse MoE remains a research branch for later because the main 4 GB problem is total weight residency, bandwidth, KV/runtime overhead, and training budget, not just FLOPs per token.

Tool RL is no longer just a paper watch item. AVA now has a first verifier-RL scaffold, but it is still a smoke path and should be treated as AVA-v2 infrastructure, not as a solved post-training recipe.

## Near-Term Experiments

- `exp-001`
  Build a LIMO-style micro-rationale set for language, math, science, and coding with short demonstrations only.
- `exp-002`
  Run domain-adaptive pretraining on math, science, and code before aggressive tool learning.
- `exp-003`
  Distill compact calculator traces with iterative ToolACE-R style filtering.
- `exp-004`
  Turn the new verifier-RL scaffold into a serious branch by warm-starting from a recurrent-depth checkpoint and expanding rewards beyond arithmetic into science, code, and tool-policy verification.
- `exp-005`
  Enable budget forcing only on hard reasoning prompts.
- `exp-013`
  Extend the new recurrent-depth AVA-v2 scaffold into a real training line: longer unsupervised pretraining, warm-start transfer, and recurrent-vs-transformer ablations on public benchmarks.
- `exp-014`
  Reboot AVA-v2 around a strong open tokenizer, with Qwen-style segmentation as the current lead candidate.
- `exp-015`
  Distill knowledge with RAFT-style retrieved evidence plus distractors so the student learns what to ignore, not just what to copy.
- `exp-016`
  Upgrade retrieval toward HippoRAG2-style structured memory instead of flat sparse or dense lookup alone.
- `exp-006`
  Keep Titans-inspired memory as a product-side augmentation, not as a claim that the base model has infinite context.
- `exp-007`
  Explore BitNet or related low-bit deployment only after the dense teacher is stable.
- `exp-008`
  Run a tiny MoE feasibility branch only after the dense checkpoint is strong, and judge it on narrow language, math, science, and coding wins plus real 4 GB residency.
- `exp-009`
  After tool SFT works, test ICRL or similar tool-RL warm starts that anneal away in-context tool demonstrations.
- `exp-010`
  Add DeepPlanning-style planning evaluation and SkillNet-style skill instrumentation before making agentic product claims.
- `exp-011`
  Use Penguin-VL as the first reference when AVA opens a compact multimodal branch; treat InternVL-U as a scale reference, not a target.
- `exp-012`
  Only invest in AReaL-style asynchronous RL infra if rollout throughput becomes the real bottleneck.
- `exp-017`
  Add a terminal-proof capture branch using `asciinema` plus optional `ffmpeg` rendering so experiments can ship with compact human-readable replay artifacts without replacing the structured audit trail.

