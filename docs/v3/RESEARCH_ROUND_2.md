# Research Round 2 — June 2026 additions

> Second independent research sweep (2026-06-11), triggered by three user-supplied
> leads (sia, turbovec, cuda-oxide), the June 2026 NVIDIA release wave, and a
> directive to examine every layer of the LLM stack (tokenizer → embeddings →
> blocks → decoding) plus model transparency. Everything here is additive to
> [CODE_PIVOT.md](CODE_PIVOT.md); nothing changes the donor-transplant spine.

---

## 1. The three leads, evaluated

### 1.1 SIA (hexo-ai/sia, MIT) — self-improvement harness pattern

A meta-agent generates a task-specific agent, a target agent executes, a
feedback agent analyses results and proposes improvements — a closed loop over
**code and harness, not weights**. Reported: 91.9% runtime reduction on GPU
kernel optimization, #1 on MLE-Bench hard.

**Verdict: adopt the pattern, not the framework.** Maps directly onto our
self-play phase C6 and the S1–S4 ladder:

- The propose→verify→distill loop gains an explicit **feedback agent role**
  (critique of failed trajectories becomes training signal, not just discard).
- SIA's headline gains came from *measurable-performance* tasks (kernel
  runtime). We add a **perf-verifiable task family** to self-play: optimize
  code against a timed harness (sandboxed `hyperfine`-style reward). Reward =
  measured speedup — un-gameable, free, and infinitely generatable.
- Confirms the INVENTION.md position: self-improvement of *pipeline and
  harness* (S3/S4) is where published evidence is strongest; weight-level
  self-modification stays out of scope.

### 1.2 turbovec (TurboQuant, MIT) — 2-bit vectors for the RAM tier

Data-oblivious 2-bit/4-bit vector quantization (random rotation → Beta-
distributed coordinates → Lloyd-Max buckets), SIMD nibble-LUT kernels.
16× compression; beats FAISS PQ on recall at 1536-dim; 10M docs: 31 GB → 4 GB.

**Verdict: direct upgrade to edge E1 (RAM-tier PKM memory).**

- PKM **value vectors quantized to 2-bit TurboQuant**: the ~1.89 B-param memory
  tier drops from ~3.8 GB fp16 RAM to **~0.5 GB + scales**. This directly
  attacks R19 (RAM-latency risk): 7× fewer bytes per fetch is 7× less
  bandwidth pressure on the EmbeddingBag gather.
- Same library gives a **zero-VRAM repo index** for the agentic harness:
  embed the user's repository once, search it in RAM. Free retrieval
  augmentation for SWE-bench-style tasks, fully air-gapped (D3-compatible).
- Both uses are MIT-licensed and CPU-only — no new budget, no new hardware.

### 1.3 cuda-oxide (NVlabs, Apache 2.0 + NV bindings) — a domain, not a dependency

Rust→PTX compiler, alpha. We do not need it as infrastructure ($0 plan trains
no custom kernels). Its value is as **coding-domain coverage and self-play
fuel**:

- GPU-kernel code (CUDA C++, PTX-adjacent Rust) joins the language matrix —
  a domain where even big generalist models are weak, and where correctness +
  speed are *mechanically checkable*.
- KernelBench-style optimize-this-kernel tasks become a C6 self-play vertical
  (see 1.1) — on free T4s, kernel speedups are measurable for $0.

---

## 2. NVIDIA June 2026 wave — what we take

| Release | What it is | What v3 takes |
|---|---|---|
| Nemotron 3 Nano (30B-A3B MoE) | Agentic MoE, ~3B active | **Above-class rival** for agentic evals (it cannot fit 4 GB resident — 30B total params; we can. That asymmetry *is* our pitch) |
| Nemotron 3 Ultra 550B-A55B (OpenMDW 1.1) | Open weights + full recipe | Recipe transparency precedent to cite; too big to touch |
| **Nemotron open datasets (~3T tokens)** | Pretraining + post-training + **RL/agentic traces**, incl. code | **Free training data.** Post-training and agentic-trace subsets slot into T4/C5 curriculum. License (OpenMDW/permissive) allows derivative training — verify per-subset at download time |
| Nemotron Flash 1B | 1B utility/routing model | Candidate **draft model** for the speculation lane (E3) if our self-drafting head underperforms; also a μP-scale sparring partner |
| Llama Embed Nemotron 8B + data + code | Open embedding recipe | Too big to ship, but its **training data** can train a tiny code-embedder for the turbovec repo index (1.2) |
| Nemotron 3 MTP heads | Multi-token-prediction heads for speculative decoding, shipped in trained checkpoints | **Direct validation of E3.** An MTP head trained into the checkpoint raises draft acceptance; ours additionally couples to the HRM halting signal (Claim 3 stands, now with stronger precedent) |

## 3. Multi-teacher distillation — the "taught by many" design

User directive: *"a model taught from multiple different models for different
things."* The 2025–26 literature says this works **only with explicit conflict
handling** — naïve trace-mixing degrades students.

What is proven:

- **TinyLLM** (multi-teacher + rationales): +5.1–15.7 pp over fine-tuning;
  students up to +23.4 pp **over their own teachers** at 1.1–26% of size.
- **Peer-review selection** across teachers: +5.48 pp over best-single-teacher
  KD (Llama-2-7B student).
- **Local Naturalness** (stepwise trace scoring): +9.4 pp over global
  log-prob selection on math; picks the most *learnable* trace, not the most
  familiar one.
- **Ensemble-then-distill** beats per-teacher KD; multi-teacher KD also
  stabilizes GRPO-style RL for small models.
- Failure modes: logit averaging across vocabularies, unfiltered style clash
  (worst in code: API choice, error-handling idiom), global-likelihood trace
  selection.

**v3 teacher panel (all free-servable on Kaggle 2×T4 int4, all permissive):**

| Teacher | License | Serves | Specialty routed to it |
|---|---|---|---|
| Qwen3.6-27B | Apache 2.0 | ~14.5 GB int4 | Agentic/SWE traces, long-horizon repair (SWE-bench V 77.2) |
| **Devstral Small 2 24B-2512** | Apache 2.0 | ~13 GB int4 | Tool-driven codebase navigation, multi-file edits (SWE-bench V 68.0, Multilingual 55.7) |
| Qwen3-4B (donor) | Apache 2.0 | laptop | Style anchor — keeps student close to its own inductive biases for low-conflict basics |
| Nemotron open RL/agentic traces | OpenMDW | pre-generated | Agentic tool-call patterns (no serving cost at all) |

**Conflict protocol (binding, in priority order):**
1. **Execution filter first**: only traces whose final code passes tests enter
   the pool. Two teachers disagreeing but both passing = both valid (diversity).
2. **Domain routing**: each task family has a designated primary teacher;
   secondary teacher traces enter only if primary fails the execution filter.
3. **Local-naturalness scoring** (student-side, stepwise) picks among
   surviving traces — choose what the student can learn, not what looks best.
4. **One format contract**: a single trace template (analysis → plan → code →
   self-check) and one code-style normalizer pass before anything reaches the
   student. Style clash dies in preprocessing, not in the loss.
5. **Text traces only** — no logit fusion, ever (vocabularies differ; logit
   averaging is the documented failure mode).

This replaces the single-teacher assumption in CODE_PIVOT §3. Cost delta: zero
— same Kaggle serving hours, split across two teacher checkpoints.

## 4. Tokenizer / embedding layer — edge E8 (gated)

Findings: tokenizer transfer (ZeTT, TokenAdapt) is proven tech; tokenizer
quality can dominate scale (a 350M model with the right tokenizer beating a
2.7B one); byte-level/tokenizer-free remains unproven at our scale for code.

The donor's 151,936-vocab tied embedding costs **389 M params** — ~12% of the
core budget — sized for general multilingual text, not code.

**E8 (ablation-gated, off the critical path):** ZeTT-style transfer to a
~80K code-weighted vocabulary after T2 stabilizes:
- ~180 M params returned to budget (embeddings are Q8 at export — this is
  real GGUF megabytes and RAM, plus softmax/LM-head compute every token).
- Better code compression = fewer tokens per line = higher *effective* tok/s —
  multiplies with D1's CPU-decode target.
- Risk: knowledge damage to the donor → gate: ≤1 pp on the C1 500-problem
  probe after a short recovery run, else revert. Run only if T1–T4 land early.

## 5. Decoding + transparency — demand D9

Proven, cheap, and aligned with the mission:

- **min-p sampling + entropy-gated compute**: deployment defaults
  (min-p 1e-4–1e-5, top-p 0.95); high-entropy steps may trigger one extra
  HRM refinement loop — unifies published entropy-gating with our
  architecture's native compute dial.
- **Calibration**: temperature rescaling + isotonic regression reach ECE
  5–10% in practitioner reports — D2's ECE ≤ 0.15 gate is comfortably
  realistic; tighten to ≤ 0.10 at v3.1.
- **Glass-box decoding — new demand D9** (added to INVENTION.md): the model
  ships with transparency as a *product surface*, not a paper section:
  - per-token confidence exposed in the API (and surfaced as a "how sure am I"
    annotation on generated code);
  - **PKM memory attribution**: product-key lookups are discrete indices —
    log which memory slots fired per token and map slots to their training
    provenance. Retrieval-style attribution *from inside the weights*, nearly
    free because the index structure already exists. This is novelty-claim
    material (Claim 8 candidate) — no shipped open model exposes
    parametric-memory attribution today;
  - structured trace format (analysis/plan/code/self-check) as the only
    output mode in explain contexts (D4 synergy);
  - per-response compute meter (HRM steps taken, tokens drafted/accepted,
    memory hits) — the energy ledger (D6), per answer.

## 6. Realism pass — what the calendar actually looks like

Directive: "think about real time constraints, be more realistic." Revised
estimates, superseding CODE_PIVOT §6 where they conflict:

| Item | Old estimate | Revised | Why |
|---|---|---|---|
| Total free-GPU budget | 250–400 T4-h ≈ 8–12 wks | **400–700 T4-h ≈ 3–6 months** | HRM refinement ≈ 4.5× FLOPs on supervised iterates; teacher-serving hours compete with training hours on the same Kaggle quota |
| Teacher trace generation | implicit | **explicit ~25% of weekly quota** | Two-teacher panel; traces are generated weekly, not all upfront |
| Single-maintainer bandwidth | unstated | **the binding constraint** | Phases C2–C5 each have research-grade failure modes; budget 1–2 redo cycles per phase |
| Eval gate (primary) | beat ALL ≤4B on ALL evals | **beat donor +δ on full matrix; beat ALL ≤4B on ≥70% of matrix** | The all/all gate had ~30–40% odds; this gate is winnable while keeping the punch-up targets as stretch goals |
| Agentic punch-up | beat 14B-class | unchanged | Harness quality dominates there; SIA-pattern + repo-RAG (1.2) raise odds |

Calendar shape (calendar ≠ GPU time; most weeks are bandwidth-bound):
- **Weeks 1–3**: C1 donor baseline + teacher panel serving validated.
- **Weeks 4–10**: T1–T2 transplant (the risk bulge; fallback recipes ready).
- **Weeks 11–16**: T3 mount + C4/C5 curriculum, first multi-teacher rounds.
- **Weeks 17–24**: C6 self-play + D-gates + E8/E-ablations as time allows.
- Any phase ships (unchanged); each phase ends with a HF-published checkpoint.

## 7. Claims ledger delta

- Claim 3 (halting-coupled speculation): **precedent strengthened** —
  Nemotron 3 ships trained MTP heads for speculation; our coupling to the
  halting signal remains the novel part.
- Claim 4 (VRAM-free memory tier): **extended** — 2-bit TurboQuant values cut
  the RAM bill ~7×; attribution (D9) makes the memory tier *inspectable*, not
  just cheap.
- **Claim 8 (candidate, to validate)**: parametric memory attribution —
  per-token provenance from product-key indices in a consumer-grade model.
  Struck if the logged indices prove uninformative (entropy ≈ uniform) or
  cost > 5% decode speed.

## References (round 2)

- hexo-ai/sia — MIT; meta/target/feedback self-improvement triad.
- RyanCodrai/turbovec — MIT; TurboQuant 2-bit data-oblivious VQ (Google
  Research algorithm), SIMD nibble-LUT kernels.
- NVlabs/cuda-oxide — Rust→PTX, Apache 2.0 (bindings NV-licensed).
- NVIDIA Nemotron 3 family + open datasets (Dec 2025–Jun 2026), OpenMDW 1.1.
- Devstral Small 2 24B-Instruct-2512 — Apache 2.0, SWE-bench Verified 68.0.
- TinyLLM; peer-review multi-teacher KD; Local Naturalness trace selection;
  ensemble-then-distill; multi-teacher KD × GRPO (2025–26).
- ZeTT (zero-shot tokenizer transfer); TokenAdapt; multilingual-tokenizer >
  scale result (ACL).
- min-p sampling; entropy-gated test-time compute; RACER retrieval
  speculation; SDTT diffusion decoding (kept watch-listed, not adopted).
