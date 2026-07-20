# Design Spec — AVA-v2 arXiv Paper

**Title:** AVA-v2: A Reproducible Case Study of Adapting and Evaluating a 1.9B Model Under a 2 GB VRAM Budget on Consumer Hardware

**Author:** Muhammad Afsah Mumtaz (alias NAME0x0 — GitHub / Hugging Face). Affiliation: Independent Researcher.

**Venue / format:** arXiv preprint (cs.CL / cs.LG). Single-column LaTeX, `arxiv.sty` style, `natbib` + BibTeX. No anonymization.

**Date:** 2026-07-20. **Status:** design approved; open verification items (§8) must close before prose.

**Governing rule:** every quantitative claim traces to a named source file in this repo (the "evidence ledger", §3–§6). No number enters the paper from memory. Every citation is verified to exist before it enters the `.bib`.

---

## 1. Thesis

The language backbone of a 1.9B-parameter model can be QLoRA-adapted and broadly evaluated entirely on a single 4 GB laptop GPU — 1.81 GB peak training VRAM, ~100 minutes, a 42 MB adapter — and a full-set evaluation across 17 benchmarks reveals a capability profile that small benchmark subsets materially misrepresent (GSM8K falls from a 48% 50-item estimate to 35.3% on all 1,319 items).

This is a **resource-constrained systems + empirical case study**, not a new algorithm, not a SOTA claim, not a controlled superiority claim over larger models.

## 2. Research questions

- **RQ1 — Feasibility.** Can a ~2B model be meaningfully adapted under 2 GB peak VRAM on consumer hardware? → Yes (verified).
- **RQ2 — Capability profile.** What improves vs. stays weak after low-resource QLoRA? → Strong science/knowledge/MCQ; weak advanced math, advanced CS, strict instruction-following; latent tool-use; partial multilingual transfer.
- **RQ3 — Evaluation reliability.** How much do small subsets mislead for compact fine-tuned models? → 50-item GSM8K overstated full-set by ~13 pp; self-consistency (k=5) adds +8.7 pp; tool-use fires on only 0.6% of tasks.

## 3. Evidence ledger — TRAINING (verified 2026-07-20)

Source: `experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v2/training_report.json`, `adapter_config.json`, `adapter_model.safetensors`, `training_v2_full.log`.

| Fact | Value | Source |
|---|---|---|
| Base total params | 1,892,736,832 (1.89B) | training_report.json |
| Trainable params | 10,911,744 (0.5765% ≈ 0.58%) | training_report.json |
| Peak training VRAM | 1.81 GB | training_report.json (`gpu_memory_gb`) |
| Wall time | 6,027.5 s = 100.5 min | training_report.json + log |
| Train examples | 20,741 | training_report.json |
| Epochs / steps | 1 epoch / 2,593 steps | log (`2593/2593`) |
| Final train loss | 0.4145 | training_report.json + log |
| Engine | sdpa_qlora (BitsAndBytes NF4 4-bit base + LoRA + SDPA attn) | training_report.json (`engine`) |
| LoRA r / α / dropout / bias | 16 / 32 / 0.0 / none | adapter_config.json |
| Target modules | q,k,v,o,gate,up,down_proj (text transformer only) | adapter_config.json |
| Learning rate | 1.5e-4 (v2; NOT the 2e-4 v1 figure) | training_report.json |
| Max seq length | 384 | training_report.json |
| Effective batch | batch_size 1 × grad_accum 8 | derived: 20,741/8 ≈ 2,593 steps (confirm in `finetune_v2_full.py`) |
| Adapter file size | 43,672,224 bytes = 41.6 MiB (~42 MB) | adapter_model.safetensors |
| Seed | 42 | build scripts (confirm training seed in `finetune_v2_full.py`) |
| PEFT version | 0.18.1 | adapter_config.json |

**Base model identity (must-verify, §8a).** `experiments/exp4_finetune/models/Qwen3.5-2B/config.json` → `model_type: qwen3_5`, arch `Qwen3_5ForConditionalGeneration`, contains `text_config` + `vision_config`, image/video token ids, `pipeline_tag: image-text-to-text`, license apache-2.0, `transformers 4.57.0.dev0`. README declares `Qwen/Qwen3.5-2B` from `Qwen/Qwen3.5-2B-Base`. **The base is a 1.9B vision-language model; AVA-v2 adapts only its text transformer; the vision tower is untouched and unused; all evaluation is text-only.** The paper states this precisely. Qwen3.5 existence + correct citation to be web-verified before writing.

## 4. Evidence ledger — DATA & CONTAMINATION (verified 2026-07-20)

Build path: `scripts/build_finetune_corpus_v2.py` → `ava_exp4_finetune_v2.jsonl`; then `scripts/build_v2_augmented.py` (+ `tool_use_augmented.jsonl`) → `ava_exp4_finetune_v2_augmented.jsonl` (the trained corpus). Both `random.seed(42)`.

| Component | Count | Source corpus | Split |
|---|---|---|---|
| GSM8K chain-of-thought | 7,473 | `gsm8k_train_reasoning_support_v1` | GSM8K **train** |
| Science MCQ (explained) | sampled 8,000 | `public_science_support_v1` (SciQ + OpenBookQA) | **train** |
| ARC-Challenge (explained) | 4,476 | `public_benchmark_distill_v1` | ARC-C **train** ×4 choice-rotation aug (1,119×4) |
| Math CoT (breakthrough) | remainder | `ava_v3_breakthrough_distill_v1` | distilled |
| Tool-use / reasoning / identity / conversation | ~38 handwritten | inline in build script | synthetic |
| + tool_use_augmented | delta | `tool_use_augmented.jsonl` | synthetic |
| Base v2 corpus | 20,886 rows | — | — |
| Augmented corpus (file) | 20,941 rows | — | — |
| **Trained** | **20,741** | — | 200-row gap to explain (§8e) |

**Contamination finding: clean data path.** Training draws only from TRAIN splits (ARC-C train, GSM8K train, SciQ/OpenBookQA train) + hand-written synthetic. Evaluation loads TEST/validation (`loaders.py`: ARC-C/E `allenai/ai2_arc` `test`; GSM8K `gsm8k main` `test`; PIQA `validation`; etc.). SciQ/OpenBookQA are not eval benchmarks. Train/test are officially disjoint. **Disclosure to make prominently:** AVA-v2 was deliberately adapted to ARC/GSM8K-style tasks via their train splits — so headline scores are held-out results reflecting in-distribution adaptation, not zero-shot generalization. A formal n-gram / exact-match overlap test between the final training corpus and each eval test set will be run and its result reported (§8b) as belt-and-suspenders evidence.

## 5. Evidence ledger — EVALUATION (verified 2026-07-20)

Source: `experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md` + `eval_v2/results/v2_full/<bench>_{summary,details}.json` (17 files, dated 2026-05-04..06). Spot-checked against raw JSON: ARC-C 961/1172, MMLU 8318/14042, GSM8K 466/1319, HumanEval+ 32/164, MATH-500 94/500 — all match.

Eval config: Qwen3.5-2B-AVA-v2 merged→**Q8_0 GGUF**; RTX A2000 Laptop 4 GB, Win11; llama.cpp b7472-head (commit 69c28f1); llama-server, Flash Attention, Q8 KV, 4 parallel slots, continuous batching, 8K context. Wall 4h18m.

Full 17-benchmark table (n, acc, 95% Wilson CI) reproduced verbatim from the report into §Results.

**Methodology caveats to carry into the paper (verbatim honesty):**
- MCQ = single-token argmax over candidate label tokens (`n_probs=60`); differs from lm-eval-harness logprob-of-continuation. Directionally comparable, not numerically identical.
- Generative = greedy (`temp=0`), Qwen chat template, `enable_thinking=False`.
- Self-consistency = 5 samples `temp=0.7 top_p=0.95`, majority vote.
- Code = subprocess sandbox, 5 s / 4 GB caps.
- Q8_0 retains ~99.7% BF16 quality.

**Task-count reconciliation (§8d).** Report header says "16,872 tasks"; n-column sums to **52,027**; throughput table says **52,045** (MCQ 47,193 + generative 4,852). Paper + model card must define ONE accounting: (i) unique questions, (ii) model generations/forward passes, (iii) including self-consistency samples & retries. Resolve from the JSON before publishing any total.

## 6. Evidence ledger — DEPLOYMENT (verified 2026-07-20)

Source: `gguf_build/README_GGUF.md`, `gguf_build/AVA-v2.imatrix`, `quantize_all.ps1`. Perplexity on held-out slice vs Q8_0 reference:

| Quant | Size | RAM | PPL | Δ vs Q8_0 |
|---|---|---|---|---|
| IQ4_XS | 1.11 GB | ~1.6 GB | 2.5347 | +2.0% |
| Q4_K_M (recommended) | 1.19 GB | ~1.7 GB | 2.4907 | +0.25% |
| Q8_0 (reference) | 1.87 GB | ~2.4 GB | 2.4844 | — |

(Pull the FULL table incl. any Q4_0/Q5_K_M rows when writing.) imatrix-calibrated ladder; Ollama + llama.cpp compatible; repo `NAME0x0/AVA-v2-GGUF` (confirm public, §8c).

## 7. Section plan

1. **Introduction** — mismatch between growing model sizes and the hardware students / independent researchers actually own; contributions (4, below).
2. **Related work** — LoRA, QLoRA, PEFT, bitsandbytes NF4; small/edge LMs; quantized local deployment (llama.cpp/GGUF); benchmark-subset reliability; CoT + self-consistency.
3. **Training system** — base VLM identity & what is/ isn't adapted; corpus overview; QLoRA config; memory-saving decisions (NF4, SDPA, seq 384, grad-accum, eval_strategy=no); Windows/A2000/Triton environment; §3 ledger.
4. **Data & contamination** — provenance table (§4), train/test disjointness argument, formal overlap-test result, explicit in-distribution-adaptation disclosure.
5. **Evaluation methodology** — 17 benchmarks; letter-argmax vs logprob caveat; greedy; self-consistency; code sandbox; IFEval rules; Wilson CIs; deviations from lm-eval-harness.
6. **Results** — full verified table (§5) with CIs.
7. **Capability & failure analysis** — MMLU by subject (strong humanities/bio vs weak math/CS/moral_scenarios); MATH-500 by level (51%→5%); MGSM en→es→fr drop; tool-use 0.6% fire rate, 0 pp lift; self-cons +8.7 pp; **subset-vs-full estimation error (RQ3 centerpiece)**.
8. **Deployment** — adapter / merged / GGUF ladder + measured perplexity (§6); RAM footprint; Ollama/llama.cpp.
9. **Limitations & Conclusion** — no matched full-set base eval (pending); single training run; single hardware; short training context (384); tiny tool-use share; MCQ scoring differs from harnesses; MMLU 2.7% ctx-overflow; ARC-train-in-mix disclosure. Conclude: meaningful adaptation + rigorous eval are possible under severe hardware limits — NOT that AVA-v2 beats larger models.

**Contributions (stated in §1):** (1) end-to-end sub-2 GB QLoRA pipeline on a laptop GPU; (2) reproducible broad evaluation across 6 capability families; (3) evidence that small subsets over/under-estimate compact-model performance; (4) released adapter + merged weights + GGUF ladder with measured perplexity.

## 8. Open verification items — CLOSE BEFORE PROSE

- **(a)** Web-verify `Qwen/Qwen3.5-2B`: real HF repo, technical report/citation, license, param count, multimodal status. Correct base name everywhere.
- **(b)** Run n-gram / exact-match overlap test: final training corpus vs each eval test set → record leakage number.
- **(c)** Confirm HF repos public: `NAME0x0/AVA-v2` (adapter), `NAME0x0/AVA-v2-GGUF`. (Repository-existence claim for reproducibility.)
- **(d)** Reconcile 16,872 / 52,027 / 52,045 from the JSON; pick one accounting; define terms.
- **(e)** Explain the 20,941-file vs 20,741-trained 200-row gap (dedup/length filter in `finetune_v2_full.py`).
- **(f)** Build a fully real `.bib`: every reference verified to exist (authors, year, venue, arXiv id). Zero fabrication. Candidate refs: LoRA (Hu 2021), QLoRA (Dettmers 2023), bitsandbytes, PEFT, llama.cpp/GGUF, ARC (Clark 2018), GSM8K (Cobbe 2021), MMLU (Hendrycks 2021), MMLU-Pro (Wang 2024), HellaSwag (Zellers 2019), PIQA (Bisk 2020), WinoGrande (Sakaguchi 2021), BoolQ (Clark 2019), TruthfulQA (Lin 2022), MATH (Hendrycks 2021) + MATH-500 (Lightman 2023), MGSM (Shi 2022), HumanEval (Chen 2021) + EvalPlus (Liu 2023), IFEval (Zhou 2023), self-consistency (Wang 2022), CoT (Wei 2022), Wilson CI (Wilson 1927 / Brown 2001), SciQ (Welbl 2017), OpenBookQA (Mihaylov 2018), Qwen technical report (verify). Any small-LM/consumer-hardware prior work cited only if confirmed real.

## 9. Artifacts to fix (scope: paper + artifacts)

- **`MODEL_CARD.md`** — real Hugging Face card replacing the current all-`[More Information Needed]` stub at `models/Qwen3.5-2B-AVA-v2/README.md`; fields sourced from §3–§6 ledger.
- **Task-count reconciliation note** — short doc defining the count taxonomy; update report header + card to one number.
- **Reproducibility appendix** — git commit, seed, package/CUDA/driver versions, prompt/chat templates, per-table eval commands.

## 10. Claim guardrails

**Safe to claim:** trained on one 4 GB laptop GPU; 1.81 GB peak VRAM; ~100 min; 0.58% trainable; ~42 MB adapter; broad eval across 6 families; full-set ≠ small-subset; self-cons > implemented tool-use; artifacts public & reproducible.

**Must NOT claim:** controlled superiority over base Qwen / Llama / Gemma / Phi (external numbers only as caveated context, protocols differ); "definitively +16 pp over base" (base lacks matched full-set eval — pending); "first to fine-tune on 4 GB" (prior work exists). Novelty = the complete, audited, low-footprint end-to-end case study, not QLoRA itself.

## 11. Process

Prose written by Claude with the `humaniser` skill applied **in service of scholarly register** (kill AI-flatness / LLM tells, keep formal academic voice) — not delegated to Codex. Mechanical evidence-audit grinds may go to Codex only if parallelism is needed. Implementation plan via `writing-plans` skill next.

## 12. Quality & authenticity bar (non-negotiable — added at author request)

Reads like a genuine, citable scientific paper. Professional, credible, career-building. Hooks sharpened but essence unchanged; nothing false; everything backed.

- **Register.** Authentic academic voice: formal, precise, third-person, measured hedging. Prose paragraphs carry the argument; tables/lists used only where a real paper uses them (results, configs, provenance). No marketing tone. No LLM tells — no "delve / moreover / furthermore" filler, no throat-clearing openers, no over-signposting, no bullet-spam standing in for analysis. `humaniser` applied to reach scholarly register, not casual tone.
- **Credibility / citable.** Standard structure: Abstract, §1–9, Limitations, References, Appendix. Every claim backed by the evidence ledger (§3–§6). Every citation real and verified before it enters the `.bib` (§8f). Reproducibility appendix. arXiv-ready LaTeX that compiles cleanly. Tables/figures numbered, captioned, with units. Equations where they add rigor (VRAM budget; Wilson score interval). A self-citation BibTeX block so the work is easy to cite.
- **Hooks (accurate, not inflated).** Title, abstract, and introduction lead with the genuine findings — sub-2 GB feasibility and the "small subsets misrepresent compact models" result (RQ3). Sharp framing, zero overclaim. Research essence unchanged.
- **Career value.** Real author identity, affiliation, contact; optional ORCID (ask author). Links to public GitHub/HF artifacts. Positioned as rigorous, honest, reproducible systems + empirical work — the kind of preprint that survives scrutiny and reflects well on the author.
- **Anti-overclaim discipline.** §10 guardrails enforced sentence-by-sentence during drafting; each headline number cross-checked against its source file at write time, never trusted from draft memory.
