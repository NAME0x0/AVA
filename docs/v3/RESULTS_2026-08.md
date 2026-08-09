# AVA v3 — Measured Results Log (August 2026)

Parked findings from the Series-1/2 measurement campaign. Every number here was
produced on the target hardware (RTX A2000 4 GB laptop) with the matched protocol
below. Claims without a measurement do not belong in this file.

## Measurement protocol (adopt for every speed number)

Established 2026-08-08 after a first attempt showed **±26% run-to-run noise on
identical configs** and nearly produced a fake "+27% from DFlash" headline:

1. **Discard a warm-up round** before recording.
2. **Interleave arms round-robin**, never in blocks (block ordering entangles the
   arm with warm-up / page-cache state).
3. **≥5 repeats**, report mean ± sd, and treat differences under ~2σ as nothing.

Effect: standard deviation dropped ~4× (8.29 → 2.16 on the plain arm). Every
single-run number predating this — including the old "18 t/s" and "43.8 t/s"
headlines and the whole `phase1_baseline` table — is unreliable.

## Speculative decoding (Series 1)

Matched: `-c 2048 -ctk q8_0 -ctv q8_0 -b 512 -ub 128 -n 256`, greedy, 5 interleaved
rounds, Qwen3.5-4B Q4_K_M target.

| arm | gen t/s | sd | vs plain |
|---|---|---|---|
| plain | 33.2 | 2.16 | — |
| draft-mtp | 44.3 | 2.21 | ×1.33 |
| **draft-dflash** | **48.3** | 3.24 | **×1.46** |

- Both spec methods beat plain decisively (t ≈ 8). **DFlash over MTP is +4.0 t/s,
  t ≈ 2.3, p ≈ 0.06 — suggestive, not conclusive.**
- DFlash drafter: `Anbeeld/Qwen3.5-4B-DFlash-GGUF` Q4_K_M (381 MB), Apache-2.0
  upstream (`z-lab/Qwen3.5-4B-DFlash`). Needs `-ub 128` to fit 4 GB.
- **Prefill tradeoff:** DFlash ~162 t/s prompt vs plain ~216 (the `-ub 128` tax).
  For long-prompt/short-output work — i.e. whole-file editing — MTP may win
  overall. Measure the workload, not the benchmark.

## Donor comparison (Series 2)

Matched CanItEdit: n=52, descriptive, 2560 tokens, 4-bit NF4, greedy, non-thinking.

| | LFM2.5-2.6B | Qwen3.5-4B |
|---|---|---|
| **score** | **38.46%** | **53.85%** |
| adaptive (add feature) | 40.0 | **65.0** |
| perfective (improve) | 35.7 | **50.0** |
| corrective (fix bug) | **50.0** | 25.0 |
| reasons | pass 20 / tests 28 / syntax 4 | pass 28 / tests 24 / syntax 0 |
| eval VRAM | **1.79 GB** | 3.12 GB |

- **Not a token-budget artifact.** Re-ran 10 sampled failures at 4096 tokens:
  **0/10 flipped**; 8/10 had been truncated at 2560; 4/10 hit even 4096.
  Truncation was widespread but not causal.
- **Inversion worth remembering:** LFM2.5 is *better* at corrective (bug-fix)
  edits and much worse at adaptive (add-feature).
- **Verdict: keep Qwen3.5-4B as the coding donor.** Caveat: only the axis Liquid
  admits is weak has been measured. LFM2.5's agentic/tool strength is UNMEASURED —
  we own no tool-use harness.

## Verbosity / prompting (Series 1) — the first measured capability *gain*

LFM2.5 writes an analysis, re-quotes the original program, then answers.

**LFM2.5-230M** (12 problems, greedy ⇒ deterministic, 1 run each):

| variant | tokens | %base | fences | parses | pass |
|---|---|---|---|---|---|
| system | 305.7 | 69.6% | 1.00 | 12/12 | 0/12 |
| sys_terse_prefill | 307.3 | 69.9% | 1.00 | 12/12 | 0/12 |
| terse_prefill | 368.6 | 83.9% | 1.00 | 11/12 | 0/12 |
| terse | 371.6 | 84.6% | 1.00 | 11/12 | 0/12 |
| prefill | 381.2 | 86.7% | 1.00 | 12/12 | 1/12 |
| baseline | 439.5 | 100% | 1.08 | 9/12 | 0/12 |

**LFM2.5-2.6B**, same 12 problems:

| variant | tokens | %base | fences | parses | pass |
|---|---|---|---|---|---|
| prefill | 1061.5 | 55.8% | 2.25 | 11/12 | 4/12 ↓ |
| **system** | 1245.5 | 65.5% | 2.00 | 11/12 | **8/12 ↑** |
| baseline | 1900.8 | 100% | 2.42 | 12/12 | 6/12 |

- **A one-line system-role constraint cut tokens 34% AND raised pass 6/12 → 8/12.**
  Zero training. Three training runs regressed; this is the first thing that helped.
- **Role placement matters, not wording** — the same text in the user turn
  (`terse`) was clearly worse.
- Combining tricks does not stack (`sys_terse_prefill` ≡ `system`).

### Full 52-problem confirmation (2026-08-09)

| prompt | LFM2.5-2.6B | passes |
|---|---|---|
| baseline | 38.46% | 20/52 |
| **system** | **42.31%** | 22/52 |
| donor Qwen3.5-4B (baseline prompt) | 53.85% | 28/52 |

- System prompt is worth **+3.85 pp and -34% tokens** — free, keep it.
- **The n=12 sample overstated the gain 3x** (+33% relative there, +10% here).
  Another entry in the small-n ledger.
- **Donor verdict unchanged**: LFM2.5 stays ~11.5 pp behind even prompt-tuned.
- Truncation is structural for LFM2.5: **12/52 (23%) still hit the cap** after the
  token cut, and an earlier probe flipped 0/10 at double budget.

### Proxy transfer rule (important, reusable)

Small siblings **transfer token/format effects** — the 230M predicted `system` at
69.6%, actual 65.5% on the 2.6B. They **do not transfer capability effects** — the
230M nominated `prefill`, which *hurts* the 2.6B (6→4). Verbosity itself is a
capability-*scale* property: the 230M shows fences 1.08 vs the 2.6B's 2.42.

**Screen variants on the sibling; settle them on the target.**

## Harness artifacts found (running count: 6)

The recurring failure mode of this project is the measurement, not the model.

1. Body-only completions ran standalone → SyntaxError (fake −13 "regression")
2. Dropped typing imports → NameError on correct code
3. EvalPlus tests blew the OS command-line limit → temp-file sandbox
4. Data cursor key mismatch → same 12K samples retrained ~4×
5. **First-fence extraction** (`92c8f2e`) — a model that reasons quotes the
   ORIGINAL program first; we scored the unedited input. Would have reported
   LFM2.5 at ~0% and killed the donor evaluation on an artifact.
6. **Truncation invisible to `_classify`** (`6f0ca9c`) — cut-off programs often
   still compile, so they were filed as "tests" (real wrong edits). Cap-hit is now
   measured and authoritative.

Standing rule that catches these: **dump the generations before believing a score.**

## Environment gotchas (cost real time)

- `_build_qlora_model` is a **training** builder: `prepare_model_for_kbit_training`
  casts to fp32 (2.37 GiB) and OOMs on 4 GB. Use a plain 4-bit load for eval.
- `device_map="auto"` under 12 GB spills to CPU, which bnb 4-bit rejects →
  `AVA_SINGLE_GPU=1` (first real use of the Phase-1 escape hatch).
- Git-Bash paths (`/d/AVA/...`) fail in Windows Python — use `D:/AVA/...`.
- Qwen3.5-4B needs 3.12 GB just to *evaluate* on this card; LFM2.5-2.6B needs 1.79.
