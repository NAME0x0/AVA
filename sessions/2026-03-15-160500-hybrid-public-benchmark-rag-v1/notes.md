# Hybrid Public Benchmark RAG V1

## Goal
Improve AVA's real public benchmark behavior through retrieval changes that are transparent, local, and reproducible.

## Hypotheses
- Mixed support banks were underperforming partly because explicit coarse categories were being lost at load time.
- Public benchmark retrieval would benefit from retaining rotated ARC support rows instead of collapsing them to one canonical example.
- ARC hybrid scoring was overweighting full-prompt overlap and underweighting question semantics.
- A margin-gated ensemble over complementary support banks could outperform either bank alone.

## Code Changes
- Preserved explicit `category` in [src/ava/retrieval.py](/D:/AVA/src/ava/retrieval.py).
- Added reusable sparse support indexing and cached support index usage in [src/ava/external_benchmarks.py](/D:/AVA/src/ava/external_benchmarks.py).
- Removed signature-level row collapse so rotated ARC support rows remain available in the broad public bank.
- Shifted the hybrid multiple-choice scorer toward question-stem and semantic answer matching.
- Added support-bank ensemble manifests in [arc_ensemble_public_primary_v1.json](/D:/AVA/configs/support/arc_ensemble_public_primary_v1.json), [arc_ensemble_public_primary_v2.json](/D:/AVA/configs/support/arc_ensemble_public_primary_v2.json), and [arc_ensemble_arc_primary_v1.json](/D:/AVA/configs/support/arc_ensemble_arc_primary_v1.json).
- Added regression coverage in [tests/test_retrieval.py](/D:/AVA/tests/test_retrieval.py) and [tests/test_external_benchmarks.py](/D:/AVA/tests/test_external_benchmarks.py).
- Added merged support corpus [public_benchmark_plus_teacher_v1](/D:/AVA/corpora/public_benchmark_plus_teacher_v1).

## Results
- Broad public support bank, hybrid scoring:
  - `24/100` after category fix alone.
  - `29/100` after restoring rotated support rows.
  - `30/100` after question/semantic-heavy weight tuning.
- Broad public support bank, full ARC validation:
  - `84/299` after restoring rotated support rows.
  - `86/299` after weight tuning.
- Two-bank hybrid ensemble over `public_benchmark_distill_v1` and `arc_train_support_v1`:
  - `31/100` with public-primary margin gate `0.0`.
  - `31/100` with arc-primary margin gate `0.0`.
  - `88/299` with public-primary margin gate `0.0`.
- Broad public support bank plus teacher anchors:
  - `30/100`, no gain over tuned public bank.
- GSM8K nearest retrieval on the cleaned public math bank:
  - `0/50` at threshold `0.90`
  - `0/50` at threshold `0.95`
- PIQA runner status:
  - blocked locally by `datasets` runtime script support, not by model inference.

## Decision
Keep the explicit-category fix, the reusable support index, the non-collapsed public support bank, the question/semantic-heavy ARC hybrid scorer, and the margin-gated two-bank ensemble. Do not pursue teacher-anchor merging on this branch. Do not claim GSM8K retrieval progress from this work.

## Next Step
Run a separate AVA-v2 architecture branch with modern dense decoder changes and compare its raw benchmark behavior against the current retrieval-assisted baseline. In parallel, fix the PIQA loader path and build benchmark-native ensemble manifests for other multiple-choice tasks.
