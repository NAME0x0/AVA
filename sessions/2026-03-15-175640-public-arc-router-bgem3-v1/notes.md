# Public Benchmark Session: public-arc-router-bgem3-v1

## Goal

Turn the routed sparse+dense ARC result into a first-class session packet instead of leaving it only in activity logs.

## Baselines

- Sparse ensemble: `88/299` = `0.294`
- Dense `BAAI/bge-m3` hybrid: `82/299` = `0.274`
- Routed sparse+dense: `90/299` = `0.301`
- Held-out routed slice: `57/199` = `0.286`

## Why This Matters

- Dense retrieval alone is weaker than the sparse ensemble on full ARC.
- Dense retrieval is still complementary enough to matter: sparse-only rows=`31`, dense-only rows=`25`, oracle union=`113/299`.
- The router is the first AVA public-benchmark path to cross `90` correct rows on ARC-Challenge validation.

## Router Thresholds

- dense score min: `0.66`
- dense margin min: `0.014`
- sparse margin max: `0.022`
- margin gap min: `-0.02`

## Decision

Keep the routed sparse+dense ARC path as the current public-benchmark mainline.

## Next Step

Try a stronger retrieval layer on top of the same support banks before changing the student weights again. The likely candidates are reranking, graph-style memory propagation, or broader science support-bank distillation.

## Artifacts

- `results/summary.json`
- `sessions/activity/arc-hybrid-support-ensemble-299-v1.json`
- `sessions/activity/arc-dense-bgem3-299.json`
- `sessions/activity/arc-router-bgem3-299.json`
- `sessions/activity/arc-router-bgem3-test199.json`
