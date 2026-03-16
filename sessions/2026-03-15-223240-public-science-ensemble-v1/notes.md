# Public Science Ensemble V1

## Goal
Test whether a real train-split public science support bank from SciQ + OpenBookQA improves AVA on public ARC-Challenge through transparent non-parametric retrieval.

## Hypothesis
The current ARC ensemble is missing domain-specific science coverage. A larger public-science MCQ bank should add complementary retrieval support without touching model weights.

## Baseline
- ARC train support only: 30/100 ([sessions/activity/arc-support-mc-100-kindscience.json](/D:/AVA/sessions/activity/arc-support-mc-100-kindscience.json))
- Best prior sparse ensemble: 88/299
- Best prior sparse+dense router: 90/299 and 57/199 held-out

## New Corpus
- [public_science_support_v1](/D:/AVA/corpora/public_science_support_v1)
- 33,272 support examples total
- 23,358 SciQ rows including one rotation
- 9,914 OpenBookQA rows including one rotation

## Selector Search
- Public science bank alone: 28/100
- ARC + science ensemble: 31/100
- Public + ARC + science ensemble: 31/100, then 90/299 and 59/199 held-out
- Science-primary public+ARC+science ensemble: 32/100, then **91/299** and **59/199** held-out

## Conclusion
The public-science bank is useful as an ensemble bank, not as a replacement bank. The crucial extra gain came from routing discipline: making `science` the primary bank while keeping `public` and `arc` as fallback banks produced the new best public ARC score.

This is now the strongest public ARC result in the repo:
- 91/299 full ARC-Challenge
- 59/199 held-out ARC slice
- no dense encoder required

## Artifacts
- [summary.json](/D:/AVA/sessions/2026-03-15-223240-public-science-ensemble-v1/results/summary.json)
- [heldout_delta.json](/D:/AVA/sessions/2026-03-15-223240-public-science-ensemble-v1/results/heldout_delta.json)
- [arc-ensemble-science-public-arc-299-v1.json](/D:/AVA/sessions/activity/arc-ensemble-science-public-arc-299-v1.json)
- [arc-ensemble-science-public-arc-test199-v1.json](/D:/AVA/sessions/activity/arc-ensemble-science-public-arc-test199-v1.json)

## Next Step
Target the remaining held-out science miss clusters, especially life science and earth/space/weather, with compact support-bank additions or teacher-distilled label-only calibration packets.
