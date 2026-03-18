# AVA-v2 Recurrent Launch V1

## Goal
Launch a genuinely larger architectural step for AVA instead of continuing narrow support-bank or patch tuning.

## Hypothesis
The next meaningful gain will come from changing the student core and training regime, not from another small retrieval tweak. AVA needs:
- shared recurrent depth so it can spend more compute per parameter
- transformer-to-recurrent warm-start so the new branch does not restart blind
- verifier-based RL so learning is grounded in rewards, not only supervised targets

## What Was Implemented
- recurrent-depth architecture support in [model.py](/D:/AVA/src/ava/model.py) and [config.py](/D:/AVA/src/ava/config.py)
- transformer-to-recurrent warm-start remap in [train.py](/D:/AVA/src/ava/train.py)
- verifiable RL scaffold in [rl.py](/D:/AVA/src/ava/rl.py)
- CLI entrypoints in [cli.py](/D:/AVA/src/ava/cli.py)
- smoke configs in [ava-v2-recurrent-depth-smoke.yaml](/D:/AVA/configs/experiments/ava-v2-recurrent-depth-smoke.yaml) and [ava-v2-recurrent-depth-rl-smoke.yaml](/D:/AVA/configs/experiments/ava-v2-recurrent-depth-rl-smoke.yaml)

## Smoke Results
- recurrent-depth unsupervised pretraining ran end to end on GPU and produced a falling loss curve
- recurrent-depth verifier RL ran end to end on GPU from a warm-started checkpoint
- the first RL pass was initially dead because the prompts did not match the model's learned format; fixing the prompt interface produced non-zero reward

## Takeaway
This is the first AVA-v2 branch that changes the core training story instead of just changing what sits around the checkpoint. The branch is not yet strong, but it is now real and executable.

## Next Step
Run a longer recurrent-depth unsupervised job on a real mixed corpus, then a second RL phase on top of that new checkpoint.
