#!/bin/bash
# Launch v2 full corpus training after v1 fast results are verified.
# Run from D:/AVA directory.
#
# Usage: bash experiments/exp4_finetune/scripts/launch_v2_training.sh

set -e

echo "============================================"
echo "Launching AVA V2 Full Corpus Training"
echo "============================================"
echo "Corpus: 20,941 examples (v2 augmented)"
echo "Estimated: ~2586 steps, ~16 hours"
echo ""

# Check GPU is free
if nvidia-smi | grep -q "python"; then
    echo "WARNING: GPU is in use by another Python process!"
    echo "Make sure v1 training has completed before starting v2."
    read -p "Continue anyway? (y/N) " -r
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

cd D:/AVA
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u experiments/exp4_finetune/scripts/finetune_v2_full.py \
    > experiments/exp4_finetune/training_v2_full.log 2>&1 &

PID=$!
echo "Training PID: $PID"
echo "Log: experiments/exp4_finetune/training_v2_full.log"
echo ""
echo "Monitor with: tail -f experiments/exp4_finetune/training_v2_full.log"
