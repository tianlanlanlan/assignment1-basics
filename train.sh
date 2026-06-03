#!/bin/bash
set -ex
export PYTHONUNBUFFERED=1

# Step 1: Tokenize text -> binary token ID files (skip if already done)
[ -f data/TinyStoriesV2-GPT4-train.bin ] || uv run preprocess_data.py

# Step 2: Find latest checkpoint to resume from (if any)
RESUME_FLAG=""
LATEST_CKPT=$(ls -t checkpoints/iter_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    RESUME_FLAG="--resume $LATEST_CKPT"
    echo "Resuming from $LATEST_CKPT"
fi

# Step 3: Train the model
uv run train.py \
    --vocab-size 10000 \
    --train-data data/TinyStoriesV2-GPT4-train.bin \
    --val-data data/TinyStoriesV2-GPT4-valid.bin \
    --context-length 256 \
    --batch-size 32 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --rope-theta 10000.0 \
    --max-iters 40000 \
    --save-dir checkpoints \
    $RESUME_FLAG
