#!/bin/bash

# Run LLaVA 1.6 caption generation
# Using liuhaotian/llava-v1.6-vicuna-7b (Direct upgrade from 1.5)

export CUDA_VISIBLE_DEVICES=0

python generate_captions_for_training.py \
    --data_dir "/workspace/dataset/cld_dataset/snapshots/snapshot_1" \
    --output "caption_llava16.json" \
    --model "liuhaotian/llava-v1.6-vicuna-7b" \
    --load_4bit \
    --batch_size 8 \
    --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence." \
    --force

echo "Done! Check caption_llava16.json"
