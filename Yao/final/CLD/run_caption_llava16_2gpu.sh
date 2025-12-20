#!/bin/bash

# LLaVA 1.6 多 GPU 生成腳本 (使用 2 張顯卡)
# 建議 CUDA_VISIBLE_DEVICES 包含所有要用的卡

export CUDA_VISIBLE_DEVICES=0,1

python generate_captions_for_training.py \
    --data_dir "/workspace/dataset/cld_dataset/snapshots/snapshot_1" \
    --output "caption_llava16_2gpu.json" \
    --model "liuhaotian/llava-v1.6-vicuna-7b" \
    --load_4bit \
    --batch_size 8 \
    --num_gpus 2 \
    --gpu_ids "0,1" \
    --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence." \
    --force

echo "生成完成！請檢查 caption_llava16_2gpu.json"
