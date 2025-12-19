#!/bin/bash
# 并行 Caption 生成脚本
# 生成时间: $(date)

cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
conda activate llava

# GPU 0
CUDA_VISIBLE_DEVICES=0 python generate_captions_parallel.py --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data --output_dir /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD --file_range 0-3 --gpu_id 0 --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence." > caption_gpu0.log 2>&1 &
echo "Started process on GPU 0 (PID: $!)" 

# GPU 1
CUDA_VISIBLE_DEVICES=1 python generate_captions_parallel.py --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data --output_dir /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD --file_range 4-7 --gpu_id 1 --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence." > caption_gpu1.log 2>&1 &
echo "Started process on GPU 1 (PID: $!)" 

# GPU 2
CUDA_VISIBLE_DEVICES=2 python generate_captions_parallel.py --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data --output_dir /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD --file_range 8-11 --gpu_id 2 --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence." > caption_gpu2.log 2>&1 &
echo "Started process on GPU 2 (PID: $!)" 

# GPU 3
CUDA_VISIBLE_DEVICES=3 python generate_captions_parallel.py --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data --output_dir /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD --file_range 12-15 --gpu_id 3 --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence." > caption_gpu3.log 2>&1 &
echo "Started process on GPU 3 (PID: $!)" 

# GPU 4
CUDA_VISIBLE_DEVICES=4 python generate_captions_parallel.py --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data --output_dir /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD --file_range 16-19 --gpu_id 4 --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence." > caption_gpu4.log 2>&1 &
echo "Started process on GPU 4 (PID: $!)" 

echo "All processes started!"
echo "Monitor progress with: tail -f caption_gpu*.log"
echo "Check running processes: ps aux | grep generate_captions"
wait
echo "All processes completed!"

# 合并所有结果
echo "Merging results..."
jq -s 'add' /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_mapping_gpu*.json > /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_mapping_full.json
echo "✅ Done! Final output: /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_mapping_full.json"
