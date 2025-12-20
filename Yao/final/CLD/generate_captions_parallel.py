#!/usr/bin/env python3
"""
并行 Caption 生成脚本
每个进程处理指定的 parquet 文件范围
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json

def get_parquet_files(data_dir):
    """获取所有 parquet 文件"""
    data_dir = Path(data_dir)
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    return parquet_files

def split_files_into_chunks(files, num_chunks):
    """将文件列表分成 N 个块"""
    chunk_size = (len(files) + num_chunks - 1) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(files))
        if start_idx < len(files):
            chunks.append(files[start_idx:end_idx])
    return chunks

def main():
    parser = argparse.ArgumentParser(description="并行生成 captions - 按文件分割")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument("--num_gpus", type=int, default=6, help="使用的 GPU 数量")
    parser.add_argument("--gpu_ids", type=str, default=None, help="GPU IDs (e.g., '0,1,2,3,4,5')")
    parser.add_argument("--prompt", type=str, 
                        default="Describe style, main subject, and especially the background of the whole image in one short sentence.",
                        help="Caption prompt")
    parser.add_argument("--dry_run", action="store_true", help="只显示命令，不执行")
    parser.add_argument("--file_range", type=str, default=None, 
                        help="指定文件范围 (e.g., '0-3' 表示处理第 0-3 个文件)")
    parser.add_argument("--gpu_id", type=int, default=None, help="单个进程使用的 GPU ID")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    args = parser.parse_args()
    
    # 获取所有 parquet 文件
    parquet_files = get_parquet_files(args.data_dir)
    print(f"📂 Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        print("❌ No parquet files found!")
        return
    
    # 单进程模式：处理指定范围的文件
    if args.file_range and args.gpu_id is not None:
        start, end = map(int, args.file_range.split('-'))
        files_to_process = parquet_files[start:end+1]
        
        print(f"\n🎯 Single process mode:")
        print(f"   GPU: {args.gpu_id}")
        print(f"   Files: {start}-{end} ({len(files_to_process)} files)")
        for f in files_to_process:
            print(f"     - {f.name}")
        
        # 生成输出文件名
        output_file = Path(args.output_dir) / f"caption_mapping_gpu{args.gpu_id}_files{start}-{end}.json"
        
        # 构建命令
        cmd = [
            "python", "generate_captions_for_training.py",
            "--data_dir", args.data_dir,
            "--output", str(output_file),
            "--num_gpus", "1",
            "--device", f"cuda:{args.gpu_id}",
            "--prompt", args.prompt,
            "--batch_size", str(args.batch_size),
        ]
        
        print(f"\n🚀 Running command:")
        print(" ".join(cmd))
        
        if not args.dry_run:
            env = {"CUDA_VISIBLE_DEVICES": str(args.gpu_id)}
            subprocess.run(cmd, env={**subprocess.os.environ, **env})
        
        return
    
    # 多进程模式：自动分配文件
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.num_gpus))
    
    num_gpus = len(gpu_ids)
    
    # 分割文件
    file_chunks = split_files_into_chunks(parquet_files, num_gpus)
    
    print(f"\n🎮 Multi-process mode: {num_gpus} GPUs")
    print(f"   GPU IDs: {gpu_ids}")
    print(f"\n📊 File distribution:")
    
    commands = []
    for i, (gpu_id, files) in enumerate(zip(gpu_ids, file_chunks)):
        if not files:
            continue
        
        # 计算文件索引范围
        start_idx = parquet_files.index(files[0])
        end_idx = parquet_files.index(files[-1])
        
        print(f"\n   GPU {gpu_id}: Files {start_idx}-{end_idx} ({len(files)} files)")
        for f in files[:3]:
            print(f"     - {f.name}")
        if len(files) > 3:
            print(f"     ... and {len(files)-3} more")
        
        # 生成命令
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python generate_captions_parallel.py " \
              f"--data_dir {args.data_dir} " \
              f"--output_dir {args.output_dir} " \
              f"--file_range {start_idx}-{end_idx} " \
              f"--gpu_id {gpu_id} " \
              f"--prompt \"{args.prompt}\" " \
              f"--batch_size {args.batch_size}"
        
        commands.append(cmd)
    
    # 打印所有命令
    print(f"\n{'='*70}")
    print("🚀 Commands to run (in separate terminals):")
    print(f"{'='*70}\n")
    
    for i, cmd in enumerate(commands):
        print(f"# Terminal {i+1} - GPU {gpu_ids[i]}")
        print(cmd)
        print()
    
    # 生成启动脚本
    script_path = Path(args.output_dir) / "run_parallel_caption_generation.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 并行 Caption 生成脚本\n")
        f.write(f"# 生成时间: $(date)\n\n")
        f.write("cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD\n")
        f.write("conda activate llava\n\n")
        
        for i, cmd in enumerate(commands):
            f.write(f"# GPU {gpu_ids[i]}\n")
            f.write(f"{cmd} > caption_gpu{gpu_ids[i]}.log 2>&1 &\n")
            f.write(f"echo \"Started process on GPU {gpu_ids[i]} (PID: $!)\" \n\n")
        
        f.write("echo \"All processes started!\"\n")
        f.write("echo \"Monitor progress with: tail -f caption_gpu*.log\"\n")
        f.write("echo \"Check running processes: ps aux | grep generate_captions\"\n")
        f.write("wait\n")
        f.write("echo \"All processes completed!\"\n")
        f.write("\n# 合并所有结果\n")
        f.write(f"echo \"Merging results...\"\n")
        f.write(f"jq -s 'add' {args.output_dir}/caption_mapping_gpu*.json > {args.output_dir}/caption_mapping_full.json\n")
        f.write(f"echo \"✅ Done! Final output: {args.output_dir}/caption_mapping_full.json\"\n")
    
    script_path.chmod(0o755)
    
    print(f"\n✅ Startup script saved to: {script_path}")
    print(f"\n📝 To run all processes in background:")
    print(f"   bash {script_path}")
    print(f"\n📝 Or run each command in separate terminals for better monitoring")

if __name__ == "__main__":
    main()


