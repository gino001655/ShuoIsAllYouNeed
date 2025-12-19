# 多 GPU 并行 Caption 生成指南

## 🎮 GPU 配置

你的服务器有 **6 张 GPU**：
- GPU 0-3: NVIDIA GeForce RTX 4090 (24GB)
- GPU 4-5: NVIDIA GeForce RTX 3090 (24GB)

## 🚀 使用方法

### 方式 1: 使用所有 GPU (推荐)

```bash
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
conda activate llava

# 使用所有 6 张 GPU（速度提升 6 倍！）
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --num_gpus 6
```

### 方式 2: 指定特定 GPU

```bash
# 只使用 GPU 0, 1, 2（速度提升 3 倍）
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --gpu_ids "0,1,2"

# 只使用 4090 显卡（GPU 0-3）
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --gpu_ids "0,1,2,3"
```

### 方式 3: 单 GPU 模式（默认）

```bash
# 只使用一张 GPU（默认 GPU 0）
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --num_gpus 1
```

## ⚡ 性能对比

| GPU 数量 | 预估时间 (18K 图片) | 速度提升 |
|---------|-------------------|---------|
| 1 GPU   | ~10-15 小时       | 1x      |
| 2 GPUs  | ~5-7.5 小时       | 2x      |
| 4 GPUs  | ~2.5-4 小时       | 4x      |
| 6 GPUs  | ~1.7-2.5 小时     | 6x      |

## 🧪 测试多 GPU 功能

先用少量样本测试：

```bash
# 测试 4 GPU 并行（10 个样本）
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output test_multi_gpu.json \
    --num_gpus 4 \
    --max_samples 40 \
    --force
```

## 📊 工作原理

1. **数据分配**: 将所有图片路径平均分配到各个 GPU
   - 例如: 18,000 张图片 ÷ 6 GPU = 每个 GPU 处理 3,000 张

2. **并行处理**: 每个 GPU 在独立进程中：
   - 加载自己的 LLaVA 模型
   - 处理分配给自己的图片
   - 生成 captions

3. **结果合并**: 主进程收集所有结果并保存到 JSON

## 🛠️ 完整参数说明

```bash
python generate_captions_for_training.py \
    --data_dir <数据目录> \
    --output <输出文件.json> \
    --num_gpus <GPU数量> \              # 使用的 GPU 数量
    --gpu_ids <GPU_IDs> \               # 可选: 指定特定 GPU (e.g., "0,1,2")
    --prompt "你的 prompt" \            # 自定义 prompt
    --max_new_tokens 128 \              # Caption 最大长度
    --temperature 0.2 \                 # 生成温度
    --max_samples <N> \                 # 可选: 只处理 N 个样本（测试用）
    --force \                           # 强制重新生成（忽略已有 captions）
    --save_images_dir <目录>            # 可选: 保存样本图片
```

## 💡 使用建议

### 全量生成（18K 图片）

**推荐配置**: 使用所有 6 张 GPU

```bash
# 在 screen/tmux 中运行（防止断线）
screen -S caption_gen
conda activate llava

python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --num_gpus 6 \
    --prompt "Precisely describe style, subjects, text, and especially the graphic design and background of the whole image in simple but detailed sentences."

# Detach: Ctrl+A, D
# Reattach: screen -r caption_gen
```

### 增量生成（续传）

如果中途中断，可以继续：

```bash
# 自动跳过已生成的 captions（不用加 --force）
python generate_captions_for_training.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output caption_mapping_full.json \
    --num_gpus 6
```

## 🔍 监控进度

### 查看 GPU 使用情况

```bash
watch -n 1 nvidia-smi
```

### 查看生成进度

```bash
# 查看已生成的 caption 数量
jq '. | length' caption_mapping_full.json

# 查看最新生成的样本
jq 'to_entries | .[-5:] | .[].value' caption_mapping_full.json
```

## ⚠️ 注意事项

1. **内存要求**: 每个 GPU 需要 ~8-10GB VRAM（4-bit 模型）
2. **进程数**: 不要超过可用 GPU 数量
3. **中断恢复**: 脚本每 100 个样本自动保存一次
4. **错误处理**: 单个图片失败不会影响整体进程

## 🐛 常见问题

### Q: 进程卡住不动？
A: 检查是否有 GPU 内存不足，使用 `nvidia-smi` 查看

### Q: 某个 GPU 特别慢？
A: 可能该 GPU 被其他进程占用，使用 `--gpu_ids` 排除它

### Q: 想要更快的速度？
A: 可以降低 `--max_new_tokens` (e.g., 64) 或使用更多 GPU

## ✅ 测试清单

- [ ] 测试单 GPU 模式 (`--num_gpus 1`)
- [ ] 测试多 GPU 模式 (`--num_gpus 4` + `--max_samples 40`)
- [ ] 检查生成的 captions 质量
- [ ] 确认中断恢复功能正常
- [ ] 开始全量生成

