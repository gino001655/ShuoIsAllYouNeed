# 并行 Caption 生成完整指南

## 🎯 方案总结

✅ **已实现**: 文件级别并行处理  
✅ **稳定性**: 单 GPU 模式 100% 稳定  
✅ **速度**: 使用 5 个 GPU 可提速 5 倍（约 2-3 小时完成 18K 图片）  
❌ **Batch Size**: LLaVA 不支持真正的批处理（每个图片需要独立 context）

---

## 🚀 快速开始

### 方式 1: 一键启动（推荐）

```bash
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
conda activate llava

# 生成并执行启动脚本
python generate_captions_parallel.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output_dir . \
    --num_gpus 5

# 执行启动脚本（后台运行所有进程）
bash run_parallel_caption_generation.sh
```

### 方式 2: 手动在多个 Terminal 运行

更好的监控和调试：

```bash
# Terminal 1 - GPU 0
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
conda activate llava
CUDA_VISIBLE_DEVICES=0 python generate_captions_parallel.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output_dir . \
    --file_range 0-3 \
    --gpu_id 0

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 python generate_captions_parallel.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output_dir . \
    --file_range 4-7 \
    --gpu_id 1

# Terminal 3-5 类似...
```

---

## 📊 工作原理

### 文件分配

**总共 20 个 Parquet 文件**：
- 18 个 train 文件（`train-00000-of-00018.parquet` ~ `train-00017-of-00018.parquet`）
- 2 个 val 文件（`val-00000-of-00002.parquet`, `val-00001-of-00002.parquet`）

**5 GPU 分配方案**（推荐）：
```
GPU 0: Files  0-3  (4 files, ~3500 images)
GPU 1: Files  4-7  (4 files, ~3500 images)
GPU 2: Files  8-11 (4 files, ~3500 images)
GPU 3: Files 12-15 (4 files, ~3500 images)
GPU 4: Files 16-19 (4 files, ~3500 images, 包含 val set)
```

### 输出文件

每个进程生成：
- `caption_mapping_gpu0_files0-3.json` - GPU 0 的 captions
- `caption_mapping_gpu1_files4-7.json` - GPU 1 的 captions
- ... 以此类推

最终合并：
- `caption_mapping_full.json` - 所有 captions 合并

---

## 📝 详细命令说明

### 生成并行脚本

```bash
python generate_captions_parallel.py \
    --data_dir <数据目录> \
    --output_dir <输出目录> \
    --num_gpus <GPU数量> \
    --gpu_ids <可选: GPU IDs> \
    --prompt <可选: 自定义 prompt>
```

**参数说明**：
- `--data_dir`: Parquet 文件所在目录
- `--output_dir`: 输出 JSON 文件目录
- `--num_gpus`: 使用的 GPU 数量（默认 6）
- `--gpu_ids`: 指定 GPU（例如 "0,1,2"），不指定则使用 0~(num_gpus-1)
- `--prompt`: Caption 生成的 prompt（已更新为更短的版本）
- `--dry_run`: 只显示命令，不执行

### 单个进程处理指定文件

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python generate_captions_parallel.py \
    --data_dir <数据目录> \
    --output_dir <输出目录> \
    --file_range <文件范围> \
    --gpu_id <GPU_ID> \
    --prompt "<prompt>"
```

**示例**：
```bash
# GPU 0 处理文件 0-3
CUDA_VISIBLE_DEVICES=0 python generate_captions_parallel.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output_dir . \
    --file_range 0-3 \
    --gpu_id 0 \
    --prompt "Describe style, main subject, and especially the background of the whole image in one short sentence."
```

### 直接使用 generate_captions_for_training.py

如果你想更精细地控制：

```bash
python generate_captions_for_training.py \
    --data_dir <数据目录> \
    --output <输出文件> \
    --file_indices "0-3" \  # 或 "0,1,2,3"
    --device cuda:0 \
    --prompt "<prompt>" \
    --max_samples <可选>
```

---

## 🔍 监控进度

### 查看实时日志

```bash
# 查看所有 GPU 的日志
tail -f caption_gpu*.log

# 查看特定 GPU
tail -f caption_gpu0.log

# 查看最新进度
for f in caption_gpu*.log; do 
    echo "=== $f ==="; 
    tail -5 $f; 
done
```

### 查看 GPU 使用情况

```bash
watch -n 1 nvidia-smi
```

### 检查运行中的进程

```bash
ps aux | grep generate_captions
```

### 查看已生成的 captions 数量

```bash
# 查看单个文件
jq '. | length' caption_mapping_gpu0_files0-3.json

# 查看所有文件
for f in caption_mapping_gpu*.json; do 
    echo "$f: $(jq '. | length' $f) captions"; 
done
```

---

## ⚡ 性能预估

| 配置 | 预计时间 | 备注 |
|-----|---------|------|
| 1 GPU | ~10-15 小时 | 最稳定 |
| 2 GPUs | ~5-7.5 小时 | - |
| 4 GPUs | ~2.5-4 小时 | - |
| 5 GPUs | ~2-3 小时 | **推荐配置** |
| 6 GPUs | ~1.7-2.5 小时 | 文件数限制，效率略低 |

**注意**: 
- 每个图片处理时间约 3-7 秒（取决于 prompt 长度）
- 使用更短的 prompt 可以提速 20-30%
- Val set 较小（~2000 张），分配到最后一个 GPU

---

## 🛠️ Prompt 配置

### 当前 Prompt（简短版）

```
Describe style, main subject, and especially the background of the whole image in one short sentence.
```

**优点**: 
- ✅ 生成速度快（~3-5 秒/图）
- ✅ Caption 简洁（20-50 词）
- ✅ 适合训练时的文本编码

### 修改 Prompt

**方式 1**: 修改 `generate_captions_for_training.py` 第 54 行

```python
prompt: str = "你的新 prompt",
```

**方式 2**: 使用命令行参数

```bash
python generate_captions_parallel.py \
    --prompt "Your custom prompt here" \
    ...
```

---

## 🔧 故障排除

### Q: 某个 GPU 进程卡住不动？

**A**: 
```bash
# 1. 检查 GPU 内存
nvidia-smi

# 2. 查看日志
tail -100 caption_gpu<ID>.log

# 3. 如果确认卡住，kill 进程
pkill -f "gpu_id <ID>"

# 4. 重新启动该进程
CUDA_VISIBLE_DEVICES=<ID> python generate_captions_parallel.py ...
```

### Q: 合并 JSON 时出错？

**A**:
```bash
# 手动合并
jq -s 'add' caption_mapping_gpu*.json > caption_mapping_full.json

# 如果 jq 不可用
python -c "
import json
from pathlib import Path

all_captions = {}
for f in Path('.').glob('caption_mapping_gpu*.json'):
    with open(f) as fp:
        all_captions.update(json.load(fp))

with open('caption_mapping_full.json', 'w') as f:
    json.dump(all_captions, f, indent=2, ensure_ascii=False)

print(f'Merged {len(all_captions)} captions')
"
```

### Q: 中途中断后如何恢复？

**A**: 
脚本自动支持续传（跳过已有的 captions）：

```bash
# 直接重新运行相同命令即可
bash run_parallel_caption_generation.sh
```

---

## ✅ 完整工作流程

### 1. 生成并行脚本

```bash
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
conda activate llava

python generate_captions_parallel.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output_dir . \
    --num_gpus 5
```

### 2. 启动所有进程

**选项 A: 一键启动（后台运行）**

```bash
bash run_parallel_caption_generation.sh
```

**选项 B: Screen 管理（推荐）**

```bash
# 为每个 GPU 创建一个 screen
for i in 0 1 2 3 4; do
    screen -dmS caption_gpu$i bash -c "
        cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
        conda activate llava
        # 运行对应的命令（从生成的脚本中复制）
        CUDA_VISIBLE_DEVICES=$i python generate_captions_parallel.py ...
    "
done

# 查看所有 screen
screen -ls

# 进入某个 screen
screen -r caption_gpu0
```

### 3. 监控进度

```bash
# 实时监控
watch -n 5 'for f in caption_mapping_gpu*.json; do echo "$f: $(jq ". | length" $f 2>/dev/null || echo 0) captions"; done'

# 查看日志
tail -f caption_gpu0.log
```

### 4. 等待完成并合并

```bash
# 等待所有进程完成
wait

# 合并结果（如果脚本没有自动合并）
jq -s 'add' caption_mapping_gpu*.json > caption_mapping_full.json

# 检查最终结果
echo "Total captions: $(jq '. | length' caption_mapping_full.json)"
```

### 5. 使用 Caption Mapping 训练

Captions 会自动整合到训练中（`train.yaml` 已配置）：

```yaml
data_dir: "/tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1"
caption_mapping: "/tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD/caption_mapping_full.json"
```

---

## 📋 检查清单

开始前：
- [ ] 确认 llava 环境已激活
- [ ] 确认所有 GPU 可用（`nvidia-smi`）
- [ ] 确认数据目录正确
- [ ] 确认 prompt 符合需求

运行中：
- [ ] 监控 GPU 使用率
- [ ] 定期检查日志
- [ ] 监控生成进度

完成后：
- [ ] 检查所有输出文件已生成
- [ ] 验证合并后的 JSON 格式正确
- [ ] 抽查几个 captions 质量
- [ ] 更新 train.yaml 中的 caption_mapping 路径

---

## 🎊 总结

**推荐配置**: 使用 5 个 GPU 并行处理

```bash
# 一键启动
cd /tmp2/b12902041/Gino/ShuoIsAllYouNeed/Yao/final/CLD
conda activate llava

python generate_captions_parallel.py \
    --data_dir /tmp2/b12902041/Gino/cld_dataset/snapshots/snapshot_1/data \
    --output_dir . \
    --num_gpus 5

bash run_parallel_caption_generation.sh
```

**预计时间**: 2-3 小时  
**输出**: `caption_mapping_full.json`（包含 ~18,000 个 captions）

**现在可以开始全量生成了！** 🚀
