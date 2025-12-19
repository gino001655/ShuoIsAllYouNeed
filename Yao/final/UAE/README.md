# 🎨 UAE: Incentivizing Mutual Benefits for Unified Multimodal Understanding and Generation via RL

<p>
  <a href="http://arxiv.org/abs/2509.09666">📄 Paper</a> |
  <a href="https://huggingface.co/zhiyuanyan1/UAE">🤗 Model</a> |
  <a href="#benchmark-section">📊 UniBench</a> |
</p>

Official code of UAE and UniBench benchmark for our paper *"Can Understanding and Generation Truly Benefit Together — or Just Coexist?"*. 

UAE is a unified multimodal framework for image generation and understanding.

![Example](./assets/figure1_new.jpg)

🌟 Key contributions of our work:

> ✅ **UAE**: an Auto-Encoder–based unification that treats understanding as the encoder (I2T) and generation as the decoder (T2I), using reconstruction similarity as an explicit objective to quantify cross-modal coherence and operationalize unification.
> 
> ✅ **Unified-GRPO**: to our knowledge, the first RL scheme that *jointly* improves both modules via two complementary steps—*Generation for Understanding* (train the encoder to caption for higher reconstruction quality) and *Understanding for Generation* (refine the decoder to reconstruct from those captions)—forming a positive feedback loop toward unification.
>
> ✅ **Aha Moment in Multimodal**: We report an emergent "aha moment" in multimodal learning. As RL progresses, the encoder autonomously emits longer, more descriptive captions while the decoder simultaneously achieves strikingly faithful reconstructions. This co-evolution offers compelling empirical evidence for unified multimodal intelligence.
>
> ✅ **Unified-Bench**: to the best of our knowledge, the first benchmark explicitly designed to **measure the degree of unification** in UMMs, rather than individually evaluating the generation or understanding capabilities.


## 📋 TODO List

- [ ] Release unified-grpo training code (RL).
- [ ] Release the training data of long-context-700K for SFT and the training data for unified-grpo.
- [ ] Release training code for SFT (text-to-image generation).
- [☑️] Release all models' checkpoints.
- [☑️] Release inference code for both image understanding and generation.


## 🚀 Quick Start Guide

### Installation
```bash
conda create -n UAE python==3.12
conda activate UAE
pip install -r requirements.txt
pip install flash-attn --no-build-isolation 
```

### Model Setup

1. Download the required model checkpoints:
   - Stable Diffusion 3.5 Large model
   - UAE fine-tuned weights
   - Vision-language model checkpoints

2. Update the model paths in `demo.py`:
```python
model_cfg = {
    "SD3": "/path/to/stable-diffusion-3.5-large",
    "dit": "/path/to/dit/checkpoint",
    "dit_lora": "/path/to/dit/lora",
    "llm": "/path/to/llm/model",
    "llm_lora": "/path/to/llm/lora",
    "llm_processor": "/path/to/llm/processor"
}
```
Here, the items are defined as follows:  
- **"SD3"**: Path to the official weights of *Stable Diffusion 3-Large*.  
- **"dit"**: Our pre-trained weights of *DiT*.  
- **"dit_lora"**: Our pre-trained *LoRA* for DiT, obtained in **Stage-3 of unified-GRPO**.  
- **"llm"**: Our pre-trained weights of *Qwen-2.5-VL-3B*.  
- **"llm_lora"**: Our pre-trained *LoRA* for Qwen-2.5-VL-3B, obtained in **Stage-2 of unified-GRPO**.  
- **"llm_processor"**: The official configuration of Qwen-2.5-VL-3B, located at `./Checkpoints/llm_processor`.  


## 🎯 Core Functionality: demo.py

The `demo.py` script is the heart of our inference pipeline, supporting two main modes:

### 1. Text-to-Image (Generation)
Generate images directly from text descriptions:

```bash
python demo.py \
    --input_text "A serene mountain landscape with snow-capped peaks reflecting in a crystal clear lake, surrounded by pine forests under a golden sunset sky" \
    --output_path ./output/generated_image.png
```

### 2. Image-to-Text (Understanding/Captioning)
Generate detailed descriptions of images:

```bash
python demo.py \
    --input_img /path/to/input/image.jpg \
    --prompt_only
```

<a id="benchmark-section"></a>
## 📊 Evaluation Framework

Our comprehensive evaluation suite in the `Unified-Bench/` directory provides multiple similarity metrics for image-to-image generation assessment.

### Supported Metrics

- **CLIP**: Semantic similarity using CLIP vision encoder
- **DINO v2**: Self-supervised visual representation similarity
- **DINO v3**: Enhanced DINO model for improved feature matching
- **LongCLIP**: Extended context CLIP for better long-range dependencies

### Running Evaluation

#### 1. Single Model Evaluation
```bash
cd eval
python CLIP.py --image_path /path/to/generated/images --ref_path /path/to/reference/images
python DINO_v2.py --image_path /path/to/generated/images --ref_path /path/to/reference/images
python DINO_v3.py --image_path /path/to/generated/images --ref_path /path/to/reference/images
python LongCLIP.py --image_path /path/to/generated/images --ref_path /path/to/reference/images
```

#### 2. Comprehensive Multi-Model Evaluation
Use the unified evaluation script for complete assessment:

```bash
cd eval
python Score_i2i.py \
    --image_path ./Unified-Bench/UniBench/example_image \
    --ref_path ./Unified-Bench/UniBench/Image \
    --output_file ./Unified-Bench/results/example.json \
    --models clip dinov2 dinov3 longclip
```

### Unified-Bench Evaluation

The `Unified-Bench/UniBench/` directory contains our evaluation benchmark:

```
UniBench/
├── Image/           # Reference images (100 samples)
│   ├── 0.jpg
│   └── ...
└── example_image/   # Example generated images
    ├── 0.jpg
    └── ...
```

The data from the `Image` folder can be downloaded from the [link](https://github.com/PKU-YuanGroup/UAE/releases/download/v1.0/Image.zip).

#### Evaluation Results Format

The evaluation generates comprehensive statistics:
```json
{
  "clip": {
    "0.jpg": 0.8542,
    "1.jpg": 0.7893,
    "average": 0.8234,
    "min": 0.7123,
    "max": 0.9456
  },
  "dinov2": { ... },
  "dinov3": { ... },
  "longclip": { ... }
}
```

### Custom Evaluation

To evaluate your own generated images:

1. Organize your images following the UniBench structure
2. Ensure corresponding images have matching names
3. Run the evaluation script with your paths
4. Results will include per-image scores and aggregate statistics


## 📬 Contact & Feedback

For questions or feedback, please reach out:

- **Email**: [yanzhiyuan1114@gmail.com]

---

⭐️ If this repository helped your research, please star 🌟 this repo 👍!
