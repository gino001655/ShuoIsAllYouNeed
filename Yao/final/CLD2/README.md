# CLD2 (SD3.5 + UAE VLM)

This is an experimental fork of `Yao/final/CLD/` that replaces the FLUX backbone with **Stable Diffusion 3.5 (SD3/SD3.5)** and replaces the text-conditioning stack with **UAE's Qwen2.5-VL projector embeddings**.

Goals:
- Keep CLD's controllable multi-layer decomposition training objective (bbox-masked loss on multi-layer latents).
- Use UAE's SD3.5 denoiser (+ optional LoRA) and Qwen2.5-VL projected embeddings.
- Provide VRAM-saving switches (mixed precision, checkpointing, SDPA/Flash, 4-bit LLM, layer subsampling).

Run (training):
```bash
python -m train.train -c train/train.yaml
```

<div align="center">

## Controllable Layer Decomposition for Reversible Multi-Layer Image Generation

🏠 [Homepage](https://monkek123King.github.io/CLD_page)      📄 [Paper](http://arxiv.org/abs/2511.16249)      🤗 [HuggingFace](https://huggingface.co/papers/2511.16249)


</div>


### 📢 News

  * **`Nov 2025`:** The paper is now available on [arXiv](https://arxiv.org/abs/2511.16249). ☕️

-----

## 🚀 Getting Started

### 🔧 Installation

**a. Create a conda virtual environment and activate it.**
```shell
conda env create -f environment.yml
conda activate CLD
```

**b. Clone CLD.**
```
git clone https://github.com/monkek123King/CLD.git
```

### 🏋️ Train and Evaluate

**Train**

```
python -m train.train -c train/train.yaml
```

**Infer**
```
python -m infer.infer -c infer/infer.yaml
```

**Eval**

Prepare the ground-truth samples.
```
python -m eval.prepare_gt
```

Evaluate to obtain the metric results.
```
python evaluate.py --pred-dir "Path_to_predict_results" --gt-dir "Path_to_gt_samples" --output-dir "Path_to_save_eval_results"
```

-----

## ✍️ Citation

If you find our work useful for your research, please consider citing our paper and giving this repository a star 🌟.

```bibtex
@article{liu2025controllable,
  title={Controllable Layer Decomposition for Reversible Multi-Layer Image Generation},
  author={Liu, Zihao and Xu, Zunnan and Shu, Shi and Zhou, Jun and Zhang, Ruicheng and Tang, Zhenchao and Li, Xiu},
  journal={arXiv preprint arXiv:2511.16249},
  year={2025}
}
```
