import os
import yaml
from huggingface_hub import snapshot_download

# ==========================================
# 1. Setting and verifying (Configuration)
# ==========================================
# Please fill in your Hugging Face Token here (required, used for downloading FLUX.1-dev)
HF_TOKEN = "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" 

# Define the download target folder
BASE_DIR = os.getcwd() # current directory
FLUX_PATH = os.path.join(BASE_DIR, "models/FLUX.1-dev")
ADAPTER_PATH = os.path.join(BASE_DIR, "models/FLUX.1-dev-Controlnet-Inpainting-Alpha")
CLD_CKPT_DIR = os.path.join(BASE_DIR, "ckpt")

print(f"Working directory: {BASE_DIR}")

# ==========================================
# a. Download FLUX.1-dev weights
# ==========================================
print("\n--- [a] Downloading FLUX.1-dev ---")
try:
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        local_dir=FLUX_PATH,
        token=HF_TOKEN
    )
    print(f"FLUX model saved to: {FLUX_PATH}")
except Exception as e:
    print(f"Error downloading FLUX: {e}")
    print("Please check if you have filled in the correct HF_TOKEN and have signed the FLUX license agreement on the official website.")

# ==========================================
# b. Download Adapter pre-trained weights
# ==========================================
print("\n--- [b] Downloading Adapter weights ---")
snapshot_download(
    repo_id="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    local_dir=ADAPTER_PATH
)
print(f"Adapter saved to: {ADAPTER_PATH}")

# ==========================================
# c. Download LoRA weights for CLD
# ==========================================
print("\n--- [c] Downloading CLD LoRA weights ---")
cld_patterns = [
    "decouple_LoRA/*",
    "pre_trained_LoRA/*",
    "prism_ft_LoRA/*",
    "trans_vae/*"
]
snapshot_download(
    repo_id="thuteam/CLD",
    local_dir=CLD_CKPT_DIR,
    allow_patterns=cld_patterns,
    local_dir_use_symlinks=False
)
print(f"CLD weights saved to: {CLD_CKPT_DIR}")

# ==========================================
# d. Generate YAML configuration file (Generate YAML)
# ==========================================
print("\n--- [d] Generating config.yaml ---")

# Create YAML content dictionary
config_data = {
    "pretrained_model_name_or_path": FLUX_PATH,
    "pretrained_adapter_path": ADAPTER_PATH,
    "transp_vae_path": os.path.join(CLD_CKPT_DIR, "trans_vae/0008000.pt"),
    "pretrained_lora_dir": os.path.join(CLD_CKPT_DIR, "pre_trained_LoRA"),
    "artplus_lora_dir": os.path.join(CLD_CKPT_DIR, "prism_ft_LoRA"),
    "lora_ckpt": os.path.join(CLD_CKPT_DIR, "decouple_LoRA/transformer"),
    "layer_ckpt": os.path.join(CLD_CKPT_DIR, "decouple_LoRA"),
    "adapter_lora_dir": os.path.join(CLD_CKPT_DIR, "decouple_LoRA/adapter")
}

# Write to file
yaml_path = os.path.join(BASE_DIR, "config.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

print(f"Configuration file created successfully: {yaml_path}")
print("Content preview:")
print("-" * 20)
print(yaml.dump(config_data, sort_keys=False))
print("-" * 20)