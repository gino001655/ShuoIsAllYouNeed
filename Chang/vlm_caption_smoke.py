import argparse
from pathlib import Path
from PIL import Image

from vlm_caption import build_vlm_captioner, DEFAULT_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test for VLMCaptioner. Outputs one generated whole-image caption."
    )
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument(
        "--model_id",
        default="OpenBMB/MiniCPM-V-2_6",
        help="Hugging Face model id (supports MiniCPM-V / LLaVA-Next family).",
    )
    parser.add_argument("--device", default="cuda", help='Device, e.g., "cuda" or "cpu".')
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help='Torch dtype for the model, e.g., "bfloat16", "float16", "float32".',
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit to save memory (requires bitsandbytes).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature. Set 0 or negative to disable sampling.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instruction prompt for the VLM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")

    # Build config dict expected by build_vlm_captioner
    config = {
        "use_vlm_caption": True,
        "vlm_model_id": args.model_id,
        "vlm_device": args.device,
        "vlm_dtype": args.dtype,
        "vlm_load_in_4bit": args.load_in_4bit,
        "vlm_max_new_tokens": args.max_new_tokens,
        "vlm_temperature": args.temperature,
        "vlm_prompt": args.prompt,
    }

    print(f"[INFO] Loading VLM model: {args.model_id}")
    captioner = build_vlm_captioner(config)
    print("[INFO] Model loaded. Generating caption...")

    caption = captioner.generate(image)
    print("\n=== Generated whole caption ===")
    print(caption)


if __name__ == "__main__":
    main()


