import torch
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_PROMPT = (
    "Give one short global textual description of this image. Include: "
    "1) Style (e.g., cartoon style, realistic photo, 3D render); "
    "2) Main subject (e.g., a cute dog, big sale text); "
    "3) Background (location/scene, this is most important). "
    "Keep it concise, one sentence, no markdown or bullet list."
)


class VLMCaptioner:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
    ):
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = torch.device(device)

        dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else None

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto" if (load_in_4bit or self.device.type == "cuda") else None,
        )

        if not load_in_4bit:
            self.model = self.model.to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def generate(self, image):
        """Generate a whole-image caption focusing on style / subject / background."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image"},
                ],
            }
        ]

        inputs = self.processor(
            messages=messages,
            images=[image.convert("RGB")],
            return_tensors="pt",
        )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            # When processor returns dict
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)

        eos_token_id = None
        if hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "eos_token_id", None) is not None:
            eos_token_id = self.processor.tokenizer.eos_token_id

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature is not None and self.temperature > 0,
            temperature=self.temperature if (self.temperature is not None and self.temperature > 0) else 0.0,
            eos_token_id=eos_token_id,
        )

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return text.strip()


def build_vlm_captioner(config):
    if not config.get("use_vlm_caption", False):
        return None

    model_id = config.get("vlm_model_id", "OpenBMB/MiniCPM-V-2_6")
    prompt = config.get("vlm_prompt", DEFAULT_PROMPT)
    torch_dtype = config.get("vlm_dtype", "bfloat16")
    device = config.get("vlm_device", "cuda")
    max_new_tokens = config.get("vlm_max_new_tokens", 96)
    temperature = config.get("vlm_temperature", 0.2)
    load_in_4bit = config.get("vlm_load_in_4bit", False)

    return VLMCaptioner(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


