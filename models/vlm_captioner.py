from click import prompt
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image



class Captioner:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device:str="cuda"):

        # Enable TF32 on Ampere+ for speed boost
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.device = device


        # Choose the fastest safe dtype for your GPU
        if torch.cuda.is_available():
            # bf16 preferred on A100/RTX 30xx/40xx; fall back to fp16 otherwise
            try_dtype = torch.bfloat16
        else:
            try_dtype = torch.float32
            

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            device_map="auto",
            torch_dtype=try_dtype if torch.cuda.is_available() else torch.float32,
            # attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print(f"Loaded Qwen/Qwen2.5-VL-7B-Instruct on {self.model.device} dtype={next(self.model.parameters()).dtype}")


        SYS_PROMPT = (
            "You are an image captioning model. "
            "Your job is to describe exactly what you see in the image using a single short sentence. "
            "Do NOT include interpretations, guesses, emotions, opinions, style, or artistic judgment. "
            "Describe only concrete visible objects, attributes, colors, positions, and actions. "
            "Avoid hallucinating unseen objects. "
            "Be concise, factual, and literal. "
            "Return ONLY the caption text, no additional words."
        )

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]},
        ]

        self.chat_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )


    def caption(self, image) -> str:

        inputs = self.processor(
            text=[self.chat_prompt],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=self.model.device.type=="cuda"):
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,     # deterministic
                # temperature=0.0,
            )

        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        caption = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        # print("VLM output:", text)

        return caption


    def caption_batch(self, images: list[Image.Image]) -> list[str]:
        """
        Caption a batch of images in one forward pass.
        """
        prompts = [self.chat_prompt] * len(images)
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=self.model.device.type == "cuda"
        ):
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        captions = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True
        )
        return [c.strip() for c in captions]