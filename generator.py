import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from typing import List, Tuple
import time


def load_pipeline(model_name: str, device: str, use_fp16: bool = False, attention_slicing: bool = True):
    dtype = torch.float16 if use_fp16 else torch.float32
    
    # Load the pipeline WITH safety_checker enabled
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        revision="fp16" if use_fp16 else None # We'll handle safety at app level
    )
    
    # Move to device FIRST before any modifications
    if device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    
    # NOW apply scheduler and optimizations
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    if attention_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    
    return pipe


def generate_images(
    pipe,
    prompt: str,
    negative_prompt: str = None,
    num_images: int = 1,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
):
    """
    Generates images using the provided pipeline.
    Raises ValueError if prompt is empty.
    Returns list of tuples (PIL.Image, elapsed_seconds)
    """
    if not prompt or not str(prompt).strip():
        raise ValueError("Prompt is empty. Please provide a non-empty text prompt.")

    results = []
    for i in range(num_images):
        start = time.time()
        try:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
            # Check if image was flagged as unsafe
            if out.nsfw_content_detected and len(out.nsfw_content_detected) > 0:
                if any(out.nsfw_content_detected):
                    raise RuntimeError("⚠️ Generated image was flagged as unsafe by the safety filter. Please modify your prompt and try again.")
            
        except Exception as e:
            raise RuntimeError(f"Pipeline call failed: {e}") from e

        img = out.images[0]
        elapsed = time.time() - start
        results.append((img, elapsed))
    return results