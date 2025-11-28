import streamlit as st
from utils import get_gpu_info, has_enough_system_ram
from generator import load_pipeline, generate_images
from prompt_enhancer import enhance_prompt, default_negative_prompt
from storage import save_image_and_meta
import torch
import os
import time
from PIL import Image, ImageDraw, ImageFont
import io

# Page config
st.set_page_config(
    page_title="AI Image Generator",
    layout='wide',
    initial_sidebar_state="expanded",
    menu_items={"About": "Text-to-Image Generator powered by Stable Diffusion"}
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
    [data-testid="stSidebar"] { background: linear-gradient(135deg, #1e1b4b 0%, #1e0f3d 100%); }
    .main { background: linear-gradient(135deg, #0f172a 0%, #1a0f2e 100%); }
</style>
""", unsafe_allow_html=True)

# Content filter - inappropriate keywords
INAPPROPRIATE_KEYWORDS = [
    "nude", "nsfw", "explicit", "porn", "sex", "violence", "gore", "weapons",
    "hate", "racism", "terrorist", "drug", "illegal", "malware", "hacking"
]

def check_prompt_safety(prompt: str) -> tuple[bool, str]:
    """Check if prompt contains inappropriate content"""
    prompt_lower = prompt.lower()
    for keyword in INAPPROPRIATE_KEYWORDS:
        if keyword in prompt_lower:
            return False, f"⚠️ Prompt contains inappropriate content: '{keyword}'. Please modify your prompt."
    return True, ""

def add_watermark(image: Image.Image, watermark_text: str = "🤖 AI Generated") -> Image.Image:
    """Add a watermark to indicate AI generation"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    
    # Image dimensions
    width, height = img_copy.size
    
    # Add semi-transparent watermark at bottom
    watermark_box_height = 40
    box_coords = [(0, height - watermark_box_height), (width, height)]
    draw.rectangle(box_coords, fill=(0, 0, 0, 80))
    
    # Add text
    font_size = max(int(width / 25), 14)
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = height - watermark_box_height + 10
    
    draw.text((text_x, text_y), watermark_text, fill=(255, 255, 255, 200), font=font)
    
    return img_copy

def estimate_time(num_images: int, steps: int, device: str) -> str:
    """Estimate generation time based on parameters"""
    # Rough estimates (in seconds per image)
    base_time = steps * 0.5  # ~0.5s per step
    
    if device == "cpu":
        multiplier = 8  # CPU is ~8x slower
    else:
        multiplier = 1  # GPU baseline
    
    estimated_seconds = base_time * multiplier * num_images
    
    if estimated_seconds < 60:
        return f"~{int(estimated_seconds)}s"
    else:
        minutes = estimated_seconds / 60
        return f"~{int(minutes)}m {int(estimated_seconds % 60)}s"

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("### ✨ AI Image Generator")
with col2:
    st.markdown("<h5 style='text-align: center; color: #a78bfa;'>Powered by Stable Diffusion</h5>", unsafe_allow_html=True)
with col3:
    gpu_available, vram_gb, gpu_name = get_gpu_info()
    status_color = "🟢" if gpu_available else "🔴"
    st.markdown(f"<h6 style='text-align: right;'>{status_color} GPU Ready</h6>", unsafe_allow_html=True)

st.divider()

# Responsible Use Guidelines
with st.expander("📋 Responsible Use & Guidelines", expanded=False):
    st.markdown("""
    ### ⚖️ Responsible AI Generation
    
    **Content Policy:**
    - ✅ DO: Create educational, artistic, and creative content
    - ✅ DO: Use for inspiration and design prototyping
    - ❌ DON'T: Generate offensive, explicit, or harmful content
    - ❌ DON'T: Create misleading or deceptive imagery
    - ❌ DON'T: Attempt to bypass content filters
    
    **Watermarking:**
    - All generated images are automatically watermarked with "🤖 AI Generated"
    - This helps identify AI-created content for transparency
    
    **Best Practices:**
    - Be specific and descriptive in your prompts
    - Experiment with different styles and parameters
    - Respect copyright and intellectual property rights
    - Don't use generated images to impersonate real people
    """)

# Hardware detection
gpu_available, vram_gb, gpu_name = get_gpu_info()
ram_ok, total_ram = has_enough_system_ram(8)

with st.expander("💻 System Hardware", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("GPU Available", "✅ Yes" if gpu_available else "❌ No")
    with col2:
        st.metric("VRAM", f"{vram_gb:.2f} GB")
    with col3:
        st.metric("System RAM", f"{total_ram:.2f} GB" if total_ram else "Unknown")
    if gpu_name:
        st.info(f"🎮 GPU: {gpu_name}")

# Sidebar controls
st.sidebar.title("⚙️ Generation Settings")

# Prompt section
prompt = st.sidebar.text_area(
    "🎨 Prompt",
    value="a futuristic city at sunset",
    height=100,
    help="Describe what you want to generate"
)

# Style selection
style = st.sidebar.selectbox(
    "🎭 Style (optional)",
    ["", "photorealistic", "oil painting", "cartoon", "anime", "cyberpunk", "watercolor", "digital art", "steampunk", "pixel art"],
    help="Choose an art style to apply"
)

# Basic generation settings
col1, col2 = st.sidebar.columns(2)
with col1:
    num_images = st.number_input("📷 Images", min_value=1, max_value=4, value=1)
with col2:
    image_width = st.selectbox("📏 Size", [256, 384, 512], index=2)

width = image_width
height = width

# Quality sliders
steps = st.sidebar.slider(
    "🔄 Inference Steps",
    min_value=10,
    max_value=50,
    value=25,
    help="Higher = better quality but slower"
)

guidance = st.sidebar.slider(
    "🎯 Guidance Scale",
    min_value=1.0,
    max_value=15.0,
    value=7.5,
    step=0.5,
    help="How closely the model follows your prompt"
)

# Advanced settings
neg_prompt = ""
format_choice = "png"
user_confirm_gpu = gpu_available and vram_gb >= 4
user_prefer_gpu = True
add_watermark_toggle = True

with st.sidebar.expander("🔧 Advanced"):
    neg_prompt = st.text_area(
        "Negative Prompt",
        value="",
        height=80,
        help="Things to avoid in the image"
    )
    format_choice = st.radio("Format", ["png", "jpeg"], horizontal=True, index=0)
    add_watermark_toggle = st.checkbox("Add AI watermark", value=True, help="Adds watermark to indicate AI generation")
    user_confirm_gpu = st.checkbox("Use GPU (if available)", value=(gpu_available and vram_gb >= 4))
    user_prefer_gpu = st.checkbox("Prefer GPU over CPU", value=True)

st.sidebar.divider()

# Model control
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    load_model_button = st.button("🔄 Load Model", width='stretch')
with col2:
    unload_model = st.button("🗑️ Unload", width='stretch')

# Model selection logic
MODEL_V1 = "runwayml/stable-diffusion-v1-5"
MODEL_CPU_FALLBACK = "CompVis/stable-diffusion-v1-4"

use_gpu = False
if gpu_available and user_confirm_gpu and user_prefer_gpu:
    use_gpu = True

use_fp16 = use_gpu and (vram_gb >= 6)
attention_slicing = True

if 'pipe' not in st.session_state:
    st.session_state.pipe = None
    st.session_state.model_name = None

selected_model = MODEL_V1 if use_gpu else MODEL_CPU_FALLBACK

if load_model_button or st.session_state.pipe is None:
    with st.spinner(f"🚀 Loading model {selected_model}... (this may take a minute)"):
        try:
            device = "cuda" if use_gpu else "cpu"
            st.session_state.pipe = load_pipeline(
                selected_model,
                device=device,
                use_fp16=use_fp16,
                attention_slicing=attention_slicing
            )
            st.session_state.model_name = selected_model
            st.session_state.device_used = device
            st.sidebar.success(f"✅ Model loaded on {device.upper()}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load model: {e}")

if unload_model:
    st.session_state.pipe = None
    st.session_state.model_name = None
    st.sidebar.info("Model unloaded")

st.sidebar.divider()

# Time estimate
estimated_time = estimate_time(num_images, steps, "cuda" if use_gpu else "cpu")
st.sidebar.info(f"⏱️ Estimated time: {estimated_time}")

# Generate button (sidebar)
generate_button = st.sidebar.button("✨ Generate Images", width='stretch', type="primary", key="gen_btn")

st.sidebar.divider()

# Show which device will be used
device_info = "cuda" if use_gpu else "cpu"
device_emoji = "🎮" if use_gpu else "💻"
st.sidebar.info(f"{device_emoji} Using: **{device_info.upper()}**")

st.markdown("")

# Main content area
st.markdown("---")

# Generation logic
if generate_button:
    if st.session_state.pipe is None:
        st.error("❌ Model not loaded. Please click '🔄 Load Model' in the sidebar.")
        st.stop()

    raw_prompt = prompt

    if raw_prompt is None or raw_prompt.strip() == "":
        st.error("❌ Prompt cannot be empty.")
        st.stop()

    # Check prompt safety
    is_safe, safety_message = check_prompt_safety(raw_prompt)
    if not is_safe:
        st.error(safety_message)
        st.stop()

    final_prompt = enhance_prompt(raw_prompt, style=style, add_quality=True)
    final_neg = default_negative_prompt(neg_prompt)

    # Progress tracking
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    time_placeholder = st.empty()
    
    start_total_time = time.time()

    results = []
    for i in range(num_images):
        progress_percent = int((i / num_images) * 100)
        progress_bar.progress(progress_percent)
        
        elapsed_total = time.time() - start_total_time
        remaining_images = num_images - i
        avg_time_per_image = elapsed_total / (i + 1) if i > 0 else estimated_time.replace("~", "").replace("s", "").replace("m", "")
        
        status_placeholder.info(f"⏳ Generating image {i+1} of {num_images}...")
        time_placeholder.caption(f"⏱️ Elapsed: {int(elapsed_total)}s | Processing: Image {i+1}")

        try:
            image_start = time.time()
            res = generate_images(
                st.session_state.pipe,
                prompt=final_prompt,
                negative_prompt=final_neg,
                num_images=1,
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
            )
            results.extend(res)
        except RuntimeError as e:
            if "unsafe" in str(e).lower():
                st.error(f"🚫 **Safety Filter Triggered**: {e}\n\nThe generated image was flagged as containing unsafe content. Please modify your prompt and try again.")
            else:
                st.error(f"❌ Generation failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            st.stop()

    progress_bar.progress(100)
    total_time = time.time() - start_total_time
    status_placeholder.success(f"✅ Generation complete! Total time: {int(total_time)}s")
    time_placeholder.empty()
    
    # Show which device was actually used
    device_used = getattr(st.session_state, 'device_used', 'unknown').upper()
    st.info(f"🚀 Generated on **{device_used}** | ⏱️ Total time: {int(total_time)}s")

    st.divider()

    # Display results
    cols = st.columns(num_images)
    
    # Store metadata for all images
    all_images_data = []

    for idx, (img, elapsed) in enumerate(results, start=1):
        with cols[idx-1]:
            # Add watermark if enabled
            if add_watermark_toggle:
                img_with_watermark = add_watermark(img)
            else:
                img_with_watermark = img
            
            st.image(img_with_watermark, caption=f"Image {idx} · {elapsed:.1f}s", width='stretch')

            metadata = {
                "prompt": raw_prompt,
                "final_prompt": final_prompt,
                "negative_prompt": final_neg,
                "model": st.session_state.model_name,
                "height": height,
                "width": width,
                "steps": steps,
                "guidance": guidance,
                "index": idx,
                "format": format_choice,
                "watermarked": add_watermark_toggle,
            }

            img_path, _ = save_image_and_meta(img_with_watermark, metadata)
            all_images_data.append((img_with_watermark, img_path))

            with open(img_path, "rb") as f:
                st.download_button(
                    label="📥 Download",
                    data=f,
                    file_name=img_path.split("\\")[-1] if "\\" in img_path else img_path.split("/")[-1],
                    mime=f"image/{format_choice}",
                    width='stretch',
                    key=f"download_{idx}"
                )
    
    # Download all button
    if len(all_images_data) > 1:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            import zipfile
            from datetime import datetime
            
            # Create zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for img, img_path in all_images_data:
                    file_name = img_path.split("\\")[-1] if "\\" in img_path else img_path.split("/")[-1]
                    zip_file.write(img_path, arcname=file_name)
            
            zip_buffer.seek(0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📦 Download All",
                data=zip_buffer.getvalue(),
                file_name=f"generated_images_{timestamp}.zip",
                mime="application/zip",
                width='stretch',
            )