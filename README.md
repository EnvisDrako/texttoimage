# Text-to-Image (Stable Diffusion) — Local Streamlit App

A minimal Streamlit-based text-to-image generator using the Hugging Face Diffusers Stable Diffusion pipelines.

**This repository provides a simple UI to load a Stable Diffusion model, generate images from text prompts, and save them to disk.**

**Note:** The app includes basic content filtering and watermarks generated images with "🤖 AI Generated" to promote responsible use.

---

**Features:**
- Load a Stable Diffusion model (GPU or CPU).
- Adjustable generation settings: image size, steps, guidance scale, negative prompt.
- Automatic saving of images and metadata to `generated/YYYY-MM-DD/`.
- Simple safety checks and watermarking.

---

**Prerequisites**
- Windows, macOS or Linux with Python 3.10+ installed.
- Recommended: 8+ GB RAM; for decent performance and to use GPU, an NVIDIA GPU with sufficient VRAM (6+ GB recommended).
- Git (optional) if cloning the repository.

---

**Quickstart (Windows PowerShell)**

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Install PyTorch with CUDA if you have an NVIDIA GPU and want GPU acceleration.
   - Example for CUDA 13.0 (adjust if you need a different CUDA version):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

If you don't have a GPU or prefer CPU-only, a CPU wheel is available from PyTorch's website or install normally with the CPU index.

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`) in your browser.

---

**Usage Notes**
- In the sidebar you can set number of images, image size, inference steps, guidance scale, and negative prompts.
- Click `🔄 Load Model` to download/load the selected model. This may take time on the first run.
- After the model is loaded, click `✨ Generate Images` to create images.
- Generated images and a JSON metadata file are saved to `generated/YYYY-MM-DD/` (see `storage.py`).

**Model selection**
- By default the app prefers a GPU model if a compatible GPU is detected. If no GPU is available it falls back to a CPU model.

---

**GPU & Performance**
- GPU detection and VRAM checks are handled in `utils.py` (`get_gpu_info`).
- Recommended VRAM: 6+ GB to comfortably use fp16; 4 GB minimum to run smaller models but memory may be tight.
- If you experience GPU errors, check drivers with `nvidia-smi` and verify PyTorch CUDA is installed correctly.

Troubleshooting quick commands (PowerShell):

```powershell
# Check NVIDIA GPU and driver
nvidia-smi

# Activate venv then start the app
.venv\Scripts\Activate.ps1; streamlit run app.py
```

If GPU detection fails, call `utils.print_gpu_debug_info()` from a Python REPL to get diagnostics.

---

**Safety & Responsible Use**
- The app includes a list of disallowed keywords and will warn on potentially inappropriate prompts.
- All images are watermarked by default with "🤖 AI Generated". You can toggle watermarking in the UI, but keep transparency in mind.
- Do not use this tool to generate explicit, harmful, or illegal content.

---

**Files of interest**
- `app.py` — Streamlit UI and app logic.
- `generator.py` — Model loading and image generation helper functions.
- `prompt_enhancer.py` — Prompt enhancement & default negative prompt utility.
- `utils.py` — GPU and system helper functions.
- `storage.py` — Save images and metadata.

---

**Output location**
- Saved images and metadata JSON files go to `generated/<YYYY-MM-DD>/`.
