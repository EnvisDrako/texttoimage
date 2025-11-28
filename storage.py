import os
import json
from datetime import datetime
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_image_and_meta(img, metadata: dict, base_folder="generated"):
    # metadata should include a safe filename or we make one
    date = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(base_folder, date)
    ensure_dir(folder)
    base = metadata.get("filename_base") or metadata.get("prompt", "image").replace(" ", "_")[:50]
    idx = metadata.get("index", 1)
    fmt = metadata.get("format", "png").lower()
    fname = f"{base}_{idx:03d}.{fmt}"
    img_path = os.path.join(folder, fname)
    # save
    img.save(img_path)
    # metadata JSON
    meta_path = os.path.join(folder, f"{base}_{idx:03d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return img_path, meta_path