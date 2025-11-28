import torch
import sys

def get_gpu_info():
    """Return (available: bool, total_vram_gb: float or 0, name:str or None)"""
    try:
        if not torch.cuda.is_available():
            return False, 0.0, None
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False, 0.0, None
            
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024**3)
        name = props.name if hasattr(props, 'name') else "Unknown GPU"
        return True, float(total), name
    except Exception as e:
        print(f"GPU detection error: {e}", file=sys.stderr)
        return False, 0.0, None

def has_enough_system_ram(min_gb=8):
    try:
        import psutil
        available_gb = psutil.virtual_memory().total / (1024**3)
        return available_gb >= min_gb, available_gb
    except Exception:
        # psutil is optional; if missing, assume True
        return True, None

def print_gpu_debug_info():
    """Call this to debug GPU issues"""
    print("\n=== GPU Debug Info ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Current Device: {torch.cuda.current_device()}")
    else:
        print("No CUDA-capable GPU detected!")
        print("\nTroubleshooting:")
        print("1. Check NVIDIA drivers: nvidia-smi")
        print("2. Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        print("3. Verify CUDA toolkit is installed")
    print("=== End Debug Info ===\n")