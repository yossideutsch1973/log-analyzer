import os
import torch
import subprocess

def test_cuda():
    print("=== CUDA Environment Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
        print("\n=== NVIDIA-SMI Output ===")
        print(nvidia_smi)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
    
    try:
        if torch.cuda.is_available():
            # Try to initialize CUDA explicitly
            torch.cuda.init()
            device_count = torch.cuda.device_count()
            print(f"\nCUDA device count: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"\nDevice {i}: {props.name}")
                print(f"  Compute capability: {props.major}.{props.minor}")
                print(f"  Total memory: {props.total_memory / 1024**2:.0f} MB")
    except Exception as e:
        print(f"\nError during CUDA initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuda()
