import os
import torch

try:
    from pynvml import *
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


def get_gpu_memory():
    if not HAS_PYNVML or not torch.cuda.is_available():
        return 0, 0, 0
    
    try:
        torch.cuda.synchronize()
        nvmlInit()
        visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
        cuda_device_idx = torch.cuda.current_device()
        cuda_device_idx = visible_device[cuda_device_idx]
        handle = nvmlDeviceGetHandleByIndex(cuda_device_idx)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        total_memory = mem_info.total
        used_memory = mem_info.used
        free_memory = mem_info.free
        nvmlShutdown()
        return total_memory, used_memory, free_memory
    except Exception:
        # Fallback for any CUDA/pynvml errors
        return 0, 0, 0
