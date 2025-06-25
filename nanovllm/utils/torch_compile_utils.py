import os
import torch

def optional_torch_compile(fn):
    """
    Decorator to optionally apply torch.compile to a function based on device and environment variable.
    Usage:
        @optional_torch_compile
        def forward(...):
            ...
    Control with env var USE_TORCH_COMPILE (default: on for CUDA, off for MPS/CPU).
    """
    use_compile = os.environ.get("USE_TORCH_COMPILE")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
    if use_compile is not None:
        enabled = use_compile.lower() in ("1", "true", "yes", "on")
    else:
        enabled = device.type == "cuda"
    if enabled and hasattr(torch, "compile"):
        return torch.compile(fn)
    return fn
