
import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.torch_compile_utils import optional_torch_compile


class SiluAndMul(nn.Module):
    """SiLU activation followed by multiplication with input.
    
    Matches HuggingFace's implementation exactly.
    """

    def __init__(self):
        super().__init__()
        # Use nn.SiLU() to match HuggingFace exactly instead of functional
        self.act = nn.SiLU()
        
    @optional_torch_compile
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Apply activation to gate then multiply with up projection
        return self.act(x) * y
