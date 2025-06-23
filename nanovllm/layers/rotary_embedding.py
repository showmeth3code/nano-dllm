import torch
from torch import nn
from typing import Optional


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.
    
    Args:
        x: Input tensor of shape [..., d]
        
    Returns:
        Rotated tensor of same shape as input
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensors using the given cosine and sine values.
    
    This implementation is designed to match the HuggingFace transformers implementation 
    for Qwen3, which uses the rotate_half function.
    
    Args:
        x: Input tensor of shape [batch, seq_len, heads, head_dim]
        cos: Cosine tensor of shape [seq_len, 1, head_dim] or similar broadcastable shape
        sin: Sine tensor of shape [seq_len, 1, head_dim] or similar broadcastable shape
        
    Returns:
        Tensor with rotary embeddings applied, shape [batch, seq_len, heads, head_dim]
    """
    # Store original dtype for restoring later
    orig_dtype = x.dtype
    
    # Move to float32 for precise computation
    x = x.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    
    # In the Qwen3 implementation, cos/sin from the rotary embedder are [1, seq_len, 128]
    # but our nano-vllm generates [seq_len, 1, 128] - need to handle both formats 
    
    # Make the cos/sin broadcastable to x regardless of exact shape
    # For [seq_len, 1, head_dim] format from nano-vllm
    if cos.shape[0] == x.shape[1] and cos.shape[1] == 1:
        # Assume [seq_len, 1, head_dim] format, reshape to [1, seq_len, 1, head_dim]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    # For [1, seq_len, head_dim] format from HF
    elif cos.shape[0] == 1 and cos.shape[1] == x.shape[1]:
        # Assume [1, seq_len, head_dim] format, reshape to [1, seq_len, 1, head_dim]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        
    # Apply the rotation formula with rotate_half (matching HF implementation)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    
    # Restore original dtype
    x_embed = x_embed.to(orig_dtype)
    
    return x_embed


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE)"""

    def __init__(
        self, 
        head_dim: int,
        rotary_dim: Optional[int] = None,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,  # Changed to float to match HF config
    ):
        super().__init__()
        self.dim = head_dim
        self.rotary_dim = rotary_dim or head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.attention_scaling = 1.0  # Default, can be configured when needed

    def forward(
        self, 
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embedding to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, heads, head_dim]
            cos: Cosine tensor for rotary embedding
            sin: Sine tensor for rotary embedding
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Ensure inputs have the right dtype for stable computation
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Apply rotary embedding using the HF-compatible implementation
        rotated = apply_rotary_emb(x, cos, sin)
        
        # Restore original dtype
        rotated = rotated.to(orig_dtype)
        
        return rotated
