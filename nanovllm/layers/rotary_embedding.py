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

    # [batch, seq_len, num_heads, head_dim]
    # x: [batch, seq_len, num_heads, head_dim]
    # cos/sin: [batch, seq_len, head_dim] or [seq_len, 1, head_dim] or [1, seq_len, head_dim]
    # print(f"[DEBUG][rotary] x shape: {x.shape}, cos shape: {cos.shape}, sin shape: {sin.shape}")

    # Ensure cos/sin are broadcastable to x: [batch, seq_len, num_heads, head_dim]
    # We want cos/sin to be [batch, seq_len, 1, head_dim] or [1, seq_len, 1, head_dim]
    if cos.dim() == 3:
        # If cos is [batch, seq_len, head_dim] or [seq_len, 1, head_dim] or [1, seq_len, head_dim]
        if cos.shape[0] == x.shape[0] and cos.shape[1] == x.shape[1]:
            # [batch, seq_len, head_dim] -> [batch, seq_len, 1, head_dim]
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        elif cos.shape[0] == x.shape[1] and cos.shape[1] == 1:
            # [seq_len, 1, head_dim] -> [1, seq_len, 1, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.shape[0] == 1 and cos.shape[1] == x.shape[1]:
            # [1, seq_len, head_dim] -> [1, seq_len, 1, head_dim]
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        else:
            raise RuntimeError(f"[rotary] Unhandled cos/sin shape: {cos.shape}, x: {x.shape}")
    elif cos.dim() == 4:
        # Already broadcastable
        pass
    else:
        raise RuntimeError(f"[rotary] Unexpected cos/sin dim: {cos.dim()}")

    # print(f"[DEBUG][rotary] x shape: {x.shape}, cos shape (after): {cos.shape}, sin shape (after): {sin.shape}")

    # Validate broadcastability
    if not (cos.shape[0] in (1, x.shape[0]) and cos.shape[1] == x.shape[1] and cos.shape[2] == 1 and cos.shape[3] == x.shape[3]):
        raise RuntimeError(f"[rotary] After unsqueeze, cos shape {cos.shape} not broadcastable to x {x.shape}")

    # [batch, seq_len, num_heads, head_dim]
    try:
        x_embed = (x * cos) + (rotate_half(x) * sin)
    except RuntimeError as e:
        raise RuntimeError(f"[rotary] Shape mismatch: x {x.shape}, cos {cos.shape}, sin {sin.shape}. Error: {e}")

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
