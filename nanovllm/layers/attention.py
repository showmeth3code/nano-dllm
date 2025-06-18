import torch
import torch.nn as nn
import torch.nn.functional as F
from nanovllm.utils.context import get_context


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """
    替代 Triton 版本的 KV 缓存写入：纯 PyTorch 实现
    key/value: (B, H, D)
    k_cache/v_cache: (num_slots, H * D)
    slot_mapping: (B,)
    """
    B, H, D = key.shape
    flat_key = key.reshape(B, H * D)
    flat_value = value.reshape(B, H * D)
    k_cache[slot_mapping] = flat_key
    v_cache[slot_mapping] = flat_value


def pytorch_attention(q, k, v, mask=None, scale=None):
    """
    q/k/v: (B, H, D)
    return: (B, H, D)
    """
    B, H, D = q.shape
    q = q.unsqueeze(2)         # (B, H, 1, D)
    k = k.unsqueeze(1)         # (B, 1, H, D)
    scores = (q * k).sum(-1)   # (B, H, H)
    if scale:
        scores *= scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    probs = F.softmax(scores, dim=-1)
    v = v.unsqueeze(1)         # (B, 1, H, D)
    out = (probs.unsqueeze(-1) * v).sum(2)  # (B, H, D)
    return out


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.empty(0)
        self.v_cache = torch.empty(0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q: (B, H*D), k/v: (B, H*D)
        context: 从 get_context() 获取
        """
        B = q.shape[0]
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_kv_heads, self.head_dim)
        v = v.view(B, self.num_kv_heads, self.head_dim)

        context = get_context()

        # 初始化缓存（如果还没初始化）
        if self.k_cache.numel() == 0:
            num_slots = context.kv_cache.shape[0]
            D = self.num_kv_heads * self.head_dim
            self.k_cache = torch.zeros(
                (num_slots, D), device=q.device, dtype=q.dtype)
            self.v_cache = torch.zeros(
                (num_slots, D), device=q.device, dtype=q.dtype)

        # 写入 KV cache
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k = self.k_cache
                v = self.v_cache
            k = k.view(B, self.num_kv_heads, self.head_dim)
            v = v.view(B, self.num_kv_heads, self.head_dim)
            out = pytorch_attention(q, k, v, scale=self.scale)
        else:
            # decode: 每个样本只取一个 slot
            slots = context.slot_mapping
            k = self.k_cache[slots].view(B, self.num_kv_heads, self.head_dim)
            v = self.v_cache[slots].view(B, self.num_kv_heads, self.head_dim)
            out = pytorch_attention(q, k, v, scale=self.scale)

        return out.view(B, self.num_heads * self.head_dim)
