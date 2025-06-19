import torch
from torch import nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    # Create stub objects for triton
    class TritonStub:
        def jit(self, func):
            return func
        language = type('tl', (), {'constexpr': None})()
    triton = TritonStub()
    tl = triton.language

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    # Create stub functions
    def flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError("flash_attn not available")
    def flash_attn_with_kvcache(*args, **kwargs):
        raise NotImplementedError("flash_attn not available")
from nanovllm.utils.context import get_context


if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        slot = tl.load(slot_mapping_ptr + idx)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    if HAS_TRITON:
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
    else:
        # Fallback implementation without triton
        N, num_heads, head_dim = key.shape
        key_flat = key.view(N, -1)
        value_flat = value.view(N, -1)
        for i in range(N):
            slot = int(slot_mapping[i].item())
            k_cache[slot] = key_flat[i]
            v_cache[slot] = value_flat[i]


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache = self.k_cache
        v_cache = self.v_cache
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if HAS_FLASH_ATTN:
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:    # decode
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        else:
            # Fallback implementation using basic PyTorch operations
            import torch.nn.functional as F
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
            
            # Basic attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # Apply causal mask
            seq_len = q.size(-2)
            if seq_len > 1:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            o = torch.matmul(attn_weights, v)
            
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
