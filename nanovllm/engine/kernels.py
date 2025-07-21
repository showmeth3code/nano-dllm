# kernels.py

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_HEAD': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_HEAD': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_HEAD': 256}, num_warps=8),
    ],
    key=['head_size'],
)
@triton.jit
def copy_kv_prefix_kernel(
    # --- Pointers ---
    src_ptr, dst_ptr,
    # --- Arguments ---
    num_tokens_to_copy, num_layers, num_kv_heads, head_size,
    # --- Strides ---
    stride_src_l, stride_src_kv, stride_src_h, stride_src_t,
    stride_dst_l, stride_dst_kv, stride_dst_h, stride_dst_t,
    # --- Constants ---
    BLOCK_SIZE_HEAD: tl.constexpr,
):
    """
    Triton Kernel: 高效地将一个KV缓存块的部分前缀拷贝到另一个块。
    这是一个 3D Grid 的启动: (num_layers, num_kv_heads, 2 for k/v)
    """
    layer_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_idx = tl.program_id(2) # 0 for K, 1 for V

    # 遍历需要拷贝的Token
    for token_idx in range(num_tokens_to_copy):
        # 向量化处理 head_size 维度
        for head_offset in range(0, head_size, BLOCK_SIZE_HEAD):
            offsets = head_offset + tl.arange(0, BLOCK_SIZE_HEAD)
            mask = offsets < head_size

            # 计算源和目标的完整内存地址
            src_offset = (layer_idx * stride_src_l + 
                          kv_idx * stride_src_kv + 
                          head_idx * stride_src_h + 
                          token_idx * stride_src_t + 
                          offsets)
            
            dst_offset = (layer_idx * stride_dst_l + 
                          kv_idx * stride_dst_kv + 
                          head_idx * stride_dst_h + 
                          token_idx * stride_dst_t + 
                          offsets)

            # 加载和存储数据
            data = tl.load(src_ptr + src_offset, mask=mask)
            tl.store(dst_ptr + dst_offset, data, mask=mask)


def copy_kv_prefix_host(
    src_block: torch.Tensor, 
    dst_block: torch.Tensor, 
    num_tokens_to_copy: int
):
    """
    Python host function to launch the Triton kernel.
    """
    # Tensor 形状: [num_layers, 2, num_heads, block_size, head_size]
    assert src_block.dim() == 5 and dst_block.dim() == 5
    assert src_block.shape == dst_block.shape
    
    num_layers, _, num_heads, block_size, head_size = src_block.shape
    assert num_tokens_to_copy <= block_size

    # 定义 Triton Kernel 的启动网格 (Grid)
    grid = (num_layers, num_heads, 2)

    # 启动 Kernel
    copy_kv_prefix_kernel[grid](
        src_block, dst_block,
        num_tokens_to_copy,
        num_layers, num_heads, head_size,
        src_block.stride(0), src_block.stride(1), src_block.stride(2), src_block.stride(3),
        dst_block.stride(0), dst_block.stride(1), dst_block.stride(2), dst_block.stride(3),
    )