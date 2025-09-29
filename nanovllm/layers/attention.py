import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.ar_conetxt import get_context
from nanovllm.utils.diffusion_context import get_context as get_diffusion_context


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
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
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
        return o

def _repeat_kv_to_q_heads(kv: torch.Tensor, num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    # kv: [Lk, Hkv, D]  ->  [Lk, Hq, D]  by repeating along head dim
    if num_kv_heads == num_q_heads:
        return kv
    assert num_q_heads % num_kv_heads == 0, f"num_heads({num_q_heads}) must be multiple of num_kv_heads({num_kv_heads})"
    repeat = num_q_heads // num_kv_heads
    return kv.repeat_interleave(repeat, dim=1)

def _view_kcache_unified(k_cache: torch.Tensor, v_cache: torch.Tensor):
    # Expect: [num_blocks, block_size, Hkv, D]
    return k_cache, v_cache

def _view_kcache_distinct(k_cache: torch.Tensor, v_cache: torch.Tensor, head_dim: int, x: int):
    """
    k_cache: [num_blocks, Hkv, D//x, block_size, x]
    v_cache: [num_blocks, Hkv, D,      block_size]
    Return views shaped as [num_blocks, block_size, Hkv, D]
    """
    # K: [B, H, D//x, T, x] -> [B, T, H, D] by permute then reshape
    k_view = k_cache.permute(0, 3, 1, 2, 4).contiguous().view(
        k_cache.size(0),  # num_blocks
        k_cache.size(3),  # block_size (T)
        k_cache.size(1),  # Hkv
        head_dim         # D = (D//x)*x
    )
    # V: [B, H, D, T] -> [B, T, H, D]
    v_view = v_cache.permute(0, 3, 1, 2).contiguous()
    return k_view, v_view

def _gather_ctx_kv_for_one_seq(kc_view: torch.Tensor, vc_view: torch.Tensor,
                               block_table_row: torch.Tensor, ctx_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    kc_view/vc_view: [num_blocks, block_size, Hkv, D] (device = CUDA)
    block_table_row: [num_mem_blocks] (int32, on CUDA)
    ctx_len: number of cached tokens for this seq (int)
    Return K/V of shape [ctx_len, Hkv, D] (or empty tensor if ctx_len == 0)
    """
    if ctx_len <= 0:
        Hkv, D = kc_view.size(2), kc_view.size(3)
        empty = kc_view.new_empty((0, Hkv, D))
        return empty, empty

    blk_sz = kc_view.size(1)
    n_full = ctx_len // blk_sz
    rem = ctx_len % blk_sz

    k_parts, v_parts = [], []
    # full blocks
    if n_full > 0:
        slots = block_table_row[:n_full].tolist()  # safe: tiny list
        for s in slots:
            if s < 0: break
            k_parts.append(kc_view[s, :blk_sz])  # [T, Hkv, D]
            v_parts.append(vc_view[s, :blk_sz])
    # tail
    if rem > 0:
        slot = int(block_table_row[n_full].item())
        if slot >= 0:
            k_parts.append(kc_view[slot, :rem])
            v_parts.append(vc_view[slot, :rem])

    K = torch.cat(k_parts, dim=0) if k_parts else kc_view.new_empty((0, kc_view.size(2), kc_view.size(3)))
    V = torch.cat(v_parts, dim=0) if v_parts else vc_view.new_empty((0, vc_view.size(2), vc_view.size(3)))
    return K, V

def _build_global_block_mask_allow(seqs) -> torch.Tensor:
    """
    把每条序列的 seq.current_block_mask（形状通常为 [Lq_seq, Lk_seq], True=允许）
    拼成一个大块对角遮罩（2D bool），行维拼 Lq，总列维拼 Lk。
    """
    per_masks = []
    rows, cols = 0, 0
    row_offsets, col_offsets = [], []

    total_rows = 0
    total_cols = 0
    for seq in seqs:
        m = seq.current_block_mask
        # 期望 2D: [Lq, Lk]；若是 4D [1,1,Lq,Lk]，就 squeeze 掉
        while m.dim() > 2:
            m = m.squeeze(0)
        assert m.dim() == 2, f"current_block_mask must be 2D, got {m.shape}"
        Lq, Lk = m.size(-2), m.size(-1)
        per_masks.append(m)
        row_offsets.append(total_rows)
        col_offsets.append(total_cols)
        total_rows += Lq
        total_cols += Lk

    # 构建大遮罩（True=允许）
    device = per_masks[0].device if per_masks else torch.device("cuda")
    global_allow = torch.zeros((total_rows, total_cols), dtype=torch.bool, device=device)
    for m, ro, co in zip(per_masks, row_offsets, col_offsets):
        Lq, Lk = m.size(-2), m.size(-1)
        global_allow[ro:ro+Lq, co:co+Lk] = m

    return global_allow

class BlockAttention(nn.Module):
    """
    支持“块间因果、块内双向”的注意力：
      - 每个 diffusion block 内（同一 block 的行范围），对应该 block 的列范围允许双向（允许看“未来”同 block）。
      - 跨 block 仍维持因果（只能看历史块）。
    通过显式的 block_mask 实现，而不是 SDPA 的 causal 开关。
    """
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 由 ModelRunner.allocate_kv_cache 注入
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def _get_kv_cache_views(self, kv_cache_layout: str):
        """
        统一拿到 [num_blocks, block_size, Hkv, D] 的视图。
        """
        assert self.k_cache.numel() == self.v_cache.numel() or self.k_cache.numel() == 0, \
            "k_cache/v_cache inconsistent"
        if self.k_cache.numel() == 0:
            return None, None

        if kv_cache_layout == "unified":
            kc_view, vc_view = _view_kcache_unified(self.k_cache, self.v_cache)
        elif kv_cache_layout == "distinct":
            # 从上下文里取 x（head_dim split factor）
            ctx = get_diffusion_context()
            # 简化做法：从第一条序列的 config 读；若在多 rank 环境也一致
            x = getattr(ctx.seqs[0].config, "k_cache_hdim_split_factor_x", 1)
            kc_view, vc_view = _view_kcache_distinct(self.k_cache, self.v_cache, self.head_dim, x)
        else:
            raise ValueError(f"Unsupported kv_cache_layout: {kv_cache_layout}")
        # 期望 contiguous，token 维度 stride(1) == Hkv*D，方便 triton 写入
        return kc_view.contiguous(), vc_view.contiguous()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q, k, v: [N_this_step, Hq/Hkv, D]  —— N 是本步拼成的一维 token 序列（多条序列拼接）
        返回 o: [N_this_step, Hq, D]
        """
        ctx = get_diffusion_context()
        # 1) 若需要，把本步新算出的 K/V 写回 KV-cache（prefix 或 decode 的 to_cache）
        if (self.k_cache.numel() and self.v_cache.numel()
            and (ctx.need_kv_cache_store or (ctx.is_prefill and ctx.slot_mapping is not None))):
            # 注意：store_kvcache 期望 key/value [N, Hkv, D]，cache 的第二维 stride == Hkv*D
            store_kvcache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)
            print(F"storing k: {k[0, 0, :5]}, v: {v[0, 0, :5]}")

        # 2) 得到 KV-cache 的统一视图（若无 cache 则为 None）
        kc_view, vc_view = self._get_kv_cache_views(ctx.kv_cache_layout)

        # 3) 逐序列拼接 K/V，顺序必须与 block_mask 的列对齐：
        #    K_total_seq = [K_cached_seq, K_current_seq]；最后再把各序列按队列顺序拼起来
        cu_q = ctx.cu_seqlens_q.tolist()  # len = num_seqs + 1
        context_lens = ctx.context_lens.tolist() if ctx.context_lens is not None else [0] * (len(cu_q) - 1)
        block_tables = ctx.block_tables  # [num_seqs, num_mem_blocks]
        seqs = ctx.seqs

        Ks_all, Vs_all, Qs_all = [], [], []
        for si in range(len(cu_q) - 1):
            print(f"=== seq {si} ===")
            q_start, q_end = cu_q[si], cu_q[si+1]
            q_seq = q[q_start:q_end]     # [Lq_s, Hq, D]
            k_cur = k[q_start:q_end]     # [Lq_s, Hkv, D] —— 当前这步新 tokens 的 K
            v_cur = v[q_start:q_end]     # [Lq_s, Hkv, D]

            print(f" q_seq: {q_seq[0, 0, :5]}, k_cur: {k_cur[0, 0, :5]}, v_cur: {v_cur[0, 0, :5]} ")

            # 从 cache 聚合 K/V（可能为 0 长度）
            if kc_view is not None and block_tables is not None:
                K_ctx, V_ctx = _gather_ctx_kv_for_one_seq(
                    kc_view, vc_view,
                    block_tables[si],  # 一行
                    int(context_lens[si])
                )
            else:
                # 没有 cache 的情况
                Hkv = k_cur.size(1)
                D = k_cur.size(2)
                empty = k_cur.new_empty((0, Hkv, D))
                K_ctx, V_ctx = empty, empty

            # [cached_s, current_s]
            Ks_all.append(torch.cat([K_ctx, k_cur], dim=0))  # [Lk_s, Hkv, D]
            Vs_all.append(torch.cat([V_ctx, v_cur], dim=0))  # [Lk_s, Hkv, D]
            Qs_all.append(q_seq)                             # [Lq_s, Hq,  D]

        # 拼成全局一维
        K_all = torch.cat(Ks_all, dim=0)  # [Lk_total, Hkv, D]
        V_all = torch.cat(Vs_all, dim=0)  # [Lk_total, Hkv, D]
        Q_all = torch.cat(Qs_all, dim=0)  # [Lq_total, Hq,  D]

        # 4) 构造/获取 block-wise 允许遮罩（True=允许），并转为 SDPA 的屏蔽遮罩（True=屏蔽）
        #    为了稳妥，我们在前向里按当前 seqs 重建一次 2D 大遮罩（也可以直接用 ctx.block_mask，如果你保证它一定正确）。
        allow_mask_2d = _build_global_block_mask_allow(seqs)  # [Lq_total, Lk_total], True=允许
        assert allow_mask_2d.size(0) == Q_all.size(0), f"mask rows {allow_mask_2d.size(0)} != Lq {Q_all.size(0)}"
        assert allow_mask_2d.size(1) == K_all.size(0), f"mask cols {allow_mask_2d.size(1)} != Lk {K_all.size(0)}"
        attn_mask_bool = ~allow_mask_2d  # SDPA: True=屏蔽

        # 5) 头数对齐（GQA）：把 K/V 的 Hkv 扩展到 Hq
        K_all = _repeat_kv_to_q_heads(K_all, self.num_heads, self.num_kv_heads)  # [Lk_total, Hq, D]
        V_all = _repeat_kv_to_q_heads(V_all, self.num_heads, self.num_kv_heads)  # [Lk_total, Hq, D]

        # 6) 整形成 SDPA 需要的 [B=1, H, L, D]，并做缩放
        Q_bhld = Q_all.transpose(0, 1).unsqueeze(0) * self.scale  # [1, Hq, Lq, D]
        K_bhld = K_all.transpose(0, 1).unsqueeze(0)               # [1, Hq, Lk, D]
        V_bhld = V_all.transpose(0, 1).unsqueeze(0)               # [1, Hq, Lk, D]

        # 7) SDPA（非 causal；由 mask 控制全部可见性）
        O_bhld = F.scaled_dot_product_attention(
            Q_bhld, K_bhld, V_bhld,
            attn_mask=attn_mask_bool,  # [Lq, Lk] 会按 batch/head 广播
            dropout_p=0.0,
            is_causal=False
        )  # [1, Hq, Lq, D]

        # 8) 还原回 [N_this_step, Hq, D]
        O = O_bhld.transpose(1, 2).squeeze(0).contiguous()  # [Lq_total, Hq, D]
        return O