import torch

from typing import List
from dataclasses import dataclass

from nanovllm.engine.diffu_sequence import SequenceForDiffusionLM

@dataclass
class ContextBase:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

# Global context for causal language model
@dataclass
class ContextForCausalLM(ContextBase):
    kv_cache_layout: str = "unified"  # Only "unified" is supported for Causal LM currently
    need_kv_cache_store: bool = True

_CONTEXT_FOR_CAUSAL_LM = ContextForCausalLM()

def get_context_causal_lm() -> ContextForCausalLM:
    return _CONTEXT_FOR_CAUSAL_LM

def set_context_causal_lm(
    is_prefill,
    cu_seqlens_q=None, cu_seqlens_k=None,
    max_seqlen_q=0, max_seqlen_k=0,
    slot_mapping=None, context_lens=None, block_tables=None
) -> None:
    global _CONTEXT_FOR_CAUSAL_LM
    _CONTEXT_FOR_CAUSAL_LM = ContextForCausalLM(
        is_prefill, 
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, 
        slot_mapping, context_lens, block_tables
    )

def reset_context_causal_lm() -> None:
    global _CONTEXT_FOR_CAUSAL_LM
    _CONTEXT_FOR_CAUSAL_LM = ContextForCausalLM()


# Global context for diffusion language model
@dataclass
class ContextForDiffusionLM(ContextBase):
    seqs: List[SequenceForDiffusionLM] = None
    seq_lens: List[int] = None
    seq_lens_ts: torch.Tensor | None = None
    kv_cache_layout: str = "unified"  # "unified" or "distinct"
    need_kv_cache_store: bool = True
    block_mask: List[torch.Tensor] | None = None
    
    def __post_init__(self):
        if self.seq_lens_ts is not None and self.context_lens is not None:
            self.total_lens = self.seq_lens_ts + self.context_lens
        if self.seqs is not None and len(self.seqs) > 0:
            if self.is_prefill:
                masks = [seq.current_block_mask for seq in self.seqs]
                total_len = sum(mask.size(-1) for mask in masks)
                self.block_mask = torch.zeros(total_len, total_len, dtype=torch.bool)
                
                start_idx = 0
                for mask in masks:
                    seq_len = mask.size(-1)
                    end_idx = start_idx + seq_len
                    self.block_mask[start_idx:end_idx, start_idx:end_idx] = mask.clone()
                    start_idx = end_idx
                self.block_mask = self.block_mask.to(mask.device)
            else:
                masks = [seq.current_block_mask for seq in self.seqs]
                total_height = sum(mask.size(-2) for mask in masks)
                total_width = sum(mask.size(-1) for mask in masks)
                self.block_mask = torch.zeros(total_height, total_width, dtype=torch.bool)
                start_row = 0
                start_col = 0
                for mask in masks:
                    height, width = mask.size(-2), mask.size(-1)
                    end_row = start_row + height
                    end_col = start_col + width
                    self.block_mask[start_row:end_row, start_col:end_col] = mask.clone()
                    start_row, start_col = end_row, end_col
                self.block_mask = self.block_mask.to(mask.device)
    
    @property
    def block_mask_for_checking(self) -> torch.Tensor:
        for seq in self.seqs:
            seq.set_layout("unified")
            
        masks = [seq.current_block_mask for seq in self.seqs]
        total_height = sum(mask.size(-2) for mask in masks)
        total_width = sum(mask.size(-1) for mask in masks)
        block_mask = torch.zeros(total_height, total_width, dtype=torch.bool)
        start_row = 0
        start_col = 0
        for mask in masks:
            height, width = mask.size(-2), mask.size(-1)
            end_row = start_row + height
            end_col = start_col + width
            block_mask[start_row:end_row, start_col:end_col] = mask.clone()
            start_row, start_col = end_row, start_col
            
        for seq in self.seqs:
            seq.set_layout("distinct")
        return block_mask.to(mask.device)

    @property
    def total_num_seqs(self) -> int:
        return len(self.seqs) if self.seqs is not None else 0

_CONTEXT_FOR_DIFFUSION_LM = ContextForDiffusionLM()

def get_context() -> ContextForDiffusionLM:
    return _CONTEXT_FOR_DIFFUSION_LM

def set_context(
    is_prefill,
    cu_seqlens_q=None, cu_seqlens_k=None,
    max_seqlen_q=0, max_seqlen_k=0,
    slot_mapping=None, context_lens=None, block_tables=None,
    seqs= None, seq_lens=None, seq_lens_ts=None, kv_cache_layout="unified", need_kv_cache_store=True
) -> None:
    global _CONTEXT_FOR_DIFFUSION_LM
    _CONTEXT_FOR_DIFFUSION_LM = ContextForDiffusionLM(
        is_prefill,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        slot_mapping, context_lens, block_tables,
        seqs, seq_lens, seq_lens_ts, kv_cache_layout, need_kv_cache_store
    )

def reset_context() -> None:
    global _CONTEXT_FOR_DIFFUSION_LM
    _CONTEXT_FOR_DIFFUSION_LM = ContextForDiffusionLM()