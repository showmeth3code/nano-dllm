import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional


class VocabParallelEmbedding(nn.Module):
    """Parallel embedding layer that supports sharded vocab"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Create the embedding table
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        
        # Initialize with normal distribution
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input IDs to embeddings
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            
        Returns:
            output: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        return F.embedding(
            input_ids, 
            self.weight,
            self.padding_idx,
            scale_grad_by_freq=False,  # Scale grad by freq not supported
            sparse=False  # Sparse not supported
        )


class ParallelLMHead(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.hidden_size = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # Remove the extra layer norm - Qwen3 doesn't use this

    def load_weight(self, loaded_weight: torch.Tensor):
        param_data = self.weight.data
        # Handle transpose if needed
        if loaded_weight.shape != param_data.shape:
            if loaded_weight.shape == (param_data.shape[1], param_data.shape[0]):
                loaded_weight = loaded_weight.transpose(0, 1)
            else:
                raise ValueError(f"Incompatible weight shapes: param {param_data.shape}, loaded {loaded_weight.shape}")
        
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove normalization - this should be done before the LM head in the model pipeline
        
        # Compute logits directly without extra scaling
        logits = F.linear(x, self.weight)
        
        # Gather from all workers if using tensor parallelism
        if self.tp_size > 1:
            world_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            dist.all_gather(world_logits, logits)
            logits = torch.cat(world_logits, dim=-1)
        
        # Return logits without additional scaling
        return logits
