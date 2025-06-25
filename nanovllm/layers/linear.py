import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int | None = None):
        """Default weight loading behavior"""
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        output_size = sum(output_sizes)
        super().__init__(input_size, output_size, tp_dim=0)
        self.output_sizes = output_sizes
        self.output_splits = [divide(x, self.tp_size) for x in output_sizes]
        self.weight = nn.Parameter(
            torch.empty(divide(self.output_size, self.tp_size), self.input_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(divide(self.output_size, self.tp_size))
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int | None = None):
        if self.tp_size > 1:
            assert shard_id is not None
            # Handle sharding for tensor parallelism
            assert loaded_weight.dim() == 2
            out_size, in_size = loaded_weight.size()
            shard_size = out_size // self.tp_size
            loaded_weight = loaded_weight.narrow(0, shard_id * shard_size, shard_size)
        
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(MergedColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, [output_size], bias)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, tp_dim=1)
        self.input_size_per_partition = divide(self.input_size, self.tp_size)
        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int | None = None):
        if self.tp_size > 1:
            assert shard_id is not None
            # Handle sharding for tensor parallelism
            assert loaded_weight.dim() == 2
            out_size, in_size = loaded_weight.size()
            shard_size = in_size // self.tp_size
            loaded_weight = loaded_weight.narrow(1, shard_id * shard_size, shard_size)
            
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local output
        output = F.linear(x, self.weight, None)  # Bias is applied after all-gather
        
        if self.tp_size > 1:
            # All-reduce across tensor parallel group
            dist.all_reduce(output)
            
        if self.bias is not None:
            output = output + self.bias
            
        return output


class QKVLinear(nn.Module):
    """Linear layer with QKV projections and bias"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads 
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        
        # Q projection
        self.q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        
        # K projection  
        self.k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        
        # V projection
        self.v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q(hidden_states)
        k = self.k(hidden_states)  
        v = self.v(hidden_states)
        return q, k, v


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer for tensor parallelism"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer for tensor parallelism"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        
        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
