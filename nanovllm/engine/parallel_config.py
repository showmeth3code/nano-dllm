from dataclasses import dataclass


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    world_size: int = 1

    def __post_init__(self):
        self.world_size = self.tensor_parallel_size * self.pipeline_parallel_size 