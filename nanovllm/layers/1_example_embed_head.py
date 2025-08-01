import torch.multiprocessing as mp
import torch

from nanovllm.layers.embed_head import VocabParallelEmbedding


def setup(rank, world_size):
    import torch.distributed as dist
    dist.init_process_group(
        backend='gloo',  # 或 'nccl'（如果使用 CUDA 卡）
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=rank
    )


def run_tp(rank, world_size):
    setup(rank, world_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_embeddings = 128
    embedding_dim = 16

    model = VocabParallelEmbedding(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)

    # 构造一个全量 embedding matrix：第 i 行是 [i, i, ..., i]
    full_weight = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(
        1).repeat(1, embedding_dim).to(device)

    # 加载这个权重
    model.weight_loader(model.weight, full_weight)

    # 模拟输入
    input_ids = torch.tensor([74, 66, 10, 20, 30, 127],
                             device=device)  # 注意包括两边分片
    output = model(input_ids)

    print(f"[Rank {rank}] output =\n{output}")



if __name__ == '__main__':
    world_size = 2  # 模拟两个 TP 分片
    mp.spawn(run_tp, args=(world_size,), nprocs=world_size, join=True)
