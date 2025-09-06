import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        这段代码围绕 “分布式环境下高效加载并初始化大模型” 展开，通过 4 个关键步骤确保性能：
        1、建立分布式通信，让多 GPU 协同工作；
        2、配置 GPU 设备和数据类型，为模型运行做好准备；
        3、加载模型并初始化采样器，搭建推理核心组件；
        4、热身、分配缓存、捕获 CUDA 图，最大化推理速度。
        最终结果是：模型被高效部署在多 GPU 环境中，既能利用分布式提升吞吐量，又通过 KV 缓存和 CUDA 图优化实现低延迟推理。
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 空跑模型进行热身（激活CUDA内核并分配初始资源）
        self.warmup_model()
        # 分配KV-Cache缓存
        self.allocate_kv_cache()
        # 捕获CUDA图（不强制即时执行时启用，进一步加速推理）
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复默认设备为CPU（避免后续非模型代码占用GPU）
        torch.set_default_device("cpu")
        # 恢复原始默认数据类型（避免影响其他代码的精度）
        torch.set_default_dtype(default_dtype)
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        这个方法负责在模型推理前进行预热，通过运行一次完整的推理过程来初始化GPU内存、编译CUDA内核、预热缓存等，确保后续推理的性能稳定。
        1. 清空缓存
        2. 设置最大批处理token数和最大模型长度
        3. 创建num_seqs个Sequence对象
        4. 运行模型
        5. 清空缓存
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        这个方法负责分配KV-Cache缓存
        1. 获取GPU内存信息
        2. 计算KV-Cache缓存大小
        3. 分配KV-Cache缓存
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        """   
        参数计算解析  
        # K和V各占一份
        block_bytes = 2 
            * 32  # 32层注意力
            * 16  # 每个块存16个token
            * 32  # 32个KV头
            * 64  # 每个头64维
            * 2  # float16占2字节
        = 2×32×16×32×64×2 = 4,194,304 字节 = 4MB（每个块大小）
        """
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        """
        参数计算解析
        可用内存 = 总内存×利用率 - 已用内存 - 峰值冗余（peak - current）
        available = 16GB×0.9 - 6GB - (7GB - 6GB) 
        = 14.4GB - 6GB - 1GB = 7.4GB = 7.4×1024³字节 ≈ 7.948×10⁹字节

        # 块数 = 可用内存 // 每个块大小
        config.num_kvcache_blocks = 7.948×10⁹ // 4.194×10⁶ ≈ 1895 个块
        """
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
                
        assert config.num_kvcache_blocks > 0
        """
        计算结果：
        7.948×10⁹ // 4.194×10⁶ ≈ 1895 个块
        每个块大小 = 4.194×10⁶ 字节 = 4MB
        2 * num_hidden_layers * (1895块 * 4MB)
        
        生成的张量形状为 (2, 32, 1895, 16, 32, 64)，是一块连续的 GPU 内存（约 7.58GB），包含了所有层、所有 KV 头的缓存空间。
        可以理解为一个 “多层货架”：
                第 0 层（维度 0）放 Key，第 1 层放 Value；
                每个货架有 32 个 “层隔间”（维度 1），对应 32 层注意力；
                每个隔间里有 1895 个 “存储盒”（维度 2，缓存块）；
                每个盒子里有 16 个 “槽位”（维度 3，token 位置）；
                每个槽位有 32 个 “小格子”（维度 4，KV 头）；
                每个小格子里有 64 个 “数据格”（维度 5，特征维度）。
        """
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        prepare_block_tables 是KV 缓存块管理的 “格式统一器”，通过三步操作（找最大长度→填充统一→转 GPU 张量），确保不同长度的序列块表能被 GPU 
        高效处理。这是大模型实现高并发、低延迟推理的关键细节之一，直接影响批量处理时的性能。
        具体流程为：
        1. 获取最大block table长度
        2. 将block table填充到最大长度
        3. 返回block table
        原始：
        seq1:[1, 2, 5]
        seq2:[1, 2, 3, 4]
        
        填充完成：
        seq1:[1, 2, 5, -1]
        seq2:[1, 2, 3, 4]
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 不存在block_table, 不处理，首次填充，没有KV-Cache缓存进行存储
            if not seq.block_table:
                continue
            
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        # input_ids：告诉模型 “基于哪个 token 继续生成”；
        # positions：提供位置编码，让模型知道 “下一个 token 在序列中的位置”；
        # context_lens：限制注意力范围，避免模型 “看到” 还未生成的 token；
        # slot_mapping：指示新生成 token 的 KV 数据应该 “存在缓存的哪个位置”，确保后续生成能复用这些数据。
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            # 定位最后序列存储最后一块KV-Cache的位置索引
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        1. 准备参数
        2. 运行模型
        3. 采样
        4. 重置上下文
        5. 返回token_ids
        """
        # 运行阶段可以分为两个阶段：prefill和decode
        # prefill阶段：处理新的prompt序列
        # decode阶段：逐个生成token
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
