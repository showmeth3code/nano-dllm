import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.diffusion.sequence import SequenceForDiffusionLM as Sequence
from nanovllm.models.dream import DreamForDiffusionLM
from nanovllm.layers.diffu_sampler import SamplerForDream
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.diffusion_context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
import time

class ModelRunnerForDiffusionLM():
    """Model runner for Diffusion Language Models. TODO: Implement DLM-specific logic."""
    def __init__(self, config, rank, event):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", f"tcp://localhost:{config.port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = DreamForDiffusionLM(hf_config)
        # self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = SamplerForDream() if self.config.is_dllm else Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
        
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

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
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

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def warmup_model(self):
        # return
        print("Warming up model...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [Sequence(test_input_ids, config=self.config) for _ in range(num_seqs)]
        self.run(seqs, True)
        for seq in seqs:
            seq.post_process()
        torch.cuda.empty_cache()

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")
        
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize)
        get_num_kvcache_blocks = lambda gpu_memory_utilization: int(total * gpu_memory_utilization - 
                                                                    used - peak + current) // block_bytes
        try:
            num_kvcache_blocks = get_num_kvcache_blocks(config.gpu_memory_utilization)
            assert num_kvcache_blocks > 0
        except:
            gpu_memory_utilization = config.gpu_memory_utilization
            while num_kvcache_blocks <= 200: 
                print(f"Warning: GPU memory utilization {gpu_memory_utilization} is too low to allocate kv cache. "
                    f"Automatically adding 0.05, which is {gpu_memory_utilization + 0.05:.2f} now.")
                gpu_memory_utilization += 0.05
                num_kvcache_blocks = get_num_kvcache_blocks(gpu_memory_utilization)
            print(f"Set gpu_memory_utilization to {gpu_memory_utilization:.2f} to allocate kv cache.")
            config.gpu_memory_utilization = gpu_memory_utilization
            
        config.num_kvcache_blocks = num_kvcache_blocks
        print(f"Allocated {config.num_kvcache_blocks} blocks of size {self.block_size} for kv cache on rank {self.rank}.")

        if config.kv_cache_layout == "distinct":
            # k_cache: [layer_id, block_id, head, head_dim // x, block_size(segmented seq_len), x]
            # v_cache: [layer_id, block_id, head, head_dim, block_size(segmented seq_len)]
            x = config.k_cache_hdim_split_factor_x
            
            self.k_cache = torch.zeros(
                hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                num_kv_heads, head_dim // x, self.block_size, x
            )
            self.v_cache = torch.zeros(
                hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                num_kv_heads, head_dim, self.block_size
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.k_cache[layer_id]
                    module.v_cache = self.v_cache[layer_id]
                    layer_id += 1
        elif config.kv_cache_layout == "unified":
            # [kv_separated, layer_id, block_id, block_size(segmented seq_len), head, head_dim]
            self.kv_cache = torch.zeros(
                2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                self.block_size, num_kv_heads, head_dim)
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
        else:
            raise ValueError(f"Unsupported kv_cache_layout: {config.kv_cache_layout}. "
                             f"Supported values are 'distinct' and 'unified'.")
        
    def prepare_prefill(self, seqs):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        context_lens = []
        seq_lens = []

        for seq in seqs:
            seq.next_diffusion_step(is_prefill=True)

            total_seqlen = len(seq)
            # tokens and positions to run in this prefill step
            input_ids.extend(seq[seq.cached_num_tokens:])
            positions.extend(list(range(seq.cached_num_tokens, total_seqlen)))
            seq_lens.append(total_seqlen)
            context_lens.append(0)
            assert len(input_ids) == len(positions), (
                f"prepare_prefill(diffusion): len(input_ids) {len(input_ids)} != len(positions) {len(positions)}"
            )
            
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            # build slot mapping for prefix cache prompt blocks
            for i in range(0, seq.num_prompt_blocks):
                if seq.block_cache_missed[i]:
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_prompt_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_prompt_num_tokens
                    slot_mapping.extend(list(range(start, end)))
                else:
                    slot_mapping.extend([-1] * self.block_size)
            # pad to a full diffusion block
            slot_mapping.extend([-1] * seq.diffusion_block_size)

        # For diffusion prefill we always need block tables for prefix cache bookkeeping
        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # More checks to avoid downstream rotary errors
        assert cu_seqlens_q[-1].item() == input_ids.numel(), (
            f"prepare_prefill(diffusion): cu_seqlens_q[-1]={cu_seqlens_q[-1].item()} != num_tokens={input_ids.numel()}"
        )
        assert cu_seqlens_k[-1].item() == sum(seq_lens), (
            f"prepare_prefill(diffusion): cu_seqlens_k[-1]={cu_seqlens_k[-1].item()} != sum(seq_lens)={sum(seq_lens)}"
        )

        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            seqs=seqs,
            kv_cache_layout=self.config.kv_cache_layout,
            seq_lens=seq_lens,
            seq_lens_ts=seq_lens_ts,
        )
        return input_ids, positions

    def prepare_decode(self, seqs):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        context_lens = []
        seq_lens = []
        seq_id_to_queue_id = {}
        need_kv_cache_store = False
        # if sum((sum(seq.active_blocks) + sum(seq.to_cache_blocks)) * seq.diffusion_block_size for seq in seqs) == 1536:
        #     pass
        for seq_idx_in_queue, seq in enumerate(seqs): 
            seq_id = seq.seq_id
            seq_id_to_queue_id[seq_id] = seq_idx_in_queue
            seq.next_diffusion_step()
            cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()
            
            seq_lens.append(len(cur_input_ids))
            input_ids.extend(cur_input_ids)
            positions.extend(cur_positions)
            context_lens.append(cur_context_len)
            
            total_seqlen = len(seq)
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            mem_block_to_diffusion_blocks_map = seq.mem_block_to_diffusion_blocks_map
            context_len = context_lens[seq_id_to_queue_id[seq_id]]
            for mem_block_idx in range(0, seq.num_blocks):
                start_idx = mem_block_idx * seq.block_size
                end_idx = start_idx + seq.block_size
                cur_map = mem_block_to_diffusion_blocks_map[mem_block_idx]
                is_last_block = False
                meet_active_block = False
                while start_idx < end_idx and not is_last_block and not meet_active_block:
                    local_start_idx = lambda: start_idx % seq.block_size
                    diffusion_block = seq.diffusion_blocks[cur_map[local_start_idx()]]
                    if diffusion_block.block_id == 0 and diffusion_block.cursor != start_idx:
                        diffusion_block.cursor = start_idx
                    if cur_map[local_start_idx()] == seq.num_diffusion_blocks - 1:
                        is_last_block = True
                    get_step = lambda diff_blk, start_idx: (
                        diff_blk.remaining_length(start_idx)
                        if diff_blk.remaining_length(start_idx) + local_start_idx() <= seq.block_size
                        else seq.block_size - local_start_idx()
                    )
                    if diffusion_block.is_in_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        start_idx += step
                    elif diffusion_block.is_to_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        cur_diffusion_block_start = 0
                        cur_diffusion_block_end = step
                        start_idx += step
                        mem_block_start = seq.block_table[mem_block_idx] * self.block_size + context_len % seq.block_size
                        context_len += step
                        slot_mapping.extend(list(range(mem_block_start + cur_diffusion_block_start,
                                                       mem_block_start + cur_diffusion_block_end)))
                        need_kv_cache_store = True
                    elif diffusion_block.is_active:
                        meet_active_block = True
                        
                if meet_active_block:
                    # Covering all the after-active blocks
                    active = seq.active_blocks
                    first_active_idx = next((i for i, v in enumerate(active) if v), None)
                    if first_active_idx is not None:
                        num_blocks_to_pad = len(active) - first_active_idx
                        padding_slots = [-1] * (num_blocks_to_pad * seq.diffusion_block_size)
                        slot_mapping.extend(padding_slots)
                    break
            assert len(input_ids) == len(positions), f"Input IDs length {len(input_ids)} does not match positions length {len(positions)}"
            assert len(input_ids) == len(slot_mapping), f"Input IDs length {len(input_ids)} does not match slot mapping length {len(slot_mapping)}"

        # CHECK_SLOT_MAPPING(seqs, slot_mapping)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, 
                                 cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                 block_tables=block_tables, seqs=seqs, 
                                 seq_lens=seq_lens, seq_lens_ts=seq_lens_ts, 
                                 kv_cache_layout=self.config.kv_cache_layout, need_kv_cache_store=need_kv_cache_store)
        return input_ids, positions

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

    def run_verbose(self, seqs, is_prefill: bool):
        print("= =" * 20)
        print(f"Running {'prefill' if is_prefill else 'decode'} for {len(seqs)} sequences on rank {self.rank}")
        s = time.time()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        print(f"Prepared input in {time.time() - s:.2f} seconds")
        s = time.time()
        logits = self.run_model(input_ids, positions, is_prefill)
        print(f"Ran model in {time.time() - s:.2f} seconds")
        s = time.time()
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        print(f"Sampled tokens in {time.time() - s:.2f} seconds")
        reset_context()
        return sample_output

    def run(self, seqs, is_prefill: bool):
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        reset_context()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph(self):
        '''
            TODO: Varlen decoding does not support CUDA graph capture yet.
            Can be implemented, but requires drastically high overhead.
        '''
        raise NotImplementedError("CUDA graph capture for DiffusionLM is not implemented yet.")