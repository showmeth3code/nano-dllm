from collections import deque
from typing import Tuple, List, Deque
from nanovllm.config import Config
from nanovllm.engine.diffusion.sequence import SequenceForDiffusionLM as Sequence
from nanovllm.engine.ar.sequence import SequenceStatus
from nanovllm.layers.diffu_sampler import SampleOutputForDiffusionLM
from nanovllm.engine.diffusion.block_manager import BlockManagerForDiffusionLM as BlockManager

class SchedulerForDiffusionLM():
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.diffusion_block_size = config.diffusion_block_size

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)
    
    def schedule(self):
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) + seq.diffusion_block_size > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) + seq.diffusion_block_size - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            diag = {
                "phase": "decode",
                "waiting": len(self.waiting),
                "running": len(self.running),
                "max_num_seqs": self.max_num_seqs,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "diffusion_block_size": getattr(self, 'diffusion_block_size', None),
            }
            candidates = list(self.running)[:3] + list(self.waiting)[:2]
            infos = []
            for j, s in enumerate(candidates):
                try:
                    cap = self.block_manager.can_append(s)
                except Exception:
                    cap = "error"
                infos.append(
                    f"[{j}] status={s.status.name}, len={len(s)}, diff_block={getattr(s, 'diffusion_block_size', '?')}, new_tokens={getattr(s, 'new_tokens', '?')}, cached={getattr(s, 'num_cached_tokens', '?')}, can_append={cap}"
                )
            raise RuntimeError(f"SchedulerForDiffusionLM: unable to schedule any sequence in decode; state={diag}; details={' | '.join(infos)}")
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[Sequence], sample_output: SampleOutputForDiffusionLM) -> None:
        n_diff_steps = {}
        for seq in seqs:
            seq.reset_new_tokens()
            seq_id = str(seq.seq_id)
            cur_true_local_ids_sub_map = sample_output.true_local_ids_map.get(seq_id, {})
            cur_accepted_ids_sub_map = sample_output.accepted_ids_map.get(seq_id, {})
            cur_sampled_tokens_sub_map = sample_output.sampled_tokens_map.get(seq_id, {})
            for block_id, accepted_ids in cur_accepted_ids_sub_map.items():
                if len(accepted_ids) > 0:
                    diffusion_block = seq.diffusion_blocks[int(block_id)]
                    sampled_tokens = cur_sampled_tokens_sub_map.get(block_id, [])
                    true_local_ids = cur_true_local_ids_sub_map.get(block_id, [])

                    for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                        diffusion_block.modify_token(true_local_id, sampled_tokens[accepted_id])
                        if ((not seq.ignore_eos and sampled_tokens[accepted_id].item() == self.eos) 
                            or seq.num_completion_tokens >= seq.max_tokens):
                            seq.meet_eos = True
            if seq.meet_eos and seq.diffusion_blocks[-1].available_to_cache:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.free(seq)
                self.running.remove(seq)
                n_diff_steps[seq.seq_id] = seq.n_steps
            seq.post_process()
        return n_diff_steps