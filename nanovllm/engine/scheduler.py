from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        1. 从scheduler的等待队列中获取等待的序列
        2. 将序列添加到模型运行器中
        3. 从模型运行器中获取输出
        4. 将输出添加到scheduler的运行队列中
        5. 返回输出
        """
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 1. 从scheduler的等待队列中获取等待的序列
        # 优先等待队列中数据
        while self.waiting and num_seqs < self.max_num_seqs:
            # 获取等待队列中的第一个序列
            seq = self.waiting[0]
            # 2. 将序列添加到模型运行器中 
            # 检查是否有足够的KV缓存内存块来存储序列，如果不够，则跳出循环
            # 检查添加新序列后是否会超过最大批处理token数
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            # 3. 从模型运行器中获取输出
            num_seqs += 1
            # 4. 将输出添加到scheduler的运行队列中
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 5. 将序列状态设置为运行中
            seq.status = SequenceStatus.RUNNING
            # 6. 将序列从等待队列中移除，添加到运行队列中
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            
        # 如果等待队列中没有序列，则返回False
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
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
