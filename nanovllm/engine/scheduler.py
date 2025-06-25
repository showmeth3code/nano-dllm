from collections import deque
import logging
from typing import List, Tuple

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# Set up module-level logger
logger = logging.getLogger(__name__)


class Scheduler:
    """
    The Scheduler manages sequence execution by allocating KV cache blocks and
    deciding which sequences to run based on available resources.
    
    The scheduler operates in two modes:
    1. Prefill: Processing new sequences with their initial input tokens
    2. Decode: Generating new tokens for already running sequences
    
    It ensures that the total number of tokens being processed stays within
    hardware limits and manages preemption when resources are constrained.
    
    Attributes:
        max_num_seqs (int): Maximum number of sequences that can be processed in parallel
        max_num_batched_tokens (int): Maximum total tokens across all sequences in a batch
        eos (int): End-of-sequence token ID
        block_manager (BlockManager): Manages KV cache block allocation
        waiting (deque[Sequence]): Queue of sequences waiting to be processed
        running (deque[Sequence]): Queue of sequences currently being processed
    """

    def __init__(self, config: Config):
        """
        Initialize the scheduler with configuration parameters.
        
        Args:
            config (Config): Configuration containing scheduler parameters
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """
        Check if all sequences have been processed.
        
        Returns:
            bool: True if there are no waiting or running sequences
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """
        Add a new sequence to the waiting queue.
        
        Args:
            seq: Sequence to be scheduled
        """
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """
        Schedule sequences for processing, either in prefill or decode mode.
        
        This method first attempts to schedule waiting sequences for prefill.
        If no sequences can be prefilled, it schedules running sequences for
        token generation (decode). If resources are constrained, sequences
        may be preempted back to the waiting queue.
        
        Returns:
            tuple: A tuple containing (scheduled_sequences, is_prefill_mode)
            - scheduled_sequences: List of sequences to process
            - is_prefill_mode: True if in prefill mode, False if in decode mode
        """
        # Try prefill mode first (process waiting sequences)
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                logger.debug(f"Batch token limit reached: {num_batched_tokens}/{self.max_num_batched_tokens}")
                break
                
            if not self.block_manager.can_allocate(seq):
                logger.debug(f"Cannot allocate blocks for sequence {seq.seq_id}, need more KV cache blocks")
                break
                
            num_seqs += 1
            try:
                self.block_manager.allocate(seq)
                num_batched_tokens += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)
            except RuntimeError as e:
                logger.warning(f"Failed to allocate blocks for sequence {seq.seq_id}: {str(e)}")
                # Skip this sequence but keep trying others
                self.waiting.popleft()
                self.waiting.append(seq)  # Move to the end of the queue
                
        if scheduled_seqs:
            return scheduled_seqs, True  # Prefill mode

        # Decode mode (process running sequences)
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Ensure we have blocks available for the next token
            retry_count = 0
            max_retries = len(self.running) + 1  # Limit preemption attempts
            
            while not self.block_manager.can_append(seq) and retry_count < max_retries:
                if self.running:
                    # Preempt another sequence to free blocks
                    self.preempt(self.running.pop())
                else:
                    # No other sequences to preempt, so preempt this one
                    self.preempt(seq)
                    seq = None  # Mark that this sequence was preempted
                    break
                retry_count += 1
                
            if seq is None:
                break  # Current sequence was preempted
                
            try:
                self.block_manager.may_append(seq)
                num_seqs += 1
                scheduled_seqs.append(seq)
            except (ValueError, RuntimeError) as e:
                logger.error(f"Error appending to sequence {seq.seq_id}: {str(e)}")
                # If we can't append, preempt this sequence
                self.preempt(seq)
                continue
                
        if not scheduled_seqs:
            # This should not happen in normal operation, but handle it gracefully
            logger.warning("No sequences could be scheduled for processing")
            # Return an empty list for now, the engine will need to handle this case
            return [], False
            
        # Add scheduled sequences back to the running queue
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # Decode mode

    def preempt(self, seq: Sequence) -> None:
        """
        Preempt a sequence by deallocating its blocks and moving it back to the waiting queue.
        
        Args:
            seq: Sequence to preempt
        """
        if seq is None:
            return
            
        seq.status = SequenceStatus.WAITING
        try:
            self.block_manager.deallocate(seq)
        except Exception as e:
            logger.error(f"Error deallocating blocks for sequence {seq.seq_id}: {str(e)}")
            # Continue anyway, as we need to move this sequence back to waiting
            
        self.waiting.appendleft(seq)
        logger.debug(f"Preempted sequence {seq.seq_id}")

    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> List[bool]:
        """
        Process generated tokens for each sequence and update their status.
        
        Appends the new token to each sequence and checks if any sequence has finished
        (reached EOS or max tokens).
        
        Args:
            seqs: List of sequences that were processed
            token_ids: List of token IDs generated for each sequence
            
        Returns:
            List[bool]: List indicating which sequences have finished (True) or 
                       are still running (False)
        """
        results = []
        for seq, token_id in zip(seqs, token_ids):
            # Append the generated token
            seq.append_token(token_id)
            
            # Check if sequence has finished
            is_finished = False
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                try:
                    self.block_manager.deallocate(seq)
                except Exception as e:
                    logger.error(f"Error deallocating blocks for finished sequence {seq.seq_id}: {str(e)}")
                    
                try:
                    self.running.remove(seq)
                except ValueError:
                    logger.warning(f"Sequence {seq.seq_id} not found in running queue")
                    
                is_finished = True
                
            results.append(is_finished)
            
        return results
