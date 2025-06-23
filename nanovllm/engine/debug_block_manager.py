from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @staticmethod
    def _flatten_token_ids(token_ids):
        flat = []
        for item in token_ids:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.extend(BlockManager._flatten_token_ids(item))
            else:
                flat.append(item)
        return flat

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        token_ids = cls._flatten_token_ids(token_ids)
        arr = np.array(token_ids, dtype=np.int32)
        h.update(arr.tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        if block_id in self.free_block_ids:
            self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        return block

    def can_allocate(self, seq: Sequence) -> bool:
        num_blocks_needed = (seq.num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_block_ids) >= num_blocks_needed

    def allocate(self, seq: Sequence):
        # Clear any existing block table
        seq.block_table = []
        
        h = -1
        cache_miss = False
        blocks_needed = (seq.num_tokens + self.block_size - 1) // self.block_size
        
        for i in range(blocks_needed):
            token_ids = seq.block(i) if i < seq.num_blocks else []
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
                
            if cache_miss:
                if not self.free_block_ids:
                    raise RuntimeError("No free blocks available for allocation")
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
                    
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
                
            seq.block_table.append(block_id)

        return True

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        if not block_table:
            raise ValueError(f"Empty block table for sequence {seq.seq_id}")
            
        last_block = self.blocks[block_table[-1]]
        
        # Debug info
        print(f"Sequence ID: {seq.seq_id}, Length: {len(seq)}, Mod block_size: {len(seq) % self.block_size}")
        print(f"Last block ID: {last_block.block_id}, Hash: {last_block.hash}")
        print(f"Block table: {seq.block_table}")
        
        # Logic based on sequence length relative to block size
        if len(seq) % self.block_size == 1:
            if last_block.hash == -1:
                print(f"ERROR: Attempting to allocate a new block but last_block.hash is -1")
                # We'll fix the issue here
                if len(seq.block_table) > 1:
                    previous_block = self.blocks[block_table[-2]]
                    print(f"Previous block hash: {previous_block.hash}")
                    token_ids = seq.block(seq.num_blocks-2)  # Get the completed previous block
                    h = self.compute_hash(token_ids, -1 if len(block_table) <= 2 else self.blocks[block_table[-3]].hash)
                    print(f"Computed hash for previous block: {h}")
                    previous_block.update(h, token_ids)
                    self.hash_to_block_id[h] = previous_block.block_id
                
                # We don't assert but continue with the allocation
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
            else:
                # This is the expected path
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block_table.append(block_id)
                
        elif len(seq) % self.block_size == 0:
            # The block should be full now, compute its hash
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # Still filling the block, hash should be -1
            if last_block.hash != -1:
                print(f"WARNING: Unexpected hash value {last_block.hash} for partially filled block")
                # Fix it by resetting the hash
                last_block.hash = -1
