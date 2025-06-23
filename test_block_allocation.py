import traceback
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

def test_block_allocation():
    try:
        print("Setting up test...")
        block_size = 256
        num_blocks = 20  # Generous allocation
        
        # Create the block manager
        manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        print(f"Created BlockManager with {num_blocks} blocks of size {block_size}")
        
        # Create a sequence exactly block_size tokens
        tokens = list(range(block_size))
        params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=10)
        seq = Sequence(tokens, params, "test prompt")
        print(f"Created sequence with {len(seq)} tokens (exactly one block)")
        
        # Allocate blocks
        print("Allocating blocks...")
        manager.allocate(seq)
        print(f"Block table after allocation: {seq.block_table}")
        
        # Get the block
        block = manager.blocks[seq.block_table[-1]]
        print(f"Block ID: {block.block_id}, Hash: {block.hash}")
        
        # Now append a token
        print("Appending one token (should trigger new block allocation)...")
        seq.append_token(999)
        print(f"Sequence length after append: {len(seq)}")
        print(f"Sequence length % block_size = {len(seq) % block_size}")
        
        # Try may_append
        print("Calling may_append...")
        manager.may_append(seq)
        print(f"Block table after may_append: {seq.block_table}")
        print(f"Now we have {len(seq.block_table)} blocks allocated")
        
        # Append tokens until we fill the second block
        print("\nAppending more tokens to fill the second block...")
        for i in range(block_size-1):
            seq.append_token(1000 + i)
            
        print(f"Sequence length: {len(seq)}")
        print(f"Sequence length % block_size = {len(seq) % block_size}")
        print(f"Block table size: {len(seq.block_table)}")
        
        # The second block should now be full
        print("Block table:", seq.block_table)
        if len(seq.block_table) >= 2:
            block1 = manager.blocks[seq.block_table[0]]
            block2 = manager.blocks[seq.block_table[1]]
            print(f"Block 1 - ID: {block1.block_id}, Hash: {block1.hash}")
            print(f"Block 2 - ID: {block2.block_id}, Hash: {block2.hash}")
        
        # Try one more append to trigger a third block
        print("\nAppending one more token to trigger third block allocation...")
        seq.append_token(2000)
        print(f"Sequence length: {len(seq)}")
        print(f"Sequence length % block_size = {len(seq) % block_size}")
        
        print("Calling may_append again...")
        manager.may_append(seq)
        print(f"Block table after may_append: {seq.block_table}")
        
        print("\nTest completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_block_allocation()
