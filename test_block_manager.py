import os
import sys
from nanovllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.engine.debug_block_manager import BlockManager

def main():
    try:
        # Simple test case with a few token IDs
        block_size = 256
        num_blocks = 10
        
        print("Creating BlockManager...")
        manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        
        # Create a basic test sequence
        from nanovllm.engine.sequence import Sequence
        
        # Test with a sequence that has exactly block_size tokens
        tokens = list(range(block_size))
        sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=10)
        seq = Sequence(tokens, sampling_params, "test")
        
        print(f"Allocating blocks for sequence of {len(seq)} tokens...")
        manager.allocate(seq)
        
        print("Block table after allocation:", seq.block_table)
        
        # Now try to append a token and see what happens
        print("Appending a token...")
        seq.append_token(999)
        
        print(f"Sequence length after append: {len(seq)}")
        print(f"Mod block_size: {len(seq) % block_size}")
        
        print("Checking if can append...")
        can_append = manager.can_append(seq)
        print(f"Can append: {can_append}")
        
        print("Trying may_append...")
        manager.may_append(seq)
        
        print("Block table after may_append:", seq.block_table)
        
        # Clean up
        print("Deallocating...")
        manager.deallocate(seq)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
