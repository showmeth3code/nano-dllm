
import os
import time
from random import randint, seed
import pytest
from nanovllm import LLM, SamplingParams
from typing import Optional

def run_block_manager_fix_test(verbose: bool = True) -> float:
    """
    Run a minimal test to validate block_manager fix for edge-case block allocations.
    Returns the elapsed time in seconds.
    Raises Exception if the test fails.
    """
    seed(0)
    num_seqs: int = 2
    max_input_len: int = 100
    # max_output_len is not used, but kept for clarity
    max_output_len: int = 20

    path: str = os.path.expanduser("Qwen/Qwen3-0.6B")
    if verbose:
        print(f"Loading model from {path}...")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
    if verbose:
        print("Model loaded successfully")

    if verbose:
        print(f"Running validation test with {num_seqs} sequences...")
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(50, max_input_len))]
        for _ in range(num_seqs)
    ]

    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256),
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=257),
    ]

    if verbose:
        print("Starting test run...")
    t = time.time()
    _ = llm.generate(prompt_token_ids, sampling_params, use_tqdm=verbose)
    elapsed = time.time() - t
    if verbose:
        print("✅ Test completed successfully!")
        print("The block_manager fix works - no assertion errors occurred.")
        print(f"Execution time: {elapsed:.2f}s")
    return elapsed


@pytest.mark.heavy
def test_block_manager_fix_pytest():
    """
    Pytest-compatible test for block_manager fix. Marked as heavy.
    """
    try:
        elapsed = run_block_manager_fix_test(verbose=False)
    except Exception as e:
        pytest.fail(f"Block manager fix test failed: {e}")


def main():
    """
    Command-line entry point for block_manager fix test.
    """
    try:
        run_block_manager_fix_test(verbose=True)
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        print("The fix may not be working properly.")


if __name__ == "__main__":
    main()
