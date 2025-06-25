import os
import shutil
import subprocess
from typing import List, Tuple

# List of test and debug files to check and move
test_and_debug_files: List[str] = [
    'test_block_allocation.py',
    'test_block_manager_fix.py',
    'test_block_manager.py',
    'test_cache_allocation.py',
    'test_causal_mask_fix.py',
    'test_causal_mask_multi.py',
    'test_engine_debug.py',
    'test_fixed_model.py',
    'test_generation_after_fix.py',
    'test_generation.py',
    'test_hf_direct.py',
    'test_hf_manual.py',
    'test_hf_mini.py',
    'test_just_model.py',
    'test_kv_cache.py',
    'test_model_outputs.py',
    'test_nano_vllm.py',
    'test_official_hf.py',
    'test.py',
    # 'bench.py',
    'compare_model_outputs.py',
    'compare_outputs.py',
    'compare_with_llamacpp.py',
    'debug_attention_comparison.py',
    'debug_attention_detailed.py',
    'debug_attention_mask.py',
    'debug_attention.py',
    'debug_bench.py',
    'debug_causal_mask.py',
    'debug_forward_pass.py',
    'debug_heads.py',
    'debug_model_consistency.py',
    'debug_model_weights.py',
    'debug_rotary.py',
    'debug_tokenizer.py',
    'debug_vocab_size.py',
    'download_hf_model.py',
    # 'example.py',
    'inspect_output_structure.py',
    'medium_bench.py',
    'mini_bench.py',
    'mps_bench.py',
    'mps_optimized_test.py',
    'quick_bench.py',
]

SCRIPTS_DIR = 'scripts'
RESULTS_DIR = 'test_results'

os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_python_file(filename: str) -> Tuple[bool, str]:
    """
    Run a Python file and capture its output.
    Returns (success, output_path).
    """
    output_path = os.path.join(RESULTS_DIR, f'{filename}.out')
    if os.path.exists(output_path):
        # Already ran, skip rerun
        with open(output_path, 'r') as f:
            first_line = f.readline()
        return (first_line.startswith('SUCCESS'), output_path)
    try:
        # Use tee to write to both screen and file
        proc = subprocess.run(
            f"python3 {filename} 2>&1 | tee {output_path}",
            shell=True,
            executable='/bin/zsh',
            timeout=120
        )
        # Write result marker to file
        with open(output_path, 'a') as out:
            if proc.returncode == 0:
                out.write('\nSUCCESS')
                return (True, output_path)
            else:
                out.write(f'\nFAIL (exit code {proc.returncode})')
                return (False, output_path)
    except Exception as e:
        with open(output_path, 'a') as out:
            out.write(f'\nFAIL (exception: {e})')
        return (False, output_path)

def main() -> None:
    """
    Run and move test/debug scripts if they work.
    """
    for fname in test_and_debug_files:
        if not os.path.exists(fname):
            print(f"[SKIP] {fname} does not exist.")
            continue
        print(f"[CHECK] Running {fname} ...", end=' ')
        success, out_path = run_python_file(fname)
        if success:
            print("OK, moving.")
            shutil.move(fname, os.path.join(SCRIPTS_DIR, fname))
        else:
            print(f"FAIL, see {out_path}")

if __name__ == '__main__':
    main()
