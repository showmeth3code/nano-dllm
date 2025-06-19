import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import tiny_vllm.config.settings as settings


def test_default_constants():
    assert settings.DEFAULT_MAX_NUM_BATCHED_TOKENS == 32768
    assert settings.DEFAULT_MAX_NUM_SEQS == 512
    assert settings.DEFAULT_MAX_MODEL_LEN == 4096
    assert abs(settings.DEFAULT_GPU_MEMORY_UTILIZATION - 0.9) < 1e-6
    assert settings.DEFAULT_TENSOR_PARALLEL_SIZE == 1
    assert settings.DEFAULT_ENFORCE_EAGER is False
    assert settings.DEFAULT_KVCACHE_BLOCK_SIZE == 256
    assert settings.DEFAULT_NUM_KVCACHE_BLOCKS == -1
    assert settings.DEFAULT_EOS == -1
