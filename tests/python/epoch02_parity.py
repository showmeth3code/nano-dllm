import tiny_vllm_py

def test_get_device():
    assert tiny_vllm_py.get_device() == "cpu"


def test_memory_stubs():
    assert tiny_vllm_py.get_gpu_memory() == 0
    assert tiny_vllm_py.get_gpu_memory_utilization() == 0.0
