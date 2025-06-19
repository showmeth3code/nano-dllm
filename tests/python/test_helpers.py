import importlib.util
import pathlib
import tiny_vllm_py

helpers_path = pathlib.Path('nanovllm/utils/helpers.py')
spec = importlib.util.spec_from_file_location('py_helpers', helpers_path)
py_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_helpers)


def test_clamp():
    for v in [-5, 0, 5, 10, 15]:
        assert tiny_vllm_py.clamp(v, 0, 10) == py_helpers.clamp(v, 0, 10)


def test_flatten():
    data = [[1, 2], [3, 4, 5], []]
    assert tiny_vllm_py.flatten(data) == py_helpers.flatten(data)


def test_chunked():
    data = list(range(7))
    assert tiny_vllm_py.chunked(data, 3) == py_helpers.chunked(data, 3)
    assert tiny_vllm_py.chunked(data, 0) == py_helpers.chunked(data, 0)

