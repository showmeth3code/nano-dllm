use pyo3::prelude::*;
use tiny_vllm_core::cuda_utils;
use tiny_vllm_core::helpers;

fn to_py_err(err: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

#[pyfunction]
fn get_device() -> PyResult<String> {
    cuda_utils::get_device().map_err(to_py_err)
}

#[pyfunction]
fn get_gpu_memory() -> PyResult<u64> {
    cuda_utils::get_gpu_memory().map_err(to_py_err)
}

#[pyfunction]
fn get_gpu_memory_utilization() -> PyResult<f32> {
    cuda_utils::get_gpu_memory_utilization().map_err(to_py_err)
}

#[pymodule]
fn tiny_vllm_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_memory, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_memory_utilization, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    m.add_function(wrap_pyfunction!(chunked, m)?)?;
    Ok(())
}

#[pyfunction]
fn clamp(value: i64, min_value: i64, max_value: i64) -> PyResult<i64> {
    Ok(helpers::clamp(value, min_value, max_value))
}

#[pyfunction]
fn flatten(list_of_lists: Vec<Vec<i64>>) -> PyResult<Vec<i64>> {
    Ok(helpers::flatten(list_of_lists))
}

#[pyfunction]
fn chunked(lst: Vec<i64>, size: usize) -> PyResult<Vec<Vec<i64>>> {
    Ok(helpers::chunked(lst, size))
}

