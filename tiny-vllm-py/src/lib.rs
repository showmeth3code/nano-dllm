use pyo3::prelude::*;
use tiny_vllm_core::cuda_utils;

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
fn tiny_vllm_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_memory, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_memory_utilization, m)?)?;
    Ok(())
}
