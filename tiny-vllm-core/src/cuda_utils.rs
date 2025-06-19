use anyhow::Result;

/// Get the current device.
///
/// CPU-only stub always returning "cpu".
pub fn get_device() -> Result<String> {
    Ok("cpu".to_string())
}

/// Return total GPU memory in bytes.
///
/// CPU-only stub returning 0.
pub fn get_gpu_memory() -> Result<u64> {
    // TODO: Enable when CUDA is supported
    Ok(0)
}

/// Return GPU memory utilization as a fraction (0.0-1.0).
///
/// CPU-only stub returning 0.0.
pub fn get_gpu_memory_utilization() -> Result<f32> {
    // TODO: Enable when CUDA is supported
    Ok(0.0)
}
