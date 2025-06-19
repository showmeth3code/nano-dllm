# Benchmarks

Benchmarking ensures the Rust engine matches or surpasses the Python version. We use [Criterion.rs](https://github.com/bheisler/criterion.rs) for micro benchmarks and compare against the original `bench.py` script.

## Baseline
Early tests on the Python implementation (RTX 4070, Qwen3-0.6B) show around 1314 tokens/s throughput for 256 requests. This serves as our reference.

## Methodology
- Run `cargo bench` for Rust micro benchmarks.
- Use the same model weights and sampling parameters when comparing against Python vLLM.
- Measure aggregate tokens/second and per-request latency for varying numbers of concurrent prompts.

Results will be added here as the port progresses.
