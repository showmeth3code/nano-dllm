# Architecture

Tiny-vLLM mirrors the original vLLM layout. Each Python module has a corresponding Rust module so behavior can be compared directly. The table below highlights the mapping.

| Python Source | Rust Module | Notes |
|---------------|------------|-------|
| `nanovllm/config.py` | `src/config.rs` | Engine configuration struct using Serde |
| `nanovllm/engine/llm_engine.py` | `src/engine.rs` | Core engine for token generation |
| `nanovllm/engine/async_llm_engine.py` | `src/engine_async.rs` | Asynchronous wrapper for concurrency |
| `nanovllm/engine/scheduler.py` | `src/scheduler.rs` | Request scheduler for continuous batching |
| `nanovllm/engine/model_runner.py` | `src/model_runner.rs` | Loads models via `tch` and runs forward passes |
| `nanovllm/layers/...` | `src/layers/...` | Individual neural net layers |

Components communicate through a shared `VllmConfig`. PyO3 exposes a Python-facing `LLM` class that instantiates the Rust engine under the hood. The design keeps Rust logic self contained so it can be used as a native crate or via Python bindings.
