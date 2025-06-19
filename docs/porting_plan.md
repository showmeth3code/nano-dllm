Tiny-vLLM Porting Plan
======================

This plan breaks down the project into cycles and epochs. Each epoch corresponds to porting a single Python file or feature to Rust. The approach follows "one file, one epoch" so progress is incremental and easy to review.

Cycle 0: Project Setup and Infrastructure
----------------------------------------
- Epoch 0.1: Repository scaffolding and initial Rust crates.
- Epoch 0.2: PyO3 bridge bootstrap to verify building as a Python module.

Cycle 1: Core Engine MVP (Offline Inference)
-------------------------------------------
- Epoch 1: Port `vllm_config.py` to `src/config.rs`.
- Epoch 2: Port `llm_engine.py` to `src/engine.rs`.
- Epoch 3: Port model loading code (`model_runner.py` and helpers).
- Epoch 4: Port `sampling_params.py` and sampling utilities.
- Epoch 5: Integrate pieces and run an end-to-end parity test against Python.

Cycle 2: Parallelism and Serving Features
----------------------------------------
- Epoch 6: Port `async_llm_engine.py` to Rust.
- Epoch 7: Port `scheduler.py` for request batching.
- Epoch 8: Implement attention cache management (paged attention).
- Epoch 9: Add streaming support for partial outputs.
- Epoch 10: Parity and load testing with multiple requests.

Cycle 3: Feature Parity and Optimization
---------------------------------------
- Epoch 11: Port the OpenAI-compatible HTTP server.
- Epoch 12: Support distributed or quantized models as available.
- Epoch 13: Comprehensive testing and edge cases.
- Epoch 14: Performance tuning and final optimizations.

Each epoch has an associated GitHub issue using the template in `.github/ISSUE_TEMPLATE/porting_task.md`. See `ROADMAP.md` for high level milestones. As epochs finish, update this file to mark progress and note any design deviations.
