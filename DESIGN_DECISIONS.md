# Design Decisions

This document captures the rationale for key choices in the Rust port.

## tch-rs for Tensor Operations
We rely on the `tch` crate which binds to PyTorch's `libtorch`. This lets us reuse the same GPU kernels as vLLM and load existing PyTorch checkpoints without conversion. Alternatives like burn or candle were considered but lack some optimized ops we need today.

## PyO3 for Python Interop
Tiny-vLLM intends to be a drop-in replacement for the Python API. Using PyO3 allows the Rust engine to be imported as a Python module. It also enables parity tests against the original implementation.

## One File, One Epoch
Porting is organized so that each epoch corresponds to a single Python file. This keeps pull requests focused and aids verification. Stub modules may be committed first so later epochs can compile against them.

## Static Compilation
The Rust crates build as `cdylib` for Python but also support static binaries. We set `LIBTORCH_USE_PYTORCH=1` when building so `tch` links against the same libtorch as Python's `torch` package, avoiding version mismatches.
