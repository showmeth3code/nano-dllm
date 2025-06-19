# Contributing

Thank you for helping with the Rust rewrite of Tiny-vLLM! Development follows an epoch based workflow where each Python file is ported in its own pull request.

## Development Setup
1. Install the latest stable Rust toolchain.
2. Build the project with `cargo build` (set `LIBTORCH_USE_PYTORCH=1` if PyTorch is installed).
3. Run tests with `cargo test`. There are currently no automated Python tests, but parity checks are encouraged.
4. Format with `cargo fmt` and lint using `cargo clippy` before submitting a PR.

## Workflow
- Create an issue using `.github/ISSUE_TEMPLATE/porting_task.md` for the file you wish to port.
- Work in a feature branch named after the epoch (e.g. `epoch-2-engine`).
- Keep commits focused. One module per PR is preferred.
- When opening a PR, use `.github/PULL_REQUEST_TEMPLATE.md` and link the issue.

## Commit Messages
Include the epoch number and module name in the title, e.g. `Epoch 3: port model_runner`. Describe major decisions in the body.

## Testing
Run `cargo test` before pushing. If you add Python parity tests, run `pytest` as well.

## Documentation
Update `ARCHITECTURE.md` and `docs/porting_plan.md` when new modules are completed. Benchmarks should be recorded in `BENCHMARKS.md`.
