# Roadmap

This roadmap summarizes the development schedule for the Rust rewrite. Cycles group related epochs and culminate in a milestone where Tiny-vLLM reaches a usable state.

## Cycle 0 – Setup *(completed)*
- Repository scaffolding
- PyO3 bridge verification

## Cycle 1 – Core Engine MVP
Goal: basic offline generation for a single prompt.
- Epoch 1: configuration module
- Epoch 2: synchronous engine
- Epoch 3: model loader
- Epoch 4: sampling utilities
- Epoch 5: end-to-end parity test

## Cycle 2 – Parallelism
Goal: handle multiple requests with batching.
- Epoch 6: async engine
- Epoch 7: scheduler
- Epoch 8: attention cache
- Epoch 9: streaming outputs
- Epoch 10: throughput and load testing

## Cycle 3 – Feature Parity
Goal: match vLLM features and tune performance.
- Epoch 11: HTTP server
- Epoch 12: advanced model support
- Epoch 13: extensive tests
- Epoch 14: performance tuning

Future cycles may include pure Rust tokenizers, fine tuning support, and broader ecosystem integration. Refer to `docs/porting_plan.md` for detailed tasks.
