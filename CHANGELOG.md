# changelog

## v0.3.0 (06-05-2026)

Initial public release.

- 647k-parameter transformer encoder, contrastive (InfoNCE) training on synthetic equivalence pairs
- Five equivalence-preserving rewrites: insert/cancel HH-XX-CNOT pairs, commute disjoint gates, split/merge Rz angles
- Three retrieval evaluations: in-distribution synthetic (R@10 = 1.000), held-out rewrite (R@10 = 0.935 vs 0.290 baseline), QASMBench out-of-distribution (R@10 = 0.345 vs 0.172)
- `quark dedupe <directory>` CLI for grouping equivalent QASM files (default cosine threshold 0.9)
- `quark bench` for a fast end-to-end smoke benchmark, also run in CI
- PennyLane and Cirq adapters, with arbitrary wire labels supported and unrecognised gates reported instead of silently dropped
- Pretrained weights loaded with `weights_only=True`; a warning is emitted when a circuit is truncated to the 128-gate context window
- 24 tests, all passing
- arXiv-ready preprint scaffold in `paper/`
- Streamlit demo in `demo/`
- Quickstart notebook in `examples/`
