# changelog

## v0.3.0 (06-05-2026)

Initial public release.

- 647k-parameter transformer encoder, contrastive (InfoNCE) training on synthetic equivalence pairs
- Five equivalence-preserving rewrites: insert/cancel HH-XX-CNOT pairs, commute disjoint gates, split/merge Rz angles
- Three retrieval evaluations: in-distribution synthetic (R@10 = 1.000), held-out rewrite (R@10 = 0.935 vs 0.290 baseline), QASMBench out-of-distribution (R@10 = 0.345 vs 0.172)
- `quark dedupe <directory>` CLI for grouping equivalent QASM files
- PennyLane and Cirq adapters
- 24 tests, all passing
- arXiv-ready preprint scaffold in `paper/`
- Streamlit demo in `demo/`
- Quickstart notebook in `examples/`
