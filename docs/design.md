# design notes

These are personal notes on what's in the codebase and why. Not exhaustive — for the user-facing pitch see `README.md`, for the formal write-up see `paper/quark.tex`.

## what it is

A contrastive learner for quantum circuits. Encode a circuit, get a vector. Equivalent circuits get close vectors, different ones get far apart vectors. Useful as a similarity metric for search, deduplication, and clustering. That is the entire pitch.

## why this and not something hardware-flavoured

Anything noise-aware or hardware-targeted lives or dies on real-hardware fidelity numbers. Without those, the claims don't survive scrutiny. Circuit-as-data is a parallel problem with a cleaner signal: equivalence is a definable property, generating training pairs is mechanical, baselines are obvious (hashes, gate counts), evaluation is standard (Recall@k). It's a smaller story but a more defensible one.

## architecture

A small transformer encoder. Inputs per gate token:

- `gate_id` — integer in [0, 16), one for each of {h, x, y, z, s, t, sx, rx, ry, rz, cx, cz, swap, OTHER_1Q, OTHER_2Q, PAD}. Two fall-through buckets handle exotic gates.
- `q1` — first qubit, 1-indexed (0 = padding or "no qubit")
- `q2` — second qubit, 0 for one-qubit gates
- two parameter floats: `cos(θ)` and `sin(θ)` of the rotation angle (zero for non-parametric gates)

The four features map to a shared `d`-dim space via three embedding tables (gate, q1, q2) and a linear projection (params), then summed. Prepend a `[CLS]` token. Three transformer encoder layers, four heads, `d = 128`, GELU, batch_first, learned positional embeddings.

Padding mask zeros out attention to padded positions. Read off the `[CLS]` hidden state, project, L2-normalise. That's the embedding.

647k parameters at the default config. The architecture is unsurprising on purpose. The job here is to cleanly separate equivalents from non-equivalents in vector space, not invent a new transformer variant.

## training

Symmetric InfoNCE loss with temperature 0.07. Within a minibatch of `B` (anchor, positive) pairs, similarities form a `B × B` matrix where the diagonal is the correct match. Cross-entropy in both directions, summed and averaged. In-batch negatives (the other positives) act as harder negatives than explicit random ones.

Per-step data:

```
anchor   = random_circuit(n in [2, 4], depth in [8, 24])
positive = anchor after k=4 rewrites from {insert_id, cancel_consecutive,
                                            commute_swap, merge_rz, split_rz}
```

Each rewrite is mathematically identity-preserving on the unitary. Verified by `quark.verify.equiv`, which compares full `Operator(qc)` matrices up to global phase. The pair-equivalence test runs over multiple random seeds × small circuits in CI.

AdamW, lr `3e-4` with cosine annealing, weight decay `1e-5`, gradient clipping at 1.0, batch 32, 12 epochs over 1500 generated triples. Total training time: about 5 minutes on CPU.

## why the rewrites work

Each rewrite preserves the underlying unitary by construction. The five chosen ones are the simplest non-trivial identities that don't require deep gate algebra:

- **insert/cancel pairs** of self-inverse gates (HH = I, XX = I, ZZ = I, CNOT-CNOT = I)
- **commute disjoint** — `[U_AB, V_CD] = 0` when `{A,B} ∩ {C,D} = ∅`
- **split/merge Rz** — `Rz(α) Rz(β) = Rz(α + β)` on the same qubit

These five are enough for a non-trivial signal (the synthetic benchmarks demonstrate this) but limited (the QASMBench benchmark exposes how limited). Adding ZX-calculus identities and known compiler patterns is the obvious next direction.

## why the model is so small

The input space is bounded: at most 128 gates, 64 qubits, 16 vocabulary entries. There's no open-ended generative ambiguity here, only structural pattern matching. A 647k-parameter model has enough capacity to learn the patterns visible in the training distribution; making it bigger would overfit the limited unique data. The training set has on the order of a few thousand truly distinct circuit structures even after data augmentation, which constrains useful model size.

If a much larger training corpus becomes available (e.g. millions of real circuits scraped from QML repositories), scaling `d` and `layers` would make sense.

## things this can't do

- **Detect equivalences outside its rewrite repertoire.** Two circuits equivalent by some clever non-local identity will look unrelated to the encoder unless similar identities show up in training somehow. The held-out generalisation result suggests the model does generalise to *related* unseen rewrites (e.g., `insert_id` learned as the inverse of `cancel_consecutive`), but fully novel identities are out of reach.
- **Handle measurements, conditionals, or anything beyond unitary gates.** Encoder vocabulary covers ~13 named gates plus 2 fall-throughs. No `measure`, `reset`, classical-control.
- **Verify equivalence rigorously.** Embedding distance is a heuristic. `verify.equiv` is the ground-truth tool, exponential cost.
- **Process circuits longer than 128 gates without truncation.** This bites on QASMBench transpiled circuits, some of which exceed this length.

## known weaknesses

1. **In-distribution evaluation is intra-distribution.** Train and synthetic test pools are drawn from the same random-circuit / rewrite distribution. The held-out experiment partially addresses this but only for held-out rewrites, not held-out circuit families.
2. **Random negatives are too easy.** Most random circuits look very different from the anchor. Hard-negative mining (e.g. nearest-neighbour negatives in the current embedding space) would push performance higher.
3. **Brute-force verifier caps the verifiable circuit size at ~10 qubits.** Beyond that we have to trust pair-generation correctness without per-pair verification at training time.
4. **No phase tracking on Rz splits.** `split_rz` divides the angle in two; if some downstream consumer cared about global phase this would diverge. The encoder ignores global phase, so the inconsistency is invisible to it, but a user who cares about phase needs to know.
5. **PennyLane / Cirq adapter coverage is partial.** Common gates work; exotic gate sets fall through to OTHER_1Q / OTHER_2Q tokens, losing information.

## what next

Roughly in order of expected value:

- Hard-negative mining (concrete: rerank within batch by similarity, take top-K as additional negatives)
- More rewrite identities — ZX-calculus phase fusion, T-count reductions, KAK decompositions
- Evaluation on a real-world circuit corpus beyond QASMBench (e.g. OpenFermion ansatzes, QFT variants from textbooks)
- Sequence-length scaling — either retrain at `max_len = 256` or add chunked encoding
- Measurement and classical-control support
- A FAISS index in `quark.dedupe` so the directory CLI scales beyond a few thousand circuits

If any of these change benchmark numbers materially, the README and this doc get updated.
